# Table of Contents: every heading, with suggested titles

Generated 2026-07-28 for quality-control review. Covers **2,186 headings**
across all 30 chapters, at every level (`##` = N.M.K, `###` = N.M.K.L,
`####` = N.M.K.L.P). Slide sections and code fences are excluded; they are not
in the book's table of contents.

## How to read this

Every heading in the book gets a row, including the good ones, so this doubles
as the full TOC. A `—` in **Suggested title** means the current title already
names its content and no change is proposed. Each suggestion was made after
reading the actual prose under that heading, not from the heading text alone.

## The standard applied

The brief was your critique verbatim: a title has to be meaningful *in the
table of contents*, as a noun phrase a reader can understand without the
preceding paragraph. Rejected forms: half-sentences, trailing `, and why it
matters` clauses, literal questions, bare pronouns, and teasers. Vivid titles
were kept where they genuinely name their content ("Diffusion Is Transport in
Disguise"); the defect is meaninglessness out of context, not personality.

## Where the problems are

| Chapters | Headings | Suggested | Rate |
|---|---:|---:|---:|
| 1–2 Preliminaries, Linear Regression | 127 | 2 | 1.6% |
| 3–4 Linear Classification, MLP | 128 | 3 | 2.3% |
| 5–6 Computation, Convnets | 108 | 5 | 4.6% |
| 7–8 Modern Convnets, Sequence Models | 173 | 11 | 6.4% |
| 9 Optimization | 127 | 7 | 5.5% |
| 10–11 Attention, Transformers | 163 | 17 | 10.4% |
| **12–13 State Space Models, Performance** | **148** | **34** | **23.0%** |
| 14–15 RL, Deep RL | 250 | 7 | 2.8% |
| 16–19 GANs, Diffusion, Language Models | 139 | 3 | 2.2% |
| 20–23 Image Models, Attic | 209 | 8 | 3.8% |
| 24–26 Linear Algebra, Calculus, Optimization | 246 | 19 | 7.7% |
| 27–30 Probability, Info Theory, Dynamics, Tools | 368 | 11 | 3.0% |
| **Total** | **2,186** | **127** | **5.8%** |

The defect tracks how recently a chapter was written, not its subject. The
classic d2l chapters are near-clean; the newest passes carry it. Chapter 12
(State Space Models) is the outlier at 23%, and it repeats your exact flagged
patterns: "What the Hardware Bought" echoes "What It Bought" almost verbatim,
and "The Ladder, and Its Ceiling" is the same trailing-clause half-sentence.

Chapters 14–15 read low here because they were already rewritten in response to
your critique; that section of this document is a verification pass, and it
confirms all five titles you quoted now name their content. It found one
straggler (`sac.md`'s bare "The Algorithm") and raises one judgment call
(`offline-rl.md`'s file title, phrased as an indirect question).

## Recurring defects worth a global decision

1. **"What X Bought/Costs/Leaves Open"** — appears in chapters 12, 24 and 26
   after being flagged in 15.
2. **Literal questions** — "Why Notebooks?", "Colab or Kaggle?", "Why Not
   Newton?", "Do the Position Embeddings Discover the Grid?", "Why
   'cross-entropy'?", "When Does an Eigenbasis Exist?".
3. **Trailing dependent clauses** — ", Measured", ", and Its Ceiling", ", and
   Where This Leaves Us", ", and the General Fact".
4. **Title Case drift** — `convnext.md` (7 headings) and Recommender Systems
   (5) use sentence case against the book's Title Case convention.
5. **Duplicate titles at adjacent levels** — `mamba.md` §12.3.2 repeats its
   parent; `bert-pretraining.md` §18.11.1 repeats its file title.

---

# Chapter 1 — Preliminaries, Chapter 2 — Linear Regression in Neural Networks

## Chapter 1 front matter <sub>`chapter_preliminaries/index.md`</sub>

| # | Current title | Suggested title | Why |
|---|---|---|---|
| (unnumbered) | Resources and Further Reading | — | boilerplate, book-wide convention |

### 1.1 Data Manipulation <sub>`chapter_preliminaries/ndarray.md`</sub>

| # | Current title | Suggested title | Why |
|---|---|---|---|
| 1.1.1 | Getting Started | — | already descriptive |
| 1.1.2 | Indexing and Slicing | — | already descriptive |
| 1.1.3 | Operations | — | already descriptive |
| 1.1.4 | Broadcasting | — | already descriptive |
| 1.1.5 | Saving Memory | — | already descriptive |
| 1.1.6 | Conversion to Other Python Objects | — | already descriptive |
| 1.1.7 | Discussion | — | book-wide convention |
| 1.1.8 | Exercises | — | book-wide convention |

### 1.2 Data Preprocessing <sub>`chapter_preliminaries/pandas.md`</sub>

| # | Current title | Suggested title | Why |
|---|---|---|---|
| 1.2.1 | Load and Inspect | — | already descriptive |
| 1.2.1.1 | Reading the Dataset | — | already descriptive |
| 1.2.1.2 | Knowing Your Data | — | already descriptive |
| 1.2.1.3 | Separating Inputs and Targets | — | already descriptive |
| 1.2.2 | Clean and Encode | — | already descriptive |
| 1.2.2.1 | Handling Missing Values | — | already descriptive |
| 1.2.2.2 | Encoding Categorical Features | — | already descriptive |
| 1.2.3 | Scale and Convert | — | already descriptive |
| 1.2.3.1 | Numerical Features | — | already descriptive |
| 1.2.3.2 | Conversion to Tensor Format | — | already descriptive |
| 1.2.4 | Discussion | — | book-wide convention |
| 1.2.5 | Exercises | — | book-wide convention |

### 1.3 Linear Algebra <sub>`chapter_preliminaries/linear-algebra.md`</sub>

| # | Current title | Suggested title | Why |
|---|---|---|---|
| 1.3.1 | The Objects | — | names the content (scalars…tensors) |
| 1.3.1.1 | Scalars | — | already descriptive |
| 1.3.1.2 | Vectors | — | already descriptive |
| 1.3.1.3 | Matrices | — | already descriptive |
| 1.3.1.4 | Tensors | — | already descriptive |
| 1.3.2 | Arithmetic and Reductions | — | already descriptive |
| 1.3.2.1 | Basic Properties of Tensor Arithmetic | — | already descriptive |
| 1.3.2.2 | Reduction | — | already descriptive |
| 1.3.2.3 | Non-Reduction Sum | — | already descriptive |
| 1.3.3 | Products | — | already descriptive |
| 1.3.3.1 | Dot Products | — | already descriptive |
| 1.3.3.2 | Matrix–Vector Products | — | already descriptive |
| 1.3.3.3 | Matrix–Matrix Multiplication | — | already descriptive |
| 1.3.4 | Norms | — | already descriptive |
| 1.3.4.1 | Eigenvalues: A First Look | — | names topic + honest scope |
| 1.3.5 | Discussion | — | book-wide convention |
| 1.3.6 | Exercises | — | book-wide convention |

### 1.4 Calculus <sub>`chapter_preliminaries/calculus.md`</sub>

| # | Current title | Suggested title | Why |
|---|---|---|---|
| 1.4.1 | Derivatives and Differentiation | — | already descriptive |
| 1.4.2 | Partial Derivatives and Gradients | — | already descriptive |
| 1.4.3 | Chain Rule | — | already descriptive |
| 1.4.3.1 | Plotting Utilities for This Book | — | matches content (matplotlib helpers) |
| 1.4.4 | Discussion | — | book-wide convention |
| 1.4.5 | Exercises | — | book-wide convention |

### 1.5 Automatic Differentiation <sub>`chapter_preliminaries/autograd.md`</sub>

| # | Current title | Suggested title | Why |
|---|---|---|---|
| 1.5.1 | Mechanics | — | already descriptive |
| 1.5.1.1 | A Simple Function | — | already descriptive |
| 1.5.1.2 | Backward for Non-Scalar Variables | — | already descriptive |
| 1.5.2 | Controlling the Graph | — | already descriptive |
| 1.5.2.1 | Detaching Computation | — | already descriptive |
| 1.5.2.2 | Turning Off Gradient Tracking | — | already descriptive |
| 1.5.3 | Beyond the Basics | — | umbrella for 3 varied subsections; acceptable |
| 1.5.3.1 | Gradients and Python Control Flow | — | already descriptive |
| 1.5.3.2 | Higher-Order Derivatives | — | already descriptive |
| 1.5.3.3 | Forward versus Reverse Mode | — | already descriptive |
| 1.5.4 | Discussion | — | book-wide convention |
| 1.5.5 | Exercises | — | book-wide convention |

### 1.6 Probability and Statistics <sub>`chapter_preliminaries/probability.md`</sub>

| # | Current title | Suggested title | Why |
|---|---|---|---|
| 1.6.1 | A Simple Example: Tossing Coins | — | already descriptive |
| 1.6.2 | The Formal Language | The Formal Language of Probability | vague alone; "of Probability" names topic |
| 1.6.2.1 | A More Formal Treatment | Sample Spaces, Events, and Axioms | meaningless out of context; names actual content |
| 1.6.2.2 | Random Variables | — | already descriptive |
| 1.6.3 | Multiple Random Variables | — | already descriptive |
| 1.6.4 | Worked Example: HIV Testing | — | already descriptive |
| 1.6.5 | Expectations | — | already descriptive |
| 1.6.5.1 | From Means to Tail Bounds | — | both endpoints are real technical terms |
| 1.6.6 | Discussion | — | book-wide convention |
| 1.6.7 | Exercises | — | book-wide convention |

### 1.7 Documentation <sub>`chapter_preliminaries/lookup-api.md`</sub>

| # | Current title | Suggested title | Why |
|---|---|---|---|
| 1.7.1 | Discovering What Exists: `dir` | — | already descriptive |
| 1.7.2 | Reading the Signature: `help`, `?`, and `??` | — | already descriptive |
| 1.7.3 | Verifying With a Quick Run | — | already descriptive |
| 1.7.4 | Exercises | — | book-wide convention |

## Chapter 2 front matter <sub>`chapter_linear-regression/index.md`</sub>

| # | Current title | Suggested title | Why |
|---|---|---|---|
| (unnumbered) | Resources and Further Reading | — | boilerplate, book-wide convention |

### 2.1 Linear Regression <sub>`chapter_linear-regression/linear-regression.md`</sub>

| # | Current title | Suggested title | Why |
|---|---|---|---|
| 2.1.1 | Basics | — | names the section's scope (model, loss, solution) |
| 2.1.1.1 | Model | — | already descriptive |
| 2.1.1.2 | Loss Function | — | already descriptive |
| 2.1.1.3 | Analytic Solution | — | already descriptive |
| 2.1.1.4 | Minibatch Stochastic Gradient Descent | — | already descriptive |
| 2.1.1.5 | Predictions | — | already descriptive |
| 2.1.2 | Vectorization for Speed | — | already descriptive |
| 2.1.3 | The Normal Distribution and Squared Loss | — | already descriptive |
| 2.1.3.1 | A Menu of Losses | — | vivid but names content (table of losses) |
| 2.1.4 | Linear Regression as a Neural Network | — | already descriptive |
| 2.1.4.1 | Biology | — | matches content (biological neuron) |
| 2.1.5 | Summary | — | book-wide convention |
| 2.1.6 | Exercises | — | book-wide convention |

### 2.2 Object-Oriented Design for Implementation <sub>`chapter_linear-regression/oo-design.md`</sub>

| # | Current title | Suggested title | Why |
|---|---|---|---|
| 2.2.1 | Utilities | — | already descriptive |
| 2.2.2 | Models | — | already descriptive |
| 2.2.3 | Data | — | already descriptive |
| 2.2.4 | Training | — | already descriptive |
| 2.2.5 | Summary | — | book-wide convention |
| 2.2.6 | Exercises | — | book-wide convention |

### 2.3 Synthetic Regression Data <sub>`chapter_linear-regression/synthetic-regression-data.md`</sub>

| # | Current title | Suggested title | Why |
|---|---|---|---|
| 2.3.1 | Generating the Dataset | — | already descriptive |
| 2.3.2 | Reading the Dataset | — | already descriptive |
| 2.3.3 | Concise Implementation of the Data Loader | — | already descriptive |
| 2.3.4 | Summary | — | book-wide convention |
| 2.3.5 | Exercises | — | book-wide convention |

### 2.4 Linear Regression Implementation from Scratch <sub>`chapter_linear-regression/linear-regression-scratch.md`</sub>

| # | Current title | Suggested title | Why |
|---|---|---|---|
| 2.4.1 | Defining the Model | — | already descriptive |
| 2.4.2 | Defining the Loss Function | — | already descriptive |
| 2.4.3 | Defining the Optimization Algorithm | — | already descriptive |
| 2.4.4 | Training | — | already descriptive |
| 2.4.5 | Summary | — | book-wide convention |
| 2.4.6 | Exercises | — | book-wide convention |

### 2.5 Concise Implementation of Linear Regression <sub>`chapter_linear-regression/linear-regression-concise.md`</sub>

| # | Current title | Suggested title | Why |
|---|---|---|---|
| 2.5.1 | Defining the Model | — | already descriptive |
| 2.5.2 | Defining the Loss Function | — | already descriptive |
| 2.5.3 | Defining the Optimization Algorithm | — | already descriptive |
| 2.5.4 | Training | — | already descriptive |
| 2.5.5 | Summary | — | book-wide convention |
| 2.5.6 | Exercises | — | book-wide convention |

### 2.6 Generalization <sub>`chapter_linear-regression/generalization.md`</sub>

| # | Current title | Suggested title | Why |
|---|---|---|---|
| 2.6.1 | Training Error and Generalization Error | — | already descriptive |
| 2.6.1.1 | Model Complexity | — | already descriptive |
| 2.6.2 | Underfitting or Overfitting? | — | both terms named; question form matches book convention |
| 2.6.2.1 | Polynomial Curve Fitting | — | already descriptive |
| 2.6.2.2 | Dataset Size | — | already descriptive |
| 2.6.3 | Model Selection | — | already descriptive |
| 2.6.3.1 | Cross-Validation | — | already descriptive |
| 2.6.4 | Summary | — | book-wide convention |
| 2.6.5 | Exercises | — | book-wide convention |

### 2.7 Weight Decay <sub>`chapter_linear-regression/weight-decay.md`</sub>

| # | Current title | Suggested title | Why |
|---|---|---|---|
| 2.7.1 | Norms and Weight Decay | — | already descriptive |
| 2.7.2 | High-Dimensional Linear Regression | — | already descriptive |
| 2.7.3 | Implementation from Scratch | — | already descriptive |
| 2.7.3.1 | Defining $\ell_2$ Norm Penalty | — | already descriptive |
| 2.7.3.2 | Defining the Model | — | already descriptive |
| 2.7.3.3 | Training without Regularization | — | already descriptive |
| 2.7.3.4 | Using Weight Decay | — | already descriptive |
| 2.7.3.5 | Why Shrinkage Helps: The Spectral View | — | names mechanism + method (SVD-based) |
| 2.7.4 | Concise Implementation | — | already descriptive |
| 2.7.5 | Summary | — | book-wide convention |
| 2.7.6 | Exercises | — | book-wide convention |

# TOC title review — Chapter 3 (Linear Classification) and Chapter 4 (Multilayer Perceptrons)

## Chapter 3: Linear Classification in Neural Networks

### 3 Linear Classification in Neural Networks (chapter opener)  <sub>`chapter_linear-classification/index.md`</sub>

| # | Current title | Suggested title | Why |
|---|---|---|---|
| (unnumbered) | Resources and Further Reading | — | standard boilerplate, unnumbered, descriptive |

### 3.1 Softmax Regression  <sub>`chapter_linear-classification/softmax-regression.md`</sub>

| # | Current title | Suggested title | Why |
|---|---|---|---|
| 3.1.1 | Classification | — | already descriptive |
| 3.1.1.1 | Linear Model | — | already descriptive |
| 3.1.2 | The Softmax Model | — | already descriptive |
| 3.1.2.1 | Vectorization | — | already descriptive |
| 3.1.3 | Loss Function | — | already descriptive |
| 3.1.3.1 | Log-Likelihood | — | already descriptive |
| 3.1.3.2 | Softmax and Cross-Entropy Loss | — | already descriptive |
| 3.1.3.2.1 | Why "cross-entropy"? | The Information-Theoretic Origin of Cross-Entropy | question form; names the topic (entropy/coding cost), not a teaser |
| 3.1.4 | Summary and Discussion | — | standard, descriptive |
| 3.1.5 | Exercises | — | standard, descriptive |

### 3.2 The Image Classification Dataset  <sub>`chapter_linear-classification/image-classification-dataset.md`</sub>

| # | Current title | Suggested title | Why |
|---|---|---|---|
| 3.2.1 | Loading the Dataset | — | already descriptive |
| 3.2.2 | Reading a Minibatch | — | already descriptive |
| 3.2.3 | Visualization | — | already descriptive |
| 3.2.4 | Summary | — | standard, descriptive |
| 3.2.5 | Exercises | — | standard, descriptive |

### 3.3 The Base Classification Model  <sub>`chapter_linear-classification/classification.md`</sub>

| # | Current title | Suggested title | Why |
|---|---|---|---|
| 3.3.1 | The `Classifier` Class | — | already descriptive |
| 3.3.2 | Accuracy | — | already descriptive |
| 3.3.3 | Beyond Accuracy | — | names the topic (precision/recall/confusion matrix) |
| 3.3.4 | Summary | — | standard, descriptive |
| 3.3.5 | Exercises | — | standard, descriptive |

### 3.4 Softmax Regression Implementation from Scratch  <sub>`chapter_linear-classification/softmax-regression-scratch.md`</sub>

| # | Current title | Suggested title | Why |
|---|---|---|---|
| 3.4.1 | The Softmax | — | already descriptive |
| 3.4.2 | The Model | — | already descriptive |
| 3.4.3 | The Cross-Entropy Loss | — | already descriptive |
| 3.4.4 | Training | — | already descriptive |
| 3.4.5 | Prediction | — | already descriptive |
| 3.4.6 | Summary and Discussion | — | standard, descriptive |
| 3.4.7 | Exercises | — | standard, descriptive |

### 3.5 Concise Implementation of Softmax Regression  <sub>`chapter_linear-classification/softmax-regression-concise.md`</sub>

| # | Current title | Suggested title | Why |
|---|---|---|---|
| 3.5.1 | Defining the Model | — | already descriptive |
| 3.5.2 | Softmax Revisited | — | already descriptive (numerical-stability rewrite) |
| 3.5.3 | Training | — | already descriptive |
| 3.5.4 | Summary | — | standard, descriptive |
| 3.5.5 | Exercises | — | standard, descriptive |

### 3.6 Generalization in Classification  <sub>`chapter_linear-classification/generalization-classification.md`</sub>

| # | Current title | Suggested title | Why |
|---|---|---|---|
| 3.6.1 | The Test Set | — | already descriptive |
| 3.6.2 | Test Set Reuse | — | already descriptive |
| 3.6.3 | Statistical Learning Theory | — | already descriptive |
| 3.6.4 | Summary | — | standard, descriptive |
| 3.6.5 | Exercises | — | standard, descriptive |

### 3.7 Environment and Distribution Shift  <sub>`chapter_linear-classification/environment-and-distribution-shift.md`</sub>

| # | Current title | Suggested title | Why |
|---|---|---|---|
| 3.7.1 | Types of Distribution Shift | — | already descriptive |
| 3.7.1.1 | Covariate Shift | — | already descriptive |
| 3.7.1.2 | Label Shift | — | already descriptive |
| 3.7.1.3 | Concept Shift | — | already descriptive |
| 3.7.2 | Examples of Distribution Shift | — | already descriptive |
| 3.7.2.1 | Medical Diagnostics | — | already descriptive |
| 3.7.2.2 | Self-Driving Cars | — | already descriptive |
| 3.7.2.3 | Nonstationary Distributions | — | already descriptive |
| 3.7.2.4 | Further Failure Modes | — | names content: more failure case studies |
| 3.7.3 | Correction of Distribution Shift | — | already descriptive |
| 3.7.3.1 | Covariate Shift Correction | — | already descriptive |
| 3.7.3.1.1 | Covariate Shift Correction in Code | — | already descriptive |
| 3.7.3.2 | Label Shift Correction | — | already descriptive |
| 3.7.3.3 | Concept Shift Correction | — | already descriptive |
| 3.7.4 | A Taxonomy of Learning Problems | — | already descriptive |
| 3.7.4.1 | Batch Learning | — | already descriptive |
| 3.7.4.2 | Online Learning | — | already descriptive |
| 3.7.4.3 | Bandits | — | already descriptive |
| 3.7.4.4 | Control | — | already descriptive |
| 3.7.4.5 | Reinforcement Learning | — | already descriptive |
| 3.7.4.6 | Considering the Environment | — | names topic: strategy vs. an adapting environment |
| 3.7.5 | Fairness, Accountability, and Transparency in Machine Learning | — | already descriptive |
| 3.7.6 | Summary | — | standard, descriptive |
| 3.7.7 | Exercises | — | standard, descriptive |

## Chapter 4: Multilayer Perceptrons

### 4 Multilayer Perceptron (chapter opener)  <sub>`chapter_multilayer-perceptrons/index.md`</sub>

| # | Current title | Suggested title | Why |
|---|---|---|---|
| (unnumbered) | Resources and Further Reading | — | standard boilerplate, unnumbered, descriptive |

### 4.1 Multilayer Perceptrons  <sub>`chapter_multilayer-perceptrons/mlp.md`</sub>

| # | Current title | Suggested title | Why |
|---|---|---|---|
| 4.1.1 | Hidden Layers | — | already descriptive |
| 4.1.1.1 | Limitations of Linear Models | — | already descriptive |
| 4.1.1.2 | Incorporating Hidden Layers | — | already descriptive |
| 4.1.1.3 | From Linear to Nonlinear | — | already descriptive |
| 4.1.1.4 | A Concrete Win: XOR | Solving XOR with a Hidden Layer | "a win" is vague out of context; name the actual construction |
| 4.1.1.5 | Universal Approximators | — | already descriptive |
| 4.1.2 | Activation Functions | — | already descriptive |
| 4.1.2.1 | ReLU Function | — | already descriptive |
| 4.1.2.2 | Sigmoid Function | — | already descriptive |
| 4.1.2.3 | Tanh Function | — | already descriptive |
| 4.1.3 | Summary and Discussion | — | standard, descriptive |
| 4.1.4 | Exercises | — | standard, descriptive |

### 4.2 Implementation of Multilayer Perceptrons  <sub>`chapter_multilayer-perceptrons/mlp-implementation.md`</sub>

| # | Current title | Suggested title | Why |
|---|---|---|---|
| 4.2.1 | Implementation from Scratch | — | already descriptive |
| 4.2.1.1 | Initializing Model Parameters | — | already descriptive |
| 4.2.1.2 | Model | — | already descriptive |
| 4.2.1.3 | Training | — | already descriptive |
| 4.2.2 | Concise Implementation | — | already descriptive |
| 4.2.2.1 | Model | — | already descriptive |
| 4.2.2.2 | Training | — | already descriptive |
| 4.2.3 | Summary | — | standard, descriptive |
| 4.2.4 | Exercises | — | standard, descriptive |

### 4.3 Forward Propagation, Backward Propagation, and Computational Graphs  <sub>`chapter_multilayer-perceptrons/backprop.md`</sub>

| # | Current title | Suggested title | Why |
|---|---|---|---|
| 4.3.1 | Forward Propagation | — | already descriptive |
| 4.3.2 | Computational Graph of Forward Propagation | — | already descriptive |
| 4.3.3 | Backpropagation | — | already descriptive |
| 4.3.3.1 | A Worked Example | — | already descriptive |
| 4.3.3.2 | From the Chain Rule to Autograd | — | already descriptive |
| 4.3.4 | Training Neural Networks | — | already descriptive |
| 4.3.5 | Summary | — | standard, descriptive |
| 4.3.6 | Exercises | — | standard, descriptive |

### 4.4 Numerical Stability and Initialization  <sub>`chapter_multilayer-perceptrons/numerical-stability-and-init.md`</sub>

| # | Current title | Suggested title | Why |
|---|---|---|---|
| 4.4.1 | Vanishing and Exploding Gradients | — | already descriptive |
| 4.4.1.1 | Vanishing Gradients | — | already descriptive |
| 4.4.1.2 | Exploding Gradients | — | already descriptive |
| 4.4.1.3 | Breaking the Symmetry | — | already descriptive |
| 4.4.2 | Parameter Initialization | — | already descriptive |
| 4.4.2.1 | Default Initialization | — | already descriptive |
| 4.4.2.2 | Xavier Initialization | — | already descriptive |
| 4.4.2.3 | He Initialization | — | already descriptive |
| 4.4.2.4 | Watching the Variance Propagate | — | descriptive gerund phrase, matches book convention (cf. "Loading the Dataset") |
| 4.4.2.5 | Beyond | Further Initialization Heuristics | one word, meaningless out of context; names the actual content |
| 4.4.3 | Summary | — | standard, descriptive |
| 4.4.4 | Exercises | — | standard, descriptive |

### 4.5 Generalization in Deep Learning  <sub>`chapter_multilayer-perceptrons/generalization-deep.md`</sub>

| # | Current title | Suggested title | Why |
|---|---|---|---|
| 4.5.1 | Revisiting Overfitting and Regularization | — | already descriptive |
| 4.5.1.1 | Double Descent | — | already descriptive |
| 4.5.2 | Inspiration from Nonparametrics | — | already descriptive |
| 4.5.3 | Early Stopping | — | already descriptive |
| 4.5.4 | Classical Regularization Methods for Deep Networks | — | already descriptive |
| 4.5.4.1 | Implicit Regularization | — | already descriptive |
| 4.5.5 | Summary | — | standard, descriptive |
| 4.5.6 | Exercises | — | standard, descriptive |

### 4.6 Dropout  <sub>`chapter_multilayer-perceptrons/dropout.md`</sub>

| # | Current title | Suggested title | Why |
|---|---|---|---|
| 4.6.1 | Dropout in Practice | — | already descriptive |
| 4.6.2 | Implementation from Scratch | — | already descriptive |
| 4.6.2.1 | Defining the Model | — | already descriptive |
| 4.6.2.2 | Training | — | already descriptive |
| 4.6.3 | Concise Implementation | — | already descriptive |
| 4.6.4 | Summary | — | standard, descriptive |
| 4.6.5 | Exercises | — | standard, descriptive |

### 4.7 Predicting House Prices on Kaggle  <sub>`chapter_multilayer-perceptrons/kaggle-house-price.md`</sub>

| # | Current title | Suggested title | Why |
|---|---|---|---|
| 4.7.1 | Kaggle | — | intro to the platform; clear in context |
| 4.7.2 | Accessing and Reading the Dataset | — | already descriptive |
| 4.7.3 | Data Preprocessing | — | already descriptive |
| 4.7.4 | Error Measure | — | already descriptive |
| 4.7.5 | $K$-Fold Cross-Validation | — | already descriptive |
| 4.7.6 | Model Selection | — | already descriptive |
| 4.7.7 | Submitting Predictions on Kaggle | — | already descriptive |
| 4.7.8 | Summary and Discussion | — | standard, descriptive |
| 4.7.9 | Exercises | — | standard, descriptive |

# TOC title review — Chapters 5 (Builder's Guide) and 6 (Convolutional Neural Networks)

## Chapter 5: Computation (Builder's Guide)

### 5 Computation  <sub>`chapter_builders-guide/index.md`</sub>

| # | Current title | Suggested title | Why |
|---|---|---|---|
| (unnumbered) | Resources and Further Reading | — | standard boilerplate heading |

### 5.1 Modules and Model Construction  <sub>`chapter_builders-guide/model-construction.md`</sub>

| # | Current title | Suggested title | Why |
|---|---|---|---|
| 5.1.1 | The Module Abstraction | — | already descriptive |
| 5.1.2 | Sequential and Friends: Containers | — | vivid but names the topic (containers) |
| 5.1.3 | Forward Is Just Python | — | states the section's actual claim |
| 5.1.4 | Lazy Initialization: Shapes from Data | — | already descriptive |
| 5.1.5 | Building from a Config | — | already descriptive |
| 5.1.6 | Summary | — | standard |
| 5.1.7 | Exercises | — | standard |

### 5.2 Parameters, State, and Memory  <sub>`chapter_builders-guide/parameters-state-memory.md`</sub>

| # | Current title | Suggested title | Why |
|---|---|---|---|
| 5.2.1 | Accessing Parameters | — | already descriptive |
| 5.2.2 | Parameters and Buffers | — | already descriptive |
| 5.2.3 | Counting Parameters, Counting Bytes | — | names the two computations done |
| 5.2.4 | Tied Parameters | — | already descriptive |
| 5.2.5 | Freezing Parameters | — | already descriptive |
| 5.2.6 | Summary | — | standard |
| 5.2.7 | Exercises | — | standard |

### 5.3 Initialization  <sub>`chapter_builders-guide/init.md`</sub>

| # | Current title | Suggested title | Why |
|---|---|---|---|
| 5.3.1 | Defaults and When to Override Them | — | already descriptive |
| 5.3.2 | Applying Initializers | — | already descriptive |
| 5.3.3 | Modern Schemes: Truncation, Depth, and Zeros | — | names all three subtopics |
| 5.3.3.1 | Truncated Normals | — | already descriptive |
| 5.3.3.2 | Scaling Down Residual Branches | — | already descriptive |
| 5.3.3.3 | Starting a Block at Zero | — | matches the zero-init technique taught |
| 5.3.3.4 | Watching the Variance Compound | — | names the experiment (variance vs. depth) |
| 5.3.4 | Custom Initializers | — | already descriptive |
| 5.3.5 | Summary | — | standard |
| 5.3.6 | Exercises | — | standard |

### 5.4 Custom Layers and Functions  <sub>`chapter_builders-guide/custom-layers.md`</sub>

| # | Current title | Suggested title | Why |
|---|---|---|---|
| 5.4.1 | Layers without Parameters | — | already descriptive |
| 5.4.2 | Layers with Parameters: RMSNorm | — | already descriptive |
| 5.4.2.1 | The Composability Guarantee | — | matches the four properties verified |
| 5.4.2.2 | Checking against the Built-in | — | matches the reference comparison |
| 5.4.3 | Precomputed State: Buffers | — | already descriptive |
| 5.4.4 | Custom Gradients | — | already descriptive |
| 5.4.5 | Summary | — | standard |
| 5.4.6 | Exercises | — | standard |

### 5.5 Numerics: Dtypes and Mixed Precision  <sub>`chapter_builders-guide/numerics.md`</sub>

| # | Current title | Suggested title | Why |
|---|---|---|---|
| 5.5.1 | The Dtype Zoo | — | names the topic (survey of dtypes) |
| 5.5.1.1 | TF32: What Happens to fp32 Matrix Multiplication | TF32: A Reduced-Precision Matmul Mode | question-shaped; name the mechanism, not a "what happens" clause |
| 5.5.1.2 | Below 16 Bits | fp8 and int8: Below the 16-Bit Floor | names the actual formats discussed |
| 5.5.2 | Dtype Rules: Promotion, Parameters, and Casts | — | already descriptive |
| 5.5.3 | Mixed-Precision Training | — | already descriptive |
| 5.5.3.1 | Loss Scaling for fp16 | — | already descriptive |
| 5.5.4 | When Numerics Bite | — | vivid but names the topic (debugging field guide) |
| 5.5.5 | Summary | — | standard |
| 5.5.6 | Exercises | — | standard |

### 5.6 Saving, Loading, and Pretrained Weights  <sub>`chapter_builders-guide/saving-loading.md`</sub>

| # | Current title | Suggested title | Why |
|---|---|---|---|
| 5.6.1 | State, Not Code | — | states the core distinction taught |
| 5.6.2 | safetensors: the Interchange Format | — | already descriptive |
| 5.6.3 | Checkpointing a Training Run | — | already descriptive |
| 5.6.4 | Loading Weights You Did Not Train | — | already descriptive |
| 5.6.5 | Summary | — | standard |
| 5.6.6 | Exercises | — | standard |

### 5.7 GPUs, Devices, and Memory  <sub>`chapter_builders-guide/gpus-devices-memory.md`</sub>

| # | Current title | Suggested title | Why |
|---|---|---|---|
| 5.7.1 | Devices | — | already descriptive |
| 5.7.2 | Tensors, Models, and Devices | — | already descriptive |
| 5.7.2.1 | Copying Between Devices | — | already descriptive |
| 5.7.2.2 | Models on a Device | — | already descriptive |
| 5.7.3 | GPU Memory | — | already descriptive |
| 5.7.3.1 | What Fills Memory During Training | — | already descriptive |
| 5.7.3.2 | Trading Compute for Memory: Activation Checkpointing | — | names the technique explicitly |
| 5.7.4 | Don't Break the Pipeline | Asynchronous Dispatch and Synchronization Points | admonition, not a topic name; names the mechanism instead |
| 5.7.5 | The Trainer, Now with Devices | — | names what changes (Trainer gains device support) |
| 5.7.6 | Summary | — | standard |
| 5.7.7 | Exercises | — | standard |

### 5.8 Reproducibility and Inspection  <sub>`chapter_builders-guide/reproducibility-inspection.md`</sub>

| # | Current title | Suggested title | Why |
|---|---|---|---|
| 5.8.1 | Seeds and Randomness | — | already descriptive |
| 5.8.1.1 | Generator Objects | — | already descriptive |
| 5.8.1.2 | DataLoader Workers | — | already descriptive |
| 5.8.1.3 | Randomness as a Value | — | names the explicit-PRNG design principle |
| 5.8.2 | Determinism and Its Price | — | names the trade-off (speed/error cost) |
| 5.8.3 | Hooks: Looking Inside | — | already descriptive |
| 5.8.3.1 | Capturing Activation Statistics | — | already descriptive |
| 5.8.3.2 | A NaN Finder | — | matches the tool built |
| 5.8.3.3 | Backward Hooks and Beyond | Backward Hooks and Gradient Inspection | "and Beyond" is filler; name the actual second topic |
| 5.8.4 | Summary | — | standard |
| 5.8.5 | Exercises | — | standard |

## Chapter 6: Convolutional Neural Networks

### 6 Convolutional Neural Networks  <sub>`chapter_convolutional-neural-networks/index.md`</sub>

| # | Current title | Suggested title | Why |
|---|---|---|---|
| (unnumbered) | Resources and Further Reading | — | standard boilerplate heading |

### 6.1 From Fully Connected Layers to Convolutions  <sub>`chapter_convolutional-neural-networks/why-conv.md`</sub>

| # | Current title | Suggested title | Why |
|---|---|---|---|
| 6.1.1 | Invariance | — | already descriptive |
| 6.1.2 | Constraining the MLP | — | already descriptive |
| 6.1.2.1 | Translation Equivariance | — | already descriptive |
| 6.1.2.2 | Locality | — | already descriptive |
| 6.1.3 | Convolutions | Convolution versus Cross-Correlation | single generic word; content is the naming distinction |
| 6.1.4 | Channels | — | already descriptive |
| 6.1.5 | Summary and Discussion | — | standard |
| 6.1.6 | Exercises | — | standard |

### 6.2 Convolutions for Images  <sub>`chapter_convolutional-neural-networks/conv-layer.md`</sub>

| # | Current title | Suggested title | Why |
|---|---|---|---|
| 6.2.1 | The Cross-Correlation Operation | — | already descriptive |
| 6.2.2 | Convolutional Layers | — | already descriptive |
| 6.2.3 | Object Edge Detection in Images | — | already descriptive |
| 6.2.4 | Learning a Kernel | — | already descriptive |
| 6.2.5 | Cross-Correlation and Convolution | — | already descriptive |
| 6.2.6 | Convolution as Matrix Multiplication | — | already descriptive |
| 6.2.7 | Feature Map and Receptive Field | — | already descriptive |
| 6.2.8 | Summary | — | standard |
| 6.2.9 | Exercises | — | standard |

### 6.3 Padding and Stride  <sub>`chapter_convolutional-neural-networks/padding-and-strides.md`</sub>

| # | Current title | Suggested title | Why |
|---|---|---|---|
| 6.3.1 | Padding | — | already descriptive |
| 6.3.2 | Stride | — | already descriptive |
| 6.3.3 | Dilation | — | already descriptive |
| 6.3.4 | Summary and Discussion | — | standard |
| 6.3.5 | Exercises | — | standard |

### 6.4 Multiple Input and Multiple Output Channels  <sub>`chapter_convolutional-neural-networks/channels.md`</sub>

| # | Current title | Suggested title | Why |
|---|---|---|---|
| 6.4.1 | Multiple Input Channels | — | already descriptive |
| 6.4.2 | Multiple Output Channels | — | already descriptive |
| 6.4.3 | $1\times 1$ Convolutional Layer | — | already descriptive |
| 6.4.4 | Grouped, Depthwise, and Depthwise-Separable Convolutions | — | already descriptive |
| 6.4.5 | Discussion | — | standard |
| 6.4.6 | Exercises | — | standard |

### 6.5 Pooling  <sub>`chapter_convolutional-neural-networks/pooling.md`</sub>

| # | Current title | Suggested title | Why |
|---|---|---|---|
| 6.5.1 | Maximum Pooling and Average Pooling | — | already descriptive |
| 6.5.2 | Padding and Stride | — | already descriptive |
| 6.5.3 | Multiple Channels | — | already descriptive |
| 6.5.4 | Summary | — | standard |
| 6.5.5 | Exercises | — | standard |

### 6.6 Convolutional Neural Networks (LeNet)  <sub>`chapter_convolutional-neural-networks/lenet.md`</sub>

| # | Current title | Suggested title | Why |
|---|---|---|---|
| 6.6.1 | LeNet | — | already descriptive |
| 6.6.2 | Training | — | already descriptive |
| 6.6.3 | Summary | — | standard |
| 6.6.4 | Exercises | — | standard |

# TOC title review — Modern Convnets (ch. 7) and Sequence Models (ch. 8)

### 7.1 The ImageNet Moment: AlexNet  <sub>`chapter_convolutional-modern/alexnet.md`</sub>

| # | Current title | Suggested title | Why |
|---|---|---|---|
| 7.1.1 | Representation Learning | — | already descriptive |
| 7.1.1.1 | Missing Ingredient: Data | — | already descriptive |
| 7.1.1.2 | Missing Ingredient: Hardware | — | already descriptive |
| 7.1.2 | AlexNet | — | already descriptive |
| 7.1.2.1 | Architecture | — | already descriptive |
| 7.1.2.2 | Activation Functions | — | already descriptive |
| 7.1.2.3 | Capacity Control and Preprocessing | — | already descriptive |
| 7.1.3 | Training | — | already descriptive |
| 7.1.4 | Discussion | — | already descriptive |
| 7.1.5 | Exercises | — | already descriptive |

### 7.2 Blocks, Bottlenecks, and Branches: VGG, NiN, GoogLeNet  <sub>`chapter_convolutional-modern/blocks.md`</sub>

| # | Current title | Suggested title | Why |
|---|---|---|---|
| 7.2.1 | VGG: Blocks as the Unit of Design | — | already descriptive |
| 7.2.1.1 | VGG Blocks | — | already descriptive |
| 7.2.1.2 | The VGG Network | — | already descriptive |
| 7.2.1.3 | Training | — | already descriptive |
| 7.2.2 | NiN: $1 \times 1$ Convolutions and Global Average Pooling | — | already descriptive |
| 7.2.2.1 | NiN Blocks | — | already descriptive |
| 7.2.2.2 | The NiN Model | — | already descriptive |
| 7.2.2.3 | Training | — | already descriptive |
| 7.2.3 | GoogLeNet: Multi-Branch Blocks and the Stem-Body-Head Pattern | — | already descriptive |
| 7.2.3.1 | The Inception Block | — | already descriptive |
| 7.2.3.2 | Stem, Body, and Head | — | already descriptive |
| 7.2.4 | Summary | — | already descriptive |
| 7.2.5 | Exercises | — | already descriptive |

### 7.3 Normalization Layers  <sub>`chapter_convolutional-modern/batch-norm.md`</sub>

| # | Current title | Suggested title | Why |
|---|---|---|---|
| 7.3.1 | Training Deep Networks | — | already descriptive |
| 7.3.2 | Batch Normalization Layers | — | already descriptive |
| 7.3.2.1 | Fully Connected Layers | — | already descriptive |
| 7.3.2.2 | Convolutional Layers | — | already descriptive |
| 7.3.2.3 | Layer Normalization | — | already descriptive |
| 7.3.2.4 | Batch Normalization During Prediction | — | already descriptive |
| 7.3.3 | Implementation from Scratch | — | already descriptive |
| 7.3.4 | LeNet with Batch Normalization | — | already descriptive |
| 7.3.5 | Concise Implementation | — | already descriptive |
| 7.3.6 | Beyond Batch Normalization | — | already descriptive |
| 7.3.6.1 | Group Normalization | — | already descriptive |
| 7.3.6.2 | Layer Normalization in Convolutional Networks | — | already descriptive |
| 7.3.6.3 | Normalizer-Free Networks | — | already descriptive |
| 7.3.7 | Discussion | — | already descriptive |
| 7.3.8 | Exercises | — | already descriptive |

### 7.4 Residual Networks: ResNet, ResNeXt, and DenseNet  <sub>`chapter_convolutional-modern/resnet.md`</sub>

| # | Current title | Suggested title | Why |
|---|---|---|---|
| 7.4.1 | Function Classes | — | already descriptive |
| 7.4.2 | Residual Blocks | — | already descriptive |
| 7.4.3 | ResNet Model | — | already descriptive |
| 7.4.4 | Training | — | already descriptive |
| 7.4.5 | ResNeXt | — | already descriptive |
| 7.4.6 | Concatenation instead of Addition: DenseNet | Concatenation Instead of Addition: DenseNet | Title Case: capitalize "Instead" |
| 7.4.7 | Summary and Discussion | — | already descriptive |
| 7.4.8 | Exercises | — | already descriptive |

### 7.5 Efficient ConvNets: Depthwise Separability, Mobile Architectures, and Re-parameterization  <sub>`chapter_convolutional-modern/efficient-convnets.md`</sub>

| # | Current title | Suggested title | Why |
|---|---|---|---|
| 7.5.1 | Depthwise-Separable Networks | — | already descriptive |
| 7.5.1.1 | MobileNet | — | already descriptive |
| 7.5.1.2 | The Inverted Bottleneck | — | already descriptive |
| 7.5.1.3 | A Mini-MobileNet | — | already descriptive |
| 7.5.1.4 | Training and Comparison | — | already descriptive |
| 7.5.2 | Scaling and Searching | — | already descriptive |
| 7.5.3 | Structural Re-parameterization | — | already descriptive |
| 7.5.3.1 | The RepVGG Block | — | already descriptive |
| 7.5.3.2 | Fusing the Branches | — | already descriptive |
| 7.5.3.3 | From Paper to Product | Quantization and Industrial Adoption | vague; content is INT8 quantization collapse + adoption |
| 7.5.4 | Summary and Discussion | — | already descriptive |
| 7.5.5 | Exercises | — | already descriptive |

### 7.6 Training Recipes Matter  <sub>`chapter_convolutional-modern/training-recipes.md`</sub>

| # | Current title | Suggested title | Why |
|---|---|---|---|
| 7.6.1 | What Changed between 2015 and 2022 | — | concrete, names the comparison |
| 7.6.2 | Implementing the Ingredients | — | already descriptive |
| 7.6.2.1 | Label Smoothing | — | already descriptive |
| 7.6.2.2 | Cosine Schedules with Warmup | — | already descriptive |
| 7.6.2.3 | Mixup | — | already descriptive |
| 7.6.2.4 | Stochastic Depth | — | already descriptive |
| 7.6.2.5 | Averaging Weights | — | already descriptive |
| 7.6.3 | One Network, Two Recipes | — | vivid but clear: same net, two recipes |
| 7.6.4 | Reading the Scoreboard | — | vivid but clear: interpreting benchmarks |
| 7.6.5 | Summary and Discussion | — | already descriptive |
| 7.6.6 | Exercises | — | already descriptive |

### 7.7 ConvNeXt: A ConvNet for the 2020s  <sub>`chapter_convolutional-modern/convnext.md`</sub>

| # | Current title | Suggested title | Why |
|---|---|---|---|
| 7.7.1 | The Modernization Roadmap | — | already descriptive |
| 7.7.1.1 | Macro design | Macro Design | Title Case: capitalize "Design" |
| 7.7.1.2 | Depthwise convolutions and the inverted bottleneck | Depthwise Convolutions and the Inverted Bottleneck | Title Case throughout |
| 7.7.1.3 | Micro design | Micro Design | Title Case: capitalize "Design" |
| 7.7.2 | Implementation | — | already descriptive |
| 7.7.2.1 | The ConvNeXt block | The ConvNeXt Block | Title Case: capitalize "Block" |
| 7.7.2.2 | The full network | The Full Network | Title Case: capitalize "Full" |
| 7.7.2.3 | Training with the modern recipe | Training with the Modern Recipe | Title Case: capitalize "Modern", "Recipe" |
| 7.7.3 | Beyond ConvNeXt | — | already descriptive |
| 7.7.3.1 | ConvNeXt V2: pretraining and GRN | ConvNeXt V2: Pretraining and GRN | Title Case: capitalize "Pretraining" |
| 7.7.3.2 | Large Kernels and Other Spatial Mixers | — | already descriptive |
| 7.7.3.3 | ConvNeXt in 2026 | — | already descriptive |
| 7.7.4 | Summary and Discussion | — | already descriptive |
| 7.7.5 | Exercises | — | already descriptive |

### 7.8 Design Spaces and the Big Picture  <sub>`chapter_convolutional-modern/cnn-design.md`</sub>

| # | Current title | Suggested title | Why |
|---|---|---|---|
| 7.8.1 | The AnyNet Design Space | — | already descriptive |
| 7.8.2 | Distributions and Parameters of Design Spaces | — | already descriptive |
| 7.8.3 | RegNet | — | already descriptive |
| 7.8.3.1 | Squeeze-and-Excitation Gates | — | already descriptive |
| 7.8.4 | Training | — | already descriptive |
| 7.8.5 | The Big Picture: ConvNets and Transformers | — | already descriptive |
| 7.8.6 | Exercises | — | already descriptive |

### 8.1 Working with Sequences  <sub>`chapter_recurrent-neural-networks/sequence.md`</sub>

| # | Current title | Suggested title | Why |
|---|---|---|---|
| 8.1.1 | Sequential Data and Its Challenges | — | already descriptive |
| 8.1.2 | Autoregressive Models | — | already descriptive |
| 8.1.2.1 | Fixed Windows | — | already descriptive |
| 8.1.2.2 | Latent Summaries | — | already descriptive |
| 8.1.2.3 | From Conditionals to Sequences | — | already descriptive |
| 8.1.3 | Markov Models and Stationarity | — | already descriptive |
| 8.1.3.1 | The Markov Condition | — | already descriptive |
| 8.1.3.2 | Stationarity | — | already descriptive |
| 8.1.4 | Training | — | already descriptive |
| 8.1.5 | Prediction | — | already descriptive |
| 8.1.5.1 | One-Step-Ahead Prediction | — | already descriptive |
| 8.1.5.2 | Multistep Rollout | — | already descriptive |
| 8.1.5.3 | Why This Matters Everywhere | Error Accumulation Beyond Forecasting | vague teaser; content is compounding error in any autoregressive generator (LMs, world models) |
| 8.1.6 | Summary | — | already descriptive |
| 8.1.7 | Exercises | — | already descriptive |

### 8.2 From Text to Tokens  <sub>`chapter_recurrent-neural-networks/text-sequence.md`</sub>

| # | Current title | Suggested title | Why |
|---|---|---|---|
| 8.2.1 | Reading the Dataset | — | already descriptive |
| 8.2.2 | Characters, Words, and Bytes | — | already descriptive |
| 8.2.2.1 | Two Ends of a Spectrum | — | clear in context: char vs. word trade-off |
| 8.2.2.2 | Text as Bytes | — | already descriptive |
| 8.2.3 | Byte Pair Encoding | — | already descriptive |
| 8.2.3.1 | The Merge Rule | — | already descriptive |
| 8.2.3.2 | Implementation | — | already descriptive |
| 8.2.3.3 | Encoding and Decoding | — | already descriptive |
| 8.2.3.4 | Training on The Time Machine | — | already descriptive |
| 8.2.3.5 | Pre-Tokenization: Telling BPE Where Words End | — | already descriptive |
| 8.2.4 | Vocabularies and Special Tokens | — | already descriptive |
| 8.2.4.1 | From Tokens to Indices | — | already descriptive |
| 8.2.4.2 | Special Tokens | — | already descriptive |
| 8.2.5 | Tokenizers in the Wild | — | already descriptive |
| 8.2.5.1 | What a Production Tokenizer Stores | — | already descriptive |
| 8.2.5.2 | Verifying Our Implementation | — | already descriptive |
| 8.2.5.3 | Fertility, Digits, and Glitch Tokens | — | already descriptive |
| 8.2.6 | Summary | — | already descriptive |
| 8.2.7 | Exercises | — | already descriptive |

### 8.3 Language Models  <sub>`chapter_recurrent-neural-networks/language-model.md`</sub>

| # | Current title | Suggested title | Why |
|---|---|---|---|
| 8.3.1 | What a Language Model Buys You | Three Uses of a Language Model | echoes the flagged "What It Bought" pattern; names the actual content (generation, scoring, universal interface) |
| 8.3.2 | $n$-gram Language Models | — | already descriptive |
| 8.3.2.1 | Markov Models and Counting | — | already descriptive |
| 8.3.2.2 | Sampling from $n$-gram Models | — | already descriptive |
| 8.3.2.3 | The Sparsity Wall | — | already descriptive, book's own term |
| 8.3.3 | Word Frequency and Zipf's Law | — | already descriptive |
| 8.3.4 | Perplexity and Bits per Byte | — | already descriptive |
| 8.3.4.1 | Perplexity | — | already descriptive |
| 8.3.4.2 | Bits per Byte | — | already descriptive |
| 8.3.5 | Partitioning Sequences | — | already descriptive |
| 8.3.6 | Summary and Discussion | — | already descriptive |
| 8.3.7 | Exercises | — | already descriptive |

### 8.4 Recurrent Neural Networks  <sub>`chapter_recurrent-neural-networks/rnn.md`</sub>

| # | Current title | Suggested title | Why |
|---|---|---|---|
| 8.4.1 | Neural Networks without Hidden States | — | already descriptive |
| 8.4.2 | Recurrent Neural Networks with Hidden States | — | already descriptive |
| 8.4.2.1 | Constant Memory per Step | — | already descriptive |
| 8.4.3 | RNN Language Models | — | already descriptive |
| 8.4.3.1 | Teacher Forcing | — | already descriptive |
| 8.4.4 | Summary | — | already descriptive |
| 8.4.5 | Exercises | — | already descriptive |

### 8.5 Implementing RNN Language Models  <sub>`chapter_recurrent-neural-networks/rnn-implementation.md`</sub>

| # | Current title | Suggested title | Why |
|---|---|---|---|
| 8.5.1 | An RNN Language Model from Scratch | — | already descriptive |
| 8.5.1.1 | The Recurrent Cell | — | already descriptive |
| 8.5.1.2 | From Token IDs to Embeddings | — | already descriptive |
| 8.5.1.3 | The Output Layer | — | already descriptive |
| 8.5.2 | Gradient Clipping | — | already descriptive |
| 8.5.3 | Training | — | already descriptive |
| 8.5.4 | Generating Text | — | already descriptive |
| 8.5.5 | Concise Implementation | — | already descriptive |
| 8.5.5.1 | Scratch versus Concise, Measured | — | already descriptive |
| 8.5.6 | Summary | — | already descriptive |
| 8.5.7 | Exercises | — | already descriptive |

### 8.6 Backpropagation Through Time  <sub>`chapter_recurrent-neural-networks/bptt.md`</sub>

| # | Current title | Suggested title | Why |
|---|---|---|---|
| 8.6.1 | The Unrolled Graph and the Full Gradient | — | already descriptive |
| 8.6.2 | Vanishing and Exploding Gradients | — | already descriptive |
| 8.6.2.1 | From Arithmetic to Architecture | — | vivid but clear: two distinct fixes |
| 8.6.3 | Truncated Backpropagation Through Time | — | already descriptive |
| 8.6.4 | Summary | — | already descriptive |
| 8.6.5 | Exercises | — | already descriptive |

### 8.7 Decoding and Generation  <sub>`chapter_recurrent-neural-networks/decoding.md`</sub>

| # | Current title | Suggested title | Why |
|---|---|---|---|
| 8.7.1 | The Decoding Problem | — | already descriptive |
| 8.7.1.1 | A Model to Decode From | — | already descriptive |
| 8.7.2 | Greedy Decoding | — | already descriptive |
| 8.7.3 | Beam Search | — | already descriptive |
| 8.7.4 | Sampling and Its Dials | — | already descriptive, book's own term |
| 8.7.4.1 | A Unified Sampler | — | already descriptive |
| 8.7.4.2 | One Distribution, Three Cutoffs | — | already descriptive |
| 8.7.4.3 | The Same Prefix Under Every Strategy | — | already descriptive |
| 8.7.5 | Evaluation and Efficiency | — | already descriptive |
| 8.7.5.1 | Evaluating Generated Text | — | already descriptive |
| 8.7.5.2 | The Cost of Generation | — | already descriptive |
| 8.7.6 | Summary | — | already descriptive |
| 8.7.7 | Exercises | — | already descriptive |

# Chapter 9 — Optimization Algorithms (`chapter_optimization/`)

### 9 Optimization Algorithms  <sub>`chapter_optimization/index.md`</sub>

Chapter-intro file: prose overview plus the `toc` block and an
`{.unnumbered}` "Resources and Further Reading" section, so it carries no
numbered `##`/`###` headings of its own.

| # | Current title | Suggested title | Why |
|---|---|---|---|
| (unnumbered) | Resources and Further Reading | — | standard back-matter heading |

### 9.1 Landscapes  <sub>`chapter_optimization/optimization-intro.md`</sub>

| # | Current title | Suggested title | Why |
|---|---|---|---|
| 9.1.1 | The Goal of Optimization | — | already descriptive |
| 9.1.2 | Where Gradients Vanish | — | already descriptive |
| 9.1.2.1 | Local Minima | — | already descriptive |
| 9.1.2.2 | Saddle Points | — | already descriptive |
| 9.1.2.3 | Vanishing Gradients | — | already descriptive |
| 9.1.3 | Curvature and Noise | — | already descriptive |
| 9.1.3.1 | An Ill-Conditioned Valley | — | already descriptive |
| 9.1.3.2 | The Edge of Stability | — | named phenomenon, clear |
| 9.1.3.3 | Noisy Gradients | — | already descriptive |
| 9.1.4 | What Convexity Still Buys | — | names topic (convexity's residual value) |
| 9.1.5 | Summary | — | standard |
| 9.1.6 | Exercises | — | standard |

### 9.2 Gradient Descent  <sub>`chapter_optimization/gd.md`</sub>

| # | Current title | Suggested title | Why |
|---|---|---|---|
| 9.2.1 | One-Dimensional Gradient Descent | — | already descriptive |
| 9.2.1.1 | Learning Rate | — | already descriptive |
| 9.2.1.2 | Local Minima | — | already descriptive |
| 9.2.2 | Multivariate Gradient Descent | — | already descriptive |
| 9.2.3 | Newton's Method | — | already descriptive |
| 9.2.3.1 | Preconditioning | — | already descriptive |
| 9.2.4 | Summary | — | standard |
| 9.2.5 | Exercises | — | standard |

### 9.3 Stochastic Gradient Descent  <sub>`chapter_optimization/sgd.md`</sub>

| # | Current title | Suggested title | Why |
|---|---|---|---|
| 9.3.1 | Stochastic Gradient Updates | — | already descriptive |
| 9.3.2 | Dynamic Learning Rate | — | already descriptive |
| 9.3.3 | Gradient Variance and Batch Size | — | already descriptive |
| 9.3.4 | Summary | — | standard |
| 9.3.5 | Exercises | — | standard |

### 9.4 Minibatches  <sub>`chapter_optimization/minibatch-sgd.md`</sub>

| # | Current title | Suggested title | Why |
|---|---|---|---|
| 9.4.1 | Vectorization and Caches | — | already descriptive |
| 9.4.2 | Minibatch Gradients | — | already descriptive |
| 9.4.3 | Reading the Dataset | — | already descriptive |
| 9.4.4 | Implementation from Scratch | — | standard convention |
| 9.4.5 | Concise Implementation | — | standard convention |
| 9.4.6 | Summary | — | standard |
| 9.4.7 | Exercises | — | standard |

### 9.5 Momentum  <sub>`chapter_optimization/momentum.md`</sub>

| # | Current title | Suggested title | Why |
|---|---|---|---|
| 9.5.1 | An Ill-Conditioned Valley | — | already descriptive |
| 9.5.2 | The Momentum Method | — | already descriptive |
| 9.5.2.1 | Leaky Averages | — | already descriptive |
| 9.5.2.2 | Back to the Valley | — | "the valley" is the section's own running example |
| 9.5.2.3 | The Timescale of β | — | already descriptive |
| 9.5.2.4 | Acceleration and Damping | — | already descriptive |
| 9.5.3 | Implementation | — | standard convention |
| 9.5.3.1 | From Scratch | — | standard convention |
| 9.5.3.2 | Concise Implementation | — | standard convention |
| 9.5.4 | Nesterov Momentum | — | already descriptive |
| 9.5.5 | Summary | — | standard |
| 9.5.6 | Exercises | — | standard |

### 9.6 Adam  <sub>`chapter_optimization/adam.md`</sub>

| # | Current title | Suggested title | Why |
|---|---|---|---|
| 9.6.1 | From AdaGrad to Adam | — | already descriptive |
| 9.6.1.1 | Per-Coordinate Learning Rates | — | already descriptive |
| 9.6.1.2 | RMSProp: Forgetting on Purpose | — | names algorithm + its real mechanism |
| 9.6.1.3 | Adam: Both Moments, Debiased | — | names algorithm + its two components |
| 9.6.2 | A Tiny Language Model | — | already descriptive |
| 9.6.3 | Where Adam Wins | — | names the experimental question |
| 9.6.3.1 | The Race on the Language Model | — | already descriptive |
| 9.6.3.2 | The Same Race on a CNN | — | already descriptive |
| 9.6.3.3 | Why the Gap Lives Where It Does | Sources of the Adam–SGD Gap | vague "gap"/"lives"; doesn't name the mechanism |
| 9.6.4 | When the Variance Estimate Misbehaves | — | names the specific failure mode |
| 9.6.5 | Summary | — | standard |
| 9.6.6 | Exercises | — | standard |

### 9.7 AdamW  <sub>`chapter_optimization/adamw.md`</sub>

| # | Current title | Suggested title | Why |
|---|---|---|---|
| 9.7.1 | The Penalty Meets the Preconditioner | — | names the two interacting mechanisms |
| 9.7.2 | AdamW from Scratch | — | standard convention |
| 9.7.3 | Decoupling, Demonstrated | — | names the technique being shown |
| 9.7.3.1 | One Number, Two Meanings | The Same λ, Coupled versus Decoupled | teaser; doesn't name λ or the comparison |
| 9.7.3.2 | A Grid, Twice | A Joint Learning-Rate/Weight-Decay Sweep | doesn't name the grid's variables or purpose |
| 9.7.3.3 | What Weight Decay Is Actually Doing | — | names the subject (weight decay's real role) |
| 9.7.4 | What Not to Decay | — | already descriptive |
| 9.7.5 | Optimizer State and Memory | — | already descriptive |
| 9.7.6 | Summary | — | standard |
| 9.7.7 | Exercises | — | standard |

### 9.8 Schedules  <sub>`chapter_optimization/lr-scheduler.md`</sub>

| # | Current title | Suggested title | Why |
|---|---|---|---|
| 9.8.1 | A Testbed | — | standard convention for this chapter |
| 9.8.2 | Decay Schedules | — | already descriptive |
| 9.8.2.1 | Square-Root Decay | — | already descriptive |
| 9.8.2.2 | Multiplicative Decay | — | already descriptive |
| 9.8.2.3 | Piecewise Constant Decay | — | already descriptive |
| 9.8.2.4 | Cosine Decay | — | already descriptive |
| 9.8.3 | Warmup | — | already descriptive |
| 9.8.4 | Warmup–Stable–Decay | — | names the method |
| 9.8.4.1 | Branching Off the Plateau | — | names the specific workflow shown |
| 9.8.4.2 | The Current Frontier | Beyond Warmup–Stable–Decay | vague; doesn't name what's beyond WSD |
| 9.8.5 | Summary | — | standard |
| 9.8.6 | Exercises | — | standard |

### 9.9 Muon  <sub>`chapter_optimization/muon.md`</sub>

| # | Current title | Suggested title | Why |
|---|---|---|---|
| 9.9.1 | The Norm Decides the Direction | — | states the chapter's organizing thesis |
| 9.9.1.1 | Steepest Descent Under a Ball | — | already descriptive |
| 9.9.1.2 | Matrices and the Spectral Norm | — | already descriptive |
| 9.9.2 | Orthogonalization by Newton–Schulz | — | already descriptive |
| 9.9.3 | Muon from Scratch | — | standard convention |
| 9.9.3.1 | The Update | The Muon Update Rule | generic; doesn't name Muon specifically |
| 9.9.3.2 | Dividing the Census | — | reuses established book term ("census") |
| 9.9.3.3 | The Race on the Language Model | — | already descriptive |
| 9.9.3.4 | The Same Race on a CNN | — | already descriptive |
| 9.9.3.5 | Library Implementations | — | already descriptive |
| 9.9.4 | The Preconditioning Family | — | already descriptive |
| 9.9.5 | Muon in the Wild | — | standard idiom for real-world/production use |
| 9.9.6 | Summary | — | standard |
| 9.9.7 | Exercises | — | standard |

### 9.10 Batch Size  <sub>`chapter_optimization/batch-size.md`</sub>

| # | Current title | Suggested title | Why |
|---|---|---|---|
| 9.10.1 | The Gradient-Noise Scale | — | names the defined quantity |
| 9.10.1.1 | Signal versus Noise | — | already descriptive |
| 9.10.1.2 | A Two-Batch Estimator | — | already descriptive |
| 9.10.2 | Steps to a Target | — | names the experimental design |
| 9.10.2.1 | The Language Model | — | one of two named testbeds, clear from siblings |
| 9.10.2.2 | The CNN | — | one of two named testbeds, clear from siblings |
| 9.10.2.3 | The Bill in Examples | — | names the unit (examples) of the cost being tallied |
| 9.10.3 | Learning-Rate Rules | — | already descriptive |
| 9.10.3.1 | Where the Rules Break | — | clear referent (the rules just stated) |
| 9.10.4 | Growing the Batch | — | already descriptive |
| 9.10.5 | Summary | — | standard |
| 9.10.6 | Exercises | — | standard |

### 9.11 Scaling Up  <sub>`chapter_optimization/scaling.md`</sub>

| # | Current title | Suggested title | Why |
|---|---|---|---|
| 9.11.1 | The Optimum Does Not Stay Put | — | names the phenomenon (LR drifts with width) |
| 9.11.1.1 | A Family of Widths | — | already descriptive |
| 9.11.1.2 | The Sweep | — | standard convention for this chapter |
| 9.11.1.3 | Why It Moves | Why the Optimum Drifts with Width | vague pronoun "it" lacks a clear antecedent |
| 9.11.2 | Maximal Update Parametrization | — | names the method (muP) |
| 9.11.2.1 | The Rules | The muP Scaling Rules | generic; doesn't name muP specifically |
| 9.11.2.2 | The Coordinate Check | — | names the established diagnostic technique |
| 9.11.2.3 | Learning-Rate Transfer | — | already descriptive |
| 9.11.3 | The Spectral View | — | already descriptive |
| 9.11.4 | What the Big Runs Do | — | names the subject (production practice) |
| 9.11.5 | Summary | — | standard |
| 9.11.6 | Exercises | — | standard |

### 9.12 Practice  <sub>`chapter_optimization/practice.md`</sub>

| # | Current title | Suggested title | Why |
|---|---|---|---|
| 9.12.1 | The Recipe, as Disclosed | — | already descriptive |
| 9.12.2 | Gradient Clipping | — | already descriptive |
| 9.12.2.1 | A NaN, Averted | — | concrete noun (NaN) + outcome, clear under parent |
| 9.12.2.2 | The Stability Kit at Scale | — | already descriptive |
| 9.12.3 | Weight Averaging | — | already descriptive |
| 9.12.3.1 | Averaging at Scale | — | already descriptive |
| 9.12.4 | How to Tune | — | already descriptive |
| 9.12.5 | What We Did Not Teach | — | names the subject (deliberate omissions) |
| 9.12.6 | Summary | — | standard |
| 9.12.7 | Exercises | — | standard |

# Ch. 10 Attention, Ch. 11 Transformers — TOC title review

`chapter_attention/index.md` (ch. 10) and `chapter_transformers/index.md`
(ch. 11) are chapter-overview files: their only heading is an
`{.unnumbered}` "Resources and Further Reading" section, so they carry no
numbered TOC entries and are omitted below.

## Chapter 10 — Attention

### 10.1 Queries, Keys, and Values  <sub>`chapter_attention/queries-keys-values.md`</sub>

| # | Current title | Suggested title | Why |
|---|---|---|---|
| 10.1.1 | Attention as Soft Database Lookup | — | already descriptive |
| 10.1.2 | Visualizing Attention Weights | — | already descriptive |
| 10.1.3 | Attention Pooling with Fixed Kernels | — | already descriptive |
| 10.1.3.1 | Similarity Kernels | — | already descriptive |
| 10.1.3.2 | Nadaraya--Watson Regression in Action | — | already descriptive |
| 10.1.4 | Summary | — | standard section |
| 10.1.5 | Exercises | — | standard section |

### 10.2 Attention Scoring and Masking  <sub>`chapter_attention/attention-scoring.md`</sub>

| # | Current title | Suggested title | Why |
|---|---|---|---|
| 10.2.1 | Dot-Product Attention | — | already descriptive |
| 10.2.1.1 | Softmax Saturation and the $1/\sqrt{d}$ Factor | — | names the exact phenomenon and fix |
| 10.2.2 | Masking | — | already descriptive |
| 10.2.2.1 | The Masked Softmax Operation | — | already descriptive |
| 10.2.2.2 | Causal Masking | — | already descriptive |
| 10.2.2.3 | Composing Masks | — | already descriptive |
| 10.2.3 | Batched Attention | — | already descriptive |
| 10.2.3.1 | Batch Matrix Multiplication | — | already descriptive |
| 10.2.3.2 | The DotProductAttention Class | — | already descriptive |
| 10.2.4 | From Alignment to Attention | — | names the historical throughline |
| 10.2.5 | Summary | — | standard section |
| 10.2.6 | Exercises | — | standard section |

### 10.3 Multi-Head and Cross-Attention  <sub>`chapter_attention/multihead-attention.md`</sub>

| # | Current title | Suggested title | Why |
|---|---|---|---|
| 10.3.1 | One Head Must Average | — | declarative claim naming the proved limit |
| 10.3.2 | Multi-Head Attention | — | already descriptive |
| 10.3.2.1 | From One Head to $h$ Heads | — | already descriptive |
| 10.3.2.2 | Same FLOPs, More Views | — | verified: names the exact FLOP-invariance result |
| 10.3.2.3 | Implementation | The MultiHeadAttention Class | generic; sibling 10.2.3.2 names its class |
| 10.3.3 | Self-Attention and Cross-Attention | — | already descriptive |
| 10.3.3.1 | One Sequence Querying Itself | — | already descriptive |
| 10.3.3.2 | One Sequence Querying Another | — | already descriptive |
| 10.3.3.3 | An Alignment You Can Read | — | vivid but decodable, matches "alignment" vocabulary |
| 10.3.4 | Summary | — | standard section |
| 10.3.5 | Exercises | — | standard section |

### 10.4 Positional Information  <sub>`chapter_attention/positional-information.md`</sub>

| # | Current title | Suggested title | Why |
|---|---|---|---|
| 10.4.1 | Attention Ignores Order | — | already descriptive |
| 10.4.2 | Absolute Position Embeddings | — | already descriptive |
| 10.4.2.1 | Sinusoidal Encodings | — | already descriptive |
| 10.4.2.2 | Learned Positions | — | already descriptive |
| 10.4.2.3 | The Rotation Hidden in the Sinusoids | — | verified: literally names the rotation identity proved |
| 10.4.3 | Rotary Position Embeddings | — | already descriptive |
| 10.4.4 | Train Short, Test Long | — | established name for the extrapolation problem |
| 10.4.4.1 | Linear Biases, and Nothing at All | ALiBi and NoPE | dangling "nothing at all" hides NoPE; name both schemes |
| 10.4.4.2 | An Attention-Only Language Model | — | already descriptive |
| 10.4.4.3 | The Experiment | Measuring Extrapolation Across the Schemes | too generic standing alone |
| 10.4.4.4 | Stretching a Trained Model | — | vivid but decodable metaphor for context extension |
| 10.4.5 | Summary | — | standard section |
| 10.4.6 | Exercises | — | standard section |

### 10.5 The Cost of Attention  <sub>`chapter_attention/attention-at-scale.md`</sub>

| # | Current title | Suggested title | Why |
|---|---|---|---|
| 10.5.1 | CNNs, RNNs, and Self-Attention | — | already descriptive |
| 10.5.2 | The Quadratic Bill | — | established book metaphor for cost |
| 10.5.2.1 | Counting FLOPs | — | already descriptive |
| 10.5.2.2 | Counting Memory | — | already descriptive |
| 10.5.3 | Exact Attention Without the Matrix | — | already descriptive |
| 10.5.3.1 | Softmax, One Block at a Time | — | already descriptive |
| 10.5.3.2 | A Chunked Implementation | — | already descriptive |
| 10.5.3.3 | The Bottleneck is Memory Traffic | — | clear, self-contained declarative claim |
| 10.5.4 | Windowed and Sparse Attention | — | already descriptive |
| 10.5.4.1 | Attention Through a Window | — | already descriptive |
| 10.5.4.2 | Depth Restores the Reach | — | verified: names the receptive-field result exactly |
| 10.5.4.3 | A Linear-Cost Implementation | — | already descriptive |
| 10.5.5 | Linear Attention is a Recurrent Network | — | already descriptive |
| 10.5.5.1 | Kernelizing the Score | — | already descriptive |
| 10.5.5.2 | The Price of Attention, Measured | Comparing Dense, Windowed, and Linear Attention | trailing ", Measured" is exactly the banned pattern |
| 10.5.5.3 | The Bridge to State Space Models | — | matches the text's own cross-chapter phrasing |
| 10.5.6 | Summary | — | standard section |
| 10.5.7 | Exercises | — | standard section |

### 10.6 What Attention Computes  <sub>`chapter_attention/what-attention-computes.md`</sub>

| # | Current title | Suggested title | Why |
|---|---|---|---|
| 10.6.1 | The Residual Stream | — | already descriptive |
| 10.6.1.1 | Tokens as Vectors in a Shared Workspace | — | already descriptive |
| 10.6.1.2 | Where and What: The QK and OV Circuits | — | precisely names both circuits' roles |
| 10.6.1.3 | Bigrams, Skip-Trigrams, and the Limits of One Layer | — | matches the section's own vocabulary |
| 10.6.2 | Copying and Induction | — | already descriptive |
| 10.6.2.1 | Repetition as a Task | — | already descriptive |
| 10.6.2.2 | The Positional Shortcut | — | already descriptive |
| 10.6.2.3 | Two Blocks Learn to Look Things Up | — | declarative claim, ties to chapter's "lookup" theme |
| 10.6.2.4 | The Heads, Caught in the Act | The Circuit in the Attention Maps | "caught in the act" is clickbait-coded, names nothing |
| 10.6.3 | In-Context Learning as Pattern Completion | — | already descriptive |
| 10.6.3.1 | Completing Patterns It Has Never Seen | — | clear declarative claim |
| 10.6.3.2 | The Circuit Is in the Weights | — | clear declarative claim |
| 10.6.3.3 | Induction Heads in the Wild | — | standard ML idiom, decodable |
| 10.6.4 | What Attention Weights Do and Do Not Tell You | — | already descriptive |
| 10.6.5 | Summary | — | standard section |
| 10.6.6 | Exercises | — | standard section |

## Chapter 11 — Transformers

### 11.1 The Transformer Block  <sub>`chapter_transformers/transformer-block.md`</sub>

| # | Current title | Suggested title | Why |
|---|---|---|---|
| 11.1.1 | The Anatomy of a Block | — | already descriptive |
| 11.1.2 | Where the Normalization Goes | — | names the section's actual question |
| 11.1.2.1 | Two Arrangements | Pre-Norm and Post-Norm | vague without antecedent |
| 11.1.2.2 | Signal Propagation at Initialization | — | already descriptive |
| 11.1.2.3 | RMSNorm | — | names the technique directly |
| 11.1.2.4 | Normalizing Queries and Keys | — | describes QK-norm clearly |
| 11.1.3 | The Feed-Forward Network | — | already descriptive |
| 11.1.4 | A Configurable Block | — | describes the assembled class |
| 11.1.4.1 | Shapes and Parameters | — | matches the shape/param census |
| 11.1.4.2 | The Flags at Work: GELU versus SwiGLU | — | names the actual comparison |
| 11.1.5 | Summary | — | standard section |
| 11.1.6 | Exercises | — | standard section |

### 11.2 A GPT from Scratch  <sub>`chapter_transformers/gpt.md`</sub>

| # | Current title | Suggested title | Why |
|---|---|---|---|
| 11.2.1 | From Blocks to a Language Model | — | already descriptive |
| 11.2.2 | Training the Modern Configuration | — | matches content (modern flag set) |
| 11.2.2.1 | Breaking It with One Flag | Post-Norm Fails to Train | "It" has no antecedent; states the actual finding |
| 11.2.3 | Sampling from the Model | — | already descriptive |
| 11.2.4 | Loading GPT-2 | — | already descriptive |
| 11.2.5 | Summary | — | standard section |
| 11.2.6 | Exercises | — | standard section |

### 11.3 Generation and the KV Cache  <sub>`chapter_transformers/kv-cache.md`</sub>

| # | Current title | Suggested title | Why |
|---|---|---|---|
| 11.3.1 | From Recompute to Cache | — | already descriptive |
| 11.3.1.1 | The Cached Forward Pass | — | already descriptive |
| 11.3.1.2 | Same Logits, Measured | Correctness and Speed of the Cache | trailing ", Measured" clause, vague |
| 11.3.2 | The Memory Bill | — | established book metaphor for cost |
| 11.3.2.1 | Prefill Is Compute-Bound, Decode Is Memory-Bound | — | precise declarative claim |
| 11.3.3 | Sharing Keys and Values across Heads | — | already descriptive |
| 11.3.3.1 | A Pluggable Implementation | One Attention Class for MHA, GQA, and MQA | "pluggable" names nothing technical; content is the generalization |
| 11.3.3.2 | Cache Against Quality | Cache Size versus Quality | awkward preposition; "versus" fits house style |
| 11.3.4 | Compressing the Cache Further | — | already descriptive |
| 11.3.4.1 | Low-Rank Keys and Values | — | already descriptive |
| 11.3.4.2 | A Window Needs a Sink | — | declarative, decodable (sliding window + attention sink) |
| 11.3.4.3 | The Cache-Relief Map | — | established book term, reused in the chapter's own index.md |
| 11.3.5 | Summary | — | standard section |
| 11.3.6 | Exercises | — | standard section |

### 11.4 Encoders, Decoders, and Cross-Attention  <sub>`chapter_transformers/encoders-decoders.md`</sub>

| # | Current title | Suggested title | Why |
|---|---|---|---|
| 11.4.1 | Three Wirings of One Block | — | names the taxonomy directly |
| 11.4.2 | An Encoder: Predicting from Both Sides | — | clear, names the mechanism |
| 11.4.2.1 | A Bidirectional Encoder in a Dozen Lines | Implementing a Bidirectional Encoder | "in a Dozen Lines" is filler, not content |
| 11.4.2.2 | The Masked-Token Objective | — | precise technical noun phrase |
| 11.4.2.3 | What the Second Side Is Worth | The Value of Bidirectional Context | indirect-question phrasing; names the actual finding |
| 11.4.3 | An Encoder--Decoder: Cross-Attention at Work | — | already descriptive |
| 11.4.3.1 | A Task Whose Alignment We Know | — | verified: names the design choice (verifiable alignment) |
| 11.4.3.2 | The Decoder Block: One More Sublayer | — | precise, accurate incremental framing |
| 11.4.3.3 | Assembling the Masks | — | matches content (building the three attention masks) |
| 11.4.3.4 | Training and Decoding | — | plain, accurate |
| 11.4.3.5 | Reading the Alignment | — | matches content (checking the cross-attention map) |
| 11.4.4 | Cross-Attention as Interface | — | names the generalization being made |
| 11.4.4.1 | Queries Need Not Come from a Sequence | — | declarative claim, self-contained |
| 11.4.4.2 | The Cost Curve | The Cost of a Latent Bottleneck | "the cost curve" has no antecedent; names the actual quantity measured |
| 11.4.4.3 | Perceiver IO and the Idea's Descendants | — | names architecture and scope |
| 11.4.5 | Which Wiring When | Choosing Among the Three Wirings | elliptical fragment mirrors the flagged pattern; ties to 11.4.1 |
| 11.4.6 | Summary | — | standard section |
| 11.4.7 | Exercises | — | standard section |

### 11.5 Vision Transformer  <sub>`chapter_transformers/vision-transformer.md`</sub>

| # | Current title | Suggested title | Why |
|---|---|---|---|
| 11.5.1 | Patches as Tokens | — | clear, names the stem's function |
| 11.5.2 | The Vision Transformer Block | — | precise component name |
| 11.5.3 | The Full Model | — | plain, accurate |
| 11.5.4 | Training and What the Model Learns | — | names both the experiment and the probe |
| 11.5.4.1 | A First Run | — | plain, standard baseline-run title |
| 11.5.4.2 | Do the Position Embeddings Discover the Grid? | Position Embeddings Begin to Learn the Grid | rule violation: literal question; states the actual (partial) finding |
| 11.5.4.3 | A Convolutional Baseline at the Same Budget | — | precise, names the comparison |
| 11.5.5 | Summary and Discussion | Summary | minor: inconsistent with this book's plain "Summary" elsewhere |
| 11.5.6 | Exercises | — | standard section |

### 11.6 Mixture of Experts  <sub>`chapter_transformers/moe.md`</sub>

| # | Current title | Suggested title | Why |
|---|---|---|---|
| 11.6.1 | Conditional Computation | — | names the core bet clearly |
| 11.6.2 | A Mixture-of-Experts Layer | — | precise component name |
| 11.6.3 | Routing Collapse and Load Balancing | — | names failure mode and fix category |
| 11.6.3.1 | The Rich Get Richer | — | standard idiom for the feedback loop described |
| 11.6.3.2 | An Auxiliary Balancing Loss | — | precise, names the mechanism |
| 11.6.3.3 | Balancing Without a Loss | — | clear contrast to prior subsection |
| 11.6.3.4 | Three Runs, One Budget | Comparing the Balancing Strategies | names setup mechanics, not what's compared |
| 11.6.4 | Fine-Grained and Shared Experts | — | names both refinements |
| 11.6.5 | A Mixture-of-Experts GPT | — | precise, names the built model |
| 11.6.6 | Summary | — | standard section |
| 11.6.7 | Exercises | — | standard section |

### 11.7 Scaling Laws and the Modern Recipe  <sub>`chapter_transformers/scaling-laws.md`</sub>

| # | Current title | Suggested title | Why |
|---|---|---|---|
| 11.7.1 | Counting Parameters and FLOPs | — | clear, names the topic |
| 11.7.1.1 | A Parameter Census | — | apt metaphor, matches in-text usage |
| 11.7.1.2 | Six FLOPs per Parameter and Token | — | states the exact derived rule |
| 11.7.1.3 | Checking the Arithmetic | — | matches content (verifying the rule) |
| 11.7.2 | A Miniature Scaling Study | — | clear, names the experiment |
| 11.7.2.1 | A Corpus Bigger than a Novella | — | verified: apt in-book comparison (Time Machine is a novella) |
| 11.7.2.2 | Five Sizes, One Diet | — | verified: names both held/varied experimental variables |
| 11.7.2.3 | Reading the Bend | — | standard scaling-law parlance, decodable in context |
| 11.7.2.4 | The Published Form of the Law | — | names the formula being introduced |
| 11.7.3 | The Modern Recipe | — | clear, sets up the table |
| 11.7.3.1 | Convergent Evolution | — | vivid term used explicitly in the chapter's own prose |
| 11.7.3.2 | Recipe Rows as Constructor Calls | — | states the section's precise, central claim |
| 11.7.4 | Where the Field Is Moving | — | clear forward-looking survey title |
| 11.7.5 | Summary | — | standard section |
| 11.7.6 | Exercises | — | standard section |

# TOC review: chapter_recurrent-modern (ch. 12) and chapter_computational-performance (ch. 13)

Numbering from `CHAPTER_NUMBERING` in `tools/d2l_preprocess.py`, cross-checked
against `_quarto.yml` order. `index.md` files carry no section number (only
`[chapter]`); their `## {.unnumbered}` headings are shown with the chapter
number and "(unnumbered)" since they never receive a decimal — Quarto lists
them in the TOC by title alone.

## Chapter 12 — State Space Models

### 12 State Space Models <sub>`chapter_recurrent-modern/index.md`</sub>

| # | Current title | Suggested title | Why |
|---|---|---|---|
| 12 (unnumbered) | Resources and Further Reading | — | standard section, matches convention |

### 12.1 Gated Recurrence <sub>`chapter_recurrent-modern/lstm.md`</sub>

| # | Current title | Suggested title | Why |
|---|---|---|---|
| 12.1.1 | Memory Needs a Controller | — | already descriptive |
| 12.1.2 | Long Short-Term Memory | — | already descriptive |
| 12.1.2.1 | The Memory Cell and Its Gates | — | already descriptive |
| 12.1.2.2 | Implementation from Scratch | — | already descriptive |
| 12.1.2.3 | Training | — | already descriptive |
| 12.1.2.4 | Concise Implementation | — | already descriptive |
| 12.1.3 | Gated Recurrent Units | — | already descriptive |
| 12.1.3.1 | Reset and Update Gates | — | already descriptive |
| 12.1.3.2 | Implementation and Comparison | — | already descriptive |
| 12.1.4 | Depth and Direction | — | already descriptive |
| 12.1.4.1 | Deep Recurrent Networks | — | already descriptive |
| 12.1.4.2 | Bidirectional Recurrent Networks | — | already descriptive |
| 12.1.5 | Gates beyond Recurrent Networks | — | already descriptive |
| 12.1.6 | Summary | — | already descriptive |
| 12.1.7 | Exercises | — | already descriptive |

This section is the cleanest in the chapter — every title already names its
technical content.

### 12.2 Linear Recurrence and State Space Models <sub>`chapter_recurrent-modern/ssm.md`</sub>

| # | Current title | Suggested title | Why |
|---|---|---|---|
| 12.2.1 | Linearizing the Recurrence | — | already descriptive |
| 12.2.2 | Parallel Scans | — | already descriptive |
| 12.2.2.1 | An Associative Combine | — | already descriptive |
| 12.2.2.2 | Implementation | — | already descriptive |
| 12.2.2.3 | A minGRU Language Model | — | already descriptive |
| 12.2.3 | State Space Models | — | matches file's own topic |
| 12.2.3.1 | Discretization: the Step Size is a Gate | — | names the mechanism |
| 12.2.3.2 | Recurrence is Convolution | — | states the equivalence taught |
| 12.2.3.3 | What Control Theory Already Knew | Classical Control Theory: Controllability, Poles, and Stability | vague framing; content is four named classical results |
| 12.2.4 | Remembering the Past: HiPPO | — | names HiPPO explicitly |
| 12.2.4.1 | From HiPPO to S4 to S4D | — | already descriptive |
| 12.2.5 | S4D in Practice | — | already descriptive |
| 12.2.5.1 | Sequential Image Classification | — | already descriptive |
| 12.2.6 | Inference, One Token at a Time | — | already descriptive |
| 12.2.7 | Summary | — | already descriptive |
| 12.2.8 | Exercises | — | already descriptive |

### 12.3 Selective State Space Models <sub>`chapter_recurrent-modern/mamba.md`</sub>

| # | Current title | Suggested title | Why |
|---|---|---|---|
| 12.3.1 | The Selectivity Problem | — | already descriptive |
| 12.3.1.1 | A Task That Defeats Time Invariance | — | already descriptive |
| 12.3.1.2 | An LTI Baseline and a Gated One | — | already descriptive |
| 12.3.2 | Selective State Space Models | Building the Selective SSM Layer | duplicates the file's own title one level up |
| 12.3.2.1 | Making the Dynamics Look at the Data | — | descriptive gerund, names the mechanism |
| 12.3.2.2 | What Selectivity Costs, and What Survives | Losing the Convolution View, Keeping the Scan | trailing "and what X" clause; content is concrete |
| 12.3.3 | The Mamba Block | — | already descriptive |
| 12.3.3.1 | The Three Answers, Measured on One Task | Capstone Scoreboard: LSTM vs. minGRU vs. Mamba | "three answers" needs the chapter intro to parse |
| 12.3.3.2 | Stepping the Selective Model | — | already descriptive |
| 12.3.3.3 | Selective Copying, Revisited | — | already descriptive |
| 12.3.4 | Summary | — | already descriptive |
| 12.3.5 | Exercises | — | already descriptive |

### 12.4 The Matrix State: From Linear Attention to Mamba-2 <sub>`chapter_recurrent-modern/matrix-state.md`</sub>

| # | Current title | Suggested title | Why |
|---|---|---|---|
| 12.4.1 | Two Roads to One Recurrence | — | already descriptive |
| 12.4.1.1 | What the Memory Costs | The Capacity Law of a Matrix Memory | vague; content is a proved capacity proposition |
| 12.4.1.2 | The Decay Ladder | — | established chapter term |
| 12.4.2 | The State-Space Duality | — | names the concept directly |
| 12.4.3 | Chunked Computation: Mostly Matmul, a Little Scan | — | vivid but names the algorithm |
| 12.4.3.1 | What the Hardware Bought | Matmul Efficiency and the State-vs-Cache Ledger | matches Alex's flagged "What It Bought" pattern |
| 12.4.4 | The Family, So Far | The Matrix-State Family, Tabulated | "so far" is filler; this is the family table |
| 12.4.4.1 | The One That Kept the Normalizer | mLSTM: Exponential Gating with a Kept Normalizer | cryptic without context; content is mLSTM specifically |
| 12.4.4.2 | The Column Left Open | The Missing Write Rule: Editing, Not Just Adding | cryptic idiom; content is "no model edits its memory" |
| 12.4.5 | Summary | — | already descriptive |
| 12.4.6 | Exercises | — | already descriptive |

### 12.5 DeltaNet: Memory That Edits <sub>`chapter_recurrent-modern/deltanet.md`</sub>

| # | Current title | Suggested title | Why |
|---|---|---|---|
| 12.5.1 | The Trouble with Adding | — | names the topic (additive-write failure) |
| 12.5.1.1 | Trained Models Hit the Same Ceiling | Reproducing the Ceiling with End-to-End Training | full-sentence claim, not a noun phrase |
| 12.5.2 | The Delta Rule | — | already descriptive |
| 12.5.3 | Training It: the WY Trick | Chunked Training with the WY Trick | vague pronoun "It"; WY Trick alone is the content |
| 12.5.4 | Gating and the Modern Cell | Gated DeltaNet: Decay Plus the Delta Write | "the modern cell" doesn't name Gated DeltaNet |
| 12.5.4.1 | A Language Model | A Gated DeltaNet Language Model | generic; break with the chapter's own "A minGRU/Mamba LM" pattern |
| 12.5.4.2 | The Generalized Delta Rule | — | already descriptive |
| 12.5.5 | What the Transition Can Compute | The Transition's Expressive Power: Parity and State Tracking | vague pattern; content is a concrete expressivity result |
| 12.5.5.1 | The Reflection in the Delta Rule | From Erasure to Reflection: Parity by Construction | cryptic; content is a hand-built parity construction |
| 12.5.5.2 | The Ladder, and Its Ceiling | The State-Tracking Ladder and Its TC⁰ Ceiling | trailing "and X" clause; name the concrete ceiling |
| 12.5.6 | Summary | — | already descriptive |
| 12.5.7 | Exercises | — | already descriptive |

### 12.6 Learning at Test Time <sub>`chapter_recurrent-modern/test-time-regression.md`</sub>

| # | Current title | Suggested title | Why |
|---|---|---|---|
| 12.6.1 | Learning the Bandwidth | — | already descriptive |
| 12.6.1.1 | A Fixed Kernel, Revisited | — | already descriptive |
| 12.6.1.2 | Learning It, Without Cheating | Fitting the Bandwidth by Leave-One-Out | vague pronoun + clickbait phrasing |
| 12.6.1.3 | What Learning Sharpened | Visualizing the Sharpened Kernel | vague pattern; content is a before/after plot |
| 12.6.2 | One Recipe | The Recipe: One Regression Problem, Many Layers | too vague standalone; echoes the section's own table caption |
| 12.6.2.1 | Two Loops: What Learns When | The Inner Loop and the Outer Loop | vague "what learns when"; book's own terms are clearer |
| 12.6.2.2 | The Spectrum, Measured | The Solver Spectrum, from One Pass to the Batch Solve | vague standalone; names what's actually measured |
| 12.6.3 | Deriving the Gate: Longhorn | — | names Longhorn explicitly |
| 12.6.4 | Deeper Memories: Titans | — | names Titans explicitly |
| 12.6.4.1 | A Linear Memory, By Hand | — | already descriptive |
| 12.6.4.2 | A Deep Memory via Autograd | — | already descriptive |
| 12.6.5 | Regression That Tracks: the Forecasting Connection | — | vivid but clear with its subtitle |
| 12.6.6 | Summary | — | already descriptive |
| 12.6.7 | Exercises | — | already descriptive |

### 12.7 Hybrid Architectures <sub>`chapter_recurrent-modern/hybrids.md`</sub>

| # | Current title | Suggested title | Why |
|---|---|---|---|
| 12.7.1 | What a Fixed State Cannot Do | — | clear standalone; names the copy-bound theorem |
| 12.7.2 | The Economics | Only Attention Pays Rent | too vague ("economics" of what); reuses the section's own line |
| 12.7.3 | The Experiment: One Attention Layer Rescues Recall | — | already descriptive |
| 12.7.3.1 | Three Matched Models | — | already descriptive |
| 12.7.3.2 | The Recall Sweep | — | already descriptive |
| 12.7.3.3 | The Language-Modeling Panel | — | already descriptive |
| 12.7.3.4 | The Memory Bill, Measured | — | established term ("memory bill") |
| 12.7.4 | Design Rules, Measured | — | already descriptive |
| 12.7.5 | The Recipe Table | — | already descriptive |
| 12.7.6 | Distillation, and Where This Leaves Us | Distilling a Pretrained Transformer into a Hybrid | trailing "and where this leaves us" clause |
| 12.7.6.1 | The Chapter in One Table | — | already descriptive |
| 12.7.7 | Summary | — | already descriptive |
| 12.7.8 | Exercises | — | already descriptive |

## Chapter 13 — Computational Performance

### 13 Computational Performance <sub>`chapter_computational-performance/index.md`</sub>

| # | Current title | Suggested title | Why |
|---|---|---|---|
| 13 (unnumbered) | What This Chapter Is Not | — | already descriptive |
| 13 (unnumbered) | Resources and Further Reading | — | standard section, matches convention |

### 13.1 The Performance Model <sub>`chapter_computational-performance/performance-model.md`</sub>

| # | Current title | Suggested title | Why |
|---|---|---|---|
| 13.1.1 | Counting: FLOPs, Bytes, and Arithmetic Intensity | — | already descriptive |
| 13.1.2 | Measuring Without Lying | — | clear standalone; means honest GPU timing |
| 13.1.3 | The Sweep: Mapping Our GPU | — | already descriptive |
| 13.1.4 | Three Regimes | — | already descriptive |
| 13.1.5 | The Profiler | — | already descriptive |
| 13.1.6 | Summary | — | already descriptive |
| 13.1.7 | Exercises | — | already descriptive |

This section and the rest of ch. 13 are markedly better titled than ch. 12 —
almost every heading already names its content.

### 13.2 Hardware <sub>`chapter_computational-performance/hardware.md`</sub>

| # | Current title | Suggested title | Why |
|---|---|---|---|
| 13.2.1 | Where Bytes Live | — | clear; names the memory hierarchy |
| 13.2.2 | Why Compute Outruns Bandwidth | — | states the phenomenon explained |
| 13.2.3 | The GPU | — | already descriptive |
| 13.2.4 | The CPU's Role | — | already descriptive |
| 13.2.5 | Interconnects: Our Box as the Worked Example | — | "our box" is the book's own running term |
| 13.2.6 | Energy: Why Moving Bytes Is the Budget | — | already descriptive |
| 13.2.7 | Reading the Roofline: Two Workloads | — | already descriptive |
| 13.2.8 | Summary | — | already descriptive |
| 13.2.9 | Exercises | — | already descriptive |

### 13.3 Compute Graphs and Compilation <sub>`chapter_computational-performance/compilation.md`</sub>

| # | Current title | Suggested title | Why |
|---|---|---|---|
| 13.3.1 | The Graph Was Always There | The Compute Graph Hiding in Autograd | assertion-style sentence, not a noun phrase |
| 13.3.2 | Capture: Two Philosophies | — | already descriptive |
| 13.3.3 | What the Compiler Does: Fusion | — | names Fusion explicitly |
| 13.3.4 | Compiling the Training Step, Measured | — | already descriptive |
| 13.3.5 | The Overhead Regime: Capture and Replay | — | already descriptive |
| 13.3.6 | When Compilation Hurts | Costs and Limits of Compilation | clear but a clause, not a noun phrase; minor |
| 13.3.7 | Summary | — | already descriptive |
| 13.3.8 | Exercises | — | already descriptive |

### 13.4 Memory and Precision <sub>`chapter_computational-performance/memory-precision.md`</sub>

| # | Current title | Suggested title | Why |
|---|---|---|---|
| 13.4.1 | The Memory Anatomy of a Training Step | — | already descriptive |
| 13.4.2 | Measuring Memory | — | already descriptive |
| 13.4.3 | Mixed Precision | — | already descriptive |
| 13.4.4 | Activation Checkpointing | — | already descriptive |
| 13.4.5 | Gradient Accumulation | — | already descriptive |
| 13.4.6 | The Ladder So Far | The Escalation Ladder: Compile, Precision, Checkpoint, Accumulate | "so far" is filler; names the actual checklist |
| 13.4.7 | Summary | — | already descriptive |
| 13.4.8 | Exercises | — | already descriptive |

### 13.5 Multi-GPU from First Principles <sub>`chapter_computational-performance/multiple-gpus.md`</sub>

| # | Current title | Suggested title | Why |
|---|---|---|---|
| 13.5.1 | Three Ways to Split | — | already descriptive |
| 13.5.2 | Data Parallelism by Hand | — | already descriptive |
| 13.5.3 | Doing Better: Ring Allreduce | — | names ring allreduce explicitly |
| 13.5.4 | The Accounting | The Cost Model for a Data-Parallel Step | too vague standalone ("accounting" of what) |
| 13.5.5 | Summary | — | already descriptive |
| 13.5.6 | Exercises | — | already descriptive |

### 13.6 Multi-GPU in Practice <sub>`chapter_computational-performance/multi-gpu-practice.md`</sub>

| # | Current title | Suggested title | Why |
|---|---|---|---|
| 13.6.1 | What Our Hand-Rolled Loop Lacked | Three Deficits of the Hand-Rolled Loop | vague pattern; content is exactly three named deficits |
| 13.6.2 | DDP, Really Run | Launching DDP with `torchrun` | "Really Run" reads odd; names the actual mechanism |
| 13.6.3 | Sharding the Redundant: the FSDP Idea | — | already descriptive |
| 13.6.4 | JAX: Annotate the Layout, the Compiler Writes the Collectives | JAX: Declarative Sharding with `PartitionSpec` | two clauses joined, not a noun phrase |
| 13.6.5 | When One Node Is Not Enough | Scaling Beyond a Single Node | full clause; content is multi-node parallelism's scope |
| 13.6.6 | Summary | — | already descriptive |
| 13.6.7 | Exercises | — | already descriptive |

### 13.7 Case Study: Making a Transformer Fast <sub>`chapter_computational-performance/fast-transformer.md`</sub>

| # | Current title | Suggested title | Why |
|---|---|---|---|
| 13.7.1 | The Subject | The Model Under Test | too vague standalone ("the subject" of what) |
| 13.7.2 | Rung 0: Baseline, Profiled | — | already descriptive |
| 13.7.3 | Rungs, Each One Measured | Climbing the Ladder, One Rung at a Time | vague; covers five distinct named optimization rungs |
| 13.7.4 | The Waterfall | — | book's own established name for this exact plot |
| 13.7.5 | The Lore, and the Ladder Beyond | Further Rungs: The Modded-NanoGPT Speedrun | trailing "and X" clause |
| 13.7.6 | Summary | — | already descriptive |
| 13.7.7 | Exercises | — | already descriptive |

# TOC review — chapter_reinforcement-learning (14) and chapter_deep-reinforcement-learning (15)

Verification note up front: all five titles Alex quoted verbatim live in
`chapter_deep-reinforcement-learning/sac.md`, and all five have been rewritten
in the current source. They now read:

| Alex's quote | Current title | Verdict |
|---|---|---|
| "15.5.2.3 The stable form, and what the epsilon hides" | A Numerically Stable Log-Determinant | fixed, matches content |
| "15.5.3.1 Two critics, and why the minimum" | Twin Critics and the Pessimistic Minimum | fixed, matches content |
| "15.5.3.3 Nothing here needs a ratio" | Off-Policy Learning without Importance Ratios | fixed, matches content |
| "15.5.4 What It Bought" | Sample Efficiency, Entropy and Calibration | fixed, matches content |
| "15.5.4.2 The entropy the policy keeps" | The Entropy Trace and the Cost of Noise | fixed, matches content |

Below is the complete heading table for both chapters, in book order. Chapter
14 (the older chapter) was already clean before this pass and remains so.
Chapter 15 is clean except for the one remaining item flagged in its own
table (`sac.md`, 15.5.3 "The Algorithm").

---

### 14.1 Markov Decision Processes <sub>`chapter_reinforcement-learning/mdp.md`</sub>

| # | Current title | Suggested title | Why |
|---|---|---|---|
| 14.1.1 | The Model | — | already descriptive |
| 14.1.1.1 | States, Actions and the Transition Kernel | — | names the three objects |
| 14.1.1.2 | Reward Design and Potential-Based Shaping | — | names topic and technique |
| 14.1.2 | Return, Discount and Horizon | — | already descriptive |
| 14.1.2.1 | The Geometric Bound and the Effective Horizon | — | already descriptive |
| 14.1.2.2 | Episodes, Termination and Truncation | — | already descriptive |
| 14.1.3 | The Choice of State | — | already descriptive |
| 14.1.3.1 | The Markov Assumption and State Augmentation | — | already descriptive |
| 14.1.3.2 | Partial Observability | — | already descriptive |
| 14.1.4 | Bandits, Degenerate MDPs and the Model-Based Axis | — | names the three topics |
| 14.1.4.1 | The Bandit as a One-State MDP | — | already descriptive |
| 14.1.4.2 | The Degenerate MDP: Deterministic Transitions, Terminal Reward | — | already descriptive |
| 14.1.4.3 | Model-Based versus Model-Free Methods | — | already descriptive |
| 14.1.5 | Summary | — | standard closing heading |
| 14.1.6 | Exercises | — | standard closing heading |

### 14.2 Dynamic Programming <sub>`chapter_reinforcement-learning/value-iter.md`</sub>

| # | Current title | Suggested title | Why |
|---|---|---|---|
| 14.2.1 | Policies and Value Functions | — | already descriptive |
| 14.2.1.1 | State Values and Action Values | — | already descriptive |
| 14.2.1.2 | The Identity Linking $V$ and $Q$ | — | names the exact result |
| 14.2.1.3 | The Advantage Function | — | already descriptive |
| 14.2.2 | The Bellman Equations | — | already descriptive |
| 14.2.2.1 | The Expectation Form | — | already descriptive |
| 14.2.2.2 | The Optimality Equation | — | already descriptive |
| 14.2.2.3 | Backup Diagrams | — | already descriptive |
| 14.2.3 | Convergence via Contraction | — | already descriptive |
| 14.2.3.1 | The Contraction Proposition | — | already descriptive |
| 14.2.3.2 | Four Consequences: Uniqueness, Rate, Stopping Rule and the $\gamma=1$ Boundary | — | lists the four things named |
| 14.2.4 | Value Iteration, Policy Evaluation and Policy Iteration | — | names the three algorithms |
| 14.2.4.1 | Value Iteration | — | already descriptive |
| 14.2.4.2 | Policy Evaluation | — | already descriptive |
| 14.2.4.3 | Policy Iteration and Generalized Policy Iteration | — | already descriptive |
| 14.2.4.4 | The Optimal Policy on Slippery Ice | — | already descriptive |
| 14.2.5 | Summary | — | standard closing heading |
| 14.2.6 | Exercises | — | standard closing heading |

### 14.3 Learning from Demonstrations <sub>`chapter_reinforcement-learning/imitation.md`</sub>

| # | Current title | Suggested title | Why |
|---|---|---|---|
| 14.3.1 | Behavior Cloning | — | already descriptive |
| 14.3.1.1 | The Reduction to Classification | — | already descriptive |
| 14.3.1.2 | The Train-Equals-Test Assumption | — | already descriptive |
| 14.3.2 | Compounding Error | — | already descriptive |
| 14.3.2.1 | The Error Bounds: $\varepsilon T^2$ versus $\varepsilon T$ | — | states the exact result |
| 14.3.2.2 | The Divergence of the State Distributions | — | already descriptive |
| 14.3.3 | Interactive Data Collection | — | already descriptive |
| 14.3.3.1 | DAgger: Dataset Aggregation | — | already descriptive |
| 14.3.3.2 | Supervised Fine-Tuning and Offline RL | — | already descriptive |
| 14.3.3.3 | Modern Imitation Learning | — | already descriptive |
| 14.3.4 | Summary | — | standard closing heading |
| 14.3.5 | Exercises | — | standard closing heading |

### 14.4 Temporal Differences, Q-Learning and Exploration <sub>`chapter_reinforcement-learning/qlearning.md`</sub>

| # | Current title | Suggested title | Why |
|---|---|---|---|
| 14.4.1 | The Sampled Backup | — | already descriptive |
| 14.4.1.1 | From the Bellman Operator to the TD Update | — | already descriptive |
| 14.4.1.2 | Convergence and the Double-Sampling Problem | — | already descriptive |
| 14.4.1.3 | Step Sizes and the Robbins-Monro Conditions | — | already descriptive |
| 14.4.1.4 | Terminal Masking | — | already descriptive |
| 14.4.2 | Q-Learning on the Lake | — | already descriptive |
| 14.4.2.1 | The Implementation | — | clear given parent section |
| 14.4.2.2 | The Learned Table against the Exact Solution | — | already descriptive |
| 14.4.2.3 | The Self-Correcting Property | — | already descriptive |
| 14.4.3 | Exploration and Regret | — | already descriptive |
| 14.4.3.1 | The Multi-Armed Bandit | — | already descriptive |
| 14.4.3.2 | Regret and the Epsilon-Greedy Family | — | already descriptive |
| 14.4.3.3 | Optimism: UCB and Thompson Sampling | — | already descriptive |
| 14.4.3.4 | Exploration in MDPs | — | already descriptive |
| 14.4.4 | Off-Policy Learning | — | already descriptive |
| 14.4.4.1 | The Off-Policy Mechanism | — | already descriptive |
| 14.4.4.2 | Maximization Bias | — | already descriptive |
| 14.4.5 | Summary | — | standard closing heading |
| 14.4.6 | Exercises | — | standard closing heading |

### 14.5 Policy Gradient <sub>`chapter_reinforcement-learning/policy-gradient.md`</sub>

| # | Current title | Suggested title | Why |
|---|---|---|---|
| 14.5.1 | Parameterizing the Policy | — | already descriptive |
| 14.5.1.1 | Softmax Preferences | — | already descriptive |
| 14.5.1.2 | The Score Function | — | already descriptive |
| 14.5.1.3 | The Case for Stochastic Policies | — | already descriptive |
| 14.5.2 | The Policy Gradient | — | already descriptive |
| 14.5.2.1 | An Optimization Problem over Trajectories | — | already descriptive |
| 14.5.2.2 | The Log-Derivative Trick | — | already descriptive |
| 14.5.2.3 | Cancellation of the Transition Probabilities | — | already descriptive |
| 14.5.2.4 | The REINFORCE Estimator | — | already descriptive |
| 14.5.2.5 | The Policy Gradient Theorem | — | already descriptive |
| 14.5.3 | Costs and Limitations | — | already descriptive |
| 14.5.3.1 | The On-Policy Data Requirement | — | already descriptive |
| 14.5.3.2 | Unbiasedness and Variance | — | already descriptive |
| 14.5.3.3 | Nonconcavity of the Objective | — | already descriptive |
| 14.5.4 | Summary | — | standard closing heading |
| 14.5.5 | Exercises | — | standard closing heading |

### 14.6 Baselines, Advantages and Variance Reduction <sub>`chapter_reinforcement-learning/baselines.md`</sub>

| # | Current title | Suggested title | Why |
|---|---|---|---|
| 14.6.1 | A Zero-Mean Identity | — | already descriptive |
| 14.6.1.1 | The Zero-Mean Lemma | — | already descriptive |
| 14.6.1.2 | The Conditional Version | — | clear given parent lemma |
| 14.6.2 | Four Uses of One Identity | — | already descriptive |
| 14.6.2.1 | Reward-to-Go and Causality | — | already descriptive |
| 14.6.2.2 | Baselines | — | already descriptive |
| 14.6.2.3 | Control Variates | — | already descriptive |
| 14.6.2.4 | The Advantage and the Learned Baseline | — | already descriptive |
| 14.6.3 | Estimator Hygiene | — | already descriptive |
| 14.6.3.1 | Batch Centering versus Variance Scaling | — | already descriptive |
| 14.6.3.2 | The Leave-One-Out Baseline | — | already descriptive |
| 14.6.3.3 | Summing over Episodes of Different Lengths | — | already descriptive |
| 14.6.3.4 | Normalized Returns and GRPO | — | already descriptive |
| 14.6.4 | The Five-Estimator Comparison | — | already descriptive |
| 14.6.4.1 | The Ladder of Estimators | — | already descriptive |
| 14.6.4.2 | Reading the Comparison across Seeds | — | already descriptive |
| 14.6.5 | Summary | — | standard closing heading |
| 14.6.6 | Exercises | — | standard closing heading |

### 14.7 From Tables to Networks <sub>`chapter_reinforcement-learning/deep-rl.md`</sub>

| # | Current title | Suggested title | Why |
|---|---|---|---|
| 14.7.1 | Continuous States and Function Approximation | — | already descriptive |
| 14.7.1.1 | CartPole | — | names the environment |
| 14.7.1.2 | Replacing the Table with a Network | — | already descriptive |
| 14.7.1.3 | Tables as Linear Networks on One-Hot States | — | already descriptive |
| 14.7.2 | Continuous Actions and Stochastic Policies | — | already descriptive |
| 14.7.2.1 | The Gaussian Policy | — | already descriptive |
| 14.7.2.2 | The Score Function with Continuous Actions | — | already descriptive |
| 14.7.2.3 | Score-Function versus Pathwise Gradients | — | already descriptive |
| 14.7.2.4 | The Argmax over Continuous Actions | — | already descriptive |
| 14.7.3 | Generalization across States | — | already descriptive |
| 14.7.3.1 | Measuring State Coupling | — | already descriptive |
| 14.7.3.2 | Why Policy Gradients Survive Function Approximation | — | full clause, states the claim |
| 14.7.4 | The Surrogate Loss | — | already descriptive |
| 14.7.5 | Summary | — | standard closing heading |
| 14.7.6 | Exercises | — | standard closing heading |

### 15.1 Actor-Critic and the Credit-Assignment Dial <sub>`chapter_deep-reinforcement-learning/actor-critic.md`</sub>

| # | Current title | Suggested title | Why |
|---|---|---|---|
| 15.1.1 | Bootstrapping the Reward-to-Go | — | already descriptive |
| 15.1.1.1 | The TD Error | — | already descriptive |
| 15.1.1.2 | The TD Error as an Advantage Estimate | — | already descriptive |
| 15.1.1.3 | The Critic as Sampled Policy Evaluation | — | already descriptive |
| 15.1.1.4 | The Bias-Variance Trade-off | — | already descriptive |
| 15.1.2 | The Actor-Critic Algorithm | — | already descriptive |
| 15.1.2.1 | The Actor and Critic Updates | — | already descriptive |
| 15.1.2.2 | Two Timescales | — | already descriptive |
| 15.1.2.3 | A Batched Actor-Critic and A2C | — | already descriptive |
| 15.1.2.4 | The Critic's Moving Target and Data Freshness | — | already descriptive |
| 15.1.3 | n-Step Returns, Lambda-Returns and GAE | — | already descriptive |
| 15.1.3.1 | n-Step Returns | — | already descriptive |
| 15.1.3.2 | The Lambda-Return and the Telescoping Identity | — | already descriptive |
| 15.1.3.3 | GAE as a Backward Scan | — | already descriptive |
| 15.1.3.4 | Eligibility Traces | — | already descriptive |
| 15.1.4 | Measuring the Bias-Variance Trade-off | — | already descriptive |
| 15.1.5 | Summary | — | standard closing heading |
| 15.1.6 | Exercises | — | standard closing heading |

### 15.2 Trust Regions and Proximal Policy Optimization <sub>`chapter_deep-reinforcement-learning/ppo.md`</sub>

| # | Current title | Suggested title | Why |
|---|---|---|---|
| 15.2.1 | Parameter Space versus Policy Space | — | already descriptive |
| 15.2.2 | Reusing Data with Importance Sampling | — | already descriptive |
| 15.2.2.1 | The Change of Measure | — | already descriptive |
| 15.2.2.2 | The Exploding Product of Ratios | — | names the failure mode |
| 15.2.2.3 | The Per-Step Surrogate | — | already descriptive |
| 15.2.2.4 | The Length-Normalized Trajectory Ratio | — | already descriptive |
| 15.2.3 | Bounding the Step | — | already descriptive |
| 15.2.3.1 | The Performance Difference Lemma | — | already descriptive |
| 15.2.3.2 | Trust Regions and the Monotonic-Improvement Bound | — | already descriptive |
| 15.2.3.3 | The Clipped Objective | — | already descriptive |
| 15.2.3.4 | Asymmetric Clipping Bands | — | already descriptive |
| 15.2.3.5 | The Entropy Bonus | — | already descriptive |
| 15.2.4 | PPO in Practice | — | already descriptive |
| 15.2.4.1 | The Choice of Advantage Estimate | — | already descriptive |
| 15.2.4.2 | The Implementation | — | clear given parent section |
| 15.2.4.3 | Ablating the Clip | — | already descriptive |
| 15.2.4.4 | Training Diagnostics | — | already descriptive |
| 15.2.4.5 | Vectorized Collection and Minibatch Updates | — | already descriptive |
| 15.2.4.6 | Omitted Implementation Details | — | already descriptive |
| 15.2.5 | Summary | — | standard closing heading |
| 15.2.6 | Exercises | — | standard closing heading |

### 15.3 Regularized Policy Optimization <sub>`chapter_deep-reinforcement-learning/regularized.md`</sub>

| # | Current title | Suggested title | Why |
|---|---|---|---|
| 15.3.1 | Learning Rewards from Preferences | — | already descriptive |
| 15.3.1.1 | Preferences and the Bradley-Terry Model | — | already descriptive |
| 15.3.1.2 | Identifiability and the Per-Prompt Baseline | — | already descriptive |
| 15.3.1.3 | Dense versus Terminal Rewards | — | already descriptive |
| 15.3.2 | Optimizing a Proxy Reward | — | already descriptive |
| 15.3.2.1 | Reward Hacking and Goodhart's Law | — | already descriptive |
| 15.3.2.2 | True Return against the KL Budget | — | already descriptive |
| 15.3.3 | The Regularized Objective | — | already descriptive |
| 15.3.3.1 | The Closed-Form Optimum | — | already descriptive |
| 15.3.3.2 | Four Consequences of the Closed Form | — | already descriptive |
| 15.3.3.3 | Trust Region versus Penalty | — | already descriptive |
| 15.3.3.4 | The Direction of the KL Divergence | — | already descriptive |
| 15.3.4 | The Soft Backup and Its Consequences | — | already descriptive |
| 15.3.4.1 | Maximum-Entropy Reinforcement Learning and the Soft Backup | — | already descriptive |
| 15.3.4.2 | DDPG, TD3 and SAC | — | names the three methods |
| 15.3.4.3 | Connections to Later Sections | — | already descriptive |
| 15.3.5 | Summary | — | standard closing heading |
| 15.3.6 | Exercises | — | standard closing heading |

### 15.4 Deep Q-Networks <sub>`chapter_deep-reinforcement-learning/dqn.md`</sub>

| # | Current title | Suggested title | Why |
|---|---|---|---|
| 15.4.1 | Sources of Instability | — | already descriptive |
| 15.4.1.1 | The Semi-Gradient Update | — | already descriptive |
| 15.4.1.2 | Correlated Data | — | already descriptive |
| 15.4.1.3 | Moving Targets | — | already descriptive |
| 15.4.1.4 | The Deadly Triad | — | already descriptive |
| 15.4.2 | Replay and Target Networks | — | already descriptive |
| 15.4.2.1 | Experience Replay and the Off-Policy License | — | already descriptive |
| 15.4.2.2 | The Target Network | — | already descriptive |
| 15.4.2.3 | Ablating the Target Network | — | already descriptive |
| 15.4.3 | Overestimation | — | already descriptive |
| 15.4.3.1 | Measuring the Maximization Bias | — | already descriptive |
| 15.4.3.2 | Double DQN | — | names the method |
| 15.4.4 | The DQN Lineage | — | already descriptive |
| 15.4.4.1 | Extensions and Successors | — | already descriptive |
| 15.4.4.2 | DQN in Modern Practice | — | already descriptive |
| 15.4.5 | Summary | — | standard closing heading |
| 15.4.6 | Exercises | — | standard closing heading |

### 15.5 Soft Actor-Critic <sub>`chapter_deep-reinforcement-learning/sac.md`</sub>

The chapter Alex's five quotes came from. All five have been rewritten (see
the table at the top of this fragment); one further generic heading remains,
flagged below.

| # | Current title | Suggested title | Why |
|---|---|---|---|
| 15.5.1 | The Objective and Its Two Backups | — | already descriptive |
| 15.5.1.1 | The Maximum-Entropy Objective | — | already descriptive |
| 15.5.1.2 | Soft Policy Evaluation | — | already descriptive |
| 15.5.1.3 | Soft Policy Improvement | — | already descriptive |
| 15.5.2 | Bounded Actions and the Squashed Gaussian Policy | — | already descriptive |
| 15.5.2.1 | Why Clipping Breaks the Pathwise Gradient | — | full clause, states the claim |
| 15.5.2.2 | The Tanh Change of Variables | — | already descriptive |
| 15.5.2.3 | **A Numerically Stable Log-Determinant** (was: "The stable form, and what the epsilon hides") | — | fixed by the rewrite pass; names the object, not the fix's backstory |
| 15.5.3 | The Algorithm | The SAC Algorithm | only bare "The Algorithm" heading in either chapter; every analogous section elsewhere names the method ("The Actor-Critic Algorithm", "The DQN Lineage") |
| 15.5.3.1 | **Twin Critics and the Pessimistic Minimum** (was: "Two critics, and why the minimum") | — | fixed by the rewrite pass |
| 15.5.3.2 | Polyak-Averaged Target Networks | — | already descriptive |
| 15.5.3.3 | **Off-Policy Learning without Importance Ratios** (was: "Nothing here needs a ratio") | — | fixed by the rewrite pass |
| 15.5.3.4 | The Update Step | — | clear once 15.5.3 is renamed |
| 15.5.4 | **Sample Efficiency, Entropy and Calibration** (was: "What It Bought") | — | fixed by the rewrite pass |
| 15.5.4.1 | Sample Efficiency in Environment Steps | — | already descriptive |
| 15.5.4.2 | **The Entropy Trace and the Cost of Noise** (was: "The entropy the policy keeps") | — | fixed by the rewrite pass |
| 15.5.4.3 | Critic Calibration: Predicted against Delivered | — | already descriptive |
| 15.5.5 | Summary | — | standard closing heading |
| 15.5.6 | Exercises | — | standard closing heading |

### 15.6 Which Data May Drive Which Update <sub>`chapter_deep-reinforcement-learning/offline-rl.md`</sub>

Note on the section's own (h1) title, outside the table's numbering scope but
worth flagging: **"Which Data May Drive Which Update"** is an indirect
question, the one file-level title in either chapter phrased that way (every
other file title in both chapters is a dry noun phrase — compare "Deep
Q-Networks", "Soft Actor-Critic", "Regularized Policy Optimization"). It is
not meaningless — the section genuinely organizes on-policy/off-policy/offline
by exactly this question — but it sits closer to the "no questions" defect
than any other title in the two chapters. A noun-phrase alternative such as
"Data Rules: On-Policy, Off-Policy, Offline" would match the book's
convention at this level without losing content. Flagging for Alex's
judgment rather than asserting it must change, since the brief's bar is
content-fit, not literal grammar, and this title does pass the "understandable
standing alone" test.

| # | Current title | Suggested title | Why |
|---|---|---|---|
| 15.6.1 | On-Policy and Off-Policy Updates | — | already descriptive |
| 15.6.1.1 | The Two Families of Update Rules | — | already descriptive |
| 15.6.1.2 | SARSA: Bootstrapping on the Action Taken | — | already descriptive |
| 15.6.1.3 | Bounded Staleness and Importance Ratios | — | already descriptive |
| 15.6.2 | Offline Learning | — | already descriptive |
| 15.6.2.1 | Measuring Distribution Shift | — | already descriptive |
| 15.6.2.2 | The Loss of Self-Correction | — | already descriptive |
| 15.6.2.3 | The Experiment and the Behavior-Cloning Baseline | — | already descriptive |
| 15.6.3 | Pessimism | — | already descriptive |
| 15.6.3.1 | The Count-Based Penalty $\kappa/\sqrt{n}$ | — | already descriptive |
| 15.6.3.2 | Optimism Online, Pessimism Offline | — | already descriptive |
| 15.6.3.3 | Calibration after Pessimism | — | already descriptive |
| 15.6.4 | Beyond the Gridworld | — | already descriptive |
| 15.6.4.1 | Constraining the Policy, the Values, or Both | — | already descriptive |
| 15.6.4.2 | Sequence Modeling without Bootstrapping | — | already descriptive |
| 15.6.4.3 | Offline Model Selection | — | already descriptive |
| 15.6.5 | Summary | — | standard closing heading |
| 15.6.6 | Exercises | — | standard closing heading |

### 15.7 Sequences Are Trajectories <sub>`chapter_deep-reinforcement-learning/rl-sequences.md`</sub>

| # | Current title | Suggested title | Why |
|---|---|---|---|
| 15.7.1 | Text Generation as a Markov Decision Process | — | already descriptive |
| 15.7.1.1 | Prompt, Token, Prefix and Response | — | already descriptive |
| 15.7.1.2 | Deterministic Transitions | — | already descriptive |
| 15.7.1.3 | The Factorization Proposition | — | already descriptive |
| 15.7.2 | What Collapses and What Survives | — | already descriptive |
| 15.7.2.1 | What Collapses | — | terse but names the content exactly, unlike "What It Bought" |
| 15.7.2.2 | What Survives | — | terse but names the content exactly |
| 15.7.2.3 | The Smallest Instance: Four Prompts and a Verifier | — | already descriptive |
| 15.7.2.4 | The Group Mean as the Baseline | — | already descriptive |
| 15.7.3 | Where the Reward Comes From | — | already descriptive |
| 15.7.3.1 | Learned Rewards from Preferences | — | already descriptive |
| 15.7.3.2 | Checked Rewards from Verifiers | — | already descriptive |
| 15.7.3.3 | Reward Hacking as One Mechanism | — | already descriptive |
| 15.7.4 | GRPO and the Notation Contract | — | already descriptive |
| 15.7.4.1 | GRPO Assembled from Owned Parts | — | already descriptive |
| 15.7.4.2 | Notation Inherited by the Language Models Part | — | already descriptive |
| 15.7.5 | Where to Go Next | — | standard forward-looking heading |
| 15.7.6 | Capstone Projects | — | already descriptive |
| 15.7.7 | Summary | — | standard closing heading |
| 15.7.8 | Exercises | — | standard closing heading |

# TOC title review — GANs, Diffusion (placeholder), NLP Pretraining, NLP Applications

Chapters covered: 16 (`chapter_generative-adversarial-networks`), 17
(`chapter_diffusion-models`), 18 (`chapter_natural-language-processing-pretraining`),
19 (`chapter_natural-language-processing-applications`).

Numbering taken from `CHAPTER_NUMBERING` in `tools/d2l_preprocess.py`. Files whose
only content is a chapter-opening `#` title (plus, in ch. 16/18/19, a bare
`toc` directive) carry no `##`/`###`/`####` headings and so have no table —
noted inline instead of an empty one.

## Chapter 16: Generative Adversarial Networks

`chapter_generative-adversarial-networks/index.md` — chapter opener + `toc`
directive only, no headings.

### 16.1 Generative Adversarial Networks  <sub>`chapter_generative-adversarial-networks/gan.md`</sub>

| # | Current title | Suggested title | Why |
|---|---|---|---|
| 16.1.1 | Generate Some "Real" Data | — | already descriptive |
| 16.1.2 | Generator | — | already descriptive |
| 16.1.3 | Discriminator | — | already descriptive |
| 16.1.4 | Training | — | already descriptive |
| 16.1.5 | Summary | — | already descriptive |
| 16.1.6 | Exercises | — | already descriptive |

### 16.2 Deep Convolutional Generative Adversarial Networks  <sub>`chapter_generative-adversarial-networks/dcgan.md`</sub>

| # | Current title | Suggested title | Why |
|---|---|---|---|
| 16.2.1 | The Pokemon Dataset | — | already descriptive |
| 16.2.2 | The Generator | — | already descriptive |
| 16.2.3 | Discriminator | — | already descriptive |
| 16.2.4 | Training | — | already descriptive |
| 16.2.5 | Summary | — | already descriptive |
| 16.2.6 | Exercises | — | already descriptive |

## Chapter 17: Diffusion Models (placeholder)

`chapter_diffusion-models/index.md` — a two-sentence "this chapter is under
construction" stub. Chapter-opening `#` title only, no `##`/`###` headings, so
no table.

## Chapter 18: Natural Language Processing: Pretraining

`chapter_natural-language-processing-pretraining/index.md` — chapter opener +
`toc` directive only, no headings. (`legacy-attention-lib.md` in this
directory is a build-only, unlisted library file — not in `CHAPTER_NUMBERING`,
never rendered, not part of the TOC.)

### 18.1 Encoder-Decoder Models for Sequence Transduction  <sub>`chapter_natural-language-processing-pretraining/seq2seq.md`</sub>

| # | Current title | Suggested title | Why |
|---|---|---|---|
| 18.1.1 | Sequence Transduction and the Abstraction | — | already descriptive |
| 18.1.2 | The Machine Translation Dataset | — | already descriptive |
| 18.1.3 | The Seq2Seq Model | — | already descriptive |
| 18.1.3.1 | The Encoder | — | already descriptive |
| 18.1.3.2 | The Decoder | — | already descriptive |
| 18.1.3.3 | The Loss with Masking | — | already descriptive |
| 18.1.4 | Training, Decoding, and Evaluation | — | already descriptive |
| 18.1.4.1 | Greedy Translation | — | already descriptive |
| 18.1.4.2 | Evaluation with chrF | — | already descriptive |
| 18.1.4.3 | Beam Search | — | already descriptive |
| 18.1.5 | The Fixed-Vector Bottleneck | — | names the info bottleneck concept |
| 18.1.6 | Summary | — | already descriptive |
| 18.1.7 | Exercises | — | already descriptive |

### 18.2 Word Embedding (word2vec)  <sub>`chapter_natural-language-processing-pretraining/word2vec.md`</sub>

| # | Current title | Suggested title | Why |
|---|---|---|---|
| 18.2.1 | One-Hot Vectors Are a Bad Choice | — | full claim, clear standalone |
| 18.2.2 | Self-Supervised word2vec | — | already descriptive |
| 18.2.3 | The Skip-Gram Model | — | already descriptive |
| 18.2.3.1 | Training | — | scoped by parent (skip-gram training) |
| 18.2.4 | The Continuous Bag of Words (CBOW) Model | — | already descriptive |
| 18.2.4.1 | Training | — | scoped by parent (CBOW training) |
| 18.2.5 | Summary | — | already descriptive |
| 18.2.6 | Exercises | — | already descriptive |

### 18.3 Approximate Training  <sub>`chapter_natural-language-processing-pretraining/approx-training.md`</sub>

| # | Current title | Suggested title | Why |
|---|---|---|---|
| 18.3.1 | Negative Sampling | — | already descriptive |
| 18.3.2 | Hierarchical Softmax | — | already descriptive |
| 18.3.3 | Summary | — | already descriptive |
| 18.3.4 | Exercises | — | already descriptive |

### 18.4 The Dataset for Pretraining Word Embeddings  <sub>`chapter_natural-language-processing-pretraining/word-embedding-dataset.md`</sub>

| # | Current title | Suggested title | Why |
|---|---|---|---|
| 18.4.1 | Reading the Dataset | — | already descriptive |
| 18.4.2 | Subsampling | — | already descriptive |
| 18.4.3 | Extracting Center Words and Context Words | — | already descriptive |
| 18.4.4 | Negative Sampling | — | already descriptive |
| 18.4.5 | Loading Training Examples in Minibatches | — | already descriptive |
| 18.4.6 | Putting It All Together | — | conventional wrap-up heading, reads clearly against siblings |
| 18.4.7 | Summary | — | already descriptive |
| 18.4.8 | Exercises | — | already descriptive |

### 18.5 Pretraining word2vec  <sub>`chapter_natural-language-processing-pretraining/word2vec-pretraining.md`</sub>

| # | Current title | Suggested title | Why |
|---|---|---|---|
| 18.5.1 | The Skip-Gram Model | — | already descriptive |
| 18.5.1.1 | Embedding Layer | — | already descriptive |
| 18.5.1.2 | Defining the Forward Propagation | — | already descriptive |
| 18.5.2 | Training | — | already descriptive |
| 18.5.2.1 | Binary Cross-Entropy Loss | — | already descriptive |
| 18.5.2.2 | Initializing Model Parameters | — | already descriptive |
| 18.5.2.3 | Defining the Training Loop | — | already descriptive |
| 18.5.3 | Applying Word Embeddings | — | already descriptive |
| 18.5.4 | Summary | — | already descriptive |
| 18.5.5 | Exercises | — | already descriptive |

### 18.6 Word Embedding with Global Vectors (GloVe)  <sub>`chapter_natural-language-processing-pretraining/glove.md`</sub>

| # | Current title | Suggested title | Why |
|---|---|---|---|
| 18.6.1 | Skip-Gram with Global Corpus Statistics | — | already descriptive |
| 18.6.2 | The GloVe Model | — | already descriptive |
| 18.6.3 | Interpreting GloVe from the Ratio of Co-occurrence Probabilities | — | precise, matches content |
| 18.6.4 | Summary | — | already descriptive |
| 18.6.5 | Exercises | — | already descriptive |

### 18.7 Subword Embedding  <sub>`chapter_natural-language-processing-pretraining/subword-embedding.md`</sub>

| # | Current title | Suggested title | Why |
|---|---|---|---|
| 18.7.1 | The fastText Model | — | already descriptive |
| 18.7.2 | Byte Pair Encoding | — | already descriptive |
| 18.7.3 | Summary | — | already descriptive |
| 18.7.4 | Exercises | — | already descriptive |

### 18.8 Word Similarity and Analogy  <sub>`chapter_natural-language-processing-pretraining/similarity-analogy.md`</sub>

| # | Current title | Suggested title | Why |
|---|---|---|---|
| 18.8.1 | Loading Pretrained Word Vectors | — | already descriptive |
| 18.8.2 | Applying Pretrained Word Vectors | — | already descriptive |
| 18.8.2.1 | Word Similarity | — | already descriptive |
| 18.8.2.2 | Word Analogy | — | already descriptive |
| 18.8.3 | Summary | — | already descriptive |
| 18.8.4 | Exercises | — | already descriptive |

### 18.9 Bidirectional Encoder Representations from Transformers (BERT)  <sub>`chapter_natural-language-processing-pretraining/bert.md`</sub>

| # | Current title | Suggested title | Why |
|---|---|---|---|
| 18.9.1 | From Context-Independent to Context-Sensitive | — | names the contrast, clear standalone |
| 18.9.2 | From Task-Specific to Task-Agnostic | — | names the contrast, clear standalone |
| 18.9.3 | BERT: Combining the Best of Both Worlds | — | reads clearly after the two prior headings |
| 18.9.4 | Input Representation | — | already descriptive |
| 18.9.5 | Pretraining Tasks | — | already descriptive |
| 18.9.5.1 | Masked Language Modeling | — | already descriptive |
| 18.9.5.2 | Next Sentence Prediction | — | already descriptive |
| 18.9.6 | Putting It All Together | — | conventional wrap-up heading (assembles `BERTModel`) |
| 18.9.7 | Summary | — | already descriptive |
| 18.9.8 | Exercises | — | already descriptive |

### 18.10 The Dataset for Pretraining BERT  <sub>`chapter_natural-language-processing-pretraining/bert-dataset.md`</sub>

| # | Current title | Suggested title | Why |
|---|---|---|---|
| 18.10.1 | Defining Helper Functions for Pretraining Tasks | — | already descriptive |
| 18.10.1.1 | Generating the Next Sentence Prediction Task | — | already descriptive |
| 18.10.1.2 | Generating the Masked Language Modeling Task | — | already descriptive |
| 18.10.2 | Transforming Text into the Pretraining Dataset | — | already descriptive |
| 18.10.3 | Summary | — | already descriptive |
| 18.10.4 | Exercises | — | already descriptive |

### 18.11 Pretraining BERT  <sub>`chapter_natural-language-processing-pretraining/bert-pretraining.md`</sub>

| # | Current title | Suggested title | Why |
|---|---|---|---|
| 18.11.1 | Pretraining BERT | Training BERT | duplicates the section's own title verbatim |
| 18.11.2 | Representing Text with BERT | — | already descriptive |
| 18.11.3 | Summary | — | already descriptive |
| 18.11.4 | Exercises | — | already descriptive |

## Chapter 19: Natural Language Processing: Applications

`chapter_natural-language-processing-applications/index.md` — chapter opener
only, no headings.

### 19.1 Sentiment Analysis and the Dataset  <sub>`chapter_natural-language-processing-applications/sentiment-analysis-and-dataset.md`</sub>

| # | Current title | Suggested title | Why |
|---|---|---|---|
| 19.1.1 | Reading the Dataset | — | already descriptive |
| 19.1.2 | Preprocessing the Dataset | — | already descriptive |
| 19.1.3 | Creating Data Iterators | — | already descriptive |
| 19.1.4 | Putting It All Together | — | conventional wrap-up heading |
| 19.1.5 | Summary | — | already descriptive |
| 19.1.6 | Exercises | — | already descriptive |

### 19.2 Sentiment Analysis: Using Recurrent Neural Networks  <sub>`chapter_natural-language-processing-applications/sentiment-analysis-rnn.md`</sub>

| # | Current title | Suggested title | Why |
|---|---|---|---|
| 19.2.1 | Representing Single Text with RNNs | — | already descriptive |
| 19.2.2 | Loading Pretrained Word Vectors | — | already descriptive |
| 19.2.3 | Training and Evaluating the Model | — | already descriptive |
| 19.2.4 | Summary | — | already descriptive |
| 19.2.5 | Exercises | — | already descriptive |

### 19.3 Sentiment Analysis: Using Convolutional Neural Networks  <sub>`chapter_natural-language-processing-applications/sentiment-analysis-cnn.md`</sub>

| # | Current title | Suggested title | Why |
|---|---|---|---|
| 19.3.1 | One-Dimensional Convolutions | — | already descriptive |
| 19.3.2 | Max-Over-Time Pooling | — | already descriptive |
| 19.3.3 | The textCNN Model | — | already descriptive |
| 19.3.3.1 | Defining the Model | — | scoped by parent (textCNN) |
| 19.3.3.2 | Loading Pretrained Word Vectors | — | already descriptive |
| 19.3.3.3 | Training and Evaluating the Model | — | already descriptive |
| 19.3.4 | Summary | — | already descriptive |
| 19.3.5 | Exercises | — | already descriptive |

### 19.4 Natural Language Inference and the Dataset  <sub>`chapter_natural-language-processing-applications/natural-language-inference-and-dataset.md`</sub>

| # | Current title | Suggested title | Why |
|---|---|---|---|
| 19.4.1 | Natural Language Inference | — | already descriptive |
| 19.4.2 | The Stanford Natural Language Inference (SNLI) Dataset | — | already descriptive |
| 19.4.2.1 | Reading the Dataset | — | already descriptive |
| 19.4.2.2 | Defining a Class for Loading the Dataset | — | already descriptive |
| 19.4.2.3 | Putting It All Together | — | conventional wrap-up heading |
| 19.4.3 | Summary | — | already descriptive |
| 19.4.4 | Exercises | — | already descriptive |

### 19.5 Natural Language Inference: Using Attention  <sub>`chapter_natural-language-processing-applications/natural-language-inference-attention.md`</sub>

| # | Current title | Suggested title | Why |
|---|---|---|---|
| 19.5.1 | The Model | — | already descriptive |
| 19.5.1.1 | Attending | — | already descriptive |
| 19.5.1.2 | Comparing | — | already descriptive |
| 19.5.1.3 | Aggregating | — | already descriptive |
| 19.5.1.4 | Putting It All Together | — | conventional wrap-up heading |
| 19.5.2 | Training and Evaluating the Model | — | already descriptive |
| 19.5.2.1 | Reading the dataset | Reading the Dataset | Title Case inconsistency (lowercase "dataset") |
| 19.5.2.2 | Creating the Model | — | already descriptive |
| 19.5.2.3 | Training and Evaluating the Model | Running the Training Loop | duplicates parent heading (19.5.2) verbatim |
| 19.5.2.4 | Using the Model | — | already descriptive |
| 19.5.3 | Summary | — | already descriptive |
| 19.5.4 | Exercises | — | already descriptive |

### 19.6 Fine-Tuning BERT for Sequence-Level and Token-Level Applications  <sub>`chapter_natural-language-processing-applications/finetuning-bert.md`</sub>

| # | Current title | Suggested title | Why |
|---|---|---|---|
| 19.6.1 | Single Text Classification | — | already descriptive |
| 19.6.2 | Text Pair Classification or Regression | — | already descriptive |
| 19.6.3 | Text Tagging | — | already descriptive |
| 19.6.4 | Question Answering | — | already descriptive |
| 19.6.5 | Summary | — | already descriptive |
| 19.6.6 | Exercises | — | already descriptive |

### 19.7 Natural Language Inference: Fine-Tuning BERT  <sub>`chapter_natural-language-processing-applications/natural-language-inference-bert.md`</sub>

| # | Current title | Suggested title | Why |
|---|---|---|---|
| 19.7.1 | Loading Pretrained BERT | — | already descriptive |
| 19.7.2 | The Dataset for Fine-Tuning BERT | — | already descriptive |
| 19.7.3 | Fine-Tuning BERT | — | already descriptive |
| 19.7.4 | Summary | — | already descriptive |
| 19.7.5 | Exercises | — | already descriptive |

# Part: Computer Vision / Attic — TOC title review

Chapters covered: 20 Computer Vision, 21 Gaussian Processes, 22 Hyperparameter
Optimization, 23 Recommender Systems. These are long-standing, largely
unmodified chapters (translated from the classic d2l.ai corpus); as expected,
almost every title already names its content. The handful of suggestions below
are either (a) a heading vague enough to be meaningless standing alone in a
TOC, or (b) a Title Case slip relative to sibling headings in the same file —
not the ellipsis/clickbait pattern flagged in the RL chapters.

---

## Chapter 20 — Computer Vision

### 20 Computer Vision <sub>`chapter_computer-vision/index.md`</sub>

(Front matter only — no `##`/`###`/`####` headings.)

### 20.1 Image Augmentation <sub>`chapter_computer-vision/image-augmentation.md`</sub>

| # | Current title | Suggested title | Why |
|---|---|---|---|
| 20.1.1 | Common Image Augmentation Methods | — | already descriptive |
| 20.1.1.1 | Flipping and Cropping | — | already descriptive |
| 20.1.1.2 | Changing Colors | — | already descriptive |
| 20.1.1.3 | Combining Multiple Image Augmentation Methods | — | already descriptive |
| 20.1.2 | Training with Image Augmentation | — | already descriptive |
| 20.1.2.1 | Multi-GPU Training | — | already descriptive |
| 20.1.3 | Summary | — | standard heading |
| 20.1.4 | Exercises | — | standard heading |

### 20.2 Fine-Tuning <sub>`chapter_computer-vision/fine-tuning.md`</sub>

| # | Current title | Suggested title | Why |
|---|---|---|---|
| 20.2.1 | Steps | — | names the fine-tuning procedure |
| 20.2.2 | Hot Dog Recognition | — | already descriptive |
| 20.2.2.1 | Reading the Dataset | — | already descriptive |
| 20.2.2.2 | Defining and Initializing the Model | — | already descriptive |
| 20.2.2.3 | Fine-Tuning the Model | — | already descriptive |
| 20.2.3 | Summary | — | standard heading |
| 20.2.4 | Exercises | — | standard heading |

### 20.3 Object Detection and Bounding Boxes <sub>`chapter_computer-vision/bounding-box.md`</sub>

| # | Current title | Suggested title | Why |
|---|---|---|---|
| 20.3.1 | Bounding Boxes | — | already descriptive |
| 20.3.2 | Summary | — | standard heading |
| 20.3.3 | Exercises | — | standard heading |

### 20.4 Anchor Boxes <sub>`chapter_computer-vision/anchor.md`</sub>

| # | Current title | Suggested title | Why |
|---|---|---|---|
| 20.4.1 | Generating Multiple Anchor Boxes | — | already descriptive |
| 20.4.2 | Intersection over Union (IoU) | — | already descriptive |
| 20.4.3 | Labeling Anchor Boxes in Training Data | — | already descriptive |
| 20.4.3.1 | Assigning Ground-Truth Bounding Boxes to Anchor Boxes | — | already descriptive |
| 20.4.3.2 | Labeling Classes and Offsets | — | already descriptive |
| 20.4.3.3 | An Example | Anchor Box Labeling: A Worked Example | vague alone; doesn't name the subject |
| 20.4.4 | Predicting Bounding Boxes with Non-Maximum Suppression | — | already descriptive |
| 20.4.5 | Summary | — | standard heading |
| 20.4.6 | Exercises | — | standard heading |

### 20.5 Multiscale Object Detection <sub>`chapter_computer-vision/multiscale-object-detection.md`</sub>

| # | Current title | Suggested title | Why |
|---|---|---|---|
| 20.5.1 | Multiscale Anchor Boxes | — | already descriptive |
| 20.5.2 | Multiscale Detection | — | already descriptive |
| 20.5.3 | Summary | — | standard heading |
| 20.5.4 | Exercises | — | standard heading |

### 20.6 The Object Detection Dataset <sub>`chapter_computer-vision/object-detection-dataset.md`</sub>

| # | Current title | Suggested title | Why |
|---|---|---|---|
| 20.6.1 | Downloading the Dataset | — | already descriptive |
| 20.6.2 | Reading the Dataset | — | already descriptive |
| 20.6.3 | Demonstration | Visualizing the Labeled Images | vague; doesn't say what's shown |
| 20.6.4 | Summary | — | standard heading |
| 20.6.5 | Exercises | — | standard heading |

### 20.7 Single Shot Multibox Detection <sub>`chapter_computer-vision/ssd.md`</sub>

| # | Current title | Suggested title | Why |
|---|---|---|---|
| 20.7.1 | Model | — | already descriptive |
| 20.7.1.1 | Class Prediction Layer | — | already descriptive |
| 20.7.1.2 | Bounding Box Prediction Layer | — | already descriptive |
| 20.7.1.3 | Concatenating Predictions for Multiple Scales | — | already descriptive |
| 20.7.1.4 | Downsampling Block | — | already descriptive |
| 20.7.1.5 | Base Network Block | — | already descriptive |
| 20.7.1.6 | The Complete Model | — | already descriptive |
| 20.7.2 | Training | — | already descriptive |
| 20.7.2.1 | Reading the Dataset and Initializing the Model | — | already descriptive |
| 20.7.2.2 | Defining Loss and Evaluation Functions | — | already descriptive |
| 20.7.2.3 | Training the Model | — | already descriptive |
| 20.7.3 | Prediction | — | already descriptive |
| 20.7.4 | Summary | — | standard heading |
| 20.7.5 | Exercises | — | standard heading |

### 20.8 Region-based CNNs (R-CNNs) <sub>`chapter_computer-vision/rcnn.md`</sub>

| # | Current title | Suggested title | Why |
|---|---|---|---|
| 20.8.1 | R-CNNs | — | already descriptive |
| 20.8.2 | Fast R-CNN | — | already descriptive |
| 20.8.3 | Faster R-CNN | — | already descriptive |
| 20.8.4 | Mask R-CNN | — | already descriptive |
| 20.8.5 | Summary | — | standard heading |
| 20.8.6 | Exercises | — | standard heading |

### 20.9 Semantic Segmentation and the Dataset <sub>`chapter_computer-vision/semantic-segmentation-and-dataset.md`</sub>

| # | Current title | Suggested title | Why |
|---|---|---|---|
| 20.9.1 | Image Segmentation and Instance Segmentation | — | already descriptive |
| 20.9.2 | The Pascal VOC2012 Semantic Segmentation Dataset | — | already descriptive |
| 20.9.2.1 | Data Preprocessing | — | already descriptive |
| 20.9.2.2 | Custom Semantic Segmentation Dataset Class | — | already descriptive |
| 20.9.2.3 | Reading the Dataset | — | already descriptive |
| 20.9.2.4 | Putting It All Together | — | names the assembly step, standard usage |
| 20.9.3 | Summary | — | standard heading |
| 20.9.4 | Exercises | — | standard heading |

### 20.10 Transposed Convolution <sub>`chapter_computer-vision/transposed-conv.md`</sub>

| # | Current title | Suggested title | Why |
|---|---|---|---|
| 20.10.1 | Basic Operation | — | already descriptive |
| 20.10.2 | Padding, Strides, and Multiple Channels | — | already descriptive |
| 20.10.3 | Connection to Matrix Transposition | — | already descriptive |
| 20.10.4 | Summary | — | standard heading |
| 20.10.5 | Exercises | — | standard heading |

### 20.11 Fully Convolutional Networks <sub>`chapter_computer-vision/fcn.md`</sub>

| # | Current title | Suggested title | Why |
|---|---|---|---|
| 20.11.1 | The Model | — | already descriptive |
| 20.11.2 | Initializing Transposed Convolutional Layers | — | already descriptive |
| 20.11.3 | Reading the Dataset | — | already descriptive |
| 20.11.4 | Training | — | already descriptive |
| 20.11.5 | Prediction | — | already descriptive |
| 20.11.6 | Summary | — | standard heading |
| 20.11.7 | Exercises | — | standard heading |

### 20.12 Neural Style Transfer <sub>`chapter_computer-vision/neural-style.md`</sub>

| # | Current title | Suggested title | Why |
|---|---|---|---|
| 20.12.1 | Method | — | already descriptive |
| 20.12.2 | Reading the Content and Style Images | — | already descriptive |
| 20.12.3 | Preprocessing and Postprocessing | — | already descriptive |
| 20.12.4 | Extracting Features | — | already descriptive |
| 20.12.5 | Defining the Loss Function | — | already descriptive |
| 20.12.5.1 | Content Loss | — | already descriptive |
| 20.12.5.2 | Style Loss | — | already descriptive |
| 20.12.5.3 | Total Variation Loss | — | already descriptive |
| 20.12.5.4 | Loss Function | — | already descriptive |
| 20.12.6 | Initializing the Synthesized Image | — | already descriptive |
| 20.12.7 | Training | — | already descriptive |
| 20.12.8 | Summary | — | standard heading |
| 20.12.9 | Exercises | — | standard heading |

### 20.13 Image Classification (CIFAR-10) on Kaggle <sub>`chapter_computer-vision/kaggle-cifar10.md`</sub>

| # | Current title | Suggested title | Why |
|---|---|---|---|
| 20.13.1 | Obtaining and Organizing the Dataset | — | already descriptive |
| 20.13.1.1 | Downloading the Dataset | — | already descriptive |
| 20.13.1.2 | Organizing the Dataset | — | already descriptive |
| 20.13.2 | Image Augmentation | — | already descriptive |
| 20.13.3 | Reading the Dataset | — | already descriptive |
| 20.13.4 | Defining the Model | — | already descriptive |
| 20.13.5 | Defining the Training Function | — | already descriptive |
| 20.13.6 | Training and Validating the Model | — | already descriptive |
| 20.13.7 | Classifying the Testing Set and Submitting Results on Kaggle | — | already descriptive |
| 20.13.8 | Summary | — | standard heading |
| 20.13.9 | Exercises | — | standard heading |

### 20.14 Dog Breed Identification (ImageNet Dogs) on Kaggle <sub>`chapter_computer-vision/kaggle-dog.md`</sub>

| # | Current title | Suggested title | Why |
|---|---|---|---|
| 20.14.1 | Obtaining and Organizing the Dataset | — | already descriptive |
| 20.14.1.1 | Downloading the Dataset | — | already descriptive |
| 20.14.1.2 | Organizing the Dataset | — | already descriptive |
| 20.14.2 | Image Augmentation | — | already descriptive |
| 20.14.3 | Reading the Dataset | — | already descriptive |
| 20.14.4 | Fine-Tuning a Pretrained Model | — | already descriptive |
| 20.14.5 | Defining the Training Function | — | already descriptive |
| 20.14.6 | Training and Validating the Model | — | already descriptive |
| 20.14.7 | Classifying the Testing Set and Submitting Results on Kaggle | — | already descriptive |
| 20.14.8 | Summary | — | standard heading |
| 20.14.9 | Exercises | — | standard heading |

---

## Chapter 21 — Gaussian Processes

### 21 Gaussian Processes <sub>`chapter_gaussian-processes/index.md`</sub>

(Front matter only — no `##`/`###`/`####` headings.)

### 21.1 Introduction to Gaussian Processes <sub>`chapter_gaussian-processes/gp-intro.md`</sub>

| # | Current title | Suggested title | Why |
|---|---|---|---|
| 21.1.1 | Summary | — | standard heading |
| 21.1.2 | Exercises | — | standard heading |

(This file has no `##` sections other than Summary/Exercises — it is a prose
walkthrough with figures, by design.)

### 21.2 Gaussian Process Priors <sub>`chapter_gaussian-processes/gp-priors.md`</sub>

| # | Current title | Suggested title | Why |
|---|---|---|---|
| 21.2.1 | Definition | — | already descriptive |
| 21.2.2 | A Simple Gaussian Process | — | already descriptive |
| 21.2.3 | From Weight Space to Function Space | — | already descriptive |
| 21.2.4 | The Radial Basis Function (RBF) Kernel | — | already descriptive |
| 21.2.5 | The Neural Network Kernel | — | already descriptive |
| 21.2.6 | Summary | — | standard heading |
| 21.2.7 | Exercises | — | standard heading |

### 21.3 Gaussian Process Inference <sub>`chapter_gaussian-processes/gp-inference.md`</sub>

| # | Current title | Suggested title | Why |
|---|---|---|---|
| 21.3.1 | Posterior Inference for Regression | — | already descriptive |
| 21.3.2 | Equations for Making Predictions and Learning Kernel Hyperparameters in GP Regression | Prediction and Hyperparameter-Learning Equations | 14-word title; a TOC line, not a sentence |
| 21.3.3 | Interpreting Equations for Learning and Predictions | — | already descriptive |
| 21.3.4 | Worked Example from Scratch | — | already descriptive |
| 21.3.5 | Making Life Easy with GPyTorch | — | names the tool and purpose; not a tease |
| 21.3.6 | Summary | — | standard heading |
| 21.3.7 | Exercises | — | standard heading |

---

## Chapter 22 — Hyperparameter Optimization

### 22 Hyperparameter Optimization <sub>`chapter_hyperparameter-optimization/index.md`</sub>

(Front matter only — no `##`/`###`/`####` headings.)

### 22.1 What Is Hyperparameter Optimization? <sub>`chapter_hyperparameter-optimization/hyperopt-intro.md`</sub>

| # | Current title | Suggested title | Why |
|---|---|---|---|
| 22.1.1 | The Optimization Problem | — | already descriptive |
| 22.1.1.1 | The Objective Function | — | already descriptive |
| 22.1.1.2 | The Configuration Space | — | already descriptive |
| 22.1.2 | Random Search | — | already descriptive |
| 22.1.3 | Summary | — | standard heading |
| 22.1.4 | Exercises | — | standard heading |

### 22.2 Hyperparameter Optimization API <sub>`chapter_hyperparameter-optimization/hyperopt-api.md`</sub>

| # | Current title | Suggested title | Why |
|---|---|---|---|
| 22.2.1 | Searcher | — | names the API component, clear in context |
| 22.2.2 | Scheduler | — | names the API component, clear in context |
| 22.2.3 | Tuner | — | names the API component, clear in context |
| 22.2.4 | Bookkeeping the Performance of HPO Algorithms | — | already descriptive |
| 22.2.5 | Example: Optimizing the Hyperparameters of a Convolutional Neural Network | — | already descriptive |
| 22.2.6 | Comparing HPO Algorithms | — | already descriptive |
| 22.2.7 | Summary | — | standard heading |
| 22.2.8 | Exercises | — | standard heading |

### 22.3 Asynchronous Random Search <sub>`chapter_hyperparameter-optimization/rs-async.md`</sub>

| # | Current title | Suggested title | Why |
|---|---|---|---|
| 22.3.1 | Objective Function | — | already descriptive |
| 22.3.2 | Asynchronous Scheduler | — | already descriptive |
| 22.3.3 | Visualize the Asynchronous Optimization Process | — | already descriptive |
| 22.3.4 | Summary | — | standard heading |
| 22.3.5 | Exercises | — | standard heading |

### 22.4 Multi-Fidelity Hyperparameter Optimization <sub>`chapter_hyperparameter-optimization/sh-intro.md`</sub>

| # | Current title | Suggested title | Why |
|---|---|---|---|
| 22.4.1 | Successive Halving | — | already descriptive |
| 22.4.2 | Summary | — | standard heading |

(No Exercises section in this file.)

### 22.5 Asynchronous Successive Halving <sub>`chapter_hyperparameter-optimization/sh-async.md`</sub>

| # | Current title | Suggested title | Why |
|---|---|---|---|
| 22.5.1 | Objective Function | — | already descriptive |
| 22.5.2 | Asynchronous Scheduler | — | already descriptive |
| 22.5.3 | Visualize the Optimization Process | — | already descriptive |
| 22.5.4 | Summary | — | standard heading |

(No Exercises section in this file.)

---

## Chapter 23 — Recommender Systems

### 23 Recommender Systems <sub>`chapter_recommender-systems/index.md`</sub>

(Front matter only — no `##`/`###`/`####` headings.)

### 23.1 Overview of Recommender Systems <sub>`chapter_recommender-systems/recsys-intro.md`</sub>

| # | Current title | Suggested title | Why |
|---|---|---|---|
| 23.1.1 | Collaborative Filtering | — | already descriptive |
| 23.1.2 | Explicit Feedback and Implicit Feedback | — | already descriptive |
| 23.1.3 | Recommendation Tasks | — | already descriptive |
| 23.1.4 | Summary | — | standard heading |
| 23.1.5 | Exercises | — | standard heading |

### 23.2 The MovieLens Dataset <sub>`chapter_recommender-systems/movielens.md`</sub>

| # | Current title | Suggested title | Why |
|---|---|---|---|
| 23.2.1 | Getting the Data | — | already descriptive |
| 23.2.2 | Statistics of the Dataset | — | already descriptive |
| 23.2.3 | Splitting the dataset | Splitting the Dataset | Title Case break vs. sibling headings |
| 23.2.4 | Loading the data | Loading the Data | Title Case break vs. sibling headings |
| 23.2.5 | Summary | — | standard heading |
| 23.2.6 | Exercises | — | standard heading |

### 23.3 Matrix Factorization <sub>`chapter_recommender-systems/mf.md`</sub>

| # | Current title | Suggested title | Why |
|---|---|---|---|
| 23.3.1 | The Matrix Factorization Model | — | already descriptive |
| 23.3.2 | Model Implementation | — | already descriptive |
| 23.3.3 | Evaluation Measures | — | already descriptive |
| 23.3.4 | Training and Evaluating the Model | — | already descriptive |
| 23.3.5 | Summary | — | standard heading |
| 23.3.6 | Exercises | — | standard heading |

### 23.4 AutoRec: Rating Prediction with Autoencoders <sub>`chapter_recommender-systems/autorec.md`</sub>

| # | Current title | Suggested title | Why |
|---|---|---|---|
| 23.4.1 | Model | — | already descriptive |
| 23.4.2 | Implementing the Model | — | already descriptive |
| 23.4.3 | Reimplementing the Evaluator | — | already descriptive |
| 23.4.4 | Training and Evaluating the Model | — | already descriptive |
| 23.4.5 | Summary | — | standard heading |
| 23.4.6 | Exercises | — | standard heading |

### 23.5 Personalized Ranking for Recommender Systems <sub>`chapter_recommender-systems/ranking.md`</sub>

| # | Current title | Suggested title | Why |
|---|---|---|---|
| 23.5.1 | Bayesian Personalized Ranking Loss and its Implementation | — | already descriptive |
| 23.5.2 | Hinge Loss and its Implementation | — | already descriptive |
| 23.5.3 | Summary | — | standard heading |
| 23.5.4 | Exercises | — | standard heading |

### 23.6 Neural Collaborative Filtering for Personalized Ranking <sub>`chapter_recommender-systems/neumf.md`</sub>

| # | Current title | Suggested title | Why |
|---|---|---|---|
| 23.6.1 | The NeuMF model | The NeuMF Model | Title Case break (model lowercase) |
| 23.6.2 | Model Implementation | — | already descriptive |
| 23.6.3 | Customized Dataset with Negative Sampling | — | already descriptive |
| 23.6.4 | Evaluator | — | short but clear: the ranking evaluator |
| 23.6.5 | Training and Evaluating the Model | — | already descriptive |
| 23.6.6 | Summary | — | standard heading |
| 23.6.7 | Exercises | — | standard heading |

### 23.7 Sequence-Aware Recommender Systems <sub>`chapter_recommender-systems/seqrec.md`</sub>

| # | Current title | Suggested title | Why |
|---|---|---|---|
| 23.7.1 | Model Architectures | — | already descriptive |
| 23.7.2 | Model Implementation | — | already descriptive |
| 23.7.3 | Sequential Dataset with Negative Sampling | — | already descriptive |
| 23.7.4 | Load the MovieLens 100K dataset | Loading the MovieLens 100K Dataset | Title Case break + verb-mood matches ch. 23.2 |
| 23.7.5 | Train the Model | Training and Evaluating the Model | cell also computes Hit-rate/AUC; matches sibling sections |
| 23.7.6 | Summary | — | standard heading |
| 23.7.7 | Exercises | — | standard heading |

### 23.8 Feature-Rich Recommender Systems <sub>`chapter_recommender-systems/ctr.md`</sub>

| # | Current title | Suggested title | Why |
|---|---|---|---|
| 23.8.1 | An Online Advertising Dataset | — | already descriptive |
| 23.8.2 | Dataset Wrapper | — | already descriptive |
| 23.8.3 | Summary | — | standard heading |
| 23.8.4 | Exercises | — | standard heading |

### 23.9 Factorization Machines <sub>`chapter_recommender-systems/fm.md`</sub>

| # | Current title | Suggested title | Why |
|---|---|---|---|
| 23.9.1 | 2-Way Factorization Machines | — | already descriptive |
| 23.9.2 | An Efficient Optimization Criterion | — | already descriptive |
| 23.9.3 | Model Implementation | — | already descriptive |
| 23.9.4 | Load the Advertising Dataset | — | already descriptive; correct Title Case |
| 23.9.5 | Train the Model | — | already descriptive; correct Title Case |
| 23.9.6 | Summary | — | standard heading |
| 23.9.7 | Exercises | — | standard heading |

### 23.10 Deep Factorization Machines <sub>`chapter_recommender-systems/deepfm.md`</sub>

| # | Current title | Suggested title | Why |
|---|---|---|---|
| 23.10.1 | Model Architectures | — | already descriptive |
| 23.10.2 | Implementation of DeepFM | — | already descriptive |
| 23.10.3 | Training and Evaluating the Model | — | already descriptive |
| 23.10.4 | Summary | — | standard heading |
| 23.10.5 | Exercises | — | standard heading |

# Math Appendix, Part I — Linear Algebra, Calculus, Optimization (Ch. 24–26)

## Chapter 24 — Linear Algebra  <sub>`chapter_mdl-linear-algebra/`</sub>

`index.md` — front matter only ("Linear Algebra"); the sole heading in scope-range is the `{.unnumbered}` "Resources and Further Reading" section, which is out of scope per the brief. No table needed.

### 24.1 Geometry and Linear Algebraic Operations  <sub>`chapter_mdl-linear-algebra/mdl-geometry-linear-algebraic-ops.md`</sub>

| # | Current title | Suggested title | Why |
|---|---|---|---|
| 24.1.1 | Vectors and Their Geometry | — | already descriptive |
| 24.1.1.1 | Points and Directions | — | names the two vector readings |
| 24.1.1.2 | Dot Products and Angles | — | already descriptive |
| 24.1.1.3 | Projection and Orthogonality | — | already descriptive |
| 24.1.1.4 | Span, Bases, and Subspaces | — | already descriptive |
| 24.1.1.5 | Projection onto a Subspace | — | already descriptive |
| 24.1.2 | Similarity in High Dimensions | — | covers cosine similarity, concentration |
| 24.1.3 | Hyperplanes and Decision Boundaries | — | already descriptive |
| 24.1.4 | Matrices as Linear Maps | — | already descriptive |
| 24.1.4.1 | Linear Transformations | — | already descriptive |
| 24.1.4.2 | Orthogonal Matrices | — | already descriptive |
| 24.1.4.3 | Linear Dependence, Rank, and Invertibility | — | already descriptive |
| 24.1.4.3.1 | Rank | — | already descriptive |
| 24.1.4.3.2 | Invertibility | — | already descriptive |
| 24.1.4.3.3 | Numerical Issues | Numerical Issues in Matrix Inversion | vague alone; names the operation |
| 24.1.4.4 | The Determinant | — | already descriptive |
| 24.1.4.4.1 | The Determinant in General | — | contrasts with prior 2D case, n-dim def. |
| 24.1.4.4.2 | The Unifying Theorem | Determinant, Dependence, and Invertibility | doesn't say what's unified |
| 24.1.4.4.3 | Multiplicativity | — | matches det(AB)=det(A)det(B) content |
| 24.1.5 | Tensors and Einstein Summation | — | already descriptive |
| 24.1.6 | Summary | — | conventional |
| 24.1.7 | Exercises | — | conventional |

### 24.2 Eigendecompositions  <sub>`chapter_mdl-linear-algebra/mdl-eigendecomposition.md`</sub>

| # | Current title | Suggested title | Why |
|---|---|---|---|
| 24.2.1 | Eigenvalues and Eigenvectors | — | already descriptive |
| 24.2.1.1 | Definition and Geometry | — | matches def. + ellipse picture |
| 24.2.1.2 | Finding Eigenvalues | — | already descriptive |
| 24.2.1.3 | Eigendecomposition and What It Computes | Eigendecomposition: Powers, Determinants, and Traces | "What It X" pattern Alex flagged |
| 24.2.1.3.1 | When Does an Eigenbasis Exist? Multiplicity and Diagonalizability | Multiplicity and Diagonalizability | question form; content is the two multiplicities |
| 24.2.1.3.2 | Beyond Diagonalization: The Jordan Normal Form | — | colon-qualifier names the object |
| 24.2.1.3.3 | Complex Eigenvalues Are Rotations | — | self-contained claim, not a fragment |
| 24.2.2 | Non-Normal Matrices and Transient Amplification | — | already descriptive |
| 24.2.2.1 | Normality: When Eigenvalues Control Norms | — | already descriptive |
| 24.2.2.2 | A Stable Matrix That First Grows | Transient Growth in a Diagonalizable Non-Normal Matrix | names concept, not just example's punchline |
| 24.2.2.3 | Pseudospectra: Stability under Perturbation | — | already descriptive |
| 24.2.2.4 | Products of Jacobians | — | already descriptive |
| 24.2.3 | Symmetric Matrices and Positive Definiteness | — | already descriptive |
| 24.2.3.1 | The Spectral Theorem | — | already descriptive |
| 24.2.3.2 | Positive (Semi)Definiteness | — | already descriptive |
| 24.2.3.3 | The Rayleigh Quotient: Eigenvalues as Extreme Stretches | — | already descriptive |
| 24.2.4 | Localizing and Computing Eigenvalues | — | already descriptive |
| 24.2.4.1 | Gershgorin Discs | — | already descriptive |
| 24.2.4.2 | Power Iteration | — | already descriptive |
| 24.2.4.2.1 | Aside: PageRank and the Perron–Frobenius Theorem | — | already descriptive |
| 24.2.5 | Spectral Radius, Stability, and Deep Networks | — | already descriptive |
| 24.2.5.1 | What Random Matrices Look Like | Eigenvalue Distributions of Random Matrices | "What X" pattern; names circular/MP laws |
| 24.2.6 | Summary | — | conventional |
| 24.2.7 | Exercises | — | conventional |

### 24.3 Singular Value Decomposition and Low-Rank Approximation  <sub>`chapter_mdl-linear-algebra/mdl-svd-low-rank.md`</sub>

| # | Current title | Suggested title | Why |
|---|---|---|---|
| 24.3.1 | The Singular Value Decomposition | — | already descriptive |
| 24.3.1.1 | Rotate–Scale–Rotate | — | names the SVD-as-map slogan used in text |
| 24.3.1.2 | Existence via $\mathbf{A}^\top\mathbf{A}$ | — | already descriptive |
| 24.3.1.3 | The Four Fundamental Subspaces | — | already descriptive |
| 24.3.2 | Low-Rank Approximation | — | already descriptive |
| 24.3.2.1 | Eckart–Young | — | names the theorem |
| 24.3.2.2 | Principal Component Analysis | — | already descriptive |
| 24.3.3 | Solving Linear Systems with the SVD | — | already descriptive |
| 24.3.3.1 | The Pseudoinverse and Least Squares | — | already descriptive |
| 24.3.3.2 | The Condition Number | — | already descriptive |
| 24.3.4 | The SVD in Modern Deep Learning | — | already descriptive |
| 24.3.5 | Summary | — | conventional |
| 24.3.6 | Exercises | — | conventional |

## Chapter 25 — Calculus and Automatic Differentiation  <sub>`chapter_mdl-calculus/`</sub>

`chapter_mdl-calculus/index.md` is front matter only (title "Calculus and Automatic Differentiation"); its only body heading is the `{.unnumbered}` "Resources and Further Reading" section, out of scope. No table needed.

### 25.1 Single Variable Calculus  <sub>`chapter_mdl-calculus/mdl-single-variable-calculus.md`</sub>

| # | Current title | Suggested title | Why |
|---|---|---|---|
| 25.1.1 | The Derivative | — | already descriptive |
| 25.1.1.1 | Zooming In: Every Smooth Curve Looks Like a Line | — | vivid but states the actual local-linearity fact |
| 25.1.1.2 | The Difference Quotient and the Derivative | — | already descriptive |
| 25.1.1.3 | The Small-Change Identity | — | already descriptive |
| 25.1.2 | Computing Derivatives | — | already descriptive |
| 25.1.2.1 | A Table of Common Derivatives | — | already descriptive |
| 25.1.2.2 | Four Rules from One Identity | — | names the four rules and their source |
| 25.1.3 | Linear Approximation and Gradient Descent | — | already descriptive |
| 25.1.3.1 | The Tangent Line | — | already descriptive |
| 25.1.3.2 | The Gradient-Descent Step | — | already descriptive |
| 25.1.4 | Curvature and Taylor Series | — | already descriptive |
| 25.1.4.1 | Higher-Order Derivatives and Curvature | — | already descriptive |
| 25.1.4.2 | The Mean Value Theorem | — | standard named theorem |
| 25.1.4.3 | The Best Quadratic, and the Taylor Idea | The Best Local Quadratic Approximation | comma-and tail teases later content instead of naming this one |
| 25.1.4.4 | Newton's Method | — | standard named method |
| 25.1.4.5 | Taylor Series | — | already descriptive |
| 25.1.5 | When the Tangent Fails | — | already descriptive (kinks/nonsmooth points) |
| 25.1.5.1 | One-Sided Derivatives | — | already descriptive |
| 25.1.5.2 | Subgradients and Optimality | — | already descriptive |
| 25.1.5.3 | Why SGD Shrugs | Subgradients, Conservative Fields, and SGD at Kinks | colloquial; doesn't name the actual content (conservative fields, chain rule failing at kinks, Rademacher's theorem) |
| 25.1.6 | Summary | — | conventional |
| 25.1.7 | Exercises | — | conventional |

### 25.2 Multivariable Calculus  <sub>`chapter_mdl-calculus/mdl-multivariable-calculus.md`</sub>

| # | Current title | Suggested title | Why |
|---|---|---|---|
| 25.2.1 | From Partial Derivatives to the Gradient | — | already descriptive |
| 25.2.1.1 | Partial Derivatives | — | already descriptive |
| 25.2.1.2 | The Gradient | — | already descriptive |
| 25.2.1.3 | Directional Derivatives | — | already descriptive |
| 25.2.2 | The Geometry of Gradients | — | already descriptive |
| 25.2.2.1 | Steepest Descent | — | already descriptive |
| 25.2.2.2 | Gradients and Level Sets | — | already descriptive |
| 25.2.2.3 | Tangent Planes and Linearization | — | already descriptive |
| 25.2.2.4 | Critical Points and the First-Order Test | — | already descriptive |
| 25.2.2.5 | Optimizing on a Constraint | — | already descriptive |
| 25.2.3 | The Multivariate Chain Rule | — | already descriptive |
| 25.2.3.1 | The Rule as a Sum Over Paths | — | names the path-sum formulation taught here |
| 25.2.3.2 | The Backpropagation Algorithm | — | already descriptive |
| 25.2.4 | Second-Order Structure: the Hessian | — | already descriptive |
| 25.2.4.1 | The Second-Order Taylor Approximation | — | already descriptive |
| 25.2.4.2 | The Second-Derivative Test | — | already descriptive |
| 25.2.5 | Summary | — | conventional |
| 25.2.6 | Exercises | — | conventional |

### 25.3 Matrix Calculus and Automatic Differentiation  <sub>`chapter_mdl-calculus/mdl-matrix-calculus-autodiff.md`</sub>

| # | Current title | Suggested title | Why |
|---|---|---|---|
| 25.3.1 | Derivatives of Vector- and Matrix-Valued Maps | — | already descriptive |
| 25.3.1.1 | The Jacobian as the Best Linear Approximation | — | already descriptive |
| 25.3.1.2 | The Chain Rule Is Jacobian Composition | — | states the section's exact identity |
| 25.3.1.3 | Layout Conventions: Numerator vs Denominator | — | already descriptive |
| 25.3.2 | A Few Key Identities, Derived Not Tabulated | — | names both the content (identities) and the method (derived, not memorized), matches section |
| 25.3.3 | Forward-Mode AD and Dual Numbers | — | already descriptive |
| 25.3.3.1 | Dual Numbers: an Algebra That Carries Derivatives | — | already descriptive |
| 25.3.4 | Reverse-Mode AD, the Tape, and Backprop | — | already descriptive |
| 25.3.4.1 | Why Reverse Mode Is the Right Cost Model | — | states the section's exact claim (cost-model comparison), not a teaser |
| 25.3.4.2 | The Tape: Record Forward, Replay Backward | — | already descriptive |
| 25.3.4.3 | The Cost Asymmetry, Counted | Counting the Forward- and Reverse-Mode Passes | "the," relies on prior subsection; name the measured quantity directly |
| 25.3.4.4 | Never Form the Jacobian | — | imperative names the exact principle taught (compose JVP/VJP, don't materialize J) |
| 25.3.4.5 | Hessian-Vector Products: One Order Up | — | already descriptive |
| 25.3.4.6 | Differentiating Through Equations | — | already descriptive |
| 25.3.4.7 | The Memory Trade-off and Checkpointing | — | already descriptive |
| 25.3.5 | Summary | — | conventional |
| 25.3.6 | Exercises | — | conventional |

### 25.4 Integral Calculus  <sub>`chapter_mdl-calculus/mdl-integral-calculus.md`</sub>

| # | Current title | Suggested title | Why |
|---|---|---|---|
| 25.4.1 | The Definite Integral | — | already descriptive |
| 25.4.2 | The Fundamental Theorem of Calculus | — | standard named theorem |
| 25.4.2.1 | Improper Integrals | — | already descriptive |
| 25.4.2.2 | Integration by Parts | — | standard named rule |
| 25.4.2.3 | A Note on Signed Area | — | conventional aside phrasing, names its actual subject |
| 25.4.3 | Change of Variables | — | already descriptive |
| 25.4.4 | Multiple Integrals | — | already descriptive |
| 25.4.4.1 | Double Integrals | — | already descriptive |
| 25.4.4.2 | Fubini's Theorem | — | standard named theorem |
| 25.4.4.3 | Change of Variables in Many Dimensions | — | already descriptive |
| 25.4.4.4 | The Gaussian Integral | — | already descriptive |
| 25.4.5 | Integration Meets Probability | — | vivid but names the actual topic (densities, expectations, Monte Carlo) |
| 25.4.5.1 | Differentiating under the Integral Sign | — | standard named technique |
| 25.4.5.1.1 | Pathwise and Score-Function Gradients | — | already descriptive |
| 25.4.6 | Summary | — | conventional |
| 25.4.7 | Exercises | — | conventional |

## Chapter 26 — Optimization  <sub>`chapter_mdl-optimization/`</sub>

`chapter_mdl-optimization/index.md` — chapter front matter (title "Optimization"). No numbered `##`/`###`/`####` headings in scope; only an `{.unnumbered}` "Resources and Further Reading" section, out of scope.

### 26.1 Gradient-Based Optimization  <sub>`chapter_mdl-optimization/mdl-gradient-based-optimization.md`</sub>

| # | Current title | Suggested title | Why |
|---|---|---|---|
| 26.1.1 | Descent Directions | — | names exact topic |
| 26.1.2 | Gradient Descent and Smoothness | — | names method + assumption |
| 26.1.2.1 | The Iteration and the Smoothness Assumption | — | names the iteration + hypothesis |
| 26.1.2.2 | The Descent Lemma | — | names the theorem |
| 26.1.2.3 | Guarantees Without Convexity | — | states the actual result |
| 26.1.2.4 | Backtracking Line Search | — | names the algorithm |
| 26.1.3 | The Quadratic Model and the Condition Number | — | names the model + κ |
| 26.1.3.1 | Why Quadratics Tell the Truth | The Quadratic Approximation Near a Minimum | rhetorical framing, not the topic itself |
| 26.1.3.2 | Per-Mode Contraction and the 2/L Ceiling | — | names mechanism + threshold |
| 26.1.3.3 | The Optimal Step and the $(\kappa-1)/(\kappa+1)$ Law | — | names the exact result |
| 26.1.3.4 | The Valley Picture | Bowl Versus Valley: The Geometry of Conditioning | "picture" names a figure, not the content |
| 26.1.3.5 | The Edge of Stability | — | established term (Cohen et al. 2021) |
| 26.1.3.6 | From Quadratics to Convex Functions | — | names the generalization |
| 26.1.4 | Momentum and Acceleration | — | names the topic |
| 26.1.4.1 | Inertia Against the Zig-Zag | — | vivid but names mechanism + problem solved |
| 26.1.4.2 | The $\sqrt{\kappa}$ Law | — | names the exact rate |
| 26.1.4.3 | Nesterov's Look-Ahead | — | names the method + mechanism |
| 26.1.5 | Stochastic Gradients, and Why Not Newton | Stochastic Gradients and Second-Order Methods | trailing "and why" clause joins two topics |
| 26.1.5.1 | The Cost of Exactness | Unbiased Minibatch Gradients and Their Variance | undersells the proposition actually proved |
| 26.1.5.2 | The Noise Ball and Step-Size Decay | — | names the concept + topic |
| 26.1.5.3 | Coda: Why Not Newton? | Newton's Method: Curvature as Information | phrased as a question |
| 26.1.5.4 | Quasi-Newton Methods: Curvature from Secants | — | names method + mechanism |
| 26.1.5.5 | Trust Regions: Make the Model Earn Its Radius | — | vivid but names the accept/reject mechanism |
| 26.1.6 | Summary | — | conventional |
| 26.1.7 | Exercises | — | conventional |
| 26.1.8 | Discussions | — | conventional |

### 26.2 Stochastic and Adaptive Methods  <sub>`chapter_mdl-optimization/mdl-adaptive-stochastic-methods.md`</sub>

| # | Current title | Suggested title | Why |
|---|---|---|---|
| 26.2.1 | SGD Without Convexity | — | names exact topic |
| 26.2.1.1 | What the Deterministic Theorem Leaves Open | Nonconvex, Noisy Training: A Gap Between Two Theorems | question-style framing, not a content name |
| 26.2.1.2 | The Ghadimi–Lan Rate | — | names the named result |
| 26.2.2 | Per-Coordinate Step Sizes | — | names the topic |
| 26.2.2.1 | One Global Step Must Respect the Stiffest Mode | The Bottleneck of a Single Step Size | full sentence, not a title |
| 26.2.2.2 | AdaGrad: Calibrating Steps by Accumulated Evidence | — | names method + mechanism |
| 26.2.2.3 | RMSProp: Forgetting on Purpose | — | vivid but names the actual fix |
| 26.2.2.4 | Adam: Momentum, Second Moments, and Bias Correction | — | lists the three components |
| 26.2.2.5 | When Adam Fails to Converge | — | states the exact result |
| 26.2.2.6 | The Valley, Revisited | Adam on the Ill-Conditioned Valley | "revisited" needs prior section to parse |
| 26.2.3 | Decoupled Weight Decay | — | names exact topic |
| 26.2.3.1 | The Penalty Gradient Goes Through the Preconditioner | — | states the mechanism precisely |
| 26.2.4 | Schedules and Warmup | — | names the topic |
| 26.2.4.1 | What Decay Does, and Which Shape | The Schedule Zoo: Cosine and Warmup-Stable-Decay | trailing "and which" clause, vague |
| 26.2.4.2 | Warmup: Do Not Trust an Estimated Preconditioner Cold | — | vivid but names the actual reason |
| 26.2.5 | Variance Reduction for Finite Sums | — | names the topic |
| 26.2.6 | Beyond Diagonals: the Preconditioning Ladder | — | names topic + the chapter's running metaphor |
| 26.2.7 | Summary | — | conventional |
| 26.2.8 | Exercises | — | conventional |
| 26.2.9 | Discussions | — | conventional |

### 26.3 Convex Sets and Convex Functions  <sub>`chapter_mdl-optimization/mdl-convexity.md`</sub>

| # | Current title | Suggested title | Why |
|---|---|---|---|
| 26.3.1 | Convex Sets | — | names topic |
| 26.3.1.1 | Segments That Stay Inside | — | names the defining property |
| 26.3.1.2 | The Catalog Deep Learning Uses | — | names the content (a catalog) |
| 26.3.1.3 | New Convex Sets from Old | — | names the construction |
| 26.3.2 | Convex Functions: Three Lenses | — | names the three-lens framework |
| 26.3.2.1 | The Chord Lens | — | names one lens |
| 26.3.2.2 | The First-Order Lens | — | names one lens |
| 26.3.2.3 | The Second-Order Lens | — | names one lens |
| 26.3.2.4 | Strong Convexity | — | standard term |
| 26.3.2.5 | The Subgradient | — | standard term |
| 26.3.2.6 | Checking the Lenses Numerically | — | names the verification |
| 26.3.3 | Jensen's Inequality | — | standard theorem name |
| 26.3.4 | Why Convexity Matters | — | motivational but content-accurate |
| 26.3.4.1 | Every Local Minimum Is Global | — | states the theorem exactly |
| 26.3.4.2 | From Local Steps to Global Rates | — | names the upgrade proved |
| 26.3.5 | Recognizing Convexity and Its Limits | — | names the topic |
| 26.3.5.1 | A Calculus of Convex Functions | — | names the operations |
| 26.3.5.2 | Log-Sum-Exp and the Softmax Covariance | — | names the exact result |
| 26.3.5.3 | The Convex Conjugate | — | standard term |
| 26.3.5.4 | Proximal Operators | — | standard term |
| 26.3.5.5 | Coordinate and Block Coordinate Descent | — | standard term |
| 26.3.5.6 | Reality Check: Deep Networks Are Non-Convex | — | states the conclusion directly |
| 26.3.6 | Summary | — | conventional |
| 26.3.7 | Exercises | — | conventional |
| 26.3.8 | Discussions | — | conventional |

### 26.4 Constrained Optimization and Duality  <sub>`chapter_mdl-optimization/mdl-constrained-optimization-duality.md`</sub>

| # | Current title | Suggested title | Why |
|---|---|---|---|
| 26.4.1 | Equality Constraints and Lagrange Multipliers | — | names the topic |
| 26.4.1.1 | The Geometry: No Feasible Descent | — | names the geometric argument |
| 26.4.1.2 | The Lagrangian | — | standard term |
| 26.4.1.3 | Worked Example: Closest Point on a Hyperplane | — | names the exact example |
| 26.4.2 | Inequality Constraints and the KKT Conditions | — | names the topic |
| 26.4.2.1 | Active and Inactive Constraints | — | standard terms |
| 26.4.2.2 | The Karush–Kuhn–Tucker Conditions | — | standard name |
| 26.4.3 | Projections and Projected Gradient Descent | — | names the topic |
| 26.4.3.1 | Projection onto a Convex Set | — | names the operator |
| 26.4.3.2 | Projected Gradient Descent | — | names the algorithm |
| 26.4.3.3 | Projection onto the Simplex | — | names the operator |
| 26.4.4 | The Dual Problem | — | names the topic |
| 26.4.4.1 | The Lagrange Dual Function | — | standard term |
| 26.4.4.2 | Strong Duality and Slater's Condition | — | standard terms |
| 26.4.4.3 | Duality as a Saddle Point | — | names the equivalent formulation |
| 26.4.4.4 | Multipliers Are Shadow Prices | — | states the interpretation exactly |
| 26.4.4.5 | Weight Decay Is a Constraint | — | states the equivalence exactly |
| 26.4.5 | Worked Duals: SVM, Water-Filling, and a Visible Gap | — | lists the three worked examples |
| 26.4.5.1 | The Support Vector Machine Dual | — | names the exact example |
| 26.4.5.2 | Water-Filling | — | standard term |
| 26.4.5.3 | A Duality Gap You Can See | — | names a concrete, computed example |
| 26.4.5.4 | Coda: A Map of Problem Classes | — | names the content (LP/QP/SOCP/SDP map) |
| 26.4.6 | Summary | — | conventional |
| 26.4.7 | Exercises | — | conventional |
| 26.4.8 | Discussions | — | conventional |

### 26.5 Numerical Stability and Conditioning  <sub>`chapter_mdl-optimization/mdl-numerical-stability-conditioning.md`</sub>

| # | Current title | Suggested title | Why |
|---|---|---|---|
| 26.5.1 | Floating-Point Arithmetic | — | names the topic |
| 26.5.1.1 | A Number System with Gaps | — | names the property described |
| 26.5.1.2 | Overflow, Underflow, and Mixed Precision | — | lists the exact content |
| 26.5.2 | Making Softmax and Cross-Entropy Safe | — | names the exact goal |
| 26.5.2.1 | Softmax Overflows and the Shift That Fixes It | — | names the problem + the fix |
| 26.5.2.2 | The Log-Sum-Exp Sandwich | — | names the exact result (the "sandwich" bound) |
| 26.5.2.3 | Pass Logits, Not Probabilities | — | states the section's actual rule |
| 26.5.3 | Catastrophic Cancellation | — | standard term |
| 26.5.3.1 | Subtraction Annihilates Digits | — | vivid but names the mechanism |
| 26.5.3.2 | Case Study: Variance in One Pass | — | names the worked example |
| 26.5.4 | Conditioning | — | standard term |
| 26.5.4.1 | Backward and Forward Error | — | standard terms |
| 26.5.4.2 | The Condition Number of a Linear System | — | names the exact topic |
| 26.5.4.3 | Why Normal Equations Square the Pain | Normal Equations Square the Condition Number | "the pain" is vague slang; state the result plainly |
| 26.5.4.4 | Ridge Regularization as Preconditioning | — | names the exact reframing |
| 26.5.5 | Summary | — | conventional |
| 26.5.6 | Exercises | — | conventional |
| 26.5.7 | Discussions | — | conventional |

# TOC Title Review: Math Appendix (27-28), Dynamics (29), Tools for Deep Learning (30)

## Chapter 27 -- Probability and Statistical Learning

### 27 Probability and Statistical Learning (chapter opener)  <sub>`chapter_mdl-probability-statistics/index.md`</sub>

| # | Current title | Suggested title | Why |
|---|---|---|---|
| (unnumbered) | Resources and Further Reading | — | standard unnumbered boilerplate, all chapters |

### 27.1 Random Variables  <sub>`chapter_mdl-probability-statistics/mdl-random-variables.md`</sub>

| # | Current title | Suggested title | Why |
|---|---|---|---|
| 27.1.1 | From Discrete to Continuous Probability | — | already descriptive |
| 27.1.1.1 | The Density Appears: A Thought Experiment | — | already descriptive |
| 27.1.1.2 | Densities and Their Two Defining Properties | — | already descriptive |
| 27.1.1.3 | The Cumulative Distribution Function | — | already descriptive |
| 27.1.2 | Summarizing a Distribution | — | already descriptive |
| 27.1.2.1 | The Mean | — | already descriptive |
| 27.1.2.2 | Variance and Standard Deviation | — | already descriptive |
| 27.1.2.3 | What the Standard Deviation Means: Markov and Chebyshev | — | already descriptive |
| 27.1.2.4 | Means and Variances in the Continuum | — | already descriptive |
| 27.1.3 | Several Variables | — | already descriptive |
| 27.1.3.1 | Joint and Marginal Densities | — | already descriptive |
| 27.1.3.2 | Conditional Densities and Independence | — | already descriptive |
| 27.1.3.3 | Conditional Expectation and the Tower Property | — | already descriptive |
| 27.1.3.4 | Covariance | — | already descriptive |
| 27.1.3.5 | Correlation | — | already descriptive |
| 27.1.3.6 | Change of Variables for Densities | — | already descriptive |
| 27.1.4 | Summary | — | already descriptive |
| 27.1.5 | Exercises | — | already descriptive |

### 27.2 Distributions  <sub>`chapter_mdl-probability-statistics/mdl-distributions.md`</sub>

| # | Current title | Suggested title | Why |
|---|---|---|---|
| 27.2.1 | Discrete Distributions | — | already descriptive |
| 27.2.1.1 | Bernoulli | — | already descriptive |
| 27.2.1.2 | Categorical and Multinomial | — | already descriptive |
| 27.2.1.3 | Discrete Uniform | — | already descriptive |
| 27.2.1.4 | Binomial | — | already descriptive |
| 27.2.1.5 | Poisson | — | already descriptive |
| 27.2.2 | Continuous Distributions | — | already descriptive |
| 27.2.2.1 | Continuous Uniform | — | already descriptive |
| 27.2.2.2 | Exponential | — | already descriptive |
| 27.2.2.3 | Gaussian | — | already descriptive |
| 27.2.2.4 | Laplace | — | already descriptive |
| 27.2.2.5 | Multivariate Gaussian | — | already descriptive |
| 27.2.3 | The Exponential Family | — | already descriptive |
| 27.2.3.1 | Recognizing Old Friends | Familiar Distributions as Exponential-Family Members | doesn't name which distributions or the family |
| 27.2.3.2 | Where the Form Comes From: Maximum Entropy | — | already descriptive |
| 27.2.3.3 | The Moment Property | — | already descriptive |
| 27.2.4 | Conjugate Priors | — | already descriptive |
| 27.2.4.1 | Beta--Bernoulli: Counting with Pseudo-Counts | — | already descriptive |
| 27.2.4.2 | The Rest of the Tier, and the General Fact | Gamma, Dirichlet, and the General Conjugacy Theorem | "the Tier"/"General Fact" name nothing out of context |
| 27.2.5 | Summary | — | already descriptive |
| 27.2.6 | Exercises | — | already descriptive |

### 27.3 Maximum Likelihood  <sub>`chapter_mdl-probability-statistics/mdl-maximum-likelihood.md`</sub>

| # | Current title | Suggested title | Why |
|---|---|---|---|
| 27.3.1 | The Maximum Likelihood Principle | — | already descriptive |
| 27.3.1.1 | A Worked Example: The Coin | — | already descriptive |
| 27.3.1.2 | The Negative Log-Likelihood | — | already descriptive |
| 27.3.2 | Maximum Likelihood Is Minimizing a Loss | — | already descriptive |
| 27.3.2.1 | NLL Is the Cross-Entropy to the Data | — | already descriptive |
| 27.3.2.2 | From Probabilities to Densities | — | already descriptive |
| 27.3.2.3 | Gaussian NLL Is Mean Squared Error | — | already descriptive |
| 27.3.3 | Estimator Theory: Why Maximum Likelihood Works | — | already descriptive |
| 27.3.3.1 | Fisher Information and the Score | — | already descriptive |
| 27.3.4 | MAP Estimation: Priors as Regularizers | — | already descriptive |
| 27.3.4.1 | Gaussian Priors Are Weight Decay | — | already descriptive |
| 27.3.4.2 | A Beta Prior on the Coin | — | already descriptive |
| 27.3.4.3 | The Posterior Mode Is Not the Posterior | — | already descriptive |
| 27.3.5 | Latent Variables, EM, and the ELBO | — | already descriptive |
| 27.3.5.1 | Why Latent Variables Break the Recipe | — | already descriptive |
| 27.3.5.2 | The Evidence Lower Bound | — | already descriptive |
| 27.3.5.3 | Expectation--Maximization | — | already descriptive |
| 27.3.6 | Summary | — | already descriptive |
| 27.3.7 | Exercises | — | already descriptive |

### 27.4 Bayesian Computation  <sub>`chapter_mdl-probability-statistics/mdl-bayesian-computation.md`</sub>

| # | Current title | Suggested title | Why |
|---|---|---|---|
| 27.4.1 | The Target: A Posterior Distribution | — | already descriptive |
| 27.4.1.1 | A Nonconjugate Running Example | — | already descriptive |
| 27.4.1.2 | A Grid as a Reference, Not a General Algorithm | — | already descriptive |
| 27.4.1.3 | What Averaging Buys: Mean, MAP, and Prediction | — | already descriptive |
| 27.4.2 | Importance Sampling: Correcting a Proposal | — | already descriptive |
| 27.4.3 | Markov Chain Monte Carlo | — | already descriptive |
| 27.4.3.1 | Diagnostics: Mixing, Not Just Draw Count | — | already descriptive |
| 27.4.4 | Deterministic Approximations | — | already descriptive |
| 27.4.4.1 | Laplace: the Free Gaussian in Every MAP | — | already descriptive |
| 27.4.4.2 | Variational Inference: Integration Becomes Optimization | — | already descriptive |
| 27.4.5 | A Practical Decision Map | — | already descriptive |
| 27.4.6 | Summary | — | already descriptive |
| 27.4.7 | Exercises | — | already descriptive |

### 27.5 Statistics  <sub>`chapter_mdl-probability-statistics/mdl-statistics.md`</sub>

| # | Current title | Suggested title | Why |
|---|---|---|---|
| 27.5.1 | Estimators and Their Quality | — | already descriptive |
| 27.5.1.1 | Estimators | — | already descriptive |
| 27.5.1.2 | Bias and Variance | — | already descriptive |
| 27.5.1.3 | Consistency and Efficiency | — | already descriptive |
| 27.5.2 | The Bias-Variance Decomposition | — | already descriptive |
| 27.5.2.1 | Mean Squared Error and the Decomposition | — | already descriptive |
| 27.5.2.2 | The Law of Large Numbers | — | already descriptive |
| 27.5.2.3 | The Trade-off and Generalization | The Bias-Variance Trade-off and Generalization | "the Trade-off" alone doesn't name which one |
| 27.5.2.4 | The Decomposition in Code | — | already descriptive |
| 27.5.2.5 | Why the Unbiased Variance Divides by $n-1$ | — | already descriptive |
| 27.5.3 | Hypothesis Testing | — | already descriptive |
| 27.5.3.1 | The Setup: Null, Alternative, and Two Kinds of Error | — | already descriptive |
| 27.5.3.2 | Significance and Power | — | already descriptive |
| 27.5.3.3 | Test Statistics, $p$-values, and Significance | — | already descriptive |
| 27.5.3.4 | A Worked Test: Comparing Two Models | — | already descriptive |
| 27.5.4 | Confidence Intervals | — | already descriptive |
| 27.5.4.1 | Definition and Interpretation | — | already descriptive |
| 27.5.4.2 | A Gaussian Example | — | already descriptive |
| 27.5.4.3 | The Bootstrap | — | already descriptive |
| 27.5.5 | Summary | — | already descriptive |
| 27.5.6 | Exercises | — | already descriptive |

### 27.6 Concentration and Generalization  <sub>`chapter_mdl-probability-statistics/mdl-concentration-generalization.md`</sub>

| # | Current title | Suggested title | Why |
|---|---|---|---|
| 27.6.1 | From Chebyshev to Chernoff | — | already descriptive |
| 27.6.1.1 | Polynomial Tails Are Not Enough | — | already descriptive |
| 27.6.1.2 | The Chernoff Method | — | already descriptive |
| 27.6.1.3 | Hoeffding's Lemma and Hoeffding's Inequality | — | already descriptive |
| 27.6.1.4 | Sub-Gaussian and Sub-Exponential Variables | — | already descriptive |
| 27.6.1.5 | The Tail Race in Code | — | already descriptive |
| 27.6.2 | Probability in High Dimension | — | already descriptive |
| 27.6.2.1 | Norm Concentration | — | already descriptive |
| 27.6.2.2 | Near-Orthogonality Revisited | — | already descriptive |
| 27.6.2.3 | Three Consequences for Deep Learning | — | already descriptive |
| 27.6.2.4 | Measuring the Shell | — | already descriptive |
| 27.6.3 | From One Estimate to Uniform Convergence | — | already descriptive |
| 27.6.3.1 | The Function Chosen After the Data | — | already descriptive |
| 27.6.3.2 | Finite Classes: the Union Bound | — | already descriptive |
| 27.6.3.3 | Rademacher Complexity | — | already descriptive |
| 27.6.3.4 | The Linear Class in Closed Form | — | already descriptive |
| 27.6.3.5 | Why the Bounds Go Vacuous, and Why the Language Survives | — | already descriptive |
| 27.6.3.6 | Coin Flips in Code | — | already descriptive |
| 27.6.4 | Interpolation and Double Descent | — | already descriptive |
| 27.6.4.1 | The U-Curve, Revisited | — | already descriptive |
| 27.6.4.2 | The Minimum-Norm Mechanism | — | already descriptive |
| 27.6.4.3 | Double Descent in Twenty-Six Lines | — | already descriptive |
| 27.6.4.4 | Benign Overfitting | — | already descriptive |
| 27.6.5 | Summary | — | already descriptive |
| 27.6.6 | Exercises | — | already descriptive |

### 27.7 Naive Bayes  <sub>`chapter_mdl-probability-statistics/mdl-naive-bayes.md`</sub>

| # | Current title | Suggested title | Why |
|---|---|---|---|
| 27.7.1 | Bayes' Rule for Classification | — | already descriptive |
| 27.7.1.1 | The Naive Assumption | — | already descriptive |
| 27.7.1.2 | Doing It in Log Space | — | already descriptive |
| 27.7.2 | Training Is Counting | — | already descriptive |
| 27.7.3 | A Worked Example: MNIST Digits | — | already descriptive |
| 27.7.3.1 | Estimating the Model | — | already descriptive |
| 27.7.3.2 | Classifying and Evaluating | — | already descriptive |
| 27.7.3.3 | Calibration | — | already descriptive |
| 27.7.4 | Summary | — | already descriptive |
| 27.7.5 | Exercises | — | already descriptive |

## Chapter 28 -- Information Theory and Divergences

### 28 Information Theory and Divergences (chapter opener)  <sub>`chapter_mdl-information-theory/index.md`</sub>

| # | Current title | Suggested title | Why |
|---|---|---|---|
| (unnumbered) | Resources and Further Reading | — | standard unnumbered boilerplate, all chapters |

### 28.1 Entropy, Cross-Entropy, and KL Divergence  <sub>`chapter_mdl-information-theory/mdl-information-theory.md`</sub>

| # | Current title | Suggested title | Why |
|---|---|---|---|
| 28.1.1 | Information and Entropy | — | already descriptive |
| 28.1.1.1 | Surprise and Self-Information | — | already descriptive |
| 28.1.1.2 | Shannon Entropy | — | already descriptive |
| 28.1.1.3 | Entropy Is Maximized by the Uniform Distribution | — | already descriptive |
| 28.1.2 | Cross-Entropy and KL Divergence | — | already descriptive |
| 28.1.2.1 | The Kullback--Leibler Divergence | — | already descriptive |
| 28.1.2.2 | Gibbs' Inequality | — | already descriptive |
| 28.1.2.3 | Maximum Entropy and the Gaussian | — | already descriptive |
| 28.1.2.4 | Gaussians, in Closed Form | — | already descriptive |
| 28.1.2.5 | Cross-Entropy | — | already descriptive |
| 28.1.2.6 | The Classification Loss | — | already descriptive |
| 28.1.3 | The Coding View and Perplexity | — | already descriptive |
| 28.1.3.1 | Prefix Codes and the Kraft Inequality | — | already descriptive |
| 28.1.3.2 | From Symbol Codes to Arithmetic Coding | — | already descriptive |
| 28.1.3.3 | Typical Sequences and the Source-Coding Theorem | — | already descriptive |
| 28.1.3.4 | Lossy Compression and Rate--Distortion | — | already descriptive |
| 28.1.3.5 | Noisy Channels and Capacity | — | already descriptive |
| 28.1.3.6 | Perplexity | — | already descriptive |
| 28.1.4 | Modern Uses | — | already descriptive |
| 28.1.4.1 | Learning by Compression: Minimum Description Length | — | already descriptive |
| 28.1.4.2 | Label Smoothing | — | already descriptive |
| 28.1.4.3 | Knowledge Distillation | — | already descriptive |
| 28.1.4.4 | One Principle, Many Losses | — | already descriptive |
| 28.1.5 | Summary | — | already descriptive |
| 28.1.6 | Exercises | — | already descriptive |

### 28.2 Divergences and Distances Between Distributions  <sub>`chapter_mdl-information-theory/mdl-divergences-distances.md`</sub>

| # | Current title | Suggested title | Why |
|---|---|---|---|
| 28.2.1 | What Is a Divergence? The f-Divergence Family | Divergences and the f-Divergence Family | drop the question form |
| 28.2.1.1 | Axioms, Metrics, and Three Families | — | already descriptive |
| 28.2.1.2 | The f-Divergence Template | — | already descriptive |
| 28.2.1.3 | A Gallery of Generators | — | already descriptive |
| 28.2.2 | Duality: The Variational View | — | already descriptive |
| 28.2.2.1 | The Fenchel Conjugate and the f-GAN Bound | — | already descriptive |
| 28.2.2.2 | Forward vs. Reverse KL: Mode-Covering vs. Mode-Seeking | — | already descriptive |
| 28.2.3 | Metrics: Total Variation, MMD, and Optimal Transport | — | already descriptive |
| 28.2.3.1 | Total Variation and Pinsker's Inequality | — | already descriptive |
| 28.2.3.2 | Integral Probability Metrics and MMD | — | already descriptive |
| 28.2.3.3 | Optimal Transport and the Wasserstein Distance | — | already descriptive |
| 28.2.4 | Scores: Fisher Divergence, Stein Discrepancy, and the Objective Map | — | already descriptive |
| 28.2.4.1 | The Score and the Fisher Divergence | — | already descriptive |
| 28.2.4.2 | Stein's Identity | — | already descriptive |
| 28.2.4.3 | The Kernel Stein Discrepancy | — | already descriptive |
| 28.2.4.4 | The Divergence-to-Objective Map | — | already descriptive |
| 28.2.5 | Summary | — | already descriptive |
| 28.2.6 | Exercises | — | already descriptive |

### 28.3 Mutual Information and Representation Learning  <sub>`chapter_mdl-information-theory/mdl-mutual-information.md`</sub>

| # | Current title | Suggested title | Why |
|---|---|---|---|
| 28.3.1 | Mutual Information: Definitions and Properties | — | already descriptive |
| 28.3.1.1 | From One Variable to Two: Joint and Conditional Entropy | — | already descriptive |
| 28.3.1.2 | Mutual Information as a Divergence from Independence | — | already descriptive |
| 28.3.1.3 | The Gaussian Anchor | — | already descriptive |
| 28.3.1.4 | Pointwise Mutual Information | — | already descriptive |
| 28.3.1.5 | Mutual Information as Nonlinear Correlation | — | already descriptive |
| 28.3.1.6 | Conditional Mutual Information and the Chain Rule | — | already descriptive |
| 28.3.1.7 | The Data-Processing Inequality | — | already descriptive |
| 28.3.2 | Why Measuring Mutual Information Is Hard | — | already descriptive |
| 28.3.2.1 | The Curse of Estimation | — | already descriptive |
| 28.3.2.2 | A Ceiling at log N | — | already descriptive |
| 28.3.2.3 | Watching the Ceiling: a Perfect-Critic Simulation | — | already descriptive |
| 28.3.3 | Variational Bounds and InfoNCE | — | already descriptive |
| 28.3.3.1 | The Barber--Agakov Bound | — | already descriptive |
| 28.3.3.2 | Donsker--Varadhan and MINE | — | already descriptive |
| 28.3.3.3 | The NWJ Bound and the Bias--Variance Spectrum | — | already descriptive |
| 28.3.3.4 | InfoNCE: Estimation as Classification | — | already descriptive |
| 28.3.3.5 | Experiment: Learning the Critic | — | already descriptive |
| 28.3.4 | The Information Bottleneck and the Limits of Mutual Information | — | already descriptive |
| 28.3.4.1 | Compression with a Purpose: the IB Lagrangian | — | already descriptive |
| 28.3.4.2 | The Information Plane: a Gaussian Bottleneck in Closed Form | — | already descriptive |
| 28.3.4.3 | The Compression-Phase Debate | — | already descriptive |
| 28.3.4.4 | What Mutual Information Guarantees: Fano's Inequality | — | already descriptive |
| 28.3.4.5 | What Mutual Information Estimates Can and Cannot Tell You | — | already descriptive |
| 28.3.5 | Summary | — | already descriptive |
| 28.3.6 | Exercises | — | already descriptive |

## Chapter 29 -- Dynamics: Differential Equations and Generative Flows

### 29 Dynamics: Differential Equations and Generative Flows (chapter opener)  <sub>`chapter_mdl-dynamics/index.md`</sub>

| # | Current title | Suggested title | Why |
|---|---|---|---|
| (unnumbered) | Resources and Further Reading | — | standard unnumbered boilerplate, all chapters |

### 29.1 Ordinary Differential Equations and Numerical Solvers  <sub>`chapter_mdl-dynamics/mdl-odes-solvers.md`</sub>

| # | Current title | Suggested title | Why |
|---|---|---|---|
| 29.1.1 | Vector Fields, Trajectories, and Well-Posedness | — | already descriptive |
| 29.1.1.1 | Velocity Fields and Integral Curves | — | already descriptive |
| 29.1.1.2 | The Flow Map | — | already descriptive |
| 29.1.1.3 | Existence and Uniqueness | — | already descriptive |
| 29.1.2 | Linear ODEs and Stability | — | already descriptive |
| 29.1.2.1 | The Matrix Exponential | — | already descriptive |
| 29.1.2.2 | The Stability Dictionary | — | already descriptive |
| 29.1.2.3 | Linearization at Fixed Points | — | already descriptive |
| 29.1.3 | Numerical Solvers: From Euler to Runge--Kutta | — | already descriptive |
| 29.1.3.1 | Forward Euler and Its Global Error | — | already descriptive |
| 29.1.3.2 | Runge--Kutta Methods | — | already descriptive |
| 29.1.3.3 | Stiffness and Implicit Methods | — | already descriptive |
| 29.1.3.4 | Gradient Descent Is a Solver | — | already descriptive |
| 29.1.4 | Neural ODEs and the Adjoint Method | — | already descriptive |
| 29.1.4.1 | Residual Networks Are Euler Steps | — | already descriptive |
| 29.1.4.2 | Training Through the Solver | — | already descriptive |
| 29.1.4.3 | The Adjoint Method: Backpropagation in Continuous Time | — | already descriptive |
| 29.1.5 | Continuous Normalizing Flows | — | already descriptive |
| 29.1.5.1 | The Instantaneous Change of Variables | — | already descriptive |
| 29.1.5.2 | The Hutchinson Trace Estimator | — | already descriptive |
| 29.1.6 | Summary | — | already descriptive |
| 29.1.7 | Exercises | — | already descriptive |

### 29.2 Stochastic Differential Equations  <sub>`chapter_mdl-dynamics/mdl-sdes.md`</sub>

| # | Current title | Suggested title | Why |
|---|---|---|---|
| 29.2.1 | Brownian Motion | — | already descriptive |
| 29.2.1.1 | Why Add Randomness | — | already descriptive |
| 29.2.1.2 | From Random Walks to the Wiener Process | — | already descriptive |
| 29.2.1.3 | Simulating Path Ensembles | — | already descriptive |
| 29.2.2 | Itô Calculus | — | already descriptive |
| 29.2.2.1 | Quadratic Variation: Why Ordinary Calculus Fails | — | already descriptive |
| 29.2.2.2 | The Itô Integral | — | already descriptive |
| 29.2.2.3 | Itô's Lemma | — | already descriptive |
| 29.2.3 | Stochastic Differential Equations and Euler--Maruyama | — | already descriptive |
| 29.2.3.1 | Drift, Diffusion, and What a Solution Is | — | already descriptive |
| 29.2.3.2 | The Euler--Maruyama Scheme | — | already descriptive |
| 29.2.3.3 | Strong and Weak Convergence | — | already descriptive |
| 29.2.4 | The Ornstein--Uhlenbeck Process | — | already descriptive |
| 29.2.4.1 | Solving the SDE with Itô's Lemma | — | already descriptive |
| 29.2.4.2 | The Stationary Distribution | — | already descriptive |
| 29.2.4.3 | The Variance-Preserving Normalization | — | already descriptive |
| 29.2.5 | Summary | — | already descriptive |
| 29.2.6 | Exercises | — | already descriptive |

### 29.3 The Fokker--Planck Equation and Probability Flow  <sub>`chapter_mdl-dynamics/mdl-fokker-planck-probability-flow.md`</sub>

| # | Current title | Suggested title | Why |
|---|---|---|---|
| 29.3.1 | From Paths to Densities | — | already descriptive |
| 29.3.1.1 | One Trajectory or the Whole Cloud | — | already descriptive |
| 29.3.1.2 | Three Identities from Vector Calculus | — | already descriptive |
| 29.3.1.3 | Watching the Cloud | Simulating the OU Marginal Density | doesn't say what's being verified |
| 29.3.2 | The Fokker--Planck Equation | — | already descriptive |
| 29.3.2.1 | From Itô's Lemma to a PDE | — | already descriptive |
| 29.3.2.2 | Diffusion Matrices and Boundary Conditions | — | already descriptive |
| 29.3.2.3 | Drift Transports, Diffusion Smooths | — | already descriptive |
| 29.3.2.4 | The Ornstein--Uhlenbeck Check | — | already descriptive |
| 29.3.3 | The Continuity Equation and the Probability-Flow ODE | — | already descriptive |
| 29.3.3.1 | Conservation of Probability | — | already descriptive |
| 29.3.3.2 | Diffusion Is Transport in Disguise | — | already descriptive |
| 29.3.3.3 | The Probability-Flow ODE | — | already descriptive |
| 29.3.3.4 | One Cloud, Two Dynamics | — | already descriptive |
| 29.3.4 | The Score Function | — | already descriptive |
| 29.3.4.1 | The One Unknown | Defining the Score Function | vague pronoun; doesn't name the score |
| 29.3.4.2 | Two Worked Scores | — | already descriptive |
| 29.3.4.3 | Why Scores Beat Densities | — | already descriptive |
| 29.3.5 | Time Reversal | — | already descriptive |
| 29.3.5.1 | Bayes on an Infinitesimal Step | — | already descriptive |
| 29.3.5.2 | Anderson's Theorem | — | already descriptive |
| 29.3.5.3 | Noise Back into Data | — | already descriptive |
| 29.3.6 | Summary | — | already descriptive |
| 29.3.7 | Exercises | — | already descriptive |

### 29.4 Score Matching, Diffusion, and Flow Matching  <sub>`chapter_mdl-dynamics/mdl-score-matching-diffusion-flow.md`</sub>

| # | Current title | Suggested title | Why |
|---|---|---|---|
| 29.4.1 | Learning the Score | — | already descriptive |
| 29.4.1.1 | Why the Score? | Matching Scores Instead of Densities | drop the question form |
| 29.4.1.2 | Denoising Score Matching | — | already descriptive |
| 29.4.1.3 | A Score Network in One Dimension | — | already descriptive |
| 29.4.2 | Score-Based Diffusion Models | — | already descriptive |
| 29.4.2.1 | From One Noise Level to All of Them | — | already descriptive |
| 29.4.2.2 | DDPM as a Discretized SDE | — | already descriptive |
| 29.4.2.3 | Langevin Dynamics and Predictor–Corrector Sampling | — | already descriptive |
| 29.4.2.4 | DDIM: Trading Noise for Speed | — | already descriptive |
| 29.4.2.5 | Guidance: Steering with Bayes' Rule | — | already descriptive |
| 29.4.3 | Flow Matching and Rectified Flow | — | already descriptive |
| 29.4.3.1 | Probability Paths and Velocity Fields | — | already descriptive |
| 29.4.3.2 | The Conditional Flow Matching Theorem | — | already descriptive |
| 29.4.3.3 | Score, Noise, and Velocity Are One Function | — | already descriptive |
| 29.4.3.4 | Rectified Flow and Straight Paths | — | already descriptive |
| 29.4.3.5 | Gaussian to Two Moons, Four Ways | — | already descriptive |
| 29.4.3.6 | One Reflow Round, Measured | — | already descriptive |
| 29.4.4 | Optimal Transport and Straightness | — | already descriptive |
| 29.4.5 | Sampling Is Solving the Learned Dynamics | — | already descriptive |
| 29.4.5.1 | A Unifying Table | — | already descriptive |
| 29.4.6 | Summary | — | already descriptive |
| 29.4.7 | Exercises | — | already descriptive |

## Chapter 30 -- Tools for Deep Learning

### 30 Tools for Deep Learning (chapter opener)  <sub>`chapter_appendix-tools-for-deep-learning/index.md`</sub>

| # | Current title | Suggested title | Why |
|---|---|---|---|
| (unnumbered) | Resources and Further Reading | — | standard unnumbered boilerplate, all chapters |

### 30.1 Notebooks  <sub>`chapter_appendix-tools-for-deep-learning/interactive-development.md`</sub>

| # | Current title | Suggested title | Why |
|---|---|---|---|
| 30.1.1 | Why Notebooks? | The Case for Notebooks | drop the question form |
| 30.1.1.1 | The Document and the Kernel | — | already descriptive |
| 30.1.1.2 | Restart and Run All | — | already descriptive |
| 30.1.2 | Running the Book Locally | — | already descriptive |
| 30.1.2.1 | Setting Up | — | already descriptive |
| 30.1.2.2 | A Quick Sanity Check | — | already descriptive |
| 30.1.3 | Working in an Editor | — | already descriptive |
| 30.1.3.1 | JupyterLab | — | already descriptive |
| 30.1.3.2 | VS Code | — | already descriptive |
| 30.1.3.3 | Debugging and Timing | — | already descriptive |
| 30.1.4 | Remote Machines | — | already descriptive |
| 30.1.5 | Summary | — | already descriptive |
| 30.1.6 | Exercises | — | already descriptive |

### 30.2 Colab and Kaggle  <sub>`chapter_appendix-tools-for-deep-learning/hosted-notebooks.md`</sub>

| # | Current title | Suggested title | Why |
|---|---|---|---|
| 30.2.1 | Launching a Book Notebook | — | already descriptive |
| 30.2.2 | Colab | — | already descriptive |
| 30.2.2.1 | What the Free Tier Buys | — | already descriptive |
| 30.2.2.2 | Saving Work | — | already descriptive |
| 30.2.3 | Kaggle | — | already descriptive |
| 30.2.3.1 | A Data-Centric Model | — | already descriptive |
| 30.2.3.2 | Quotas and Hardware | — | already descriptive |
| 30.2.4 | Choosing and Working Portably | — | already descriptive |
| 30.2.4.1 | Colab or Kaggle? | Comparing Colab and Kaggle | drop the question form |
| 30.2.4.2 | Setup Cells That Survive | — | already descriptive |
| 30.2.5 | Summary | — | already descriptive |
| 30.2.6 | Exercises | — | already descriptive |

### 30.3 Cloud Computing  <sub>`chapter_appendix-tools-for-deep-learning/cloud-instances.md`</sub>

| # | Current title | Suggested title | Why |
|---|---|---|---|
| 30.3.1 | The Rental Market | — | already descriptive |
| 30.3.1.1 | Three Tiers of Provider | — | already descriptive |
| 30.3.1.2 | What Things Cost | — | already descriptive |
| 30.3.1.3 | Cost per Result, Not per Hour | — | already descriptive |
| 30.3.2 | Working on a Rented Machine | — | already descriptive |
| 30.3.2.1 | Boot, Connect, Verify | — | already descriptive |
| 30.3.2.2 | Compute Is Disposable, Results Are Not | — | already descriptive |
| 30.3.3 | Summary | — | already descriptive |
| 30.3.4 | Exercises | — | already descriptive |

### 30.4 Hardware  <sub>`chapter_appendix-tools-for-deep-learning/hardware.md`</sub>

| # | Current title | Suggested title | Why |
|---|---|---|---|
| 30.4.1 | What Are You Buying For? | Sizing the Workload: Memory and Bandwidth | drop the question form; name the content |
| 30.4.2 | Training Boxes: Discrete NVIDIA GPUs | — | already descriptive |
| 30.4.2.1 | RTX 5070 Ti: the Smallest Serious Trainer | — | already descriptive |
| 30.4.2.2 | RTX 5090: the Enthusiast Box | — | already descriptive |
| 30.4.3 | Local Inference: the Unified-Memory Class | — | already descriptive |
| 30.4.4 | The Top End: Workstation Blackwell | — | already descriptive |
| 30.4.5 | Keeping Current | — | already descriptive |
| 30.4.6 | Summary | — | already descriptive |
| 30.4.7 | Exercises | — | already descriptive |

### 30.5 Ecosystem  <sub>`chapter_appendix-tools-for-deep-learning/software-ecosystem.md`</sub>

| # | Current title | Suggested title | Why |
|---|---|---|---|
| 30.5.1 | Where to Find Things | — | already descriptive |
| 30.5.1.1 | Models | — | already descriptive |
| 30.5.1.2 | Datasets | — | already descriptive |
| 30.5.1.3 | Papers and Code | — | already descriptive |
| 30.5.2 | Choosing a Model: Benchmarks and Leaderboards | — | already descriptive |
| 30.5.3 | Staying Current | — | already descriptive |
| 30.5.4 | Using What You Found | — | already descriptive |
| 30.5.4.1 | Pin the Identity | Pin the Revision | "the Identity" is vague; content is pinning a commit |
| 30.5.4.2 | Trust and Licenses | — | already descriptive |
| 30.5.5 | Summary | — | already descriptive |
| 30.5.6 | Exercises | — | already descriptive |

### 30.6 Model Training  <sub>`chapter_appendix-tools-for-deep-learning/training-systems.md`</sub>

| # | Current title | Suggested title | Why |
|---|---|---|---|
| 30.6.1 | From One GPU to Many | — | already descriptive |
| 30.6.1.1 | The Scaling Ladder | — | already descriptive |
| 30.6.1.2 | Where the Memory Goes | — | already descriptive |
| 30.6.2 | The Library Landscape | — | already descriptive |
| 30.6.2.1 | What to Use at Which Scale | — | already descriptive |
| 30.6.3 | Keeping a Long Run Alive | — | already descriptive |
| 30.6.4 | Summary | — | already descriptive |
| 30.6.5 | Exercises | — | already descriptive |

### 30.7 Model Serving  <sub>`chapter_appendix-tools-for-deep-learning/model-serving.md`</sub>

| # | Current title | Suggested title | Why |
|---|---|---|---|
| 30.7.1 | Know Your Workload | — | already descriptive |
| 30.7.2 | Serving on Your Own Machine | — | already descriptive |
| 30.7.3 | Serving as a Service | — | already descriptive |
| 30.7.3.1 | vLLM and SGLang | — | already descriptive |
| 30.7.3.2 | One Client Contract | — | already descriptive |
| 30.7.4 | Why These Engines Are Fast | — | already descriptive |
| 30.7.5 | Operating Notes | — | already descriptive |
| 30.7.6 | Summary | — | already descriptive |
| 30.7.7 | Exercises | — | already descriptive |

### 30.8 Developers Guide  <sub>`chapter_appendix-tools-for-deep-learning/developers-guide.md`</sub>

| # | Current title | Suggested title | Why |
|---|---|---|---|
| 30.8.1 | How the Book Is Built | — | already descriptive |
| 30.8.1.1 | One Source, Many Artifacts | — | already descriptive |
| 30.8.1.2 | The Repository at a Glance | — | already descriptive |
| 30.8.2 | Git in Five Minutes | — | already descriptive |
| 30.8.3 | Working with a Coding Agent | — | already descriptive |
| 30.8.4 | Contributing Your Changes | — | already descriptive |
| 30.8.5 | Summary | — | already descriptive |
| 30.8.6 | Exercises | — | already descriptive |

### 30.9 Utility Functions and Classes  <sub>`chapter_appendix-tools-for-deep-learning/utils.md`</sub>

| # | Current title | Suggested title | Why |
|---|---|---|---|
| 30.9.1 | A Note on Framework Coverage | — | already descriptive |

### 30.10 The `d2l` API Document  <sub>`chapter_appendix-tools-for-deep-learning/d2l.md`</sub>

| # | Current title | Suggested title | Why |
|---|---|---|---|
| 30.10.1 | Classes | — | already descriptive |
| 30.10.2 | Functions | — | already descriptive |

