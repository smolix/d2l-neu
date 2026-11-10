# Dynamics: Differential Equations and Generative Flows (chapter introduction)

Source: `chapter_mdl-dynamics/index.md`. About 380 words of prose before the
resources list, which is within the 400-word limit of guide section 12.1.

## Verdict

A good chapter introduction with one template defect and one gap. The first
two paragraphs do what the guide asks: they start from two things the reader
already does (gradient steps, residual updates), name the object behind both,
and pose two questions a reader can feel ("How few steps can such a model
take, and where does the direction come from when all we have are example
images?"). The paragraph on what the Basics tools cannot do is the best
statement of purpose in the chapter. The route paragraph is then six
sentences of one shape, which is the defect the guide names explicitly, and
the introduction never says that the chapter also serves the State Space
Models chapter, nor which appendix chapters it needs. The most important fix
is to rewrite the route as three movements.

## Scores

| Explanation | Thought | Pedagogy | Language | Enjoyment |
|---|---|---|---|---|
| 4 | 3 | 3 | 4 | 4 |

## What works

- The second paragraph, in particular "A cloud of noisy particles is
  unpredictable one particle at a time, yet its density changes
  deterministically, and running that change backward turns noise into data."
  This is the whole chapter in one sentence and should be protected.
- The three limits of the Basics tools ("Backpropagation differentiates a
  fixed computation, not the solution of an equation. Probability describes a
  random variable at one time, not a density that moves. And noise whose size
  grows like the square root of time breaks the ordinary chain rule."). Each
  maps to one later section (adjoint, Fokker-Planck, Itô). This is the
  author judging what matters, as guide section 10.2 asks.
- The scope sentence is honest about leaving out measure theory.

## Findings

**IDX-1** (moderate, pedagogy). Anchor: "asks when a differential equation has" (route paragraph).
The route is six consecutive sentences of the form ":numref: asks how ...",
one per section. Guide section 12.1 names exactly this ("a run of sentences
of one shape ... readers skip such a paragraph"). The chapter has three
movements and the route should say so: (1) deterministic motion and how to
compute and differentiate it (ODEs and solvers, Neural ODEs); (2) noise and
the density it produces (SDEs, Fokker-Planck), ending in the discovery that
reversal needs one unknown field; (3) learning that field from samples
(score matching, flow matching). Draft: "The first two sections stay
deterministic: :numref:`sec_mdl-odes-solvers` asks how accurately and stably
a computer can follow a differential equation, and
:numref:`sec_mdl-neural-odes-cnf` trains a model defined by one and follows a
density along it. The next two add noise. :numref:`sec_mdl-sdes` shows how
noise carries any data distribution to a Gaussian, and
:numref:`sec_mdl-fokker-planck-probability-flow` finds that undoing this
needs a single unknown field, the score. The last two sections learn that
field, or a velocity in its place, from samples."

**IDX-2** (moderate, pedagogy). Anchor: "We assume calculus, Gaussians and conditional expectation at the level of".
The stated prerequisites are too low. The first section uses complex
eigenvalues, the eigendecomposition, the operator norm and Taylor's theorem
in several variables, and points to :numref:`sec_mdl-eigendecompositions`,
:numref:`sec_mdl-multivariable_calculus` and
:numref:`sec_mdl-gradient-based-optimization`. A reader who arrives with
"the Basics part" only is lost at the matrix exponential. Name the appendix
chapters the chapter builds on (linear algebra for eigendecompositions,
calculus for Jacobians and automatic differentiation, probability for
densities and change of variables).

**IDX-3** (moderate, thought). Anchor: "image generators are treated in :numref:`chap_diffusion`".
The introduction says what the chapter is for in one direction only
(diffusion). It is also the natural background for the State Space Models
chapter, which builds its recurrence from a differential equation sampled at
discrete steps and discretizes it with a zero-order hold
(`chapter_recurrent-modern/ssm.md`), and for the learning-rate ceiling of
the optimization chapters. Neither is named, and the SSM chapter does not
point here. One sentence in the scope paragraph should say which main-text
chapters draw on which sections. See also ODE-S3 in the review of
`mdl-odes-solvers.md`: the material the SSM chapter needs (a linear ODE with
an input, and its exact discretization) is not in the chapter at all.

**IDX-4** (minor, language). Anchor: "Every network in this book was trained by taking small steps downhill".
The first section opens with almost the same sentence ("Every network in
this book is trained by gradient descent") and repeats the same three
examples (gradient descent, residual block, diffusion sampler) in its third
paragraph. A reader going front to back meets the same opening twice within
two pages. Keep it here and let the section open directly on the bowl
$25x^2$ (see ODE-1).

**IDX-5** (minor, thought). Anchor: "It calls a network a few dozen times".
"A few dozen" is a fact about current samplers that will date, and the
few-step sections of the diffusion chapter aim at one to four calls. "It
calls a network repeatedly" loses nothing.

## For the chapter review

- The introduction is close to the guide's template; only the route
  paragraph needs rewriting (IDX-1). The same "asks how" run should be
  checked in the other five mathematics chapters.
- The resources list is well annotated and every entry points to a section
  by label. It uses `---` between title and author, which is a list
  convention and not prose, so I do not count it against the dash rule.
- The chapter's stated reader ("the Basics part") and its real reader
  (someone who has read appendix chapters 38 to 41) differ. This probably
  holds for the part: each chapter should say which earlier appendix
  chapters it needs.
