# Ordinary Differential Equations and Numerical Solvers

Source: `chapter_mdl-dynamics/mdl-odes-solvers.md`, lines 1 to 1468 (up to the
slides marker). About 9,300 words outside code, of which about 2,000 are
exercises and about 450 are the chapter notation table. Five code cells, all
NumPy and shared by the four frameworks.

## Verdict

This is one of the better sections I have read in the book, and for long
stretches it meets the bar. It opens on a failure every reader has seen (a
run that blows up) and removes the obvious explanation in four lines; it
states its one idea early ("the eigenvalues of a differential equation decide
both how its exact motion behaves and how large a step can follow it"); and
it uses the book's own best method throughout: predict a number by hand, run
the cell, compare. I recomputed every worked number, every proof step and
every claim about a stored output, and found no mathematical error of
substance (three small imprecisions are listed below). What keeps it from a
5 is structure, not sentences. The section is three sections under one
title, it re-teaches two things that have a home elsewhere in the appendix
(the learning-rate sweep of the optimization chapter, transient growth from
the linear algebra chapter), and it says each result three times (body,
closing recap, summary). The single most important fix is to cut the
duplicated gradient-descent experiment and the closing recap, and to decide
whether the linear-stability material that later sections do not use
(six phase portraits, Jordan blocks, transient growth) earns 1,500 words.

## Scores

| Explanation | Thought | Pedagogy | Language | Enjoyment |
|---|---|---|---|---|
| 4 | 4 | 4 | 4 | 4 |

Explanation and pedagogy are close to 5 in the first and third parts. The
middle part (linear stability) is a 3: it turns into a catalogue.

## What works

- **The opening.** "The equation is stable; the steps are not." The reader
  is given $\dot x=-50x$, the factor $1-50h$, and two step sizes, and has the
  whole section in miniature before any definition. The October review also
  marked this to keep. The example is then reused honestly: the stability
  figure marks the two steps, and the stiffness table's $h=0.05$ row "is the
  opening's factor $-1.5$, applied twenty times".
- **Concrete before abstract, everywhere.** Decay and rotation are solved by
  hand before "vector field" is defined; the fan of solutions of
  $\dot x=\sqrt{|x|}$ comes before the Lipschitz condition; Picard iteration
  is run on $\dot x=x$ (and produces the Taylor terms of $e^t$) before the
  theorem; Euler is the compound-interest table before it is a formula. This
  is the order MacKay uses and it should be preserved in revision.
- **Bounds are checked against the truth.** After the Euler convergence
  proof: "the bound is $\tfrac e2(e-1)h=2.3\,h$, against the true error
  $1.36\,h$: the right order, with a pessimistic constant." Few textbooks
  bother, and it teaches what a bound is.
- **The failure cases are explained from the proof.** For $\dot x=x^2$:
  "The restart length $1/(2L)=1/(4\rho)$ therefore shrinks as the solution
  grows, and the restarts can add up to a finite time." The reader sees which
  line of the proof breaks.
- **Sentences with voice:** "A particle at rest may leave at any time"; "This
  is compound interest with a matrix interest rate"; the pendulum "which
  comes to rest balanced like a pencil on its tip"; "so stability does not
  bring accuracy".
- **The cells teach.** Each one confirms a number the prose derived
  ($n\times\text{error}=0.90$, slopes $1.04$ and $4.02$, the factor $-1$ at
  $h=0.04$), and the stored outputs match the prose in every digit I checked.
  The fair comparison of RK4 and Euler at equal work ("with the same $512$
  field evaluations") is the right experiment.
- **Exercises** are the strongest set I have seen in the appendix
  (see below).

## Structural findings

**ODE-S1. Three sections under one title.** The three `##` parts each carry a
`sec_` label (`sec_mdl-vector-fields-trajectories`,
`sec_mdl-linear-odes-stability`, `sec_mdl-euler-runge-kutta`), each opens
with its own roadmap, and each is 2,000 to 2,700 words. The main text points
only at the third (`sec_mdl-euler-runge-kutta`, three times from the
diffusion chapter). A reader arriving from there lands in the middle of a
notebook whose running example and variables were set up 600 lines earlier.
Consider splitting after "Linearization at Fixed Points" into "Ordinary
Differential Equations" (flows, existence, linear systems) and "Numerical
Solvers" (Euler, Runge-Kutta, stiffness, gradient descent). If it stays one
section, the solver part needs a two-line reminder of the spiral
($\mathbf A$, its eigenvalues, its exact solution) at its start.

**ODE-S2. Material that nothing downstream uses.** The chapter's later
sections need: the flow map and its invertibility, Lipschitz, the matrix
exponential, "real part decides decay", Euler and its order, Heun, and
stability and stiffness. They do not need the six-row portrait table and
figure, the Jordan-block paragraph, Hartman-Grobman with the pendulum, or
transient growth (about 1,500 words and one cell together). These are good
pages, but they are a detour in a chapter whose stated purpose is generative
flows. I would keep the pendulum (it is the only nonlinear example and four
exercises use it), keep the classification as three bullets, drop the table
(the figure carries it), cut the Jordan paragraph to a clause, and reduce
transient growth to a short paragraph that cites
:numref:`subsec_mdl-transient-growth` (see ODE-10).

**ODE-S3. Missing: the linear ODE with an input, and its exact step.** The
State Space Models chapter starts from $\dot h=\lambda h+bx$, holds the input
fixed over a step, and solves the step exactly (zero-order hold,
$\bar a=e^{\Delta\lambda}$); one of its exercises compares this with forward
and backward Euler. That is this section's subject, yet the section never
treats a driven linear equation (variation of constants), never mentions
exact discretization, and the SSM chapter does not point here. Half a page
would serve both that chapter and the diffusion samplers (exponential
integrators): solve $\dot{\mathbf x}=\mathbf A\mathbf x+\mathbf b$ over one
step exactly, note that the exact factor $e^{h\lambda}$ lies inside the unit
circle for every $h$ whenever $\operatorname{Re}\lambda<0$, and compare with
Euler's $1+h\lambda$ in the stability figure. The SDE section already uses an
integrating factor "that solves the decay ODE" (`mdl-sdes.md`, the OU proof)
without this section having shown one.

**ODE-S4. Thin connection to the main text.** The section names three uses
(gradient descent, residual blocks, diffusion sampling) but references no
main-text section by label: the only `:numref:` targets are appendix
sections and figures. Add a label reference for each use where it is first
named (the residual-network section, the gradient-descent section of the
optimization chapter, the samplers section of the diffusion chapter, which
uses Heun's method). `chapter_convolutional-modern/resnet.md` does not say
that a residual block is an Euler step, so the reader has not met the idea
before; one sentence there with a forward reference would connect the two.

**ODE-S5. The notation table for the whole chapter sits in this section.**
Thirty-four rows, of which twenty-one belong to later sections and mean
nothing yet ($\bar\alpha_t$, responsibilities, coupling). It interrupts the
opening between the roadmap and the first cell, and a reader of the
flow-matching section must know to look in the ODE section for it. Move it
to the chapter introduction or to the end of the chapter, and keep here only
the rows marked "ODEs and solvers".

## Passage-level findings

**ODE-1** (minor, language). Opening. Anchor: "Every network in this book is trained by gradient descent, and anyone who has".
The chapter introduction opens with nearly the same sentence, and the third
paragraph here ("Deep learning meets this gap between an equation and its
steps in three places") repeats its three examples. Open directly on the
bowl: "Anyone who has trained a network has seen a run blow up because the
learning rate was too large." Keep the "three places" paragraph but give
each place its label (ODE-S4).

**ODE-2** (minor, thought). Opening and "Gradient Descent as a Solver". Anchor: "We can now return to the question of the opening."
The opening's question ("If the loss is not the problem, what is?") is
answered completely in the next paragraph, so there is nothing to return to
1,100 lines later. What the last subsection adds is the step from one
dimension to many (the steepest direction sets the step, the flattest sets
the time). Say that: "The opening treated one coordinate. With several, the
same factor appears once per eigenvalue of the Hessian."

**ODE-3** (moderate, pedagogy). "The Initial-Value Problem" and "The Flow Map". Anchor: "assuming that each initial value has a unique solution (we prove this next)".
The flow map is defined, its composition law is argued "by uniqueness", and
its invertibility is promised "once we have shown ... (the corollary below)",
all before existence and uniqueness. The reader holds three IOUs for a page.
The order "fan of solutions, Picard-Lindelöf, corollary, then the flow map
and its group law" needs no forward promises and puts the motivating failure
($\sqrt{|x|}$) first, which is the stronger opening for the part.

**ODE-4** (minor, pedagogy). "Existence and Uniqueness", proof. Anchor: "(This is the one place where the".
The proof is the classical short one and is worth its place, because the
discussion of $\dot x=x^2$ afterwards reads the failure off its last step.
For the stated reader the uniform-convergence step is the hard one, and it
is handled in a parenthesis about "the completeness of the continuous
paths", a term the appendix does not define (I found no definition in the
linear algebra or calculus chapters). Either say in plain words what
is used ("a sequence of continuous paths whose successive differences are
summable has a continuous limit") or mark the step as taken on trust.

**ODE-5** (minor, correctness, sure). "Linear ODEs and Stability", first paragraph. Anchor: "they describe every other field near a point where it vanishes".
Not every: only near hyperbolic fixed points, as the same part says 340
lines later ("When an eigenvalue lies on the imaginary axis, the
linearization decides nothing"). Write "they describe most other fields near
a point where the field vanishes".

**ODE-6** (minor, thought). "The Matrix Exponential". Anchor: "define the **matrix exponential** by the power series that".
The linear algebra chapter already states the series, the eigenvalue formula
and the stability reading in one paragraph
(`mdl-eigendecomposition.md`, "The *matrix exponential*
$e^{\mathbf A t}=\sum_{j\ge0}(\mathbf A t)^j/j!$ solves the differential
equation"). This section is the right home for the definition and proofs
(the October review's complaint about a double definition is therefore half
resolved). What is missing is the acknowledgement: one clause here
("stated without proof in :numref:`sec_mdl-eigendecompositions`") and the
earlier paragraph cut to a pointer.

**ODE-7** (minor, language). Pattern, 4 instances in this file and the next. Anchors: "The one step that is not routine pulls the Lipschitz bound through the"; "The one step that needed care was the split of the exponential."; "(This is the one place where the"; and in `mdl-neural-odes.md` "The step that is not routine is to apply uniqueness backward".
Telling the reader which step is the hard one is exactly what guide section
10.2 asks for, and the first use is good. Four uses of one formula make it a
tic. Keep the first, vary or drop the others.

**ODE-8** (moderate, explanation). "Stability from Eigenvalues". Anchor: "suitable real basis of the plane spanned by the real and imaginary parts of".
The sentence "In a suitable real basis of the plane spanned by the real and
imaginary parts of the eigenvector, $\mathbf A$ acts there as
$a\mathbf I+b\mathbf K$" does not parse on first reading ("In a suitable
basis of the plane ..., A acts there as") and asserts the one fact about
complex eigenvalues that a reader with college linear algebra does not have.
Two lines suffice: write $\mathbf v=\mathbf p+\mathrm i\mathbf q$, take real
and imaginary parts of $\mathbf A\mathbf v=(a+\mathrm ib)\mathbf v$ to get
$\mathbf A\mathbf p=a\mathbf p-b\mathbf q$ and
$\mathbf A\mathbf q=b\mathbf p+a\mathbf q$, and read off the block. Also,
"the first subsection" (twice, here and under the matrix exponential) is a
positional reference; name the subsection or the equation.

**ODE-9** (minor, pedagogy). "Stability from Eigenvalues". Anchor: "give six standard pictures. (A zero eigenvalue gives a whole line of fixed".
The table, the figure and the three bullets above it say the same thing
three ways. The figure is the best of the three (I rendered it; the panels
are clear and each carries its eigenvalue condition as a title). Drop the
table and let the caption carry the names.

**ODE-10** (moderate, thought). "Transient Growth". Anchor: "The cause is the geometry of the eigenvectors. They are".
The mechanism (nearly parallel eigenvectors, a short vector as the
difference of two long ones, cancellation undone by different decay rates)
is taught in full in :numref:`subsec_mdl-transient-growth` with its own
triangular matrix, closed form, figure and cell. Here it is taught again in
full with a second triangular matrix, closed form, cell and exercise. By
guide section 7.7 one of them is the home. The continuous-time closed form
$x_1(t)=10(e^{-t}-e^{-2t})$ is short and worth keeping; the eigenvector
paragraph, the normal-matrix sentence and the cell can become "the reason is
the one of :numref:`subsec_mdl-transient-growth`: the eigenvectors are
$5.7^\circ$ apart". The subsection also does not say why a deep learning
reader needs it; the only use named is an exercise on momentum. Either give
the use in one sentence in the text (the first iterations of a momentum run
can grow although the run converges) or shorten further. The plot in this
cell is the only one in the section that does not carry a prediction the
printout does not already confirm.

**ODE-11** (minor, correctness, sure but pedantic). "Transient Growth". Anchor: "only $5.7^\circ$ apart, so the short".
With the signs given, $\mathbf v_1=(1,0)$ and
$\mathbf v_2=(-0.995,0.0995)$ are $174.3^\circ$ apart as vectors; it is the
lines they span that are $5.7^\circ$ apart, and
$\mathbf x_0=10.0\,\mathbf v_1+10.05\,\mathbf v_2$ is written as a sum while
the next clause calls it "the difference of two long ones". Take
$\mathbf v_2=(0.995,-0.0995)$ and write
$\mathbf x_0=10.0\,\mathbf v_1-10.05\,\mathbf v_2$, so that the picture and
the words agree (the linear algebra chapter says "The two lines are ...
apart", which is the careful wording).

**ODE-12** (minor, explanation). "Forward Euler". Anchor: "The ratios of consecutive errors are $1.53,1.69,1.82,1.90,1.95$".
A good table, but the sentence after it ("For large $n$ the error is about
$e/(2n)=1.36\,h$") arrives unexplained. One clause gives it:
$(1+1/n)^n=\exp(n\log(1+1/n))=\exp(1-\tfrac1{2n}+\dots)\approx e(1-\tfrac1{2n})$.
The reader then owns the constant that the proof's bound is compared with
later.

**ODE-13** (minor, explanation). "Runge--Kutta Methods". Anchor: "These weights are Simpson's rule,".
Simpson's rule appears nowhere else in the appendix (I searched the six
chapters), so for many readers this explains one unknown by another. Either add "which integrates cubics exactly" so the remark carries
its point, or cut it. In the same subsection "Many diffusion samplers use
this two-stage method" should point to :numref:`sec_diffusion-samplers`,
which uses Heun's method and already points back here.

**ODE-14** (minor, explanation). "Runge--Kutta Methods". Anchor: "Euler needs about $10^6$ field".
"ignoring constants" does a lot of work in "Euler needs about $10^6$ field
evaluations and RK4 about $4\times10^{1.5}\approx130$". The cell that
follows gives real numbers on the spiral that make the same point without
the caveat (Euler $3.6\times10^{-3}$ at 512 evaluations, RK4
$1.3\times10^{-9}$). Consider moving the cost sentence after the cell and
quoting those.

**ODE-15** (minor, correctness, sure but pedantic). "Stiffness and Implicit Methods". Anchor: "$\{z:|R(z)|<1\}$ where its update decays".
The stability region is defined with a strict inequality and A-stability
with $|R(z)|\le1$ on $\operatorname{Re}z\le0$, and then "Backward Euler is
A-stable, since its region is everything outside the unit disc around $+1$".
With the strict definition the region excludes $z=0$ and the boundary
circle, so the "since" does not quite follow. Define the region with
$\le$, as Hairer and Wanner do, or state A-stability on the open half-plane.

**ODE-16** (minor, pedagogy). "Stiffness and Implicit Methods". Anchor: "A system is **stiff** when some Jacobian eigenvalues have large negative real".
The definition of stiffness comes after the proposition, the stability
regions, A-stability and the cost of implicit steps, although it is the
problem that motivates all of them. The two-mode example
($\mathrm{diag}(-50,-1)$: "The fast mode is gone ... yet it still sets" the
step) is the felt problem; put it and the definition first, then the
regions, then backward Euler as the remedy.

**ODE-17** (major, thought). "Gradient Descent as a Solver". Anchor: "How do the two directions share the work as $\eta$ approaches the ceiling?".
This cell is the same experiment as the `gradient-based-optimization-eta-sweep`
cell of :numref:`sec_mdl-gradient-based-optimization`: the same quadratic
($\mathrm{diag}(10,1)$ there written $\mathrm{diag}(1,10)$), the same start
$(1,1)$, a sweep of $\eta$ across the same ceiling $0.2$, the same two
per-mode factors printed, and the norm after a fixed number of steps. The
prose that reads it repeats that section's conclusions ("Stability fixes
$\eta$ through $\lambda_{\max}$, and $\lambda_{\min}$ then sets the speed").
The text even says the other section "derived the learning-rate ceiling".
What this subsection owns is the three-arrow derivation (linearize, Euler,
stability) and the two identifications "condition number = stiffness" and
"divergence = solver instability". Keep those and cut the cell and its
paragraph. If a cell is wanted, let it show something only the solver view
predicts: for example that backward Euler on the same gradient flow (the
proximal-point step $\mathbf x_{k+1}=(\mathbf I+\eta\mathbf H)^{-1}\mathbf x_k$)
converges at $\eta=0.25$ and at $\eta=25$, or that an RK4 step of the
gradient flow moves the ceiling from $2/\lambda_{\max}$ to
$2.79/\lambda_{\max}$ at four gradient evaluations per step, which is a
worse trade. Either is a prediction the reader cannot get from the
optimization chapter.

**ODE-18** (moderate, language). End of "Gradient Descent as a Solver". Anchor: "We can now read from the eigenvalues of a field whether its motion decays,".
A recap paragraph in the "We can now ..." formula, followed immediately by a
Summary that restates the same results a third time. Guide section 12.2
excludes both the separate recap and the formula. All six sections of this
chapter close this way. Keep only the last two sentences ("Every field in
this section was given to us. The next section ... defines a model whose
layers are Euler steps of a field that is *learned*"), which belong to the
argument and are good.

**ODE-19** (minor, language). Pattern: stacked roadmaps, 5 instances. Anchors: "We first ask when a differential equation has exactly one solution, and we"; "We start from two simple motions, write down the general"; "We solve them first and name the"; "This part solves them,"; "This part".
The section has a roadmap in the opening, one at the head of each `##` part,
and one under the first `###`. The part-level ones are needed only because
the section is three sections long (ODE-S1). After a split, one roadmap per
section is enough.

**ODE-20** (minor, language). Pattern: every cell is introduced by a question, 5 of 5 here and 6 of 6 in the next section. Anchors: "Do the series, the eigendecomposition and the compound-interest limit"; "Does the closed form agree with the eigendecomposition formula, and how large"; "Are the slopes $1$ and $4$ measurable?"; "How do the two directions share the work as".
Posing the question before the cell is the right method and I would not
remove it. As the only way to introduce a cell it becomes a visible
template by the third use. Vary it: sometimes state the prediction as a
claim ("The slopes should be 1 and 4."), which is also stronger.

## Exercises

Ten problems, about 2,000 words, within the appendix band. This is an
excellent set: almost every problem produces a result a reader would want to
know (the RK4 stability polynomial on the imaginary axis; heavy ball as a
discretized damped oscillator with $\eta=h^2/m$, $\beta=1-ch/m$; the
modified loss of implicit gradient regularization; the exactly conserved
quantity $q^2+p^2+hqp$ of symplectic Euler; why an adaptive Heun controller
settles at $h=0.04$ on the stiff system). I worked the algebra of exercises
2c, 3, 5a to 5c, 6a to 6e and 10a and found them correct and answerable.

- Exercise 2a ("find the largest step size for which forward Euler decays"
  on the spiral) is answered in the text ("which is below $1$ if and only if
  $h<0.8$"). Cut it or change the field.
- Exercise 8a asks the reader to solve $\dot x=-x^3$ and $\dot x=x^3$, whose
  solutions the text gives in the last paragraph before "Numerical Solvers".
  Keep only $\dot x=x^2$ and the question about Hartman-Grobman.
- Exercise 3c leans on an exercise of another chapter for its starting
  inequality. It states the inequality, so it is answerable; fine.
- The difficulty marks are uneven: the five-part symplectic exercise and the
  transient-growth exercise carry no mark while the adaptive-step exercise
  carries one dot. Review the marks.
- Exercise 5 (implicit gradient regularization) is the best connection to
  deep learning in the section and is hidden in the exercises. One sentence
  in "Gradient Descent as a Solver" should say that the discretization error
  of gradient descent is itself a regularizer and point to the exercise.

## Still open from October

The October review asked to keep the opening example and "a diverging
learning rate is a solver instability"; both survive. Its one complaint
about this file (the matrix exponential defined twice) is half open: see
ODE-6.

## For the chapter review

- This section carries the chapter's deterministic foundations and sets up
  the running example (the spiral) that the next section reuses for the
  adjoint and for densities. That reuse works well and should be kept.
- It assumes the eigendecomposition and operator norm of the linear algebra
  chapter and the per-mode analysis of gradient descent from the
  optimization chapter, and it repeats part of both (ODE-10, ODE-17). I
  suspect the same holds across the part: results with a home in one
  appendix chapter are re-derived, with a new example, in another.
- It is the only section of the chapter with main-text inbound references
  (the diffusion chapter points at `sec_mdl-euler-runge-kutta`), and it has
  no outbound references to the main text at all.
- The "We can now ..." closing recap before the Summary occurs in all six
  sections of the chapter, and the question-before-every-cell template in
  both sections I read.
- A stale reference elsewhere:
  `chapter_mdl-calculus/mdl-matrix-calculus-autodiff.md` says "The adjoint
  method for differentiating through ODE solvers rests on the same idea
  (:numref:`sec_mdl-odes-solvers`)". The adjoint method now lives in
  :numref:`sec_mdl-adjoint-method`.
- The chapter lacks the driven linear ODE and its exact discretization,
  which the State Space Models chapter uses (ODE-S3).
