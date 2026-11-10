# Single Variable Calculus

Source: `chapter_mdl-calculus/mdl-single-variable-calculus.md`. About 6,600
words of prose before the exercises and 2,400 words of exercises (12
problems). Ten code cells, three of them in four framework tabs.

## Verdict

This is good textbook writing with one real idea running through it: a slope
is a prediction, and the second derivative says how far the prediction can be
trusted. The opening computes one number, the gap of $0.00125$ between the
logistic loss and its tangent, and the section explains that number three
times at increasing depth (measured, as a Taylor coefficient, as a descent
guarantee). That is the method of the model texts. I found no mathematical
errors; every worked number, every proof, and every exercise I recomputed is
correct. What keeps the section from the bar is scope. It is a calculus
section for the first half and an optimization section for the second
(gradient descent, the descent lemma, Newton's method with its error
recursion, subgradients), and chapter 40 proves the descent lemma, Newton's
quadratic convergence, and the subgradient again. The first thousand words,
in turn, repeat the Basics section at the same level. The single most
important thing to fix is to decide what this section owns: I would keep the
line "linear model, its error, the step it permits" and give Newton's
convergence theory and most of the subgradient material one home in
chapter 40.

## Scores

| Clarity of explanation | Clarity of thought | Pedagogy | Language | Ease and enjoyment |
|---|---|---|---|---|
| 4 | 4 | 4 | 4 | 3 |

## What works

- **The opening example and its three returns.** "the line is off by only
  $0.00125$" is predicted to be $\tfrac12\cdot\tfrac14\cdot0.1^2$, then
  measured (gap$/\epsilon^2$ prints $0.1249$, $0.1250$, $0.1250$), then
  explained by the Taylor coefficient $\tfrac18$, then used in the descent
  lemma ("so $L=\tfrac14$ and every $\eta<8$ decreases $\ell$ from any
  starting weight"). This is a prediction confirmed by a number, on a
  function the reader has trained with. Protect it in revision.
- **The proofs are short and the right ones.** The chain rule proof with the
  auxiliary $\rho(t)$ avoids the classical division by zero without drawing
  attention to itself. The second-derivative test is three lines and is
  followed by the sentence that tells the reader what was hard: "The step
  that is not routine is the last one: a sign pattern of the slope at points
  becomes a minimum on an interval only through the Mean Value Theorem."
  The optimality proof for convex functions ends "Both directions are this
  one sentence." These are sentences with an author in them.
- **Counterexamples arrive with the hypotheses.** $|x|$ on $[-1,1]$ for the
  Mean Value Theorem, $x^3$ and $x^{1/3}$ for the inverse-function rule,
  $x^3$, $x^4$, $-x^4$ for the silent case of the second-derivative test.
- **The Newton failure.** The caveat that "cannot be dropped, even for a
  strictly convex function", followed by $x\arctan x-\tfrac12\log(1+x^2)$
  diverging from $1.4$ and converging from $1.3$. A sharp, memorable example.
- **The corner of ReLU.** $\varphi(x)=\mathrm{ReLU}(x)-\mathrm{ReLU}(-x)$,
  the identity for which autograd reports slope $0$, is the best single
  demonstration in the section, and the tab prose correctly records that JAX
  returns $1$ for $|x|$ where the others return $0$ (checked against the
  stored PyTorch and JAX outputs).
- **Exercises.** Among the best I have seen in the book: the entropy problem
  that asks in which parameterization the slope is Lipschitz, the SVM whose
  minimizer sits at the kink for every $\lambda\le1$ (which refutes the
  measure-zero argument of the text), the step sizes $0.125$ and $0.1$
  on $\varphi$, and the period-2 orbit past $2/L$.

## Structural findings

1. **Half the section is optimization, and chapter 40 teaches it again.**
   "Gradient Descent in One Variable", "The Descent Lemma", "Newton's
   Method", and "Nondifferentiable Points" together are about 2,900 words.
   Chapter 40 has `**Proposition (descent lemma).**`
   (`mdl-gradient-based-optimization.md:224`), a proposition on quadratic
   convergence of Newton's method with damping (`:992` to `:1140`), and a
   subsection "Subgradients" that redefines the subgradient and the
   subdifferential and redraws the fan of supporting lines at $|x|$
   (`mdl-convexity.md:471`). By the guide's §7.7 each fact has one home. My
   recommendation:
   - Keep here: the gradient-descent step, the quadratic case, and the
     one-variable descent lemma. They answer the question the section opens
     with, and chapter 40 can cite the lemma in a clause.
   - Keep here: Newton's update as "minimize the parabola" and the divergence
     example (about 250 words). Move the error-recursion proposition and its
     proof to chapter 40, or keep it here and have chapter 40 cite it; not
     both.
   - Keep here: one-sided derivatives, what autograd returns, and the
     $\varphi$ example (these are about differentiation). Give the
     subdifferential, the optimality condition, and "A subgradient step need
     not be a descent step" one home, here or in `sec_mdl-convexity`.
2. **The first thousand words re-teach the Basics at the same level.** The
   difference quotient with a secant figure, a polynomial whose quotient is
   expanded by hand, $|x|$ at $0$ (nearly the same sentence), the table of
   derivatives, and the rectangle picture of the product rule are all in
   `chapter_preliminaries/calculus.md`. The chapter introduction says "We
   assume school calculus and the Basics part." What is new here starts with
   the $o(\epsilon)$ form and its converse. The section would lose nothing by
   compressing "The Difference Quotient" to a paragraph that recalls the
   definition and goes straight to the linear approximation (SVC-2).
3. **It is not two sections, but it is 800 to 1,000 words too long.**
   Candidates to cut, in order: the real-analytic digression with
   $e^{-1/x^2}$ and its figure (SVC-12), the plot-only Taylor cell (SVC-10),
   the autograd check cell (SVC-4), the closing recap paragraph (SVC-20),
   and whatever moves to chapter 40 under finding 1.
4. **Promissory notes.** Because the opening example needs derivatives that
   are computed later, the first third of the section asks the reader to wait
   again and again: "a value we derive below", "a computation below measures
   how it shrinks", "The last part of this section returns to such corners",
   "which the curvature part of this section studies", "the Taylor polynomial
   below explains the $\tfrac18$". The words "below" and "above" occur 17
   times in the body. Each is harmless; together they make the reader feel
   that the explanation is always somewhere else (SVC-21).
5. **Lookup use is good.** The subsections on the Mean Value Theorem, Taylor
   polynomials, Newton's method, and nondifferentiable points carry labels
   and restate their setting, so a reader arriving from a reference can start
   there. "The Descent Lemma" has an equation label but no section label; it
   is the most likely target of a reference and should get one.

## Passage-level findings

**SVC-1** (minor, correctness; sure). Under the opening. Anchor: "The error
has a simple form. It equals". The gap is $0.0012495$, not
$\tfrac12\cdot\tfrac14\cdot0.1^2=0.00125$; the stored output prints
gap$/\epsilon^2=0.1249$ at $\epsilon=0.1$. Write "To the digits shown it
equals", or "It is very nearly".

**SVC-2** (moderate, pedagogy). Heading "The Difference Quotient". Anchor:
"We want to measure how a function $f$ responds to a small change of its
input". The subsection (zoom figure, secant figure, a hand-expanded
polynomial, the limit) is the Basics treatment again. For a reader with
"school calculus and the Basics part" it is a slow start to a section whose
own opening was fast. Fix: keep the zoom figure and one sentence of
definition with a reference to :numref:`sec_calculus`, drop the secant figure
(Basics has the same one), and move the $1701$ example to where it earns its
place, as the warning that a finite $\epsilon$ can be badly wrong ($25.11$
against $8$).

**SVC-3** (minor, explanation). Same heading. Anchor: "The smallest
interesting case is a polynomial whose quotient we can also". The example is
not the smallest interesting case, and the reader learns only afterwards why
the coefficient is $1701$ ("The large coefficient $1701$ slows the approach
to the limit"). Say the purpose first: "We choose a polynomial with a large
cubic coefficient, so that the secant slope at $\epsilon=0.1$ is far from the
limit."

**SVC-4** (moderate, pedagogy; code). Heading "The Softplus, the Sigmoid, and
the Logit". Anchor: "Automatic differentiation applies these same rules
mechanically. Does it". The cell (four tabs) asks whether autograd returns
$8$ and $\tfrac12$. Nothing is in doubt and nothing is learned; the sentence
after it ("Both values agree with the hand computation") says so. The same
point was already made in prose under "The Difference Quotient" ("Automatic
differentiation reaches the same number without any $\epsilon$"). The chapter
has a whole section on automatic differentiation. Cut the cell and one of the
two remarks. (A code change; under the author's decision 3 it may wait, but
the prose can stop presenting it as a question.)

**SVC-5** (moderate, thought; candour). Heading "A Table of Common
Derivatives". Anchor: "The following entries hold wherever the right-hand
sides are defined". The section says of the rules "here each is proved", and
a reader will assume the same of the table. In fact only the exponential is
argued, and that argument assumes that $a^x$ is differentiable at $0$ and
that a base with constant $1$ exists. The power rule for real $n$, the sine
and the cosine are quoted. Say so in one sentence ("We take the sine and
cosine entries, and the existence of the limit that defines $e$, from a first
calculus course"). Then add the one line that the text is by now able to
give: after the inverse-function rule, $x^n=e^{n\log x}$ and the chain rule
give $nx^{n-1}$ for every real $n$ and $x>0$. That closes the only gap a
careful reader will notice.

**SVC-6** (minor, thought). Headings "A Table of Common Derivatives" and
"Inverse Functions". Anchor: "The inverse-function rule below identifies
$c_a$ for". The constant is identified by the chain rule applied to
$a^x=e^{x\log a}$, as the later paragraph itself shows ("the chain rule gives
$c_a=\log a$"); the inverse-function rule only supplies the logarithm. Reword
both places: "Once the logarithm is available, the chain rule identifies
$c_a$."

**SVC-7** (moderate, explanation). Heading "The Mean Value Theorem". Anchor:
"We have been reading the derivative as if it described the function nearby".
Two problems. First, the section has not yet read the derivative that way;
the sentence describes what the Basics did. Second, "a point with slope zero
is a candidate minimum" is Fermat's theorem (at an interior extremum of a
differentiable function the derivative vanishes). It is the fact an
optimization reader uses most, and here it is proved inside the proof of
Rolle's theorem and never stated. Much later the text has to cite it as "as
the proof of Rolle's theorem showed". Fix: state it as a short lemma with the
four-line proof now inside the Rolle argument, and let both the Mean Value
Theorem and the later remark on $f'(x)=0$ cite the lemma.

**SVC-8** (minor, pedagogy). Same heading. The reader is not told what the
theorem will be used for beyond monotonicity. It is the engine of everything
quantitative in the section: the Lagrange remainder, hence the descent lemma
and Newton's error. One sentence in the lead would raise its standing: "Every
error bound in the rest of this section is this theorem applied to a suitable
function."

**SVC-9** (minor, language; repetition). Headings "The Quadratic Case", "The
Descent Lemma". Anchors: "Here $1=2/f''$, since $f''\equiv2$", "On $x^2$ we
found the threshold $\eta<1=2/f''$", "For $x^2$ we have $f''\equiv2$, so
$L=2$". The same identification is made three times. Keep the first as a
forward pointer and the third as the check; delete the second.

**SVC-10** (moderate, pedagogy; code). Heading "The Taylor Polynomial".
Anchor: "Each higher degree tracks the curve farther from the base point."
The cell above this sentence only draws $P_1$, $P_2$, $P_5$ against $e^x$. It
computes nothing the prose reads, and the next cell (maximum error on
shrinking windows, ratios $4$, $8$, $16$) makes the same point with numbers.
By the house rule that code teaches and does not draw, this cell should go;
the error-rate cell is the one to keep, and it is very good.

**SVC-11** (minor, pedagogy). Same heading. Anchor: "For $f(x) = \sin(x)$,
with $\sin'=\cos$ and $\sin''=-\sin$, the best parabola". The best parabola
is introduced on $\sin$, the Taylor polynomials on $e^x$, and the running
function gets a paragraph in between and $P_4$ later. Three functions for one
idea. The softplus at $0$ would serve for the parabola and its figure, and
then the sentence "Its quadratic coefficient $\tfrac18$ is the constant that
$\text{gap}/\epsilon^2$ settled at" would land in the main line and not in an
aside.

**SVC-12** (moderate, enjoyment; length). Heading "The Lagrange Remainder".
Anchor: "A function is **real analytic** at $x_0$ if its Taylor". From here
to the figure of $e^{-1/x^2}$ (about 170 words and a figure) the section
answers a question no deep learning reader has asked. Nothing later in the
chapter uses "real analytic". The convergence of the exponential series is
worth its four lines; the rest is the "real-analysis hygiene" that the
October review found throughout the chapter. Cut to one sentence ("Not every
smooth function equals its Taylor series; $e^{-1/x^2}$ is the standard
example, and in practice we use the finite polynomials $P_n$"), or move the
example and figure to an exercise.

**SVC-13** (minor, explanation). Same heading, proof of the Lagrange
remainder. Anchor: "Fix $x$ and make the base point the variable". The proof
is correct and as short as it can be, but $G$ arrives without a reason. One
clause helps: "$G(t)$ is the error at $x$ of the Taylor polynomial based at
$t$, minus a term of the shape we expect the error to have." The reader then
sees why $G(x)=0$ before computing it.

**SVC-14** (moderate, thought; overlap). Heading "Newton's Method". Anchor:
"**Proposition (Newton's error recursion).**". The proposition and its proof
are neat (one application of the Lagrange remainder to $f'$), and the check
$e_{t+1}/e_t^2\approx0.42$, $0.64$, $0.87$, $0.99$ is exactly right. But
chapter 40 proves quadratic convergence again in $n$ dimensions and develops
damping (`mdl-gradient-based-optimization.md:1037`). See structural finding
1. If it stays here, chapter 40 should cite it for the idea and prove only
what is new.

**SVC-15** (minor, thought; claims not shown). Same heading. Anchors: "From
$x_0=1.3$ it converges" and "the factor $\alpha=\tfrac12$ lands at $-0.007$
in one step". Both numbers are correct (I recomputed them, and the boundary
$1.39$), but the cell runs only $x_0=1.4$, so the reader cannot see them.
Either say "(not printed)" or, when code is next touched, loop over the two
starts.

**SVC-16** (minor, language). Heading "One-Sided Derivatives". Anchor: "Each
returned value is one of the candidate slopes that we define next." A
transition that announces. The next subsection opens well on its own ("A
corner has no single tangent, but it has many lines that touch the graph
there and stay below it"); delete this sentence.

**SVC-17** (minor, explanation). Heading "Subgradients and Optimality".
Anchor: "We work with *convex* functions, those for which". Convexity is
defined in a clause and never referenced. Add the pointer to
:numref:`sec_mdl-convexity`, which a reader arriving by lookup will need.

**SVC-18** (minor, explanation; figure). Same heading. Anchor: "and $0.35$
and $0.65$ for $\beta=64$". The numbers are correct, but the figure the
sentence points to draws $\beta=1$, $4$, $16$. Quote a value the reader can
find in the figure, or add the curve.

**SVC-19** (moderate, thought). Heading "Kinks in Stochastic Training".
Three problems in one short subsection.
(a) The heading promises stochastic training and the subsection contains
none; its cell is even named `why-sgd-shrugs`, a title the text no longer
delivers. Rename ("What Autograd Returns for a Composition") or deliver.
(b) Anchor: "A subgradient step need not be a descent step, even for a convex
function." This paragraph sits between two paragraphs about chaining
per-kink choices and interrupts them. It belongs at the end of "Subgradients
and Optimality".
(c) Anchor: "measure how rarely, and whether it matters for training". The
subsection hands its conclusion to two exercises, one of them an [extended]
project. A reader who does not do the project never learns the answer, and
exercise 12(e) then refers to "the claim in ... that training tolerates the
occasional incorrect chained value", a claim the text does not make. State
the finding of :citet:`Bertoin.Bolte.Gerchinovitz.ea.2021` in two sentences
here (how often exact zeros occur at each precision, and what they observed
in training), and let the exercise reproduce it.

**SVC-20** (moderate, language; repetition). End of "Kinks in Stochastic
Training". Anchor: "We can now predict how a function changes from its
slope". This closing paragraph lists the section's results, and the Summary
directly below lists them again. The guide (§12.2) names both the recap
closing and the formula "We can now ..."; the next section closes with the
same words ("We can now linearize a function of many variables"). Cut the
first sentence and keep the bridge: "A network, however, has millions of
weights that move together. ..."

**SVC-21** (minor, language; pattern). Promissory cross-references by
position: 10 "below", 7 "above" in the body. Instances: "a value we derive
below", "a computation below measures how it shrinks", "the Taylor polynomial
below explains the $\tfrac18$", "The descent lemma below shows that a bound".
Fix: in the opening, say once that the two constants $-\tfrac12$ and
$\tfrac14$ will be derived when the rules are available, and then stop
promising. Where a pointer is needed, use the label.

**SVC-22** (minor, language; pattern). A question in front of every code
cell: eight in this section. The method (predict, run, compare) is the right
one and several questions are real predictions ("Does the gap ... shrink like
$\epsilon^2$, with constant $\tfrac12\ell''(0)=\tfrac18$?"; "halving the
window should divide the worst error by $2^{n+1}$ ... Does it"). Others have
only one possible answer ("Does it reproduce the hand results"; "Do ten steps
from $x_0=1$ show these regimes?"). As a fixed form it becomes a template the
reader sees through. Keep the question where the reader could be wrong;
elsewhere say what the cell computes.

**SVC-23** (minor, thought). Summary. Anchor: "autograd returns one fixed
element of it per kink". The Summary is accurate and the right length. It is
set as one paragraph with an irregular line break ("for example $s'=\sigma$
for the softplus. The Mean Value Theorem turns slopes" runs on one long
source line), which suggests a late edit; reflow when revising.

## Exercises

Twelve problems, 2,400 words (36% of the body), graded from a routine
derivation to two research-flavoured projects. The level and the connection
to deep learning are well above the rest of the book. Specific points:

- **Exercise 9 (central differences) overlaps twice.** The Basics section
  "Finite Differences" already works the truncation-against-rounding
  trade-off with a float32 and float64 sweep, and exercise 7 of the
  multivariable section ("Gradient checks in single precision") asks for the
  same sweep and the same balance argument again, with different notation
  ($u=2^{-53}$ here, $\varepsilon_{\textrm{mach}}\approx2.2\times10^{-16}$
  there). Keep parts (a) and (b) here (the order of the error from the
  Lagrange remainder, which is this section's tool) and leave the
  floating-point sweep to the gradient-check exercise.
- **Exercise 9(a)** needs one more hypothesis: merging the two remainder
  points into a single $\xi'$ uses the intermediate value property of
  $f'''$. Say "three times continuously differentiable". (Minor; sure.)
- **Exercise 12(e)** cites a claim the section does not make; see SVC-19.
- **Exercise 10 (argument reduction for exp)** is a good numerical-analysis
  problem, but it is the one with the least to do with this section's line or
  with deep learning. First candidate if the set is trimmed.
- **Exercise 4** carries an attribution to MIT 18.01SC "Cube Root of x";
  the problem as written (Newton on $|x|^p/p$) has moved far from that
  source. Keep or drop the credit deliberately.
- I recomputed the stated values in exercises 1, 4, 5, 6, 7, 8, 9, 11(b),
  11(c) and 11(e); all are correct. I could not check the figures quoted
  from Table 1 of Bertoin et al. in exercise 12(b).

## Still open from October

"The descent lemma, Newton's method, and the subgradient also appear in
ch. 26" (now chapter 40): still open, and the main structural finding above.
"Real-analysis hygiene throughout": much reduced in this section; what
remains is the real-analytic passage (SVC-12).

## For the chapter review

- This section sets up the chapter's method well: one worked number, a local
  model, a bound on its error, a check in code. The multivariable section
  follows the same order (linear model, then Hessian), so the two read as a
  pair; both end with the same "We can now ..." recap.
- It assumes the Basics calculus section and then repeats its first third.
  The chapter introduction's "We assume school calculus" should decide this.
- It owns, or should own, the one-variable Lagrange remainder and the
  one-variable descent lemma; the multivariable section and chapter 40 cite
  both. Newton's convergence theory and the subdifferential have a second
  home in chapter 40, and the chapter review should assign one.
- Patterns I expect across the chapter: a question before every cell; cells
  that confirm a hand derivative with autograd; position words ("below",
  "above", "the opening's") in place of labels; a recap paragraph before the
  Summary.
- Tab fragmentation is light here (three cells in four tabs, one pair of
  tab-specific sentences), and the one framework difference that matters
  (JAX on $|x|$) is handled correctly.
