# Multivariable Calculus

Source: `chapter_mdl-calculus/mdl-multivariable-calculus.md`. About 7,600
words of prose before the exercises and 2,100 words of exercises (11
problems). Nine code cells, two of them in four framework tabs.

## Verdict

The parts are good; the whole is three sections in one. Taken paragraph by
paragraph this is careful, correct mathematics with well-chosen numbers: the
linear model of log-sum-exp is checked and its error is later predicted to
three digits from the Hessian, the backward pass is carried out on integers
the reader can follow by hand, and the closing subsection on saddles is
honest about what random-matrix theory does and does not say about networks.
I found no mathematical errors. I found one place where notation changes
meaning in a way that will confuse (the symbol $\partial f/\partial a$ is an
edge derivative in one paragraph and a total derivative in the next) and one
theorem restated with a hypothesis different from the one that was proved.
The structural problem is the order. The section treats the gradient, then
critical points and Lagrange multipliers, then spends 3,000 words on the
chain rule and backpropagation, and only then returns to critical points with
the Hessian. The story of "what kind of point has training stopped at" is cut
in two by a different story, which the next two sections of the chapter then
tell twice more. The most important fix is to put the critical-point material
together and to give the forward-versus-backward cost argument one home in
the chapter.

## Scores

| Clarity of explanation | Clarity of thought | Pedagogy | Language | Ease and enjoyment |
|---|---|---|---|---|
| 4 | 3 | 4 | 4 | 3 |

## What works

- **The definition of differentiability is motivated by a doubt.** "It is
  natural to guess that the two changes add ... but this guess is a claim
  about $f$, and it needs a hypothesis." Then the hypothesis, then the
  counterexample $2xy/(x^2+y^2)$ in four sentences. This is the appendix
  doing what the Basics could not, and saying so.
- **The linear-model check that is paid back.** The first cell measures a
  gap of $1.786\times10^{-4}$; 800 lines later the Hessian predicts
  $\tfrac19(0.04)^2=1.778\times10^{-4}$, and the text accounts for the
  remaining $10^{-6}$ as the third-order term. A prediction, a number, and an
  explanation of the residue.
- **The existence step for the global minimum.** "To conclude which is the
  global minimum we also need to know that a global minimum *exists*", with
  the disc of radius $2$ and the bound $0.04$, and then "Without the
  existence step the argument fails: $x^3$ has a single critical point and no
  minimum at all." Most texts skip this; here it takes six lines.
- **Edge derivative against total sensitivity.** The skip-edge graph where
  "The *edge* derivative $\partial f/\partial p$ is $1$, but the *total*
  sensitivity of $f$ to $p$ ... is $2p + 1 = 7$" isolates the one thing
  people get wrong about the multivariable chain rule. (The notation that
  follows then undoes some of the good; see MVC-12.)
- **The backward pass on integers.** $73472$ and $-68224$ by hand, by plain
  Python, and by autograd, with the count of 24 multiplications against 12,
  and the remark that "All the numbers involved are integers below $2^{24}$,
  so single precision represents them without rounding."
- **The proof of sufficiency in the second-derivative test** says where the
  difficulty is: "This is the step that needs care, because the error term
  must be controlled in all directions at once."
- **"Saddles in High Dimension"** is the best page of the section. It states
  the coin-flip heuristic, says that it "*overestimates* the chance of a
  minimum", gives the exact value $0.146$ at $n=2$, the rate
  $\exp(-\tfrac{\ln3}{4}n^2)$, the reason ("eigenvalues repel one another"),
  and then limits the claim to what was measured on small networks.
- **Exercises**: the two-tanh-unit backward pass (saturation, float32
  underflow, symmetry), paths through a residual stack, Peano's function
  $(x-y^2)(x-2y^2)$, random Hessians, and the saddles of a linear network.

## Structural findings

1. **Critical points are split in two by the chain rule.** "Critical Points"
   and "Constrained Optimization and Lagrange Multipliers" sit under "The
   Geometry of Gradients". The first of them ends "The second-derivative test
   later in this section separates them." That test is about 570 source lines
   later, after "The Multivariable Chain Rule". A reader following the
   question "is this a minimum?" has to hold it across the whole of
   backpropagation. Two orders would work:
   - *Reorder within the section:* gradient and its geometry (through tangent
     planes); the chain rule and the backward pass; then one movement
     "Critical points" holding the first-order condition, the Hessian, the
     quadratic model, the second-derivative test, saddles, and Lagrange
     multipliers last.
   - *Split into two sections* of about 3,500 words each: "Gradients and the
     Chain Rule" and "Second Derivatives and Critical Points". At 7,600 words
     plus exercises, and with three unrelated top-level topics, this section
     is the clearest candidate for a split in my half of the chapter.
2. **Backpropagation is taught here and twice more.** This section defines
   forward mode, the adjoint, the backward pass, and the cost comparison. The
   matrix calculus section has "The Order of Multiplication"; the automatic
   differentiation section has "Two Orders of One Product", "The Tape", and
   "Cost of the Two Modes", and reopens with the same count of $10^8$
   forward passes. The Basics have :numref:`sec_backprop`. The October review
   said "backprop is explained three times"; that is still so. What is unique
   here, and worth keeping, is the scalar picture: the sum over paths, the
   edge-versus-total distinction, and the adjoint recursion "grouped by the
   first edge out of $a$". The cost argument (one pass per input
   against one pass in all) should live in the automatic differentiation
   section and be cited here in a sentence.
3. **The running example is dropped for half the section.** Log-sum-exp
   carries partials, gradient, directional derivatives, level sets, the
   tangent plane, the Hessian, and the Taylor check. But critical points use
   $\psi=xe^{-x^2-y^2}$, the Lagrange example is a hyperbola, the chain rule
   uses nested squares, and the second-derivative test adds
   $x^4+y^4-4xy$ and $x^2+y^4$. Some of this is forced (log-sum-exp has no
   critical point, as the text notes). Two changes would restore the thread:
   the Lagrange example in MVC-11, and an honest sentence that the nested
   squares are a toy chosen so that every number is an integer.
4. **Material a deep learning reader does not need.** "Tangent Planes" (150
   words and a 3-D figure) is used by nothing later in the section or, as far
   as I can see from the headings, the chapter. Cut it, or keep the one
   sentence that the linear model is the equation of a plane.
5. **Overlap with the Basics.** The gradient-descent "recipe" list and the
   $\cos\theta$ derivation of steepest descent repeat
   `chapter_preliminaries/calculus.md` ("Steepest Descent") almost step for
   step, before the proposition proves the same thing again by
   Cauchy–Schwarz (MVC-5).
6. **Lookup.** `subsec_mdl-gradient`, `subsec_mdl-steepest-descent`,
   `subsec_mdl-lagrange`, and `subsec_mdl-second-derivative-test` have
   labels. "Backpropagation", "The Chain Rule as a Sum over Paths",
   "Symmetry of the Hessian", and "The Second-Order Taylor Approximation" do
   not, and they are likely targets from the main text.
7. **Tabs** fragment almost nothing here; most cells are plain NumPy, which
   the text says once. Good.

## Passage-level findings

**MVC-1** (minor, language; repetition). Opening. Anchor: "evaluations of the
loss for a single step". The cost of probing one weight at a time is the
opening of the Basics calculus section ($10^8$), of the chapter introduction
("a hundred million"), of this section ($10^6$), and of the automatic
differentiation section ($10^8$), and this section returns to it twice more
("the same scaling as probing one weight at a time in the opening", "one
backward pass replaces a million and one evaluations"). One home (the
chapter introduction), one number, and a clause elsewhere.

**MVC-2** (minor, explanation). Heading "Partial Derivatives". Anchor:
"Freeze every weight except $w_1$." The subsection restates the Basics
definition, the slice picture, and the one-variable approximation. Three
sentences and the running example's partials would do; the new material
starts at "Differentiability and the Gradient".

**MVC-3** (minor, thought). Heading "Differentiability and the Gradient".
Anchor: "The gradient is the unique vector whose dot product with a step
gives the". This sentence, "The first-order change is a dot product of the
step with the gradient", and the sentence after the display of $\Lambda$ say
the same thing three times in ten lines, and the Summary opens with it again.
Keep the "unique vector" form, which is the one that generalizes.

**MVC-4** (moderate, pedagogy; code; pattern). Two cells whose answer is
foregone.
(a) Heading "Differentiability and the Gradient". Anchor: "Does each
framework's autograd return". Four tabs to confirm $[\tfrac13,\tfrac23]$.
The sentence before it ("A hand-derived gradient is routinely checked against
automatic differentiation") is a fair remark about practice, but the cell
teaches nothing the reader doubts.
(b) Heading "Critical Points". Anchor: "Does the gradient vanish at the two
hand-derived points". The cell evaluates a hand-typed formula for
$\nabla\psi$ at the roots of that same formula and prints $1.3\times10^{-16}$.
It checks the algebra against itself.
Fix: drop (a), or merge it into the first cell as one line. Replace (b) by
something the reader could get wrong, for instance the values of $\psi$ on
the circle of radius $2$ against the bound $0.04$ used in the existence
argument, or cut it. Seven cells in the section are introduced by a question
("Does ...?", "How fast ...?"); keep the form for the real predictions (the
gap from the Hessian; the factor of eight for the cubic error) and state
plainly what the others compute. (Code changes may wait under the author's
decision 3; the prose can already stop asking.)

**MVC-5** (moderate, thought; repetition). Heading "Steepest Descent".
Anchor: "The direction enters only through $\cos\theta$, so the best
direction makes". The result is derived by $\cos\theta$, then stated as a
proposition and proved by Cauchy–Schwarz, then translated back ("In the
$\cos\theta$ form this is $\theta = 0$ against $\theta = \pi$"). The Basics
gave the $\cos\theta$ argument. The geometric form of the dot product is
itself Cauchy–Schwarz, so the first derivation adds nothing to the second.
Keep the proposition and its proof (it gives uniqueness), lead with one
sentence of intuition, and delete the four-step recipe list and the
$\cos\theta$ display.

**MVC-6** (minor, thought). Same heading. Anchor: "The learning rate $\eta >
0$ absorbs the length of the gradient into the step." The recipe steps by
$\eta\mathbf v$ with a unit vector; the update steps by
$-\eta\nabla\ell$. These are different algorithms (normalized gradient
descent and gradient descent), and "absorbs" passes over the difference. Say
it: "Gradient descent does not normalize: its step has length
$\eta\|\nabla\ell\|$, which shrinks as the gradient does."

**MVC-7** (minor, language). Same heading. Anchor: "Practical optimizers also
build their steps from gradients without". This sentence follows "develops
steepest descent in other norms." on the same source line with a double
space, and reads as a late insertion. The paragraph on other norms is good
(sign descent as steepest descent in the max norm); give the optimizer remark
its own sentence and connect it: Adam's sign-like step is the natural
example.

**MVC-8** (moderate, thought; order). Heading "Critical Points". Anchor: "The
second-derivative test later in this section separates them." See structural
finding 1. If the order is not changed, at least give the label
(:numref:`subsec_mdl-second-derivative-test`) in place of "later in this
section".

**MVC-9** (minor, explanation). Heading "Critical Points". Anchor:
"Throughout, $f$ is defined on all of $\mathbb{R}^n$ and differentiable at
the". The argument that follows is clean. The first paragraph of the
subsection is weaker: "For small problems this condition is enough to find
the minimum by hand" is immediately qualified by the existence discussion.
Lead with the claim that will be proved (a necessary condition) and let the
$\psi$ example show what else is needed.

**MVC-10** (minor, correctness; sure). Heading "Constrained Optimization and
Lagrange Multipliers". Anchor: "At a feasible point the admissible moves are
the directions $\mathbf{v}$". The heuristic paragraph and the proof are
consistent, and the regularity hypothesis $\nabla g\neq\mathbf 0$ is stated
and its necessity shown in an exercise. One imprecision: the sentence "By the
level-set proposition, so is $\nabla g(\mathbf{x}^\star)$" needs $g$
differentiable at $\mathbf x^\star$, which is assumed only in the proposition
that follows. Harmless; move the hypotheses up one paragraph.

**MVC-11** (moderate, pedagogy). Same heading. Anchor: "As a worked example,
find the points of the hyperbola $xy = 1$ closest to the". The subsection
opens by naming "weights normalized to unit length, probabilities that sum to
one" and then solves a textbook geometry problem with no connection to
either. The running example offers a far better one. Maximize
$\mathbf z\cdot\mathbf p-\sum_i p_i\log p_i$ subject to $\sum_i p_i=1$. The
Lagrange condition is $z_i-\log p_i-1=\mu$, so $p_i\propto e^{z_i}$:
the maximizer is $\operatorname{softmax}(\mathbf z)$, and substituting back
gives the value $\log\sum_i e^{z_i}$. The reader then sees that the function
the chapter has been differentiating is itself the answer to a constrained
problem, and that its gradient is the maximizer. With $\mathbf z=(0,\log2)$
the numbers are the $(\tfrac13,\tfrac23)$ and $\log3$ already on the page.
The hyperbola (and its good figure) can move to the exercises next to the
Rayleigh quotient. (I checked the algebra; the positivity constraints are
inactive.)

**MVC-12** (moderate, correctness; sure that the notation collides).
Headings "The Chain Rule as a Sum over Paths" and "Backpropagation". Anchors:
"The *edge* derivative $\partial f/\partial p$ is" and "the total derivative
of the output with respect to that node". The first passage teaches the
reader that $\partial f/\partial p$ means the edge derivative, equal to $1$,
and that the total sensitivity, $7$, is a different thing. The adjoint is
then defined as $\bar a=\frac{\partial f}{\partial a}$ and described as "the
total derivative". Under the convention just established, $\bar p=7$ while
$\partial f/\partial p=1$, so the defining equation of the adjoint is false
for the very node that was used to make the distinction. Fix: define the
adjoint by words and by the recursion, and give the total its own symbol
throughout ($\bar a$, or $\mathrm df/\mathrm da$); reserve $\partial$ for
edge derivatives. Also, in the display :eqref:`eq_mdl-adjoint` the letter $a$
is the generic node, while everywhere around it $a$ is a particular node of
the example graph; use $v$ or $n$ for the generic node.

**MVC-13** (minor, explanation). Heading "The Chain Rule as a Sum over
Paths". Anchor: "Reading off the coefficient of $\epsilon$ gives the
multivariable chain rule". The rule is obtained with $\approx$ and the next
paragraph rightly says which hypothesis the step needs. Say in one clause
that the proof is the one-variable proof with the error term measured in
norm, and that :numref:`sec_mdl-matrix-calculus` proves it for Jacobians. As
written, a reader who has just seen the careful $\rho(t)$ proof in the
previous section will wonder whether this is a proof.

**MVC-14** (minor, explanation; notation). Whole section. The letter $f$ is
the log-sum-exp, then $(u+v)^2$, then $r+p+s$, then a general function, then
a quadratic polynomial; $x,y$ are the coordinates of the running example and
also two of the four inputs $w,x,y,z$; $s$ is a node here and was the
softplus one section earlier; $\varphi$ is a slice here (twice, with
different meanings) and was the ReLU identity there. Each reuse is locally
clear. A reader who arrives by lookup at "Backpropagation" meets
$f=26896$ in a section whose $f$ is $\log 3$. Rename the toy output (say $F$
or $o$) and the skip-edge output.

**MVC-15** (moderate, language; repetition). Heading "Backpropagation".
Anchor: "With $N$ inputs that feed a shared graph, forward mode costs $N$
passes". See structural finding 2. Keep the count of 24 against 12 (it is
concrete and it is this section's own), and replace the general cost
statement and the return to "the network of the opening" by one sentence
with a reference to the automatic differentiation section.

**MVC-16** (minor, language; repetition). Heading "Symmetry of the Hessian".
Anchor: "a result known as the Clairaut–Schwarz theorem". The sentence states
the theorem with a citation; the proposition directly below states it again
with its name. Delete the sentence, and move the reason the reader should
care ("Symmetry lets the spectral theorem ... apply to the Hessian"), which
now comes after the proof, to before it.

**MVC-17** (moderate, explanation). Same heading, proof. Anchor: "applied
first along $x_i$ to $\varphi$ and then along $x_j$ to the". The whole proof
is in this one sentence: two applications of the mean value theorem with no
intermediate line. The idea (both mixed partials are limits of one symmetric
second difference) is elegant and well put; the execution loses the reader
at the only step that needs work. Write the two lines:
$\varphi(h)-\varphi(0)=h\,\varphi'(\tau)
=h\bigl[\tfrac{\partial f}{\partial x_i}(\mathbf x+\tau\mathbf e_i+h\mathbf e_j)
-\tfrac{\partial f}{\partial x_i}(\mathbf x+\tau\mathbf e_i)\bigr]
=h^2\,\tfrac{\partial^2 f}{\partial x_j\partial x_i}(\mathbf x+\tau\mathbf e_i+\sigma\mathbf e_j)$
with $\tau,\sigma\in(0,h)$. The proof is correct as it stands (I checked the
hypotheses against the two applications).

**MVC-18** (minor, correctness; sure it is inconsistent, the mathematics is
not wrong). Heading "The Second-Order Taylor Approximation". Anchor: "if
$\varphi'$ is continuous on $[0, 1]$ and $\varphi''$ exists on $(0, 1)$,
then". The previous section proved the Lagrange remainder for a function
"$(n+1)$-times differentiable on an open interval containing $x_0$ and $x$".
The restatement here has a different (weaker) hypothesis, attributed to that
section. Since $f$ is assumed twice continuously differentiable near the
segment, the proved version applies; cite it as proved, in a clause, and do
not restate (guide §7.7).

**MVC-19** (minor, language; repetition). Heading "The Second-Derivative
Test". Anchor: "The proposition gives four cases at a critical point:". The
classification is given by the proposition, by this bullet list, by the
figure caption, by the paragraph after the cell, and by the Summary. The
bullets add two things (a saddle "whatever its zero eigenvalues", and the
name "inconclusive"); fold those into the proposition and delete the list.

**MVC-20** (minor, pedagogy; code). Same heading. Anchor: "Does a program
classify these points correctly?" The cell is worth keeping for one reason
the question does not name: a finite-difference Hessian never returns an
exact zero, so the "inconclusive" case is a tolerance, and $x^2+y^4$ is
reported inconclusive although it is a strict minimum. Ask that: "What does a
numerical test report at a flat direction?"

**MVC-21** (minor, explanation). Heading "The Second-Order Taylor
Approximation". Anchor: "A second example has a curved, not a flat,
direction." The contrast is with the zero eigenvalue of the log-sum-exp
Hessian, six paragraphs back; the reader has to reconstruct it. Say "In the
log-sum-exp example one direction did not bend at all. In the next example
both do."

**MVC-22** (minor, language). End of "Saddles in High Dimension". Anchor: "We
can now linearize a function of many variables". The same recap formula as
the end of the previous section ("We can now predict ..."), followed directly
by the Summary, which lists the same results. The guide (§12.2) rules out
both the recap and the formula. End the body on the saddle paragraph, which
is a strong last word ("An optimizer that stalls at high loss is therefore
more likely to be near a saddle than near a bad local minimum"), and keep
only the three bridging sentences that lead to the Jacobian.

**MVC-23** (minor, thought). Heading "Saddles in High Dimension". Anchor:
"For random landscapes this holds at typical, high values of the function."
"This" has to reach back over three sentences to "a critical point is almost
never a minimum". State the subject: "For random landscapes, critical points
at typical (high) values are almost all saddles."

## Exercises

Eleven problems, 2,100 words (28% of the body). Graded, mostly tied to deep
learning, and with several that would stand in any good text. Points:

- **5(a)** asks for three functions with Hessian $\operatorname{diag}(2,0)$
  and a strict minimum, a non-strict minimum, and no minimum. The text gives
  exactly these ("The functions $x^2 + y^4$, $x^2$, and $x^2 - y^4$ all have
  Hessian ..."). Cut 5(a) or cut the sentence; 5(b) and 5(c) are the
  substance.
- **7 (gradient checks in single precision)** repeats exercise 9 of the
  single-variable section and the Basics subsection "Finite Differences"
  (the same sweep, the same balance of truncation and rounding, in different
  notation). Keep this one, since a gradient check is what a practitioner
  runs, and trim the other.
- **11 (Rayleigh quotient)** comes after the ●● extended project; move it up
  to follow exercise 2, in the order of the text.
- If MVC-11 is not adopted in the text, add the entropy-regularized problem
  there as an exercise: it is the one Lagrange computation every deep
  learning reader should have done.
- A natural missing problem: show that
  $\operatorname{diag}(\mathbf p)-\mathbf p\mathbf p^\top\succeq0$ for every
  probability vector and identify its null space, so that log-sum-exp is
  convex in any dimension. (Check first that the matrix calculus section,
  which treats softmax and cross-entropy, does not already ask it.)
- I recomputed exercises 1, 2, 4, 5, 6(c), 8, 9(a), 10(a) and 11; the stated
  facts are correct. For 6(c), the paths through 5 to 17 of 54 blocks are
  about 0.45% of all paths, and with $c=11/43$ they carry about 97% of the
  derivative, so the exercise has a clean answer.

## Still open from October

"backprop is explained three times": still open (structural finding 2).
"Keep: ... the sum over paths (`mdl-multivariable-calculus.md:437`)": agreed,
and it is still there. The October note on real-analysis hygiene applies only
lightly to this section (the Clairaut hypotheses, which are needed).

## For the chapter review

- This section is the hinge of the chapter and is carrying too much: it
  finishes the "local model" story of the previous section in many
  variables, and it starts the "derivative of a computation" story that the
  next two sections own. Splitting it along that line, or at least reordering
  it, would make the chapter's two movements visible.
- The chapter should decide where forward mode against reverse mode is
  explained. My suggestion: the scalar sum over paths and the adjoint
  recursion here; the parenthesization and the cost rule in the automatic
  differentiation section; the matrix calculus section only names the order.
- It leans on chapter 38 for Cauchy–Schwarz and the spectral theorem (by
  label, correctly) and hands Lagrange multipliers, Newton's method, and
  steepest descent in other norms forward to chapter 40 (by label). The
  Lagrange condition is stated as a proposition here and again in
  `mdl-constrained-optimization-duality.md:107`, which defers the proof to
  this section; that division is acceptable but should be stated once in each
  place.
- Patterns shared with the single-variable section: a question before every
  cell, autograd cells that confirm a hand result, the "We can now ..." recap
  before the Summary, and results stated three to five times (the
  classification of critical points; the cost of probing). A pattern specific
  to this section: letters reused for unrelated objects ($f$, $x$, $y$, $a$,
  $\varphi$).
- The log-sum-exp thread promised by the chapter introduction is real in the
  first and third movements of this section and absent from the second.
