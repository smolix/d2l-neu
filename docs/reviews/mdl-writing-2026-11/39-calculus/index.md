# Calculus and Automatic Differentiation (chapter introduction)

Source: `chapter_mdl-calculus/index.md`. About 400 words of introduction, plus
a "Resources and Further Reading" list of 17 annotated entries.

## Verdict

A good chapter introduction, close to what the guide asks for in §12.1. It
starts from the training loop the reader has run, poses two questions the
reader can feel (how far can a slope be trusted, and why is the gradient
cheap), gives the origin of the subject in four sentences, and states scope
and prerequisites. The one defect is the route paragraph, which is a run of
five sentences of one shape. The more important observation is for the
chapter and not for this page: the introduction makes two promises (one
recurring function, and "We assume school calculus") that the first two
sections keep only in part.

## Scores

| Clarity of explanation | Clarity of thought | Pedagogy | Language | Ease and enjoyment |
|---|---|---|---|---|
| 4 | 4 | 4 | 4 | 4 |

## What works

- The opening two questions are concrete and are the questions the chapter
  answers. "The gradient is still correct, but the step has left the region
  where the slope predicts the loss" states the idea of the descent lemma in
  plain words before any mathematics. Keep it.
- "Calculus answers these questions, although it was created for others." A
  sentence with a voice; it also tells the reader that the history paragraph
  will be short and why it is there.
- "Besides the loss value, it is all an optimizer knows" is the best
  one-clause reason in the chapter for why a deep learning reader needs the
  derivative.
- The scope sentence names what is left out (divergence, Stokes, measure
  theory) and where differential equations go, by label.
- Each resource names the section it accompanies, by label.

## Structural findings

- **The promise of one recurring function is kept unevenly.** The page says
  "One function recurs throughout, the log-sum-exp". In the single-variable
  section the softplus slice is used well. In the multivariable section
  log-sum-exp carries the gradient and the Hessian, but the chain rule and
  backpropagation use an unrelated polynomial toy, critical points use
  $xe^{-x^2-y^2}$, and the Lagrange example is a hyperbola. Either the
  sections are brought closer to the promise (see MVC-11 in the multivariable
  review for a Lagrange example that is the log-sum-exp) or the sentence is
  weakened to "recurs in every section".
- **"We assume school calculus and the Basics part"** does not match the
  first 1,000 words of the single-variable section, which redefine the
  difference quotient, the secant, and the table of derivatives at the level
  of `chapter_preliminaries/calculus.md`. Either the assumption or the
  section opening should change (SVC-2).
- **The probing argument is told too often.** "Nudging a hundred million
  weights one at a time would take a hundred million network runs" appears
  here; the Basics calculus section makes the same count with $10^8$; the
  multivariable section opens on it again with $10^6$ and returns to it at
  the end of its backpropagation subsection; and the automatic differentiation
  section opens on it again with $10^8$. One home, one number (guide §7.7).
  The introduction is a fair place for it; the sections should then refer to
  it in a clause.

## Passage-level findings

**IDX-1** (moderate, language). Anchor, under the unnamed opening:
"asks how a function of one" and the four sentences that follow, each
beginning with a section reference followed by "asks how" or "asks in which".
This is the pattern the guide names as a defect in §12.1 item 3 ("seven
consecutive '§x asks how ...'"). A reader skips it. Fix: group the five
sections into the three movements the chapter has (local models of a function
of one and of many variables; derivatives of layers and of programs; the
integral and the derivative of an expectation), in two to four sentences of
different shape, naming a section only where the pointer helps. Draft:
"The first two sections build the linear and the quadratic model of a
function, in one variable (:numref:...) and in many (:numref:...), and use
them to decide how large a step may be and what kind of point training has
stopped at. The next two differentiate what a network is made of: layers
whose inputs are vectors and matrices (:numref:...), and whole programs
(:numref:...), where the order of the chain rule decides the cost. The last
section (:numref:...) turns to the integral, which is needed as soon as the
loss is an expectation."

**IDX-2** (minor, thought). Anchor: "asks in which direction many weights
should move, how one backward sweep". The route gives the backward sweep to
the multivariable section and "how a program computes these derivatives" to
the automatic differentiation section. Read together, the two sentences show
the overlap that the October review reported (backpropagation told three
times in the chapter). When the overlap is resolved, this paragraph should
say which section owns the backward pass.

**IDX-3** (minor, explanation). Anchor: "The integral enters once data".
The sentence "training through random sampling needs the derivative of that
area" is too compressed for a reader who has not yet met the
reparameterization trick; "random sampling" could be minibatch sampling.
Name one use: "a variational autoencoder, for instance, has to differentiate
an expected loss with respect to the parameters of the distribution it
samples from".

**IDX-4** (minor, language). Anchor: "shows why automatic differentiation is
correct and cheap, and checks each". That the chapter "checks each result in code" is true and
is a strength of the chapter, but in the first two sections a third of the
checks have a foregone answer (does autograd reproduce a hand derivative).
See the pattern findings SVC-22 and MVC-4.

## Exercises

None, correctly.

## Still open from October

The October review's description of the chapter ("first-year calculus with
proofs, interleaved with optimization and backprop told three times") is
still accurate for the first two sections; the introduction itself is not
the problem.

## For the chapter review

- The introduction can stay almost as it is. Rewrite the route paragraph
  (IDX-1) after the chapter decides where backpropagation and the
  forward-versus-reverse cost argument live.
- Two promises to check against every section: the log-sum-exp as recurring
  example, and "school calculus assumed".
- The cost-of-probing argument should have its single home here.
