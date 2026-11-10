# Neural ODEs, Adjoints, and Continuous Flows

Source: `chapter_mdl-dynamics/mdl-neural-odes.md`, lines 1 to 1340 (up to the
slides marker). About 7,000 words outside code, of which about 1,900 are
exercises. Ten code cells; three are written four times (one per framework),
the rest are shared NumPy.

## Verdict

A good section with a clear three-part argument (train through the solver;
differentiate the flow; follow a density) and unusually careful mathematics.
I checked every derivation, proof and worked number, compared every claim
with the stored PyTorch outputs, and re-ran the two "for any draw" claims on
several hundred random draws; all hold, and I found no mathematical error
(two small imprecisions of wording are listed). Its best quality is candour:
it trains a ten-step model, integrates the same field with a thousand steps,
and reports that the error *rises* ("Training fitted the ten-step program").
Most texts on Neural ODEs do not show this. Against the bar it falls short
in three places. The Hutchinson experiment undercuts the claim it is there
to support and the text does not resolve the tension (one probe has a
standard deviation of about 5 for a trace of about 0.2). The section tells
the reader three times what a Neural ODE gains before showing any of it.
And the prose, though clean, is more uniform than in the previous section:
every one of the six experiment cells is introduced by a question, and the
section closes with the same "We can now ..." recap as its neighbour. The
single most important fix is the Hutchinson passage (NODE-14).

## Scores

| Explanation | Thought | Pedagogy | Language | Enjoyment |
|---|---|---|---|---|
| 4 | 4 | 4 | 4 | 3 |

## What works

- **The scalar adjoint.** "Before the general statement we work one example
  to the end, an example in which every quantity has a closed form." The
  gradient $T\,x(T)^2=e^2$ is computed directly, then by the adjoint, and the
  integrand turns out to be constant ("$x$ grows forward, $a$ shrinks by the
  same factor when read forward"). The figure (I rendered it) shows exactly
  this: two curves, a constant product, and the gradient as a shaded area.
  The general proof then "keeps the shape of the scalar one". This is the
  model for how a proposition in this appendix should be prepared.
- **The discrete adjoint against the continuous one.** The two recursions
  are put side by side and differ "only in where the Jacobian is evaluated".
  The scalar check ($6.116$, $7.244$, $7.374$ against $e^2=7.389$; I
  recomputed all three) and the two cells then separate two questions that
  are usually blurred: is the backward sweep the exact gradient of the
  program (yes, to $10^{-10}$), and how far is that from the gradient of the
  flow (first order in $h$).
- **Order preservation.** The proposition is short, its proof uses only the
  intermediate value theorem and uniqueness, and the three-panel figure
  carries it. The reconciliation with the residual block that does realize
  $x\mapsto-x$ ("Here $hL=2$, so the step may reverse order. A finite step is
  not a flow") is the kind of objection-and-answer the model texts are
  built from. "on a line there is no room for two points to pass" is a
  phrase a reader will remember.
- **Honest experiment.** "With finer steps the endpoint error *rises* to
  about $0.02$ and then stops changing." The stored output shows it
  ($2.0\times10^{-3}$, $1.7\times10^{-2}$, $1.9\times10^{-2}$), and the round
  trip ($5.5\times10^{-2}$, $5.6\times10^{-3}$, $5.6\times10^{-4}$) is
  predicted in the next paragraph by a one-line argument.
- **The drift numbers.** $10^{-16}$ becoming $5\times10^5$ at $\lambda=50$,
  $T=1$ makes the cost of the memory-free adjoint concrete, and it reuses the
  stiff example of the previous section.
- **Ranges, not digits.** Unseeded results are reported as "typically a few
  times $10^{-6}$" and "a few thousandths", in line with the house rule.
- **The proofs are the short ones.** The Lemma by the Leibniz expansion, the
  change of variables in a dozen lines, Hutchinson in one.

## Structural findings

**NODE-S1. The gains are announced three times before one is shown.** The
opening lists them ("the model gains an invertible map, a computable
density, and a step count that can be chosen after training"), the first
subsection lists them again ("The exact flow of an ODE has three properties
that a finite stack is not guaranteed to have. It is invertible. Its effect
on probability densities can be written down ... And the number of steps ...
can be chosen at evaluation time"), and each is then treated in its own
part. Meanwhile the opening's question has two halves ("what do we gain ...
and what does it cost?") and the costs are never listed in one place,
although the section finds four: no order reversal on the line, a trained
field that is tied to its training step count, drift in backward
reconstruction, and a noisy trace. State the gains once (in the first
subsection, where the limit is taken) and let the opening name gains and
costs together in one sentence.

**NODE-S2. The third promised gain is contradicted by the section's own
experiment, and the text does not close the loop.** "A step count that can
be chosen after training" is the property the diffusion samplers rely on,
as the opening says. The only experiment on it shows the error growing
tenfold when the step count changes. The text explains why (the loss was
computed on the ten-step program) but stops at "The field defines a flow
close to its ten-step program, not equal to it." The reader needs one more
sentence: what must be different for the promise to hold (train against the
flow, for instance with a higher-order or finer solver, or learn the field
by a regression that involves no solver at all, which is what
:numref:`sec_mdl-score-matching-diffusion-flow` and
:numref:`sec_mdl-flow-matching-transport` do). That sentence is also the
best motivation the chapter has for simulation-free training.

**NODE-S3. Length and proportion are right; the tabs cost little.** At about
5,000 words of body this is the right size, and it is one section, not two.
Only three cells are per-framework. One paragraph is tab-driven and can go
(NODE-16).

**NODE-S4. Lookup use.** The main text and the probability chapter point at
`sec_mdl-continuous-normalizing-flows`. That part stands on its own: it
restates the change-of-variables formula, works the scalar case, and does
not depend on the adjoint part. Good. It would help a reader landing there
to have :eqref:`eq_mdl-ode-cnf-likelihood` say in words which way time runs
when the model is used as a density estimator (from data to the base
density, that is, along the direction in which the trace is integrated).

## Passage-level findings

**NODE-1** (minor, thought). Opening. Anchor: "The previous section ended with two observations about methods we have used".
The previous section did not end with the observation that "a residual
block is one forward-Euler step of size $h=1$". It mentions residual
networks in one sentence of its opening and in its hand-off sentence, and
never writes the block or $h=1$. The comparison is made for the first time
in this section, two paragraphs later. Also, a section opening that recaps
its predecessor is one of the repetition sources of guide section 7.7.
Start from the question: "A residual block adds a correction to its input,
and so does a step of Euler's method. If a residual network is a numerical
solution of a differential equation, what do we gain by taking the equation
itself as the model, and what does it cost?"

**NODE-2** (minor, thought). Pattern: a number that recurs, 3 instances. Anchors: "and its determinant costs about $2\times10^{10}$ operations, thousands of"; "operations, about $2\times10^{10}$ at $d=3072$. Discrete flows therefore"; "$d=3072$ that is the cost we set out to avoid".
The cost of the $3072\times3072$ determinant is stated in the opening and
again, with the same explanation of how discrete flows avoid it, at the
start of "Density Along a Flow". Guide section 7.7: derive once. Keep the
opening's version short ("the determinant of a $3072\times3072$ Jacobian at
every sample") and give the operation count only where the formula is.

**NODE-3** (minor, correctness, fairly sure). "The Hutchinson Trace Estimator", first paragraph. Anchor: "$d=3072$ that is the cost we set out to avoid".
The cost "we set out to avoid" was the $O(d^3)$ determinant, about
$2\times10^{10}$ operations. What this sentence describes is $d$
reverse-mode passes for the diagonal of the Jacobian, a different and much
smaller cost. The honest statement is stronger: the trace already replaced
$O(d^3)$ by $d$ passes, and the estimator replaces $d$ passes by one.

**NODE-4** (minor, explanation). "From Residual Blocks to a Continuous Model". Anchor: "Right: the same updates read as forward-Euler steps of size $h=1$ on the exact flow of the field."
The caption's "on the exact flow" is wrong as written: the rendered figure
shows the Euler polygon leaving the exact trajectory, which is its point.
Write "beside the exact flow of the same field, from which they drift".
The figure also labels every block $\mathbf f_\theta$ (shared weights),
while the sentence that cites it describes "A ResNet whose blocks have
different weights". Cite the figure after the weight sharing is introduced,
or say in the caption that the blocks share one field.

**NODE-5** (minor, explanation). "From Residual Blocks to a Continuous Model". Anchor: "To take the limit, share one field across all steps and let the steps".
Two changes are made in one clause: weights become shared across layers,
and the step shrinks. Only the second is a limit. A reader may wonder why
sharing is needed, since the sentence before allowed a time-dependent field.
Say that the field may still depend on $t$ (so a ResNet with distinct blocks
corresponds to a field that varies in time), and that "share" means one
parameter vector $\boldsymbol\theta$ for all times.

**NODE-6** (minor, explanation). "Order Preservation on the Line". Anchor: "Adding dimensions in this way is called *augmentation*".
The example rotates the plane; it does not yet say how a model on
$\mathbb R$ uses it. One clause: "the input $x$ is embedded as $(x,0)$, the
flow runs in the plane, and the first coordinate is read out".

**NODE-7** (moderate, pedagogy). "Training Through the Solver". Anchor: "Can ten shared-weight residual updates learn a prescribed deformation of the".
The "prescribed deformation" is affine: the circle is scaled by $(1.2,0.4)$
and shifted by $1.5$. The exact time-one flow of an affine field does this
exactly ($\dot{\mathbf x}=\mathrm{diag}(\log1.2,\log0.4)\,\mathbf x+\mathbf c$),
so the experiment says nothing about what a tanh field of 32 units can
learn, and the question as posed is a weak one. The real subject of the
two cells is the second question ("Is what we trained a continuous flow or
a ten-step program?"), which is a good one. Lead with it, describe the task
as what it is ("a stretch and a shift"), and tell the reader that an exact
flow with zero error exists, so that the $0.02$ gap at $N=1000$ is seen to
come from training against the ten-step program and not from lack of
capacity. This is deferred if it needs a new run; the prose change needs
none. (Trivial: the loop runs 401 iterations, the prose says 400.)

**NODE-8** (minor, language). "Training Through the Solver". Anchor: "The loss falls from order one to well below $10^{-4}$, typically a few times".
"one small field, integrated by ten shared-weight residual updates,
reproduces the prescribed endpoints" restates the setup as a finding. The
finding is the number; the sentence can end at "$10^{-6}$".

**NODE-9** (minor, explanation). "The Adjoint Method", lead paragraph. Anchor: "Backpropagating through $N$ solver steps stores the $N$ intermediate states,".
This paragraph motivates the adjoint by memory, and two pages later
"Memory and Drift" shows that the memory-free variant is the fragile one
and that checkpointing "is the sturdier choice". A reader is left unsure
what the adjoint is for. Say at the start that the adjoint equation is
first of all what backpropagation through a solver converges to (which the
next subsection proves and measures), and second an alternative
implementation with a memory advantage and a stability cost.

**NODE-10** (minor, explanation). "The Adjoint Equations", proof. Anchor: "(The same telescoping, started at an intermediate time $s$ instead of $0$,".
The proposition says "the adjoint state satisfies the linear ODE", where
the adjoint state was defined as $\partial\mathcal L/\partial\mathbf x(t)$.
The proof defines $\mathbf a$ by the ODE instead and establishes that it
equals $\partial\mathcal L/\partial\mathbf x(s)$ only in a closing
parenthesis. That identification is half of the statement and deserves a
sentence of its own, not a parenthesis: run the same argument with
$\delta\boldsymbol\theta=\mathbf 0$ and a perturbation $\delta\mathbf x(s)$
injected at time $s$; then $\mathbf a^\top\delta\mathbf x$ is constant on
$[s,T]$, which is the claim. Also the proposition writes
$\mathbf f(\mathbf x,\boldsymbol\theta,t)$ where the rest of the section
writes $\mathbf f_{\boldsymbol\theta}(\mathbf x,t)$.

**NODE-11** (minor, language). "Backpropagation as the Discrete Adjoint". Anchor: "Backpropagation through a solver differentiates the discrete program, and".
The conclusion of this subsection is stated four times in a page and a
half: "So backpropagation through a solver is a *discrete adjoint*, and a
solver applied to the adjoint ODE is the *continuous adjoint*, discretized";
"The discrete adjoint is the exact gradient of the program, and the
program's gradient converges to the flow's at the solver's order"; "the
backward sweep is the exact gradient of the unrolled program"; and the
anchor sentence with "The two gradients agree to the solver's order". Then
again in the recap and in the Summary. State it once after the two
recursions and let the numbers after each cell speak ("The gap falls
tenfold per tenfold $N$, as in the scalar case." is enough).

**NODE-12** (minor, explanation). "Memory and Drift". Anchor: "lone scalar escapes this, because its tiny stored value is rounded relative to".
The parenthesis is correct and anticipates a sharp reader's objection (a
stored $e^{-50}$ has a relative, not an absolute, error of $10^{-16}$), but
it takes three sentences to say and breaks the paragraph. Put the example
in two dimensions from the start, as the exercise does ("in a state whose
other coordinates are of size one, rounding leaves an error of about
$10^{-16}$ in the fast direction"), and the objection does not arise.

**NODE-13** (minor, correctness, moderately sure; terminology). "Density Along a Flow". Anchor: "This is Jacobi's formula, which also".
The text calls $\det e^{\mathbf At}=e^{t\operatorname{tr}\mathbf A}$
"Jacobi's formula". Exercise 9 calls
$\tfrac{d}{dt}\det\mathbf M=\operatorname{tr}(\mathbf M^{-1}\dot{\mathbf M})\det\mathbf M$
"Jacobi's formula" (the usual meaning) and then says of the first identity
"For a linear field the result is :eqref:`eq_mdl-ode-jacobi`, Jacobi's
formula". One name for two statements. Call the first "the identity
$\det e^{\mathbf A}=e^{\operatorname{tr}\mathbf A}$" (a consequence of
Jacobi's formula, or Liouville's formula for a constant matrix) and keep
the name for the derivative.

**NODE-14** (major, pedagogy and thought). "The Hutchinson Trace Estimator", after the cells. Anchor: "The estimate is noisy, however. For Rademacher probes one".
The section's opening promises that "the trace can be estimated with one
backward pass", and the subsection sets the target as beating $d$ passes.
The experiment has $d=50$, so the exact trace costs 50 passes. It then uses
$10^5$ probes to obtain $0.19\pm0.015$ for a trace of $0.17$, and the prose
adds that one probe has a standard deviation "of about $4$ to $5$" (I
confirm $4.2$ to $5.2$ over 200 draws) and that a standard error of $0.01$
"would take about $2\times10^5$ probes". Read plainly, the demonstration
shows that the estimator is four thousand times more expensive than the
exact computation it replaces, and a single probe carries no information
about a trace of a few tenths. The text states the noise and moves on to
the recap. A careful reader concludes that the method does not work.

What is missing is the reason one probe per step is nevertheless enough in
a CNF: the estimate is unbiased, the log-likelihood is an integral over many
steps and an average over many samples, and training uses it inside a
stochastic gradient, which needs an unbiased estimate and tolerates
variance. Per-sample likelihood *evaluation* with one probe is indeed poor,
which is why evaluations average several probes or compute the exact trace.
Add this in three or four sentences after the result. Two smaller repairs
would make the experiment itself show something: print the single-probe
standard deviation beside the estimate so the prose's "$4$ to $5$" is read
from the output (it is not printed now, only implied by
$0.0154\times\sqrt{10^5}$), and note that the random field chosen here is a
hard case, since its diagonal entries have random signs and nearly cancel,
whereas a contracting field has a trace of order $d$ (exercise 6c asks
exactly this; the text should say it). The second repair needs a changed
cell and is deferred under the author's decision; the prose can be fixed
now.

**NODE-15** (minor, thought). "Checking the Trace Integral". Anchor: "than $10^{-8}$ for any draw of the weights".
"for any draw" is a universal claim about an unseeded cell. I ran 300 draws:
the largest gap was $2\times10^{-9}$ and the median $5\times10^{-11}$, so it
holds in practice, but the stored output cannot show it. "in every run we
tried" or simply "to about $10^{-10}$" is safer. The same phrase occurs just
above ("agree in every printed digit for any draw"), where it is a
consequence of an exact identity and is fine.

**NODE-16** (minor, language). "The Hutchinson Trace Estimator". Anchor: "Each framework's reverse mode computes the rows $\boldsymbol\epsilon_k^\top \mathbf J$".
This sentence exists to bridge a shared NumPy cell and a per-framework cell,
and it says again what the paragraph before the proposition's use said
("one reverse-mode pass through $\mathbf{f}$ with $\boldsymbol\epsilon$ as
the seed"). Cut it and merge the two cells' lead-in.

**NODE-17** (minor, pedagogy). "Density Along a Flow", last sentences. Anchor: "the same law is the continuity equation of fluid mechanics".
A good hand-off with no destination. Add the label:
:numref:`sec_mdl-fokker-planck-probability-flow` derives it and adds the
term that noise contributes.

**NODE-18** (moderate, language). End of the body. Anchor: "We can now write a residual network as an Euler program and train it. We".
The same "We can now ..." recap as in the previous section, three sentences
of "We can", followed by a Summary that restates the same results (with the
$5\times10^5$ number a third time). Guide section 12.2 excludes both the
recap and the formula. Keep the last three sentences ("Everything here
moved deterministically. A diffusion model adds fresh noise at every step,
and increments that scale like $\sqrt{h}$ break the Taylor expansions this
section relied on."), which are a real ending.

**NODE-19** (minor, language). Pattern: 6 of 6 experiment cells are introduced by a question. Anchors: "Is what we trained a continuous flow or a ten-step program?"; "Does a hand-written backward sweep through the unrolled Euler program give the"; "How far is that gradient from the gradient of the exact flow?"; "Can we estimate the trace from vector--Jacobian products alone, without".
See ODE-20. Here the template is more visible because two of the questions
have an answer the reader already knows from the paragraph before (the
Hutchinson proposition has just proved that the answer is yes). Where the
outcome is known, state the prediction and say what the cell measures: "The
cell measures how many probes this takes."

**NODE-20** (minor, explanation). Notation across the two sections. Anchor: "map, a solver takes $N$ steps of size $h$ to the horizon $T=Nh$,".
The previous section counts steps with $n$ (and the notation table lists
$t_n=nh$); this one uses $N$ in prose, `n` in code, $n$ again as the step
index in :eqref:`eq_mdl-ode-neural-ode`, $k$ as the step index in the
discrete adjoint and $l$ for layers. In code, `B` is the generic matrix of
`expm_eig` after being the transient-growth matrix, and `H` is the number of
hidden units while the notation table reserves $\mathbf H$ for the Hessian.
None of this stops a reader, but one convention (say $N$ steps indexed by
$k$) across both sections would be cleaner.

## Exercises

Nine problems, about 1,900 words, within the band, graded from one dot to
two, and almost all of them extend the text instead of drilling it. The best
are 1 (the trained field at other step counts, with a hint that explains a
non-obvious plateau), 2 (the contraction bound $LT\ge\ln2.5$ checked against
the trained weights), 5 (the log-determinant series of an invertible
residual step, which connects to i-ResNets), 7 (discrete adjoints of Heun
and RK4) and 9 (Liouville's formula as a second proof). I checked the
algebra of 2b, 2c, 4c, 5b, 7c, 8b and 8c; all are correct and answerable.

- Exercises 1, 2d and 2e require retraining and reason about unseeded
  outcomes ("this error settles near $0.03$"). That figure will vary by
  framework and run; say "a few hundredths".
- Exercise 6c is where the reader learns when the Hutchinson estimator is
  good or bad. Its conclusion belongs in the text (NODE-14); the exercise
  can stay.
- Exercise 3 (running loss) is the only place the kinetic-energy regularizer
  appears; fine as an exercise.
- No exercise needs cutting. If one must go, 8 (checkpointing arithmetic)
  is closest to routine.
- Cosmetic, invisible to readers: seven of the nine exercise labels carry
  the prefix `ex_mdl-odes-solvers-`, a leftover from before the section was
  split off, and the label of exercise 7 (`re-derive-adjoint-equations`) no
  longer describes it.

## Still open from October

The October review did not discuss this file separately (the section was
then part of the ODE section). Nothing to carry over.

## For the chapter review

- The section follows naturally from the ODE section (it reuses the spiral,
  RK4, the stiff rate 50 and the Lipschitz corollary) and hands over to the
  SDE section with a real question. Its order within the chapter is right.
- It is the chapter's only treatment of likelihoods for flows, and the
  diffusion chapter relies on it (`design-space.md` cites the instantaneous
  change of variables). The continuity equation is named here and derived
  two sections later; the forward pointer is missing (NODE-17).
- The section makes the chapter's best case for simulation-free training
  without saying so (NODE-S2). A sentence here and a back-reference from
  the flow-matching section would tie the chapter together.
- Patterns likely to hold chapter-wide: the "We can now ..." recap before
  every Summary (confirmed in all six sections); a question before every
  cell; results restated after each cell, in the recap and in the Summary;
  "for any draw" claims on unseeded cells.
- Main-text connection: `resnet.md` does not mention the Euler reading of a
  residual block, and this section does not point to the residual-network
  section by label. One reference in each direction is enough.
