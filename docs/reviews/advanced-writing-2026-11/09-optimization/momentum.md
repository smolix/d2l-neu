# Momentum

Source: `chapter_optimization/momentum.md` (lines 1 to 628, the slides after
the marker not read). About 3,600 words of prose including 590 of exercises,
and 117 lines of code. Stored outputs checked: `outputs/pytorch` and
`outputs/jax`, `chapter_optimization/momentum.json`. Both figures rendered.

## Verdict

This is one of the better sections I expect to find in the part: one
question, one running example, a prediction before every number, and a
candid statement of where the theory stops. The valley, the table of four
settings, and the reading of $\beta$ as a timescale and as a damping knob
are good textbook teaching, and every number in the prose that I recomputed
matches the stored output. It falls short of the bar in one place that
matters. The reader arrives having trained convnets with `momentum=0.9` and
is told "We have used the setting without asking what it does"; the section
then shows momentum helping only on a two-dimensional quadratic. The one
experiment on data (the airfoil regression) shows no gain, the concise run
reproduces that non-result, and the Nesterov run on the same data shows
nothing again. **The most important fix is to replace the airfoil
experiments by one in which curvature, not noise, limits progress, so that
the section shows momentum doing on data what it does on the valley.** The
second fix is to bring two short arguments out of the exercises into the
body: why the stable range of $\eta$ widens with $\beta$ while the effective
step grows, and why momentum at a matched effective step does not reduce the
noise of the iterates.

## Scores

| Dimension | Score | Note |
|---|---|---|
| Clarity of explanation | 4 | The damping model is clear; the recurrence it rests on arrives as a parenthesis before a code cell |
| Clarity of thought and focus | 4 | One line of argument; the "agree along the floor, alternate across" observation is told four times |
| Pedagogy | 3 | Excellent on the valley; three of the four runs on data show no effect |
| Language | 4 | Plain, exact, free of the banned patterns; a little even |
| Ease and enjoyment | 4 | Moves quickly and predicts before it measures; no page a reader would quote a week later except the damping passage |

## What works

- **The opening earns the section.** "We have used the setting without
  asking what it does." One sentence that turns a recipe value into a
  question. Keep it.
- **Prediction before measurement, with the failure included.** The cell
  `momentum-acceleration-and-damping-1` prints gradient descent at its best
  rate, the plotted setting, the tuned setting and a deliberately bad one
  ($\eta = 0.3$, $\beta = 0.8$), and the prose reads every row. The bad row
  is the best part: "Its factor $0.894$ is barely better than that of
  gradient descent", followed by the model that explains it. This is the
  shape an experiment should have (decision 4): the fourth row teaches
  something the theorem alone does not.
- **Concrete critical values.** "At $\eta = 0.3$ the two critical values of
  the valley are $0.57$ and $0.01$, so $\beta = 0.8$ is underdamped in
  both." A reader can check this by hand, and it explains the plot that
  follows.
- **The two readings of $\beta$** (window of $1/(1-\beta)$ gradients;
  effective step $\eta/(1-\beta)$) are stated early, used in the airfoil
  comparison, and reused by `adam.md`. "Raising $\beta$ from $0.9$ to $0.99$
  at a fixed $\eta$ therefore makes the steps ten times longer, not only
  smoother" is the sentence a practitioner needs.
- **Candour about the theory.** "The $\sqrt{\kappa}$ law is a statement
  about quadratics. On general objectives it holds near a minimum ... and
  the evidence that momentum speeds up the training of deep networks is
  empirical." And about the experiment: "On this small, noisy problem
  momentum brings no speedup."
- **The Nesterov motivation** names the cause in one sentence: "the gradient
  computed at the current point only notices the overshoot one step after it
  has happened." The figure `mdl-opt-nesterov-lookahead.svg` shows one step
  of each method from the same state and carries the explanation by itself.
- **Use of the appendix follows decision 3.** The main text states the
  tuning and the rate, tests them by counting steps, and sends the proof and
  the lower bound to :numref:`subsec_mdl-momentum-acceleration`. Both
  references point to passages that hold what is claimed (the proposition
  and corollary for the law, the Nemirovski and Yudin bound, the Nesterov
  proposition on quadratics with the general result cited). Nothing long is
  written out twice.

## Structural findings

1. **The section never shows momentum helping on data.** See MOM-9. The
   valley carries the whole case. That is a sound choice for the theory, but
   the reader's question was about networks, and the only evidence offered
   for networks is a citation.
2. **Two results the body needs are delegated to exercises.** The widening
   of the stable range (exercise 4) and the behaviour under minibatch noise
   (exercise 6, cited twice from the body with "shows why"). Each is three
   sentences. See MOM-3 and MOM-10.
3. **"An Ill-Conditioned Valley" is mostly a recap.** The introduction has
   already said why gradient descent is slow on the valley (paragraph 2) and
   what the history of gradients shows (paragraph 3); `gd.md` ran gradient
   descent on the same valley. The 280 words and two cells of this
   subsection repeat it. Keep the divergent run at $\eta = 0.6$ (the
   momentum run answers it) and reduce the rest to a clause. See MOM-2.
4. **Length is right.** About 3,000 words of body for the second most used
   idea in the chapter. Nothing needs to be split. With the airfoil cells
   replaced and the recap cut, the section could be 300 words shorter.
5. **Order within the chapter is right**: after SGD (it needs the minibatch
   gradient and `train_with_rule`), before Adam (which needs the leaky
   average, the window, and the effective step).

## Passage-level findings

**MOM-1** (moderate, thought). Introduction. Anchor: "gradient descent needs
47 steps to" and, one paragraph later, "it needs 40 steps on the valley
where". The first count is to 1% of the start, the second pair (40 against
139) is to $10^{-6}$. A reader who compares 47 with 40 concludes that
momentum saves seven steps. Use one tolerance in the introduction. The
stored table gives 18 against 47 at 1%, and 40 against 139 at $10^{-6}$;
either pair works, and the second matches the Summary.

**MOM-2** (moderate, thought; repetition, guide 7.7). The observation that
gradients agree along the floor and alternate across the valley is stated
four times: introduction ("Along the floor of the valley they point the same
way, step after step"), end of "An Ill-Conditioned Valley" ("Along $x_1$
successive gradients agree: they are small, but they all point the same
way"), after the first momentum run ("the alternating components along $x_2$
partly cancel, which tames the overshoot"), and the Summary. The counts 40
and 139 appear three times (introduction, body, Summary). State the
observation once in the introduction as the idea, show it once in the
momentum run, and let the Summary carry the result only.

**MOM-3** (moderate, explanation). "The Timescale of $\beta$". Anchor: "When
$\beta$ goes up, $\eta$ has". The reader has just watched the opposite:
gradient descent diverged at $\eta = 0.6$, and adding $\beta = 0.5$, which
doubles the effective step to $1.2$, made it converge. The body never
resolves this; exercise 4 does ("reconcile the two"). The resolution is
short and belongs here, because it is the reason momentum works at all. In a
direction where the gradient alternates in sign, the velocity sums the
series $1 - \beta + \beta^2 - \cdots = 1/(1+\beta)$, so the step there is
$\eta/(1+\beta)$, smaller than $\eta$. In a direction where it persists, the
sum is $1/(1-\beta)$. Momentum lengthens the step where gradient descent was
too slow and shortens it where gradient descent overshot, and the stability
limit moves from $\eta\lambda < 2$ to $\eta\lambda < 2(1+\beta)$ (checked:
at $\beta = 0.5$ the valley is stable up to $\eta = 0.75$). Add these three
sentences after the display of $\eta_\textrm{eff}$, and keep exercise 4 as
the derivation of the limit. The rule "when $\beta$ goes up, $\eta$ comes
down" then reads as a statement about persistent directions and noise, which
is what the airfoil run tests.

**MOM-4** (minor, pedagogy). Cell `momentum-effective-sample-weight`. The
cell prints three closed-form numbers for three values of $\beta$. The
half-life column is printed and never read. Either read it (a gradient's
weight halves in under 7 steps at $\beta = 0.9$) or drop the column. A
sentence would also connect the timescale to the tuning rule that follows:
by :eqref:`eq_momentum-optimal`, $\beta = 0.9$ is the best value for a
quadratic with $\kappa \approx 1{,}400$ and $\beta = 0.99$ for
$\kappa \approx 1.6 \times 10^5$ (from $\sqrt{\kappa} = (1+\sqrt{\beta})/(1-\sqrt{\beta})$).
That is the concrete answer to "why 0.9" which the section opens with and
which the figure caption only gestures at ("which is why large momentum
values are the common default").

**MOM-5** (moderate, correctness; confident). "Acceleration and Damping".
Anchor: "because the factor describes the error only after the". The
explanation of why the step counts differ by 3.5 and not 4.5 is not right.
At the tuned setting both extreme modes have a double root, so the error
behaves like $t\,\rho^t$, not $\rho^t$, at every $t$, not only in the first
few steps. Numerically: $\rho^t = 10^{-6}$ at $t = 31$ for $\rho = 0.635$,
while the run needs 40, and $40 \cdot 0.635^{40} \approx 5 \times 10^{-7}$.
The appendix says this correctly ("The error behaves like $k\rho^k$").
Rewrite: "the step counts differ by a smaller ratio, 3.5, because at the
tuned setting the two roots of each extreme mode coincide and the error
falls as $t\,\rho^t$: the factor $t$ costs about ten of the 40 steps."

**MOM-6** (moderate, explanation). Same subsection. Anchor: "In a mode of
curvature $\lambda$ the". The recurrence
$x_{t+1} = (1 + \beta - \eta\lambda)x_t - \beta x_{t-1}$ is the object that
the whole damping explanation rests on, and it enters as the last sentence
of a paragraph that describes a code cell, without derivation. Three
paragraphs later it is called back as "the recurrence above". Give it its
own short paragraph before the cell: eliminate the velocity
($\eta \mathbf{v}_t = \mathbf{x}_{t-1} - \mathbf{x}_t$), display the
recurrence, say that its solutions are powers of the roots of the quadratic,
and that the larger root in absolute value is the contraction factor. Also
settle the vocabulary: the same thing is "direction", "coordinate", "mode"
and "eigendirection" within a page, and its curvature is $h$ on the valley
and $\lambda$ in the theory. On the valley the coordinates are the modes;
say so once.

**MOM-7** (minor, correctness). Anchor: "both roots are real and one of them
is close to". This describes the flat mode. In a steep mode with
$\eta\lambda > 1$ and $\beta$ below the critical value the roots are real
and negative, and the iterate alternates across the valley as it does under
gradient descent; it does not creep. One clause fixes it ("in a flat mode
one of them is close to 1 ...; in a steep mode the roots are negative and
the iterate still zigzags").

**MOM-8** (minor, explanation; figure). :numref:`fig_opt_critical_damping`.
The vertical axis is labelled "convergence rate per step" and a larger value
is slower, which the figure has to repair with an arrow marked "slower". The
text calls the quantity the contraction factor everywhere else. Relabel the
axis "contraction factor per step" and start the caption with that term. The
tick labels are visibly smaller than the in-figure labels (house style item
2).

**MOM-9** (major, pedagogy). "Implementation" and the end of "Nesterov
Momentum". Anchors: "On this small, noisy problem momentum brings no
speedup" and "On the airfoil data the loss curve is nearly the same as with
plain". Stored final losses: 0.244 (SGD), 0.245 (momentum at the matched
step), 0.263 (momentum at ten times the step), then 0.245 for the framework
optimizer and 0.244 for Nesterov. Four runs on data, and the only visible
effect is that too large an effective step hurts. The cells
`momentum-concise-implementation` and `momentum-nesterov-momentum-2` each
reproduce a curve already on the page. The closing claim of the Nesterov
part, "The correction matters when curvature rather than noise limits
progress, as in full-batch or large-batch training", is then asserted with
no run behind it.
Direction: keep the matched-step comparison as one arm (it is a real lesson:
at a matched effective step momentum changes little when noise dominates),
and add the regime in which momentum pays. Candidates, to be tried on the
development machine: (a) full-batch training of the same airfoil regression
with the features left unstandardized or with squared features added, so
that the condition number is large, comparing gradient descent at its best
rate with momentum at the tuned pair and counting steps as on the valley;
(b) a network from Basics at a large batch size, SGD against SGD with
momentum, each at its best rate from a small grid. Then run Nesterov once in
that regime, or cut the Nesterov data cell. If neither experiment shows a
clear gain, say so and cut the data experiments to the single matched-step
run; do not keep three null results.

**MOM-10** (moderate, explanation). Anchors: "the question of Exercise" and
"shows why". Twice the body states a result about noise and sends the reason
to exercise 6. The reason is short and the main text should hold it (guide
7.6 by analogy: the argument must be followable without the exercise).
Draft: "A noise term $\boldsymbol{\xi}_t$ in one gradient enters the
velocity at step $t$ and stays in it with weights $1, \beta, \beta^2,
\ldots$, so in total it moves the iterate by $\eta/(1-\beta)$ times
$\boldsymbol{\xi}_t$: the same displacement as one step of plain SGD at the
effective rate, spread over about $1/(1-\beta)$ steps. Momentum at a matched
effective step therefore delays the noise but does not remove it." I checked
the claim behind it: on $f = \frac{\lambda}{2}x^2$ the ratio of stationary
variances (momentum 0.9 at the matched step over plain SGD) is 0.98 at
$\alpha\lambda = 0.05$ and 0.90 at $0.2$, in simulation and in closed form.

**MOM-11** (minor, language). "Nesterov Momentum", the paragraph that begins
"At the same $\eta = 0.3$ and $\beta = 0.8$". Eight sentences carry the
reading of the plot, the damping interpretation, the $O(1/k^2)$ rate, the
lower bound, the strongly convex rate, and two appendix pointers. Split
after "where the overshoot is largest." so that the guarantees get their own
paragraph, and say which of them the practitioner should remember (the
method costs nothing extra and damps the ringing; the rates are background).

**MOM-12** (minor, thought; decision 6). The hand-off to Adam is written
twice: the paragraph before the Summary ("The next section gives each
coordinate a step size of its own.") and the last sentence of the Summary
("What momentum cannot do is give different coordinates different step
sizes."). Keep the Summary sentence and delete the paragraph, or the
reverse.

**MOM-13** (minor, language; notation). $\epsilon$ is the target accuracy
here and Adam's denominator constant in the next section; $\mathbf{v}$ is
the velocity here and the second-moment average there; the update is written
$\mathbf{x}_t \leftarrow \mathbf{x}_{t-1} - \ldots$ with the gradient taken
at $\mathbf{x}_{t-1}$ here and $\mathbf{x}_{t+1} = \mathbf{x}_t - \ldots$
in `adam.md`. See "For the chapter review".

## Code and experiments

- `momentum-an-ill-conditioned-problem-1`: repeats the run of `gd.md` at a
  slightly different rate. Could go (structural finding 3).
- `momentum-effective-sample-weight`: prints closed-form values; half-life
  column unread (MOM-4).
- `momentum-acceleration-and-damping-1`: the model cell of the section. All
  numbers in the prose match (0.905, 0.635, 0.707, 0.894; 47, 139, 40, 42,
  16, 18). Keep as is.
- `momentum-implementation-from-scratch-3`: prose matches the stored losses
  in both tabs. Shows that a tenfold effective step hurts, not that momentum
  helps (MOM-9).
- `momentum-concise-implementation`, `momentum-nesterov-momentum-2`: each
  reproduces an existing curve. Cells that only confirm (MOM-9).
- `momentum-nesterov-momentum-1`: the claim "the overshoot along the floor
  is smaller" holds (largest overshoot of $x_1$: 1.13 for heavy ball, 0.67
  for Nesterov, recomputed), although at step 20, the number the cell
  prints, Nesterov's $x_1$ (0.218) is farther from the minimum than heavy
  ball's (0.196). The prose is right; a reader who compares the two printed
  lines may think otherwise. Say that the printed step is not the place to
  compare.
- No TensorFlow or MXNet material. The two tabs fragment the prose only at
  the from-scratch update (two one-sentence tab notes, both fine).

## Exercises

Six, graded, and the best of them (4, 6) are real problems with a surprise
in the answer. Exercise 2 is a clean computation (the ratio
$\sqrt{(1-\beta)/(1+\beta)}$ is correct). Exercise 3 (the library form of
Nesterov) is well chosen and the checkpoint question is the kind of thing
practitioners trip over. Exercise 1 is a grid whose second part saves it.
Changes: once MOM-3 and MOM-10 move the short arguments into the body,
exercises 4 and 6 keep their derivations and lose the role of holding up
the text. Add one short exercise relating :eqref:`eq_momentum` to the
averaged form $\mathbf{m}_t = \beta\mathbf{m}_{t-1} + (1-\beta)\mathbf{g}_t$
and to PyTorch's `dampening`, since `adam.md` relies on that equivalence in
one sentence.

## Still open from October

The October review asked that `momentum.md:63–65` be kept; the line numbers
have moved and I could not identify the passage with certainty. Its
chapter-level findings that touch this section are open: the valley is
re-run in several sections, and $\lambda$ changes meaning along the chapter.

## For the chapter review

- Fits its place: needs `sgd.md` (minibatch gradient, `train_with_rule`,
  airfoil) and `gd.md` (valley, `train_2d`); sets up `adam.md` (leaky
  average, window $1/(1-\beta)$, effective step). `adam.md` uses all three
  correctly.
- Against the appendix (`mdl-gradient-based-optimization.md`, "Momentum and
  Acceleration"): compliant with decision 3. The appendix writes heavy ball
  with the learning rate inside the velocity; a reader who follows the
  pointer meets a different parameterization without warning. One clause at
  the pointer would help. The appendix has the $k\rho^k$ fact that MOM-5
  gets wrong.
- Chapter-wide patterns to check: (1) the valley is re-motivated at the top
  of each section with the same numbers (here, and again in the introduction
  of `adam.md`); (2) notation drifts between adjacent sections ($h$ and
  $\lambda$ for curvature, $\mathbf{v}$ for two things, the time index of
  the update); (3) results that the body uses are parked in exercises; (4)
  experiments on data that show no effect are kept and explained instead of
  redesigned.
- If the test models of `adam.md` get their own section (see the review of
  `adam.md`), placing it before this section would give momentum a network
  to be tested on.
