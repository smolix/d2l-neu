# Writing pass log: Stochastic Gradient Methods

File: `chapter_mdl-optimization/mdl-stochastic-gradient-methods.md`. Not split.

Prose words (checker count, body and exercises): 7519 before, 7518 after.
Checker: "OK: only prose changed", no em-dash warning, no labels added or
removed. `tools/lint_source.py`: clean. Verification scripts were run with the
system `python3` and NumPy (there is no `.venv-pytorch` on this machine).

## Correctness changes

- **SGM-8.** The proposition on the constant of a $1/k$ schedule was followed
  by a leading-order matching and an approximation, closed with a proof mark.
  Now: the statement stays a Proposition; a sentence after it says "We do not
  prove the proposition here", that the constant is derived by matching
  leading terms, that the other cases are argued informally, and that
  exercise 2 supplies the proof of the two rates. The second paragraph is
  headed "Below the threshold (informally)". The proof mark is gone. Checked:
  exercise 2(a) to (c) covers an $O(1/k)$ bound whose constant tends to $A$,
  and the $k^{-2c\mu}$ rate. Note: the exercise bounds $k e_k$; it does not
  literally prove the limit $k e_k \to A$. The text says "the proof of the
  two rates", which is what the exercise gives.
- **SGM-7.** Added the boundary case to the statement ("at the boundary
  $2c\mu = 1$ like $\log k/k$") and one sentence of informal argument (noise
  injected at step $j$ survives as about $c^2\sigma^2/(jt)$; the sum over $j$
  gives the logarithm). Verified: iteration at $c = 0.5$, $\mu = \sigma = 1$
  gives slope $-0.98$ between $10^5$ and $10^6$; $k e_k/\log k$ is $0.88$,
  $0.27$, $36$ for three starts, that is, the leading constant
  $c^2\sigma^2 = 0.25$ plus a start-dependent $O(1/k)$ term.
- **SGM-15.** "the noise term is 140 times the transient" is now "140 times
  the deterministic rate $2L\Delta/K$" ($0.0283/0.0002 = 141$).
- **SGM-17 / S3.** The $\tanh$ experiment no longer claims that a slope of
  $-0.6$ is "consistent with" the bound. The lead-in says an upper bound
  cannot be confirmed by an experiment, only not violated; the reading says
  the measurement falls a little faster than the bound's slope, as a bound
  permits, and that the best iterate falls much faster but cannot be
  returned.
- **SGM-16.** "but the sum is not convex" is now "and none is convex".
- **SGM-19.** Proposition title: "the variance is bounded by the distance to
  the snapshot". The gloss said the noise is "proportional to the distance";
  it now says the noise power is at most a multiple of the squared distance.
- **SGM-20.** "below about $0.008$" is now "below $1/(4L_{\max})$, the second
  number printed, between about $0.006$ and $0.01$" (stored output: $0.0066$).
- **SGM-21.** "so the theorem's constants are loose" is now "so on this
  problem the theorem's step condition is not necessary". Added that SVRG
  loses early: after its first epoch its gap is far above SGD's. Verified on
  8 fresh draws: SVRG $0.4$ to $1.4$ after epoch 1, SGD $0.01$ to $0.04$.
- **SGM-14.** "such as the logistic loss" now reads "such as the logistic
  loss without a quadratic penalty".
- **SGM-25, a number I changed.** The text said the convex bound
  overestimates the gap on the logistic problem "by more than an order of
  magnitude". I ran exercise 10 as stated (4 draws of the data, 20 runs each,
  $b = 1$, $K = 4000$, $G$ bounded on the ball of radius $2D$): the factor is
  $9.4$ to $9.9$. The text now says "about an order of magnitude" and "as
  exercise ... measures".
- **SGM-2.** "which serves all the measurements of this section" (false) is
  now "which the step-size experiment of the next part uses again".
- Small: "always within a factor of two" (of the SVRG cell's SGD plateau) was
  an unverifiable "always" and is dropped; "does not carry over to a network"
  is now "need not carry over".

Findings I did not follow: none on correctness. **SGM-24** (rename the
constant $A$) is not done: $A$ is used in exercises 2 and 6 and in the
slides, and the brief says to list notation clashes instead (below).

## Unprinted numbers kept (all recomputed)

Floors $0.053$, $0.176$, $0.088$; 36 and 9 steps; $3{,}450$ steps;
valley floor $7.5 \times 10^{-4}$ with shares $2.5$ and $5 \times 10^{-4}$;
$k e_k = 1.0001$ at $c = 1$; slopes $-0.50$ and $-0.20$; $A = 4/3$;
$0.008$ and $4.2 \times 10^{-3}$ for the $1/k$ schedule on the valley;
$A = 225/29$ and $1.9 \times 10^{-3}$; $2.7 \times 10^{-3}$ for full linear
decay; $0.014$, $2 \times 10^{-4} + 0.028$. Dropped, to save words: $A = 1.125$
at $c = 0.75$; the $K = 10^6$ instance of the nonconvex bound; the worked
$D = G = 1$ numbers that the figure caption already gives; the loss floor
$0.026$ and root mean squares $0.23$, $0.42$ in the body (the opening keeps
$0.053$ and $0.23$).

## Six before/after pairs

1. Before: "Does the error of the minibatch gradient fall like $1/b$? The
   next cell computes ..." After: "The proposition is exact, so sampling can
   only agree with it. We sample once, at $\mathbf{x} = \mathbf{0}$ with
   $10{,}000$ minibatches of each size, mainly to learn how large
   $\operatorname{tr}\boldsymbol{\Sigma}$ is on this problem."
2. Before: "The cell below runs $4{,}000$ steps with $b = 8$ for two fixed
   steps ..." After: "If the quadratic model is a guide, halving a fixed step
   should halve the gap at which the run stalls. We run $4{,}000$ steps ..."
3. Before: "Does $\mathbb{E}\|\nabla f(\mathbf{x}_R)\|^2$ fall like $K^{-1/2}$
   when $\eta \propto 1/\sqrt{K}$? We measure it on a small nonconvex
   problem". After: "The proposition is an upper bound, so an experiment
   cannot confirm its rate, only fail to violate it. We measure the
   random-iterate quantity, and beside it the best iterate, on a small
   nonconvex problem".
4. Before: "Does a fixed step converge without a floor once the noise is
   cancelled? The cell runs SVRG ... It also predicts SGD's floor". After:
   "We run SVRG ... For SGD the first part of this section makes a prediction
   that the run can miss: its floor should be the sum of the loss floors".
5. Before: "This is the step that is not routine: the inner product of the
   gradient ..." After: "Everything rests on this inequality: the inner
   product of the gradient ..." (the third instance became "Conditioning on
   the iterate does two things at once"; the first, in the $1/k$ derivation,
   is kept).
6. Before: "We can now measure the noise of a minibatch gradient, predict
   where a fixed step stalls ..." After (end of the last subsection): "A
   second limit has been with us since the valley: all of these methods share
   one step size across coordinates, which the stiff coordinate of the
   running example capped at $2/1000$. The next section lets each coordinate
   set its own step from the gradients it has seen."

## Repetition and ownership

Kept because this file owns them (chapter table items 3 to 5, and the
assignment): the second-moment recursion, its solution and the floor in
full; the Robbins--Monro conditions and the remark that the quadratic needs
only the first; the $1/k$ constant; the exact schedule comparison (the
section's complete version; nothing removed from it).

Chapter 9: `sec_sgd` is now cited once at the start of "The Noise Floor"
("measured the cloud ... and derived its size. Here we solve the whole
trajectory"), and `sec_scheduler` once for schedules on trained networks. The
floor formula is identical to `eq_sgd-noise-ball`. Two paragraphs that retold
chapter 9's intuition (transient against floor; kick against pull) are merged
into one.

Removed inside the file: the second derivation of $0.053$ and $0.23$ (SGM-5);
the operational reason for WSD, now stated once (SGM-12); the orphaned
noise-model paragraph, moved to the valley paragraph (S5, SGM-3), with the
tie of $\sigma^2$ to $\boldsymbol{\Sigma}/b$ moved to where the scalar model
is introduced; the subgradient definition, reduced to a clause citing
`sec_mdl-convexity`; the sentence in the opening that the closing repeats;
the momentum comparison and a third of the SAGA paragraph and of the
Johnson-Zhang hypotheses (S4); the "We can now" recap (S6). The Summary lost
its last clause, which repeated the closing.

Ranking: no Proposition block was demoted (the four that remain are cited or
are main results). The opening now names the three results to remember and
says variance reduction can be skipped on a first reading. Labels added:
none.

## Cells to cut or redesign on the development machine

- `gradient-based-optimization-sgd-variance`: confirms an exact proposition
  by sampling (overview finding 4). Prose now uses it to report the size of
  $\operatorname{tr}\boldsymbol{\Sigma}$. Could print that one number only.
- `adaptive-stochastic-methods-ghadimi-lan` (S3, SGM-17): cannot fail.
  Redesign as the two terms of the bound against $\eta$ at fixed $K$.
- `mdl-stochastic-gradient-methods-svrg-and-saga-1` and `mdl-adaptive-svrg`
  (S2): run SVRG on the logistic problem, removing the ridge data cell.
- `adaptive-stochastic-methods-schedules` (SGM-13): the legend says `c/k`;
  prose now says "the $1/k$ decay $c/(k + k_0)$" and the caption "(labelled
  $c/k$)". The figure `mdl-opt-schedule-zoo.svg` draws a warmup that the cell
  omits (SGM-11); the caption and text now say so.
- Imports cell loads frameworks no cell uses; stale cell IDs and exercise
  labels (review, "Code").

## Other deferred findings

- S1: aligning the valleys (curvatures $0.2$ and $4$ in chapter 9, $1$ and
  $1000$ here) and the step index ($t$ there, $k$ here) needs cell and
  main-text changes.
- Exercises: missing difficulty marks, exercise 1(c) overlap with
  `sec_batch_size`, exercise 4(c), a new exercise for $(n-b)/(n-1)$: left.
- SGM-25: "exercise" stays lower case here; the next section uses
  "Exercise". One of the two files must change.
- Notation clashes (SGM-24): $A$ (constant) against the data matrix `A`;
  $\lambda$, $\sigma^2$, and $k$ against $t$ with the next section.

## Claims added or changed

- "the curvature $\mu$ is rarely known, and a constant chosen too small loses
  the rate itself. This is one reason practical schedules decay more slowly
  than $1/k$" (replaces the "Intuitively" sentence; the reason given is new).
- "Large networks can fit their training sets, which is one reason why
  constant-step SGD works better on them than the floor formula suggests"
  (SGM-18; hedged as one reason, strong convexity stated in the sentence
  before).
- "The baseline of policy gradients (`sec_baselines`) is the control variate
  that this book relies on most."
- The warmup is "left out of the computation, because the quadratic model is
  stable at $\eta_0$ from the first step" ($\eta_0\lambda = 0.3 < 2$).
- SVRG "loses early and wins late"; "its first epoch ... takes only $n$ steps
  with a control variate anchored at the starting point".
- "Our code takes the last one, which in practice behaves the same" (was a
  parenthesis, SGM-22).
- The stiff coordinate "holds two thirds of the floor" ($5/7.5$).
- The boundary case and "about an order of magnitude" (above).

## Slides against the body

No slide contradicts the revised body. The slides still call the schedule
"$c/k$" and state the $1/k$ threshold without the boundary case; the slide
titled "The Measured Slope Is About $-0.6$ Against the Guaranteed $-1/2$"
agrees with the new reading.
