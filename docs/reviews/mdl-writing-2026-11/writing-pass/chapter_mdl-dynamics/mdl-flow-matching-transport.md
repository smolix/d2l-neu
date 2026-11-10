# Writing pass log: Flow Matching and Optimal Transport

File: `chapter_mdl-dynamics/mdl-flow-matching-transport.md`. Not split (no
markers). No code cell, cell ID, figure include, label or slide changed.
Checker: "OK: only prose changed", no em-dash warning; `tools/lint_source.py`
clean (exit 0). Numeric checks were done with `python3` + NumPy in the
scratchpad (no `.venv-pytorch`, no SciPy, no PyTorch on this machine) or by
hand; which is noted per item.

## Words

Checker count (body, captions, exercises): 8,693 before, 8,323 after (-370).
Display-equation delimiters 46 to 42 (the regression lemma's second display
and the sampler display are gone). The model-families table (5 rows) is also
gone; the checker does not count table cells, so the visible reduction is
larger than 370 words.

## Correctness changes

| Finding | Before | Now | Verified |
|---|---|---|---|
| FMT-1 | "a noise predictor trained in the previous section turns out to be a velocity field already"; "The previous section trained a network to predict the noise."; "A trained noise predictor is therefore also a velocity field, and the samplers of this section can run on it unchanged." | opening sentence removed (the ranking paragraph names the dictionary instead); "Suppose a network has been trained to predict the noise, as diffusion models are."; "A noise predictor is therefore also a velocity field: the samplers of this section apply to it after the conversion of :numref:`tab_mdl-score-noise-velocity`." Also in the first paragraph: "each of which calls the network once" is now "each of which evaluates the score once". | read the previous section's claim in the review; no cell here runs a sampler on a noise predictor |
| FMT-5 | reflow proposition applied to the two points and read against the two-moons numbers although its Lipschitz hypothesis excludes both | hypotheses of the proposition unchanged; a new paragraph after the corollary says that for the two points, and for any finite training set paired independently with Gaussian noise, the field is Lipschitz on $[\delta, 1]$ but not up to $t = 0$, that the proof then gives (i) for $t \ge \delta$ and (ii) for the displacement from $1$ to $\delta$ divided by $1 - \delta$, and that $\delta \to 0$ is not proved. The two-point paragraph now opens "For the two points the conclusion can be verified directly." | by hand: $\partial_x u_t(0) = (1 - (1-t)a^2/t^2)/t \to -\infty$; for bounded data the Jacobian is $(\mathbf I - (1-t)\,\mathrm{Cov}(\mathbf x_0 \mid \mathbf x_t)/t^2)/t$, bounded on $[\delta,1]$; (ii) on $[\delta,1]$: Jensen for the average over an interval of length $1-\delta$ gives $\mathbb E\,c((\mathbf Z_1 - \mathbf Z_\delta)/(1-\delta)) \le \mathbb E\,c(\mathbf X_1 - \mathbf X_0)$ |
| FMT-6 | uniqueness for the continuity equation in one clause, as if proved | "This uniqueness is the one step we do not prove in full: :numref:`sec_mdl-probability-flow-ode` gave the argument for smooth positive densities ... and we use it here without redoing it in the weak form." | I did not follow the reviewer's wording that an exercise "extends it to the weak form": exercise 9(a) proves the weak continuity equation, not uniqueness |
| FMT-14 | one run-on statement ending "so" + display with $\min$ | parts (i) and (ii); the display is introduced by "and then, with the minimum over admissible pairs"; after the gloss: the equality holds in general with the minimum over weak solutions with square-integrable velocity, not proved here, and the examples need it because the optimal two-point field has slope $1/t$. The proof says which step gives which part. | by hand: optimal two-point pairing $x_0 = a\,\mathrm{sign}(\epsilon)$, field $(x - a\,\mathrm{sign}\,x)/t$ on the support. The general statement is the standard one (Ambrosio, Gigli, Savaré; Villani), quoted from memory, no citation key added |
| FMT-9 | "The trained network and its 32 Euler steps only approximate that flow, and the measured drop is far larger than the bound." | reason first (Jensen for the time average is loose on curved trajectories), approximation in a parenthesis | n/a |
| FMT-12 | two paragraphs of mechanism with about ten uncomputed numbers; "Where the steps go matters more than the order of the solver." | one paragraph with three recomputed facts (below); "On this field, where the steps go matters more than the order of the solver."; Summary limited to "on the exact field of the running mixture" | NumPy, script `scratchpad/fmt/s.py` |
| FMT-13 | minibatch numbers stated as if a cell above produced them; two opening sentences about curvature | opening sentences deleted; "the exercise trains the two moons with this pairing. The loss plateau falls from 1.4 to about 0.06 ..." | numbers are the reviewer's run (plateau 0.056, one step 0.004); I could not rerun (no SciPy, no PyTorch) |
| FMT-8 | "energy distance" and "squared energy distance" mixed | definition paragraph ends "every score quoted below is this squared distance"; the plot sentence says "squared energy distance"; the minibatch sentence says "scores". The print label in cell `mdl-flow-matching-transport-reflow-1` still reads "energy distance" (code, unchanged) | n/a |

Findings I disagreed with: none outright. FMT-12's suggested sentence "on
equal steps Euler's cancel and Heun's do not, and on the crowded grid the
reverse holds" is too strong by my computation (see below), so I wrote what
the numbers show.

## Numbers no cell prints

Recomputed with NumPy and kept:

- solver table ratios $2.8$, $3.3$, $3.6$ (got 2.84, 3.32, 3.66) and all
  table entries (identical to the stored output, the cell is seeded);
- "By $t = 0.5$ the paths have covered less than a tenth of their length"
  (0.085 to 0.092 on a 800-step Heun reference);
- Euler's signed step errors on the uniform grid "cancel to about an eighth
  of their total size" (ratio of mean $|\sum_k c_k|$ to mean $\sum_k |c_k|$:
  0.12 at 10 and at 40 evaluations), "more than half of Heun's survive"
  (0.58, 0.70), "On the squared grid about a third of each survives" (Euler
  0.31, 0.34; Heun 0.25, 0.40). Step contributions were computed to first
  order: exact flow to $t_k$, one coarse step, exact flow to $0$;
- sign pattern "Euler falls short while the paths speed up and overshoots
  while they slow down" (mean speed peaks at $t = 0.24$; Euler's mean step
  contributions are negative above $t \approx 0.2$ and positive below);
- quarter-turn pairing cost $4$ (by hand, $\|\mathbf R\mathbf x - \mathbf
  x\|^2 = 2\|\mathbf x\|^2$; Monte Carlo 4.004) and $\det((1-t)\mathbf I +
  t\mathbf R) = (1-t)^2 + t^2 > 0$;
- unchanged and already checked by the reviewer: $-2.86$, $0.97$, the slopes,
  $t^*$, $1.81$, the floor $1.38$, $1.24$, $\pi^2/2$.

Dropped: "fastest near $t = 0.25$" (true, 0.24, but not needed), Heun's
overshoot/shortfall ranges ("above about 0.4", "between 0.1 and 0.4"), "by
about a tenth for Euler", the slope $-33$ at $t = 0.1$ (true, $-32.7$), the
EDM facts ($\rho = 3$, $\rho = 7$, 35 evaluations; `samplers.md` in the
diffusion chapter has the grid), "a single batch of 256 scatters by about
$\pm 0.1$".

## Six before/after pairs

1. Before: "Does the closed form agree with a direct posterior average, where
   does the origin stop attracting, and where does one Euler step of the
   exact field land? The cell checks all three, the slope by a finite
   difference."
   After: "The cell compares the closed form with a direct posterior average
   at three points, takes the slope at the origin by a finite difference, and
   applies one Euler step of the exact field to $\epsilon = \pm 1$."
2. Before: "Where does the loss settle when noise and data are paired
   independently?"
   After: "The first run pairs noise and data independently. By the CFM
   theorem its loss cannot reach zero, and where it stops measures how much
   the segments cross."
3. Before: "How does the quality of the samples depend on the number of Euler
   steps? The cell integrates the same noise array `z_eval` with ..."
   After: "One step of the exact field would give the point mass that scored
   $0.9$. For the trained field, the cell integrates the noise array `z_eval`
   with ..."
4. Before: "How does this change sampling, and how far did the pairing cost
   drop? The next cell compares the two models from the same noise array and
   prints the mean squared length of the old and the new pairs."
   After: "A field without crossings is nearly constant along its
   trajectories, and then a few Euler steps should do. The next cell samples
   from both models with $1$, $2$ and $32$ steps from the same noise array,
   and prints the mean squared length of the old and the new pairs, the cost
   that the proposition bounds."
5. Before: "Are the reflowed trajectories straight? The cell measures the
   mean distance of each $32$-step trajectory from the chord joining its two
   ends."
   After: "Straightness can be measured as well, by the mean distance of each
   $32$-step trajectory from the chord joining its two ends."
6. Before: "The theorem needs no hypothesis on the path beyond a
   square-integrable target. The proposition is what guarantees ..."
   After: "The proof is one line because the lemma does the work, and the
   theorem needs no hypothesis on the path beyond a square-integrable target.
   It is the proposition that guarantees ..."

Kept as questions because the reader can get them wrong: the three flows
("The first must be cheapest, but which of the other two comes second?") and
"Is it worth two evaluations?" before the solver cell.

## Repetition

Removed or reduced:

- regression lemma display and gloss (FMT-3): now one sentence with
  :eqref:`eq_mdl-regression-lemma` (owner: score-matching section);
- probability-flow ODE and reverse SDE display, and the paragraph on the
  trace integral (FMT-11): one sentence with :eqref:`eq_mdl-score-samplers`.
  The unlabeled heading "Deterministic and Stochastic Samplers" is gone; its
  sentence opens `sec_mdl-sampling-learned-dynamics`;
- EDM paragraph: two sentences, citing `sec_diffusion-samplers`;
- model-families table (FMT-11): removed; the subsection (label
  `sec_mdl-unifying-table` kept) now cites `tab_diffusion_models` and keeps
  the one-sentence reading and the open question;
- "We can now ..." recap (FMT-15): deleted; "What remains open ..." is the
  end of the last subsection;
- plumbing before the training cell (FMT-7) and the `euler_path` description;
- the quarter-turn remark after the Benamou-Brenier proof is one clause now
  that the example is in the body.

Kept because this file owns them: marginal-velocity proposition, CFM
theorem, two-point example, score-velocity identity and
`tab_mdl-score-noise-velocity`, reflow proposition, couplings and $W_2$,
Benamou-Brenier with proof. The restatement of the coupling vocabulary from
`sec_mdl-optimal-transport` stays (small, needed for the statement).

Diffusion chapter: the labels it cites from this file are
`sec_mdl-flow-matching-transport`, `sec_mdl-flow-matching` (twice),
`sec_mdl-rectified-flow`, `subsec_mdl-reflow`, `subsec_mdl-fm-two-point`
(twice), `tab_mdl-score-noise-velocity`, `fig_mdl-dyn-time-conventions`,
`sec_mdl-sampling-learned-dynamics`, and the exercises on late decisions and
target conditioning; the GAN chapter cites `sec_mdl-ot-connection`. All
passages still carry what is cited (the solver cell and its reading stay for
`samplers.md`; `further-reading.md` cites `sec_mdl-ot-connection` for the
quarter-turn pairing, which is now in the body).

**For another file (not edited):** `chapter_diffusion-models/gaussian-paths.md`
near line 600 cites `sec_mdl-ddim-guidance` for "$\mathbf u_1(\mathbf x) =
\mathbf x - \mathbb E[\mathbf x_0]$: one step sends every start to the mean
of the data". The statement lives here, in the first bullet of
`subsec_mdl-fm-two-point` ("This holds for any data under independent
pairing ..."), which I left intact so that it bears the citation. The
reference there should become `subsec_mdl-fm-two-point`.

## Ranking

No Proposition block demoted: the four blocks are all cited later or from
other chapters. Added to the opening: "Three results carry the section" (the
posterior-mean velocity and its regression, reflow, Benamou-Brenier) and
which two parts can be skipped on a first reading (the dictionary and the
solver comparison). In the reflow subsection: "This is the result of the
section that practice leans on most." (a judgment; soften if too strong). In
the reflow proof the uniqueness step is named as the one not proved in full.

Labels added: none. New references, all resolving: `eq_mdl-regression-lemma`,
`eq_mdl-score-samplers`, `sec_diffusion-samplers`, `tab_diffusion_models`,
and this file's own `sec_mdl-flow-matching`, `subsec_mdl-reflow`,
`sec_mdl-ot-connection`, `sec_mdl-score-velocity-dictionary`,
`sec_mdl-sampling-learned-dynamics`, `sec_mdl-rectified-flow`.

## Split markers

None (not to be split).

## Cells to cut or redesign on the development machine

- `mdl-score-matching-diffusion-flow-score-noise-and-velocity-are-one-function`
  (FMT-4): confirms an identity proved six lines above. Prose reduced to two
  sentences before and two after. Candidate for an exercise.
- `mdl-flow-matching-transport-example-two-points-in-one-dimension`: also a
  confirmation, but the one worth keeping (the slides use it; the one-step
  collapse is the section's hinge). Prose after it shortened.
- `score-matching-diffusion-flow-euler-vs-heun` (FMT-12, structural finding
  1): on the previous section's object; home is next to DDIM there or
  `samplers.md`. Kept with its reading because `samplers.md` cites this
  section for it.
- Exercise 7 (minibatch pairing, FMT-13): promote to a cell if the body is to
  quote its numbers as its own.

## Other deferred findings

- FMT-4 (move "Score, Noise, and Velocity" before the two-point example):
  not done, because its cell would move ahead of the two-point cell. The
  hand-off is repaired instead: the two-point example now ends by pointing to
  `sec_mdl-rectified-flow` and announcing the dictionary, and the opening
  marks the dictionary as skippable.
- FMT-10, figure: panel (c) of `mdl-dyn-couplings-energy` still draws the
  rotating flow. The body keeps the rotation (the figure and its caption need
  it) and adds the quarter-turn pairing after it; a chord panel is a figure
  change.
- FMT-12: an exercise or cell printing the signed step contributions (the
  three numbers above come from my scratch script, not from the notebook).
- Structural finding 4 (four framework tabs): policy question, untouched.
- Exercise 2's label prefix (`ex_mdl-score-matching-diffusion-flow-...`):
  left, labels may not be renamed.
- FMT-2: the opening is still about 400 words before the first heading (the
  callout and notation paragraph included); I merged the fifth paragraph into
  the fourth and spent the saving on the ranking paragraph.

## Claims added or changed

- "A sample costs one network call per solver step" (opening) and "a sample
  costs one field evaluation per solver stage" (sampling part).
- The FMT-5 paragraph (Lipschitz on $[\delta, 1]$; what the proof gives
  there; $\delta \to 0$ not proved).
- Benamou-Brenier: equality in general over weak solutions with
  square-integrable velocity, unproved here; the optimal two-point field has
  slope $1/t$ and is not admissible.
- The quarter-turn pairing paragraph: never crosses, constant speed, $C = 0$,
  reflow leaves it unchanged, cost $4$ against $W_2^2 = 0$.
- "A loss of $1.4$ therefore does not mean a poor fit."
- "A field without crossings is nearly constant along its trajectories."
  (Exactly constant when $C = 0$; "nearly" covers the trained case.)
- The mechanism paragraph after the solver table (numbers listed above).
- "Diffusion samplers crowd their steps toward the data in the same way, and
  choose how much by measurement" (kept from the old EDM paragraph, now
  without its three numbers).
- Summary: "a pairing can be straight and crossing-free without being
  optimal"; the solver sentence limited to the running mixture.
- Removed sentence: "Gaussian conditional paths with general
  $(\alpha_t, \sigma_t)$ recover the diffusion targets of the previous
  section instead." (said by the dictionary subsection).

## Slides that now disagree with the body

- "Five Model Families, One Regression": the body no longer has this table
  (it cites `tab_diffusion_models`).
- "Where the Steps Go Matters More Than Solver Order" (title) and the
  wrap-up bullet: the body now limits the claim to this field. The bullet
  "Step errors of opposite sign cancel: Euler's on uniform steps, Heun's on
  $t = u^2$" is stronger than what I computed for Heun on the squared grid
  (about a third survives for both solvers there). `samplers.md` line 347
  carries the same sentence.
- "Two Moons: The Loss Settles ..." lists the network's initializer and
  schedule, which the body no longer describes (still true of the code).
- Benamou-Brenier slides state the $\min$ without the admissibility
  condition; the reflow slide has no word on the hypotheses (FMT-5).
- Cover kicker lists "solvers" as a fifth topic; the body marks that part as
  skippable.
