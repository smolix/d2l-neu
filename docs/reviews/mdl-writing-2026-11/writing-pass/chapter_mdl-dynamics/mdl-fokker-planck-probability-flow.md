# Writing pass log: The Fokker--Planck Equation and Probability Flow

Source: `chapter_mdl-dynamics/mdl-fokker-planck-probability-flow.md`. Prose
only; no code cell, cell ID, figure include, label or slide changed. Not
split, no markers. Numeric checks were run with `python3` and NumPy (no SciPy,
no PyTorch on this laptop), with the notebook's own formulas copied into a
scratch script; nothing was re-executed in the notebook and no output was
re-captured.

## Words

Prose words (checker count, body and exercises): 8681 before, 8841 after
(+160). Body without exercises (own count): 6650 before, 6810 after. The
checker ends "OK: only prose changed", no em-dash warning;
`tools/lint_source.py` exits 0.

The section is longer, against the brief. The additions are almost all
repairs the assignment asked for: the uniqueness statement and the hypotheses
of the two propositions (about 120 words), the status of Anderson's theorem
(about 40), the backward-heat-equation objection FPK-12 (about 60), the
honest reading of the mass split (about 70), and the conditioning argument
for the mixture (about 50). Cuts (about 400 words): the notation paragraph of
the introduction, most of "Pathwise and Ensemble Views", the superposition
proposition block, the "We can now" recap, the third and fourth statements of
"the score is the only unknown", repeated glosses of advection and diffusion,
and several single sentences. The review itself says that this section "has
little that could be cut". If parity is required, the next candidates are the
ranking sentences in the introduction (55 words) and the likelihood paragraph
(nothing in the body uses it).

## Correctness changes

- **FPK-4, FPK-7, FPK-11 (uniqueness).** A new paragraph at the end of "From
  Itô's Lemma to the PDE" says that the derivation gives one direction only
  (marginal implies solution), and states the converse as an assumption that
  is cited and not proved: for $g>0$ and a drift Lipschitz in $\mathbf x$
  uniformly in $t$ the equation has exactly one solution among probability
  densities with a given initial law (`Bogachev.Krylov.Rockner.ea.2015`, the
  reference the section already used inside the family proof). It is referred
  to at the mixture ("by the uniqueness stated after
  :eqref:`eq_mdl-dyn-fokker-planck`") and in the family proof for
  $\lambda>0$. **Not verified against the book**: the statement is from
  memory of chapter 9 of Bogachev et al. (uniqueness of probability solutions
  under a Lyapunov condition, which a globally Lipschitz drift with constant
  nondegenerate diffusion satisfies with $V=1+\|\mathbf x\|^2$). The exact
  solution class in that theorem should be checked by someone with the book.
- **FPK-4 (mixture).** Said: superposition proposition, then "the mixture of
  the evolved components solves the equation", read as the marginal. Now: the
  marginal is obtained first by conditioning on the component (by the kernel,
  $X_t$ is $\alpha_tX_0$ plus an independent $\mathcal N(0,\sigma_t^2)$
  variable, so given the component it is Gaussian with mean $\alpha_t\mu_k$
  and variance $\alpha_t^2v_k+\sigma_t^2$); linearity of the PDE is then "the
  same fact" read off the equation, with uniqueness named. Check by hand:
  $0.25e^{-2t}+1-e^{-2t}=1-0.75e^{-2t}$. The read-out sentence "the
  simulation above showed that the SDE's particles follow such a solution"
  (it referred to the point-start cloud) is removed.
- **FPK-7 (twin).** Statement now reads "Let $p_t$ be the marginal density of
  ..., assumed smooth and positive, so that it solves the Fokker--Planck
  equation." The unused hypothesis "with the decay of the previous
  subsection" is removed. Proof unchanged.
- **FPK-11 (family).** Statement now: "Let $p_t$ be the marginal density of
  the forward SDE, under the hypotheses of the deterministic-twin proposition
  and with $\mathbf f$ Lipschitz in $\mathbf x$." The proof's last step names
  uniqueness as the step that is not proved and says why it applies:
  $\mathbf b'=-\mathbf f+(1+\lambda^2)(\mathbf f-\mathbf u_t)$, so
  $\mathbf b'$ is Lipschitz when $\mathbf f$ and $\mathbf u_t$ are (by hand,
  from $g^2\nabla\log p_t=2(\mathbf f-\mathbf u_t)$).
- **FPK-9.** Subsection retitled "The Reverse-Time SDE" (it had no label; no
  other file refers to the old heading). It now opens: the one-step
  calculation is a heuristic, Anderson proved the result for the whole
  process, the theorem is stated without proof, and what is proved below is
  the statement about marginals. The later sentence "Anderson's theorem
  itself is stronger than the proposition" is removed as a duplicate. The
  Summary carries the same distinction in a parenthesis.
- **FPK-8.** The justification for evaluating $\mathbf f$ at $\mathbf y$ now
  precedes the completion of the square, and the exponent is written with
  $\mathbf f(\mathbf y,t)$. The trailing sentence about order $\tau^{3/2}$ is
  removed.
- **FPK-13 and the symmetric check.** The clause "splits the mass within
  about $0.01$ of one half" is gone. A new paragraph says that start, score
  and noise are symmetric, so the output law is symmetric whatever the
  sampler's errors, the fraction is one half up to the binomial standard
  deviation $0.0035$ ($\sqrt{0.25/20000}$, by hand), and the check can show
  sampling noise only; unequal weights are referred to the exercise "Prior
  mismatch, ODE versus SDE". "KS ... usually below $0.01$" is left as it was.
  The stored output (`mass on the left: 0.487`, `KS 0.0148`) still disagrees
  with typical runs: deferred, re-capture.
- **FPK-14.** "removes nearly nine tenths of the bias" kept and completed:
  "$1{,}500$ steps in all ... $1{,}500$ uniform steps remove six tenths".
  Recomputed with the cell's recursion on a non-uniform grid: excess $0.00474$
  at 600 uniform steps, $0.00061$ with ten times finer steps on $t\le0.5$
  (87% removed), $0.00189$ at 1500 uniform steps (60% removed).
- **FPK-5.** "This is the truncation error of the finite differences" now
  carries its evidence: "halving the grid spacing divides it by four at all
  three times". Recomputed: $4.42\times10^{-6}$, $1.10\times10^{-6}$,
  $2.76\times10^{-7}$ at $t=0.25$ for 4001, 8001, 16001 points, and the same
  ratio at $t=1$ and $t=3$.
- **Width excess.** "Part of the width excess is the bias" is now "The width
  excess is the bias", and after the recursion "the bias, $0.005$, accounts
  for the excess of the sampling run". Basis: recursion $0.5047$ at 600 steps
  against "around $0.505$" in prose and stored $0.506$, $0.507$ (Monte Carlo
  error $0.0035$). This is the judgment I am least sure is not too strong;
  the recursion is for one mode started from the exact $p_T$.
- **FPK-6.** "lands anywhere in $1\pm1.73$" is now "lands at $1$ plus a draw
  of standard deviation $\sqrt3\approx1.73$".
- **FPK-2.** "The paths differ from run to run, but the density does not" is
  replaced by a statement of what one run shows: it matches three curves that
  contain no randomness.
- **FPK-1.** The "forgotten where it started" sentence is replaced by the
  reviewer's wording (no map from endpoint to start; the density of the cloud
  can be run the other way).

No finding disagreed with. FPK-3 only partly applied, see notation.

## Six before/after pairs

1. Before: "A single noisy path cannot be reversed, because it has forgotten
   where it started, but its density can."
   After: "No map takes the endpoint of a noisy path back to its start, since
   many starts lead to the same endpoint. But we know how the density of the
   whole cloud changed, and we can build processes whose density changes the
   other way."
2. Before: "The opening claimed that a large cloud has the same histogram on
   every run. To test it, $100{,}000$ particles start at $x_0 = 2$ ..."
   After: "The SDE section compared the moments of such a cloud with the
   kernel. Here we look at the whole histogram: $100{,}000$ particles start
   at $x_0 = 2$ ..."
3. Before: "No proposition about a single Gaussian covers the mixture;
   linearity does. We check numerically that the closed-form mixture density
   satisfies the PDE, ..."
   After: "With the closed form in code, we can put it into the PDE directly."
   and, after the cell, "The argument did not need this computation; the cell
   guards against a slip in the code on which all later experiments rely."
4. Before: "The next cell implements :eqref:`eq_mdl-dyn-mixture-score` with a
   numerically stable softmax for the responsibilities, checks it against the
   tanh form, and locates the zeros and the slope at $0$."
   After: "In code we use the general formula ..., with a numerically stable
   softmax for the responsibilities, because the later experiments need the
   score of $p_t$ at every time. The tanh form, the zeros and the slope $60$
   serve as checks on the implementation."
5. Before: "The proposition makes a strong claim: ... We test it on the
   mixture, whose marginal and score we have in closed form."
   After: "The heat flow was a stretch of one Gaussian. On the mixture the
   proposition has more to do: the density changes shape, from two modes to
   one, and the jagged and the smooth process must still agree on it at every
   time. ... Before looking at the output, guess which cloud's distance
   changes more from one snapshot to the next."
6. Before: "The one-step result extends to the whole process
   :cite:`Anderson.1982`."
   After: "The one-step calculation is a heuristic: it expands to first order
   and starts from an approximate kernel. :citet:`Anderson.1982` proved that
   it gives the right answer for the whole process. We state his theorem
   without proof: ..."

## Repetition

Removed or reduced:

- OU kernel (owner: SDE section). The display keeps its label
  `eq_mdl-dyn-ou-marginal` (exercises 4 and 10 cite it) but is now introduced
  as "the Gaussian kernel :eqref:`eq_mdl-sde-ou-kernel`, which we write as".
  The sentence re-describing mean and variance is cut.
- "Pathwise and Ensemble Views" is cut to its one new idea, the time-marginal
  (FPK-2). The heading stays.
- The score (owner: `sec_mdl-fisher-divergence`): the pointer is one clause;
  the definition and the geometry paragraph stay because the plot and the
  later arguments use them.
- "The only unknown is the score": stated in the introduction, at the reverse
  SDE and in the Summary; the copies after the likelihood formula and in the
  closing recap are removed.
- "The drift carries mass ..., the diffusion pushes it down the gradient"
  after the current; "Each particle follows the drift, corrected by a push
  ..." after the ODE; the second description of jagged against smooth paths:
  each was the second statement of a sentence a page earlier.
- The "We can now ..." recap (FPK-15). Its last sentences and the hand-off to
  score matching stay as the end of the last subsection.
- "The smallest case" opens one worked example now instead of four (FPK-16).

Kept because this file owns them: the Fokker--Planck derivation, the vector
calculus preliminaries, diffusion as transport, the probability-flow ODE, the
reverse SDE and the $\lambda$ family (displayed here; the later sections
should cite `eq_mdl-dyn-pf-ode`, `eq_mdl-dyn-reverse-sde`).

## Ranking; propositions demoted; labels

- Demoted: "Proposition (superposition)" with its one-line proof is now two
  sentences. It had no label.
- Kept as Proposition blocks: Gaussian solutions of the OU equation,
  Fokker--Planck is a continuity equation, the deterministic twin, one
  family.
- Ranking added: the introduction names the three results that carry the
  section and what can be skipped; the twin is announced as "the first of the
  two results that generative models take from this section"; the family
  proof says which step needs uniqueness.
- Labels added: none. Labels removed: none.
- New references, all resolve: `eq_mdl-sde-ou-kernel` (mdl-sdes.md),
  `sec_diffusion-samplers` (chapter_diffusion-models/samplers.md).

## Chapter 17 citations of this file (checked with git grep)

`sec_mdl-time-reversal` (samplers.md, gaussian-paths.md: "a family of reverse
processes with the same marginals", "the more noise a sampler injects, the
larger the score correction"), `fig_mdl-dyn-lambda-family`,
`sec_mdl-probability-flow-ode` ("deterministic twin"),
`sec_mdl-fokker-planck-probability-flow` (design-space.md, the exercise by its
title "Prior mismatch, ODE versus SDE"), and the mode-merging exercise three
times (for "a single mode exactly when $\mu^2\le v$"). All passages, the
exercise title and the sentence "The more noise a sampler injects, the larger
its score correction" are intact. The mode-merging criterion is now also a
sentence in the body: "The mixture has two modes exactly when $\mu^2>v_0$;
for $\mu^2\le v_0$ they merge into one."

## Cells to cut or redesign on the development machine

- `#fokker-planck-reverse-sde` (FPK-13; SMD-6 in the score-matching review):
  re-capture (stored draw 0.487, KS 0.0148 is atypical); redesign with
  unequal weights so that "mass on the left" tests something, and drop or
  relabel the "(target 0.5)" line.
- `#fokker-planck-cloud-marginals` (FPK-2): repeats `#sdes-ou-cloud`; keep
  only the histogram, or overlay two runs.
- `#fokker-planck-ou-residual` (FPK-4, FPK-5): confirms a proved fact; if
  kept, print the residual at two grid spacings so that the factor four is on
  the page.
- `#fokker-planck-diffusion-identity`: the first print confirms a chain-rule
  identity by finite differences and can go; the flow-against-Brownian half
  teaches.
- `#mdl-fokker-planck-probability-flow-reverse-time-sampling-1` (FPK-14): let
  `em_reverse_width` take a grid and print the non-uniform result (87%), or
  make it an exercise.

## Other deferred findings

- FPK-3: the scalar field $h$ (step size elsewhere) is not renamed, because
  exercise 1 and the slides use it; one clause now says that $\Delta$ in
  front of a function is the Laplacian. $q$ (three meanings) and $\lambda$
  (eigenvalue in the chapter table) unchanged.
- Structural 3: pointer from the calculus chapter to
  `sec_mdl-dyn-vector-calculus` (other file).
- Structural 5: only `sec_diffusion-samplers` was added. "How likelihoods of
  diffusion models are reported" still has no main-text label; I did not find
  the chapter 17 passage to point to.
- Exercises 8, 11 and 12 have no `:exlabel:`; the body refers to exercise 12
  by its title.
- Framework import tabs on NumPy-only code; second import cell for `erf`.
- Suggested new exercise from FPK-14 (place the steps to minimize the bias).

## Claims added or changed

- Uniqueness of the Fokker--Planck solution among probability densities for
  Lipschitz drift and $g>0$ (cited, from memory; see above).
- The mixture marginal by conditioning on the component.
- Residual falls by four per halving of the grid (recomputed).
- 1,500 non-uniform steps remove nearly nine tenths of the width bias, 1,500
  uniform steps six tenths (recomputed).
- The width excess of the sampling run is the Euler--Maruyama bias.
- The mass split is one half by symmetry and tests sampling noise only.
- "The mixture has two modes exactly when $\mu^2>v_0$" (standard; equal
  weights and variances, modes separate when the half-distance exceeds the
  standard deviation; agrees with the slope $(\mu^2-v_0)/v_0^2$ at $0$).
- The backward heat equation cannot be solved from an arbitrary final
  density; the reverse processes visit a given forward family (FPK-12).
- A grid with 100 cells per axis has $100^d$ unknowns (from exercise 8d).
- $\mathbf b'$ is Lipschitz when $\mathbf f$ and $\mathbf u_t$ are.
- Sampling error of the cloud standard deviation "of order $0.01$"
  ($2/\sqrt{2\cdot20000}=0.01$; stored values 1.984 and 2.005).

Numbers quoted that no cell prints and that I kept: 87% / 60% (as "nine
tenths", "six tenths"), the factor four, the merging time $0.78$
($\tfrac12\ln4.75=0.779$), the slope $-7$, $0.0035$ twice.

## Slides against the body

- Slide "The Equation Is Linear: Mixtures of Solutions Solve It" still gives
  the closed-form marginal from linearity alone (the gap of FPK-4).
- Slide "Anderson: The Reverse Process Is Again a Diffusion" is consistent,
  but the body subsection is now titled "The Reverse-Time SDE".
- Slide "The Exact Score Generates the Data From Noise" shows the stored
  output with "mass on the left: 0.487 (target 0.5)" without the body's
  caveat.
- The cover kicker's section number is rewritten by the build; not checked.
