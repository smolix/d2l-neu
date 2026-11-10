# Writing pass log: Entropy, Cross-Entropy, and KL Divergence

File: `chapter_mdl-information-theory/mdl-information-theory.md`. Not split (no
markers). No code cell, cell ID, figure include or slide changed; cells and
figures keep their order. Numeric checks were done with `python3` and NumPy
(no SciPy, no PyTorch on this machine) or by hand, as noted.

## Words

Checker count (body, captions and exercises, to the slides marker): 7,387
before, 7,367 after (-20). The section is not shorter in total because the
savings were spent as the assignment asked: the new subsection "Joint and
Conditional Entropy" (about 220 words), the two corollaries of Gibbs
(uniform bound, conditioning), the general maximum-entropy argument, the
missing hypotheses and the sentence that a questioning strategy is a code.
What paid for them: the Jensen proof of the uniform bound, the Gaussian
Proposition/Proof block, the label-smoothing Proposition/Proof block, the
"We can now" closing, three of the four tellings of units against ratios,
and the logits paragraph. Display-equation delimiters 54 to 58.

## Correctness changes

| Finding | Before | Now | Verified |
|---|---|---|---|
| INF-13, INF-S1, INF-1 | ":numref:`sec_mdl-exponential-family` used the answer"; "Here we take that quantity apart" | Opening says chapter 41 "defined the KL divergence and proved Gibbs' inequality :eqref:`eq_mdl-exp_family_gibbs`"; "There the inequality was a tool. Here it is the subject." The Gaussian case cites `subsec_mdl-max-entropy-derivation` as the same argument "told there from the side of the exponential family" | Read the finished chapter 41 section; labels exist. |
| INF-15 | symmetric-noise bullet stated the floor and "best accuracy is $1 - \rho = 90\%$" with no hypothesis | "the clean class is determined by the input"; accuracy "As long as $1 - \rho > \rho/(k-1)$" | $H_b(0.1) + 0.1\ln 9 = 0.3251 + 0.2197 = 0.5448$ (python3). |
| INF-18 | "The raw gradient decays like $1/\tau^2$" (not printed) | "the unscaled gradient is the printed row divided by $\tau^2$, a hundred times smaller at $\tau = 10$" | True by construction of the printed column; recomputed the table with NumPy, matches the stored PyTorch output. |
| INF-7, INF-S2 | "This average log-ratio is the KL divergence, defined below." | "Differential Entropy" now follows "Cross-Entropy"; the ratio is called the KL divergence, already defined | Prose-only move, no cell or figure between old and new position; checker passes. |
| INF-9 | "a code built for $Q$ wastes $0.57$ bits per symbol" (no code yet) | "a questioner who believes $Q$ asks $0.57$ more questions per outcome than necessary, on average over long runs" | $0.3961/\ln 2 = 0.571$. |
| Exercise 8(c) (review, Exercises) | asked what the formula says about a loss of 4.1 | adds "Given what you found in (b), is the formula still trustworthy at that scale?" | Simulation (NumPy, 400,000 draws): $s = 2$ gives 3.77 against the formula's 4.10. |

No finding I disagreed with on mathematics. One disagreement of fact: INF-S4
says no main-text section teaches distillation, so the derivation has no
reader. `chapter_lm-reasoning/distillation.md` (`sec_lm_distillation`, a
skeleton) and two other LM briefs now cite `subsec_mdl-distillation` for the
$\tau^2$ derivation and say "do not re-teach". I therefore kept the
derivation and cited `sec_lm_distillation`.

## Six before/after pairs

1. Opening. Before: "Here we take that quantity apart. Two questions remain
   open at the end." After: "There the inequality was a tool. Here it is the
   subject."
2. Self-information cell. Before: "The output reproduces the four numbers of
   the text. Naming the order of a shuffled deck carries as much information
   as $226$ fair coin flips." After: "Naming the order of a shuffled deck
   carries $156$ nats, as much information as $226$ fair coin flips." (The
   text now gives only the three values done by hand.)
3. Monte Carlo cell. Before: "Does the Monte Carlo estimate agree with the
   formula for the unit mean shift?" After: "We run both on the unit mean
   shift. How far from $\tfrac12$ should an average of $2 \cdot 10^5$
   samples land?"
4. Irreducible-loss cell. Before: "The next cell evaluates the floor for the
   noisy case and the loss of three predictors: the uniform one, a confident
   one that puts $0.99$ on the clean class, and the Bayes predictor $q = p$."
   After: "Before running the next cell, guess the loss of a predictor that
   puts $0.99$ on the clean class in the noisy case. It is right nine times
   in ten, and nearly certain each time."
5. Proof gloss. Before: "The step that matters is the second line." After:
   "Everything happens in the second line."
6. Ranking. Before: "The floor is the average entropy of the label given the
   input, called the *conditional entropy*." After: "The floor is the
   conditional entropy of the label given the input
   (:numref:`subsec_mdl-joint-conditional-entropy`). ... Of the results of
   this section, it is the one to have in mind when reading a loss curve."

## Repetition

Removed or reduced:

- Gibbs as a "new result": the cross-entropy bound is a clause ("by Gibbs'
  inequality, $\textrm{CE}(P,Q) \ge H(P)$"); proper scoring is a clause
  ("smallest only at $\mathbf{q} = \mathbf{p}$"; "Gibbs' inequality
  restated" deleted); label smoothing is a clause ("It is smallest at
  $\mathbf{q} = \mathbf{p}^\epsilon$"); distillation is a clause ("zero only
  when the student matches the softened teacher"). Lines naming Gibbs before the exercises: 14
  before, 12 after, most of them now in the proposition, its proof and the
  two corollaries.
- Units against ratios: told once, in "Differential Entropy" (the uniform
  example, the Jacobian shift, invariance of KL, the constant shift of the
  cross-entropy). The paragraph after the decomposition cell and the
  subsection's closing restatement are gone; the Summary keeps one clause.
- Maximum entropy: one mechanism. The uniform bound is one line after Gibbs
  ($D_{\textrm{KL}}(P\|U) = \log k - H(P)$); the general statement (constant
  cross-entropy on the constraint set) is given once in "Maximum Entropy and
  the Gaussian"; the Gaussian is three lines and cites chapter 41.
- Worked numbers: $0.545$ is derived once (third bullet), printed by the
  cell, and otherwise appears only in exercise 7(d). $\ln 91$ once. $\ln 10
  = 2.30$ in the opening and one back-reference. The four card values are no
  longer both stated and printed.
- Maximum likelihood as KL minimization: one sentence with a reference to
  `subsec_mdl-nll-crossentropy`.
- Logits against probabilities (INF-14): a clause with
  `subsec_softmax-implementation-revisited`; the double-softmax warning is
  left to the main text.
- The closing "We can now ..." paragraph; the opening's preview of it.

Kept because this file owns it: entropy from surprisal, the questions game,
KL with its questions reading, Gibbs' inequality with the Jensen proof, the
decomposition, the irreducible-loss proposition, differential entropy.

## Ranking; propositions; labels

- Proposition blocks kept: Gibbs' inequality; the irreducible classification
  loss.
- Demoted to text: uniform maximum (statement before, one-line proof after
  Gibbs); Gaussian maximum entropy (the labelled display
  `eq_mdl-gaussian-max-entropy` stays, with the three-line argument);
  optimal smoothed prediction (INF-17).
- The opening names the three results that carry the section and the two
  subsections that can be skipped.
- **Labels added:** `subsec_mdl-joint-conditional-entropy` (new subsection
  after "Entropy") and `eq_mdl-entropy-chain-rule` (the chain rule
  $H(X,Y) = H(X) + H(Y \mid X)$). No label removed.
- References added, all resolving: `subsec_mdl-max-entropy-derivation`,
  `eq_mdl-exp_family_gibbs`, `sec_mdl-mutual-information`,
  `sec_mdl-mi-estimation`, `subsec_softmax-implementation-revisited`,
  `sec_lm_distillation`.

## Order

"Differential Entropy" (with its label) moved from before "KL Divergence and
Cross-Entropy" to after "Cross-Entropy", directly before "KL Divergence
Between Gaussians". It holds no cell and no figure, so cell and figure order
are unchanged. Jensen's inequality moved from the uniform-bound subsection to
"Gibbs' Inequality", its only use.

## Split markers

None (not to be split).

## Cells to cut or redesign on the development machine

- `information-theory-self-information` (INF-S5, INF-3): keep only the
  $\ln 52!$ line.
- `information-theory-definition` (INF-S5): confirms 1.75 bits; merge with
  the next definition cell.
- `information-theory-definition-2`, first print line (INF-S5): hand
  arithmetic; the $Q'$ line teaches.
- `information-theory-kl-categorical` (INF-S5): confirms the decomposition.
- `information-theory-formal-definition-2` and the four-tab
  `information-theory-cross-entropy-as-an-objective-function-of-multi-class-classification`
  (INF-S5, INF-14): print one number twice; prose reduced to two sentences.
- `information-theory-example-3` (not in the review): evaluates a closed
  form; now introduced by a question, could print the core and tail
  contributions that the prose quotes.
- `information-theory-distillation` (INF-18): print the unscaled gradient
  next to the scaled one.

## Other deferred findings

- INF-2 (notation table): kept whole; `chapter_notation/index.md` and the
  divergences section point to it as the chapter's table. Only the first row
  gained "(surprisal)". The caption's "last six rows belong to later
  sections" is now slightly off: conditional entropy is defined here.
- INF-S4 (shrink distillation to a paragraph): not done, see above; prose
  trimmed only.
- Exercise order, and a new exercise on the chain rule: not done (exercises
  are otherwise left alone).
- `mdl-mutual-information.md` still defines joint and conditional entropy in
  full with its own labels (`eq_mdl-joint_ent_def`, `eq_mdl-cond_ent_def`)
  and proves the chain rule and "conditioning cannot increase entropy" as
  propositions; it should now cite `subsec_mdl-joint-conditional-entropy`
  and `eq_mdl-entropy-chain-rule`. It also says the Gaussian entropy is
  "computed inside the maximum-entropy proof" of
  `subsec_mdl-gaussian-max-entropy`; there is no Proof block any more, the
  value is computed in the text of that subsection.
- `softmax-regression.md` says this section gives "the interpretation of
  entropy as the length of the shortest code"; that is the coding section
  (chapter review, stale reference). Main text not edited.

## Claims added or changed

- $-\log p$ "The main text calls it the *surprisal*" (grep: 8 uses in
  `softmax-regression.md`, 6 in `language-model.md`).
- A questioning strategy is a binary code; codewords $1, 01, 001, 000$ for
  the dyadic source (by hand).
- Joint entropy, conditional entropy as $\sum_x p(x) H(Y \mid X = x)$, chain
  rule and its iterated form (standard; one-line proof given).
- Example: $Y$ = answer to "Is it $a$?"; $H_2(Y) = 1$, the remainder
  $(\tfrac12, \tfrac14, \tfrac14)$ has 1.5 bits, $H_2(X \mid Y) = 0.75$,
  $1.75 = 1 + 0.75$ (by hand).
- $D_{\textrm{KL}}(P\|U) = \log k - H(P)$ (by hand).
- $H(Y) - H(Y \mid X) = D_{\textrm{KL}}(p_{X,Y}\|p_X p_Y) \ge 0$, equality
  only under independence; holds on average, not for each $x$ (by hand).
- General maximum-entropy argument: if $\textrm{CE}(P, r)$ is the same finite
  value for all $P$ in a set containing $r$, then $H(P) = H(r) -
  D_{\textrm{KL}}(P\|r)$ (by hand; it is the old Gaussian proof with $r$ for
  $\varphi$). The condition holds when $-\log r$ is, up to a constant, linear
  in the constrained statistics.
- Gaussian panels: core and tail contributions $+0.424$, $-0.106$ (a) and
  $-0.240$, $+1.046$ (b), crossing at 1.36, ratio 2.5: recomputed by
  numerical integration (NumPy); the text now reads panel (a) first.
- Confident predictor: "its confidence costs more on the tenth of the labels
  that are flipped than it saves on the rest": saves $0.9(0.1054 - 0.0101) =
  0.086$, costs $0.1 \ln 10 = 0.230$, net $0.144 = 0.689 - 0.545$.
- Overconfident smoothed prediction "which one-hot targets would score as
  nearly perfect": one-hot loss at logit gap 10 is $\ln(1 + 9e^{-10}) \approx
  4 \cdot 10^{-4}$.
- Sample averages of a log-ratio as the way KL is estimated without a closed
  form, with a pointer to `sec_mdl-mi-estimation` (INF-11).
- Exercise 4 reworded (INF-S1): compares the bounds $1 - Q(\textrm{supp}\,P)$
  and $-\log Q(\textrm{supp}\,P)$ ($-\log t \ge 1 - t$, so Jensen's is
  larger), asks for $P \ne Q$ with Jensen's bound tight (any $q = c\,p$ on
  the support, $c < 1$) and whether the tangent bound can be tight for $P \ne
  Q$ (no). Exercise 5 now cites chapter 41 for the statement and asks the
  reader to check the condition of this section's argument; parts (b), (c)
  unchanged.
- Numbers quoted in prose that no cell prints and that I kept, recomputed:
  0.571 and 0.527 bits; 1.294 and 0.898 (printed later); 0.325 + 0.220;
  standard error 0.0022; the four core/tail values; $(2\pi e)^{-1/2} = 0.242$;
  68 digits of $52!$.

## Slides now disagreeing with the body

- "On $k$ Outcomes, Entropy Peaks at $\log k$": shows the Jensen proof of the
  uniform bound; the body proves it from Gibbs.
- "With Mean and Variance Fixed, the Gaussian Has the Largest Entropy": a
  two-step proof in the old form; consistent in content, but the body now
  states the general argument first.
- Slide order: differential entropy comes before KL in the deck, after
  cross-entropy in the body.
- "Surprise Is $-\log p$": lists all four card values; the body leaves the
  fourth to the cell. No slide uses the word surprisal or mentions joint and
  conditional entropy.
- Distillation slide: "The gradient ... shrinks like $1/\tau^2$" stands as a
  derived statement; the body no longer says the table shows it.
