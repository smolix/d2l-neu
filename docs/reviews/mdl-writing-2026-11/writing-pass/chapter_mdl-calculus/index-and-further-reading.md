# Writing pass: chapter 39 introduction and further reading

Files: `chapter_mdl-calculus/index.md`, `chapter_mdl-calculus/further-reading.md`.
No code cells, no slides sections, no labels or toc lines touched. Both pass
`check_mdl_prose.py` ("OK: only prose changed", no em-dash warning) and
`tools/lint_source.py`.

## Words

| File | `wc -w` before | after | checker before | after |
|---|---|---|---|---|
| `index.md`, whole file | 958 | 957 | 1066 | 1076 |
| `index.md`, introduction only (title to toc) | 405 | 404 | | |
| `further-reading.md` | 1417 | 1414 | 1461 | 1467 |

The checker counts each piece of a label name as a word
(`sec_mdl-single_variable_calculus` is several), so its small increases are
the added `:numref:` references (five in the introduction, four in further
reading), not prose.

## Introduction (`index.md`)

Kept verbatim: the first paragraph (both questions, and the one "hundred
million" sentence, which is now the only probing argument the introduction
makes), "Calculus answers these questions, although it was created for
others", "Besides the loss value, it is all an optimizer knows", and the scope
sentence. The "Resources and Further Reading" list is unchanged.

- **IDX-1.** The five "asks how" sentences are gone. In their place: one
  paragraph that says what the chapter is for and ranks three results, and
  one route paragraph in three movements. The route names "the first part",
  "the second", and "last", never a section count, and describes the chain
  rule as going "from a sum over the paths of a graph to a product of layer
  Jacobians ... and to an algorithm", with references only on the two
  sections whose labels are stable. "The integral and expectations come
  last" has one reference; when the expectations half moves to chapter 41
  that clause needs a second label or a cut.
- **IDX-2.** Ownership as agreed: sum over paths (multivariable, not named by
  number), product of Jacobians (matrix calculus), algorithm (automatic
  differentiation). The sentence "backpropagation is the chain rule evaluated
  in its cheapest order" moved from the history paragraph into the ranking
  ("cheapest when multiplied from the loss backward, which is
  backpropagation"), so it is said once.
- **IDX-3.** "training through random sampling needs the derivative of that
  area" now names a use with a main-text label: "a policy-gradient method
  (:numref:`sec_policygradient`) differentiates one with respect to the
  distribution it samples from". I used policy gradients and not the VAE of
  the review's draft because the book has a labelled section for the former.
- **IDX-4.** "checks each result in code" is dropped.
- **Appendix as additional reading.** "The Basics part stated the rules of
  differentiation (`sec_calculus`) and used automatic differentiation as a
  tool (`sec_autograd`); the main text needs no more. This chapter is
  additional reading, for the reader who wants the proofs."
- **"We assume school calculus"** became "The school calculus that opens
  several sections can be skimmed", which is what the revised section
  openings now say themselves.
- **Dropped:** "One function recurs throughout, the log-sum-exp". The review
  found the promise kept unevenly, each of the four derivative sections
  introduces it as its own running example, and the integral section does not
  use it. Cut for length; easy to restore as one sentence if wanted.

Main-text references added (all resolve): `sec_calculus`, `sec_autograd`,
`sec_gd` (it quotes $0<\eta<2/L$ and defers the proof to the appendix),
`sec_backprop`, `sec_policygradient`.

Judgments that may be too strong: "the main text needs no more"; the choice
of the three results (descent lemma, Jacobian product multiplied from the
loss end, the two estimators of the gradient of an expectation), which leaves
out the second-derivative test and the differential method.

## Further reading: correctness

- **FR-4 (quasi-Monte Carlo).** Was: the bound "beats $N^{-1/2}$ once $N$ is
  large compared with the dimension". Now: "As a guarantee this is weak:
  ignoring constants, the bound falls below $N^{-1/2}$ only beyond several
  thousand points for $d=2$ and $10^{15}$ for $d=5$. In practice quasi-Monte
  Carlo often wins in high dimension, which is usually explained by
  integrands that depend mostly on a few coordinates at a time." Verified by
  solving $(\ln N)^d=N^{1/2}$: $N\approx5\times10^3$ ($d=2$), $10^{15.5}$
  ($d=5$), $10^{39}$ ($d=10$). The effective-dimension explanation is stated
  as the usual one and is not attributed to a page of Owen. Owen's chapter
  numbers 15 to 17 (Quasi-Monte Carlo, Lattice rules, Randomized quasi-Monte
  Carlo) were checked against the book's web page.
- **FR-2 (subgradient rate).** The square-summable step sizes now carry only
  the convergence statement ("make the best value found converge to the
  minimum"); the rate $RG/\sqrt T$ is attached to the constant step
  $R/(G\sqrt T)$. Verified from
  $\min_t f(x_t)-f^*\le(R^2+G^2\sum\eta_t^2)/(2\sum\eta_t)$. The lower bound
  now reads "by more than a constant factor" (Nesterov's bound is
  $GR/(2(2+\sqrt{T+1}))$, not $RG/\sqrt T$ exactly).
- **FR-3.** "Frameworks therefore offer only ..." became "Frameworks do not
  attempt it: they offer forward mode, reverse mode, and their compositions,
  and leave the choice to the user."
- **FR-5.** Both bare uses of "the section" in the integration part now name
  `sec_mdl-integral_calculus`.
- **FR-6.** The parenthesis "(definable functions)" is removed (the review's
  second option); a gloss cost twenty words.
- **FR-7.** Verified against the arXiv abstract of Bertoin et al.: variations
  "around half of the time in 32 bits", the effect "disappears with double
  precision" and is "systematic at 16 bits"; SGD is affected, batch
  normalization and Adam buffer it. Since the single-variable section now
  reports these numbers in its body, this file keeps one sentence that points
  there.
- **Subdifferential and the chain rule.** "it does not obey the chain rule for
  nonconvex compositions" became "its sum and chain rules hold only as
  inclusions once the pieces are not all convex" (for
  $\mathrm{ReLU}(x)-\mathrm{ReLU}(-x)$ the sum of the Clarke subdifferentials
  at $0$ is $[0,2]$, the subdifferential of the sum is $\{1\}$).
- **Forward gradients.** "multiplies the gradient noise by about $n$" became
  "the error is about $\sqrt n$ times the length of the gradient"
  (root of $(n+1)\|\nabla\ell\|^2$). The exercise in the automatic
  differentiation section that derives both facts is now cited, as the
  Taylor-mode topic cites its exercise.
- **Softened or dropped because not verified:**
  - "nesting the marks gives a recursive schedule" is now claimed for JAX
    only (its documentation shows the nested use); not checked for
    `torch.utils.checkpoint`.
  - Giles's rules "derived by the differential method of the section":
    dropped.
  - Mohamed et al. "recommend which to use for which model": dropped; the
    sentence keeps "compare all three families in one framework and state the
    conditions each needs".
  - "for a scalar loss reverse mode is already optimal" (the review's reason
    in FR-3) was not added.
- Not re-verified, left as the reviewer checked them: Rudin's Theorem 5.19,
  the Kronecker identity, the checkpointing figures, the Taylor-mode costs,
  the statement of Bolte and Pauwels's convergence result (from memory of the
  paper, conditions kept as "the usual conditions").

## Further reading: opening, ranking, repetition

- **FR-1.** The template paragraph is replaced by an opening in the chapter's
  terms (hypotheses that practice does not meet: kinks, a tape that does not
  fit, a grid in a thousand dimensions). It ranks: "Two bear on everyday
  training and are worth reading first: conservative fields ... and recursive
  checkpointing. The rest is for lookup." It keeps the pointer to the
  resource list and "This section can be skipped without loss."
- **Against the resource list.** The Matrix Cookbook sentence is removed here
  (the introduction lists it). Mohamed et al. remains in both places with
  different jobs: the list recommends it for the two estimators the chapter
  teaches, this file cites it for the third family.
- **Survey facts owned here** and kept in full: NP-completeness of optimal
  accumulation, recursive checkpointing, the subgradient step sizes and rate,
  forward gradients. The sections still carry a one-clause version of the
  first three (`mdl-matrix-calculus.md` "the problem is NP-complete",
  `mdl-matrix-calculus-autodiff.md` "Recursive schedules ... reach
  $O(\log T)$ memory", `mdl-single-variable-calculus.md` "converges by
  shrinking its step sizes"). Those clauses have no reference to this file,
  which is consistent with the rule that nothing depends on it. The last one
  is now slightly loose against this file (the rate uses a constant step).
- The opening of the Taylor-mode topic lost its clause on forward over
  reverse (taught in the section).

## Before and after

1. Before: ":numref:`sec_mdl-single_variable_calculus` asks how a function of
   one variable responds to a small change, how large a step its local model
   allows, and what a slope means at a corner."
   After: "The first part builds the linear and the quadratic model of a
   function, in one variable (...) and in many (...), which bound a safe step
   and tell a minimum from a saddle."
2. Before: ":numref:`sec_mdl-matrix-calculus-autodiff` asks how a program
   computes these derivatives."
   After: "The second follows the chain rule from a sum over the paths of a
   graph to a product of layer Jacobians (...) and to an algorithm (...)."
3. Before: "This chapter proves the rules of differentiation that the Basics
   part used, shows why automatic differentiation is correct and cheap, and
   checks each result in code."
   After: "This chapter is additional reading, for the reader who wants the
   proofs. Three results matter most: ..."
4. Before: "Each topic states the problem it addresses and what it changes
   relative to the chapter."
   After: "The topics below say what is known when a hypothesis fails."
5. Before: "... their error is $O(N^{-1}(\log N)^d)$, which beats $N^{-1/2}$
   once $N$ is large compared with the dimension."
   After: "As a guarantee this is weak: ignoring constants, the bound falls
   below $N^{-1/2}$ only beyond several thousand points for $d=2$ and
   $10^{15}$ for $d=5$."
6. Before: "It removes the tape and multiplies the gradient noise by about
   $n$, which is acceptable for small models and prohibitive for large ones."
   After: "The tape is gone, but the error is about $\sqrt n$ times the
   length of the gradient, which is acceptable for small models and
   prohibitive for large ones."

## Deferred

- Coverage gap named by the review (normalizing flows, importance sampling):
  not added, the file may not grow; the integral section already points to
  `sec_mdl-bayesian-computation` for importance sampling.
- `sec_backprop` still sends readers to the automatic differentiation section
  for the layer rules (main-text file, not edited).
- After the two splits: give the route's "sum over the paths of a graph" and
  "expectations" their new labels, and repoint this file's references for
  Monte Carlo and the estimators if that half moves to chapter 41.
