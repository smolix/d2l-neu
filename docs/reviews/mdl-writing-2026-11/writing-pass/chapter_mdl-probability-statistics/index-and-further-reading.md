# Writing pass: chapter introduction and Further Reading (chapter 41)

Files: `chapter_mdl-probability-statistics/index.md` and
`chapter_mdl-probability-statistics/further-reading.md`. No code cells, no
labels added or removed, toc block and Resources list untouched, no split
markers (neither file is split).

## Words

| File | Checker count (before, after) | Rendered words (a reference counted as one word) |
|---|---|---|
| `index.md`, whole file | 898, 904 | 851, 851 |
| `index.md`, introduction before the toc | not reported | 405, 405 |
| `further-reading.md` | 1814, 1816 | 1716, 1713 |

The checker counts the pieces of a label name as words. The introduction now
carries five references into the main text where it had one, which is the
whole of its +6; further reading gained two references. In rendered words
neither file grew.

Checker: "OK: only prose changed" for both. `tools/lint_source.py` exits 0
for both. The checker still prints "WARN: em dash in prose" for `index.md`.
It fires on the ` --- ` between title and author in the thirteen entries of
the Resources list, which the brief says to leave ("Resources lists stay");
the same format is in all six chapter introductions, so it is a part-level
decision. The new prose has no em dash.

## Introduction (`index.md`)

### What changed

- **New opening (STA-1, "Every model in this book").** The Statistics
  section keeps the shared paragraph. The introduction now opens from what
  the main text does with probability without arguing for it: it adopts two
  losses, reports an accuracy from a finite test set, and selects by
  validation score. This also supplies the main-text labels the review asked
  for (IDX-3, second structural finding).
- **The coin is kept as the chapter's thread, in two sentences,** with its
  number (IDX-4): "a fair coin does at least this well more than one time
  in eight". The sentence "A test accuracy is the same coin, flipped once
  per test question" is kept (the review lists it as the one to protect).
- **The four objects are kept, and "easy to confuse" is paid off (IDX-2):**
  the estimator's scatter over samples against the posterior's spread for
  one sample.
- **The route (IDX-1) is rewritten around the four objects,** not the table
  of contents. It names a section only where a pointer helps, uses no
  section numbers and no counts of sections, and refers to
  `sec_mdl-latent-em-elbo` by label, so it reads the same after the two
  splits (random variables into two; maximum likelihood into two). It says
  which result we regard as central (the curvature of the loss sets the
  error bar, in maximum likelihood) and that it fails for a deep network.
- **What can be skipped,** and what a reader of `sec_prob` gains: the
  opening sections repeat parts of it with proofs and for densities
  (third structural finding); Bayesian computation and naive Bayes can be
  skipped at first.
- **Division with chapter 42** stated in one sentence: entropy and KL appear
  only as tools; the theory is in `chap_mdl-information-theory`.
- Dropped: the sentence on Pascal, Fermat and Fisher (for length), and the
  nine-sentence route.

### Before and after (verbatim)

1. Before: "Every model in this book was judged by a number computed on
   data, such as a test accuracy or a validation loss."
   After: "The main text uses probability constantly and seldom stops to
   argue about it."
2. Before: "The obvious estimate of its bias is $9/13\approx0.69$, but a
   fair coin does this often enough that 13 flips cannot rule it out."
   After: "We would estimate its bias as $9/13\approx0.69$, yet a fair coin
   does at least this well more than one time in eight."
3. Before: "Answering them needs four objects that are easy to confuse:"
   After: "The middle two are easily confused: the scatter of an estimate
   over the samples we did not draw is not the spread of a posterior given
   the sample we have, although the two often produce similar error bars."
4. Before: ":numref:`sec_mdl-distributions` asks which few distributions
   model most labels, and :numref:`sec_mdl-exponential-family` why almost
   all of them share one form that makes their losses convex."
   After: "Distributions come first, ending with the exponential family
   (:numref:`sec_mdl-exponential-family`), whose common form is why linear,
   logistic and softmax regression train with the gradient "prediction
   minus observation"."
5. Before: ":numref:`sec_mdl-maximum_likelihood` asks why maximizing
   likelihood produces the losses of the main text and how precisely it
   pins down the parameters."
   After: "... :numref:`sec_mdl-maximum_likelihood` derives the losses of
   the main text from one principle and shows that the curvature of a loss
   sets the error bar of the fitted parameters. We regard this as the
   chapter's central result, although it fails for a deep network."
6. Before: "We assume the probability of :numref:`sec_prob` and calculus."
   After: "The opening sections repeat parts of :numref:`sec_prob`, now
   with proofs and for densities, and a reader who knows them can skim."

### Claims added or changed, and how each was checked

- "A fair coin does at least this well more than one time in eight":
  $P(\ge 9 \text{ of } 13)=(715+286+78+13+1)/8192=1093/8192=0.133>0.125$.
  The distributions section computes the same $1093/8192$ (per IDX-4).
- Main-text references, all resolved with `git grep`: squared error
  `sec_linear_regression`, cross-entropy `sec_softmax`, model selection by
  validation score `sec_generalization_basics` (its "Model Selection"
  section), test-set reuse and uniform convergence
  `chap_classification_generalization`.
- "Linear, logistic and softmax regression train with the gradient
  prediction minus observation": the opening and Summary of the
  exponential-family section (canonical link; all three are canonical).
- "It fails for a deep network": the maximum-likelihood Summary ("These
  guarantees fail for a misspecified or non-identifiable model, and a deep
  network is both").
- "Hoeffding's inequality ... valid at every sample size": the
  concentration Summary.
- "Bayesian computation and naive Bayes ... no other section needs them":
  `git grep` finds no reference to `sec_mdl-bayes*` or `sec_mdl-naive_bayes`
  from the concentration, generalization-bounds or (for the latter) any
  other section body of the chapter. Naive Bayes uses the Beta posterior
  (exponential family) and the bootstrap (statistics), not Bayesian
  computation.
- "The two often produce similar error bars": the standard large-sample
  agreement of posterior and sampling distribution; stated with "often".
- "Entropy and the KL divergence appear only as tools": they are used for
  maximum entropy (exponential family), the entropy floor (maximum
  likelihood) and the ELBO gap; none of these sections develops their
  theory after the pass.

### Left open

- IDX fourth structural finding (the measure-zero exercise against "Measure
  theory ... left out"): the scope sentence is kept; the exercise is in
  another file.
- Resources list: the two Stat 110 entries could be one (review); left, as
  the brief says the list stays.
- Judgment that may be too strong: "We regard this as the chapter's central
  result". The maximum-likelihood section calls asymptotic normality "the
  one to remember" for itself; promoting it to the chapter is mine.

## Further Reading (`further-reading.md`)

### What changed

- **Opening (FR-1, FR-S1, FR-S4).** The template paragraph and the list of
  topics are gone. The new opening says what separates this section from
  the Resources list in the introduction (books and courses for what the
  chapter teaches, against topics it leaves out), and ranks: conformal
  prediction and confidence sequences are of immediate practical use; the
  VC dimension is mainly of theoretical interest; heavy-tailed gradient
  noise is disputed.
- **FR-2.** Billingsley: the unspecified "results hold without the
  regularity assumptions" became one named result (an expectation is one
  integral against the distribution; the chapter's sum and integral are two
  cases).
- **FR-3.** "the stable law with $\alpha=1$" is now "the symmetric stable
  law with $\alpha=1$". Checked: skewed 1-stable laws exist; the Cauchy is
  the symmetric one.
- **FR-4.** McFadden "stated this property" is now "discussed this
  property, Luce's *independence of irrelevant alternatives*". No citation
  for Luce: `d2l.bib` has no entry, and I did not edit the bibliography.
- **FR-5.** Bootstrap entry: a reason to care, "An accuracy on thousands of
  test examples does not need them; a small evaluation set or a ratio of
  two error rates does." Paid for by cutting "and when each is worth its
  extra computation".
- **FR-6.** Daxberger et al.: "show that these give calibrated predictive
  uncertainty ... at little extra cost" is now "report that these are
  competitive with more expensive approximations at a small fraction of
  their cost".
- **FR-7.** The random-label experiment now carries
  `:numref:`subsec_random_labels``.
- **FR-8 (half).** VC entry: "(halfspaces in $\mathbb{R}^m$ have $d=m+1$)".
- Stability: "For smooth convex losses" is now "For convex losses that are
  smooth and Lipschitz" (the Hardt, Recht and Singer bound needs both).
- Cut for length: "and several other probabilistic programming systems"
  (after Stan), "so the experimenter may stop at any time" (implied by
  "hold simultaneously at every sample size").

### Before and after (verbatim)

1. Before: "Each topic states the problem it addresses and what it changes
   relative to the chapter."
   After: "Two are of immediate practical use, conformal prediction and
   confidence sequences, which repair the intervals and tests of
   :numref:`sec_mdl-statistics` under the conditions of real experiments."
2. Before: "Resources for the material the chapter does teach are listed in
   its introduction".
   After: "The introduction of this chapter ... lists books and courses for
   what the chapter teaches. This section is about what it leaves out: each
   entry, in the chapter's order, starts from a limit of something the
   chapter did."
3. Before: "... builds probability on this footing, in which the chapter's
   results hold without the regularity assumptions it states along the
   way."
   After: "... builds probability on this footing, in which an expectation
   is one integral against the distribution and the sum and the integral of
   the chapter are two cases of it."
4. Before: "... show that these give calibrated predictive uncertainty for
   trained networks at little extra cost."
   After: "... report that these are competitive with more expensive
   approximations at a small fraction of their cost."
5. Before: "... derived the logit from Gumbel noise and stated this
   property, the *independence of irrelevant alternatives*, as its main
   restriction."
   After: "... derived the logit from Gumbel noise and discussed this
   property, Luce's *independence of irrelevant alternatives*, as its main
   restriction."
6. Before: "... obtained the first nonvacuous bounds for stochastic
   networks".
   After: "... obtained nonvacuous bounds for stochastic networks".

### Claims about papers: softened, dropped, or not verified

No paper was opened in this pass; everything below is from memory.

- Softened: Daxberger et al. 2021 (FR-6, above).
- Dropped: "the first" nonvacuous bounds (Dziugaite and Roy 2017). The
  paper's own claim is nonvacuous bounds for networks with millions of
  parameters on tens of thousands of examples; priority is not something I
  could check.
- Not added: the PAC-Bayes number proposed in FR-8 (a bound of roughly 0.16
  to 0.2). The review itself says to check the figure before quoting.
- Kept unchanged and unverified against the sources: Hall 1992 (coverage
  error $1/\sqrt n$ to $1/n$ for one-sided studentized and BCa intervals);
  Howard et al. 2021 and Johari et al. 2017; Simsekli et al. 2019 and
  Martin and Mahoney 2021; Hoffman and Gelman 2014 (now "the default
  sampler of Stan" only); Welling and Teh 2011; McAllester 1999; Bousquet
  and Elisseeff 2002; Zadrozny and Elkan 2002; Guo et al. 2017; Nigam et
  al. 2000; that McFadden 1974 discusses the restriction with the red and
  blue bus. All agree with what I know of these papers.
- Added, from memory: "whether gradient noise is heavy-tailed is still
  disputed" (later work argued against the $\alpha$-stable model). The
  entry itself already hedged; the opening now says it outright. No
  citation for the dispute was added.
- Added: halfspaces in $\mathbb{R}^m$ (with offset) have VC dimension
  $m+1$, which is the $d$ of the entry's convention ("cannot realize all
  labelings of any $d+1$ points"). Standard, and consistent with the
  main-text exercise `ex_generalization-classification-linear-vc`.

### Deferred

- **FR-S1** (two lists with one name): the opening now separates them, but
  the title "Further Reading" stays; renaming is a decision for all six
  chapters and the writing guide (§12.3 fixes the title).
- **FR-S2** (no entry for maximum likelihood; natural gradient): a new
  entry would grow the file.
- **FR-S3** (deep ensembles, proper scoring rules, Ng and Jordan,
  graphical models) and **FR-9**: I drafted one sentence on proper scoring
  rules for the calibration entry and removed it to keep the length.
  `d2l.bib` has no entry for Lakshminarayanan et al. or for a scoring-rule
  reference.
- **FR-8** for PAC-Bayes and stability: still no number.
- The main text already teaches the VC dimension
  (`chap_classification_generalization`), so the entry "VC Dimension and
  Covering Numbers" overlaps it in its first half; the opening's "mainly of
  theoretical interest" is my ranking and may be too strong for the author.

## Consistency with neighbouring files

- The Statistics section still opens with "Every model in this book has
  been judged by a number computed on held-out data"; by decision it keeps
  that paragraph, and the introduction no longer shares it. The sentence "A
  test accuracy is a coin with one flip per question" is in both, on
  purpose (the coin is the thread).
- The introduction says the one-variable material repeats `sec_prob`; the
  random-variables opening says the same.
- After the two splits nothing in either file needs editing: the route
  names no section number or count, and `sec_mdl-latent-em-elbo` already
  exists as a label.
