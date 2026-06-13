# Review — chapter_linear-classification/environment-and-distribution-shift.md  (§4.7 "Environment and Distribution Shift")

**Role in the chapter:** The conceptual capstone of the linear-classification
chapter. It steps back from fitting models to ask where data comes from and what
we do with predictions: it gives the covariate/label/concept-shift taxonomy,
derives the importance-weighting corrections for covariate and label shift via
empirical-risk minimization, surveys learning settings (batch/online/bandit/
control/RL), and closes on fairness/accountability. No code; pure prose + 3
figures + 1 citation.

**Verdict:** This is the justly-admired, beautifully-written section the watch-points
flag — the prose breathes, the cat/dog covariate-shift picture and the cancer-startup
and predictive-policing anecdotes land hard, and the two reweighting derivations are
clean and honest about their assumptions (overlapping support; deterministic-label
degeneracy). It is close to best-in-class *as conceptual writing*. What keeps it from
a top-program "assignable as-is" mark is **scholarly attribution and currency**, not
clarity: the label-shift correction is the classic Saerens/Lipton-Wang-Smola method
presented with **zero citation**; the famous "tanks" story is stated as fact when it
is almost certainly apocryphal; the "operator-theoretic" covariate-shift sentence
hand-waves past the one concrete method (kernel mean matching / MMD) whose citation
is *already in the repo's `.bib`*; and there is no nod to the fact that, post-2021,
distribution shift is the central operational problem of foundation-model deployment.
The single highest-value change: **add attribution + one modern WILDS/foundation-model
forward-pointer** (DS-1, DS-2), which the task explicitly scopes in.

**Grade:** B+. Excellent, much-loved exposition; held below A by missing attribution
on a method it teaches in full, an uncaveated apocryphal anecdote, a vague/uncited
"operator-theoretic" sentence, a non-sequitur citation, and thin exercises.

**Top priorities (ranked):**
1. [P1] DS-3 — Caveat the apocryphal "tanks in the forest" story (it is repeated as
   fact; the literature treats it as a parable of unknown provenance).
2. [P1] DS-1 — Attribute the label-shift correction (Saerens et al. 2002;
   Lipton, Wang & Smola 2018 "BBSE"); it is taught in full but cited to no one.
3. [P1] DS-2 — Add the *one* sanctioned modern forward-pointer: shift is now central
   to foundation-model deployment; cite the WILDS benchmark (Koh et al. 2021).
4. [P1] DS-5 — Rebuild the four thin exercises into a graded set worthy of a top course.
5. [P2] DS-4 — Name the concrete covariate-shift method (kernel mean matching / MMD,
   `Gretton.Borgwardt.Rasch.ea.2012`, already in `.bib`) in the vague "operator-theoretic" line.
6. [P2] DS-6 — Cut or repair the ControlVAE (`Shao.Yao.Sun.ea.2020`) non-sequitur in §Control.
7. [P2] DS-7 — Explain the `min(exp(h),c)` weight-clipping (variance control under near-violated support).
8. [P2] DS-8 — Name Goodhart's law at the loan/footwear hook (one-clause lift).

---

## 1. Coverage

### Add

- **Attribution for the label-shift estimator (P1, DS-1).** The algorithm in
  §4.7.3.2 — estimate the target label prior by inverting the source confusion
  matrix, $p(\mathbf y)=\mathbf C^{-1}\mu(\hat{\mathbf y})$ — is not folklore; it is a
  specific, citable method: the confusion-matrix / EM estimator of
  **Saerens, Latinne & Decaestecker (2002)**, put on a modern consistency footing
  as **Black-Box Shift Estimation (BBSE)** by **Lipton, Wang & Smola (2018)** and
  refined by Azizzadenesheli et al. / Garg et al. (MLLS). A top-program text cannot
  present someone's named method in full with no citation. The repo's `.bib` has
  **none of these keys** — they must be added (mechanical bib edit in DS-1). This is
  the clearest scholarship gap in the file.

- **One modern forward-pointer on shift in the foundation-model era (P1, DS-2).**
  Per the task's currency lens, exactly one in-scope pointer is wanted: the section
  predates the LLM/foundation-model era, where distribution shift (deployment
  domains unlike pretraining data, temporal drift, spurious correlations) is *the*
  central operational risk. Add one or two sentences + the **WILDS** benchmark
  citation (**Koh et al. 2021**, *"WILDS: A Benchmark of in-the-Wild Distribution
  Shifts"*), which is the standard modern empirical anchor. Do **not** import a
  shift-methods survey — that is out of scope (the cover page already lists
  Quiñonero-Candela and Sugiyama as the deeper references). `Koh.2021` is **not in
  the `.bib`** and must be added.

- **Name the concrete covariate-shift detector method (P2→P1, DS-4).** Line 358–361's
  "some fancy operator-theoretic approaches that attempt to recalibrate the
  expectation operator directly using a minimum-norm or a maximum entropy principle"
  is exactly the kind of hand-wave a strong reader distrusts. The concrete method is
  **kernel mean matching (KMM)** / the **maximum mean discrepancy (MMD)** two-sample
  framing — and its canonical citation **`Gretton.Borgwardt.Rasch.ea.2012`
  ("A kernel two-sample test") is already in this repo's `.bib`** and used elsewhere.
  Naming it costs one clause and converts vagueness into a real pointer; it also
  directly grounds Exercise "implement a covariate shift detector."

- **The honest "shift detection ≈ a classifier you can't beat" idea is present and
  good** (lines 374–385) — keep. It is, in modern terms, the *classifier two-sample
  test* (López-Paz & Oquab 2017); a one-clause name-drop is optional polish, not
  required.

Out-of-scope, do **not** add (flagged so a well-meaning agent doesn't): domain-
adversarial training (DANN), test-time adaptation/TENT, conformal prediction under
shift, full importance-weighting theory. These belong to later/other parts; a
forward-pointer is the most this section should carry.

### Remove / trim

- **The ControlVAE citation in §Control (P2, DS-6).** Lines 573–577 graft "control
  theory (e.g., PID variants) has also been used to automatically tune
  hyperparameters … the diversity of generated text and the reconstruction quality
  of generated images `:cite:Shao.Yao.Sun.ea.2020`" onto the PID-controller
  paragraph. It is the file's only citation and it is a tangential 2020 VAE paper
  that has aged and does not illuminate distribution shift. Either cut it (cleanest)
  or replace with one sentence on control-as-a-learning-setting. As written it reads
  as a citation looking for a home.

- **The "More Anecdotes" bullet list (§4.7.2.4, lines 282–287)** partly duplicates
  the Nonstationary list above it. The face-detector and 1000×1000-ImageNet bullets
  are good (the latter is a clean label-shift teaser); the "deploy a US search engine
  in the UK" bullet is thin. Consider folding the strongest two into §4.7.2.3 and
  dropping the separate heading — minor (P2), not blocking.

### Reorder / restructure

- The macro-spine is sound: *taxonomy → examples → corrections → learning-problem
  zoo → ethics → summary*. Keep it.
- §4.7.4 "A Taxonomy of Learning Problems" carries **six** `###` subsections
  (batch/online/bandit/control/RL/considering-the-environment) — a long flat run that
  brushes the guide's "3–5, nested" preference. It reads fine because each is short,
  and it is a deliberate "zoo." No restructure required, but if touched, group as
  *passive (batch/online)* vs *interactive (bandit/control/RL)* with a one-line lead-in
  — optional (P2).

## 2. Teaching quality

### Structure & flow

Strong. Each `##` earns its place. The "impatient reader may skip" signpost (lines
302–304) before the technical §4.7.3 is exactly the right move and should stay. The
one true blemish is the §4.7.3 lead-in (lines 306–313): it *re-states* the
empirical-risk/risk distinction it just cross-referenced (`:numref:` +`:eqref:` to
`generalization.md`). Verified: those anchors (`subsec_empirical-risk-and-risk`,
`eq_empirical-risk-min`, `eq_true-risk`) resolve correctly to
`chapter_linear-regression/generalization.md`. The restatement is acceptable as a
courtesy recap; leave it.

### Figures

Three figures; all render with visible images + captions (confirmed on the served
page as Figures 4.7.1–4.7.3). All are *photographic/illustrative* (cat-dog photos vs
cartoons; the PopVsSoda choropleth), which is correct here — they are *evidence of a
phenomenon*, not schematics to be redrawn by a generator, so the house "draw schematics
with a committed generator" rule does **not** apply. No inline figure-drawing code
exists (no code cells at all). Caption integrity is fine (no `]` inside captions;
attributions present). PopVsSoda has `:width:400px`.

One **missing figure would unlock the hardest idea** (P2, optional, DS-9): §4.7.3.1's
covariate-shift correction is the section's conceptual peak and is currently
text-and-equations only. A single schematic — two overlapping input densities
$q(\mathbf x)$ (training) and $p(\mathbf x)$ (test) on a 1-D axis, with the
importance weight $\beta(\mathbf x)=p/q$ drawn as a curve that blows up where $q\to0$
— would make *both* the reweighting identity and the "nonzero probability / support"
caveat (lines 426–431) visual in one glance. This is a genuine schematic and would be
a committed generator SVG (`tools/gen_mdl_*`-style), not inline code. Worth it but not
blocking; specced in DS-9.

### Prose & clarity

Largely excellent — quote-worthy in places ("by introducing our model-based decisions
to the environment, we might break the model"). Specific issues:

- **The "tanks" story is told as fact and is apocryphal (P1, DS-3).** Lines 258–268:
  "A similar thing happened to the US Army when they first tried to detect tanks…
  The classifier appeared to work *perfectly*." This is one of ML's most-repeated
  cautionary tales and also one of its least-verified — Gwern's well-known
  investigation finds no solid primary source; it functions as a *parable*. A
  best-in-class text should not assert it as documented history next to the
  *verified* synthetic-roadside anecdote. The fix is small and preserves the
  pedagogy: reframe as a famous (possibly apocryphal) cautionary tale. Drafted in DS-3.

- **The "operator-theoretic … minimum-norm or maximum entropy" sentence (P1/P2, DS-4)**
  — see Coverage. Vague where it could name KMM/MMD with an in-`.bib` citation.

- **`min(exp(h_i), c)` appears unexplained (P2, DS-7).** Line 423 introduces clipped
  weights "$\beta_i = \min(\exp(h(\mathbf x_i)), c)$ for some constant $c$" with no
  reason. One clause — *clipping caps the variance of the weighted risk estimator when
  the support assumption is nearly violated and a few $\beta_i$ explode* — closes the
  loop, and it ties directly to the very next paragraph's "$p>0, q=0 \Rightarrow$
  weight $=\infty$" caveat. Drafted in DS-7.

- **Goodhart's law goes unnamed (P2, DS-8).** The loan/footwear opener (lines 23–45)
  is a textbook instance — "when a measure becomes a target, it ceases to be a good
  measure." Naming it (verified: "Goodhart" appears nowhere in the book) is a
  one-clause lift that gives readers a hook to the wider literature. Drafted in DS-8.

- Minor: line 28 "We might be inclined " has a trailing space (cosmetic, ignore unless
  an agent is already in the paragraph).

### Exercises

The weakest part of the file relative to its content (P1, DS-5). Four exercises:
(1) search-engine behavior change — good, open-ended, keep; (2) "implement a covariate
shift detector. Hint: build a classifier." and (3) "implement a covariate shift
corrector." — both are one-liners that *restate the algorithm already given in
§4.7.3.1*, with no dataset, no acceptance criterion, no framework; (4) "besides
distribution shift, what else could affect how empirical risk approximates risk?" —
good. For a top course this set should *build* and *connect to the derivations*. A
graded rebuild (mechanical recall → implement-on-data → derive → think) is drafted in DS-5.

## 3. Code & examples

**There is no code in this file** (verified: zero code fences, no `<!-- slides -->`
section, and no `outputs/<fw>/.../environment-and-distribution-shift.json` manifests
exist — confirmed by directory listing). The per-framework checklist, output
cross-checks, and "does the code teach" review therefore **do not apply**. The single
shared `[Discussions]` link (no per-framework `:begin_tab:` blocks) is correct for a
code-free section.

The only code-adjacent items are the two "implement…" exercises; converting them into
real, dataset-backed tasks is a teaching upgrade and is folded into DS-5 (a worked
hint, not a notebook cell, keeps this section code-free as intended).

### PyTorch / JAX / TensorFlow / MXNet
N/A — no code cells in this file.

### Cross-framework consistency & d2l conventions
N/A for code. Markdown conventions are clean: labels well-formed, `:numref:`/`:eqref:`
targets resolve, equations render, captions safe. **No MXNet liability here** — there
is no MXNet tab or MXNet-specific claim to de-emphasize (unlike code chapters). Note
for the overview: the chapter's *code* siblings still carry MXNet tabs; this file does
not, so it needs no MXNet action.

---

## 4. Implementation spec (downstream agents act on THIS)

### DS-3 — Caveat the apocryphal "tanks" anecdote  ·  [P1] · [S] · [authored]
- **Type:** prose / currency
- **Where:** `chapter_linear-classification/environment-and-distribution-shift.md`,
  §4.7.2.2 Self-Driving Cars. Anchor — the paragraph beginning
  `A similar thing happened to the US Army` (≈ lines 258–268), ending
  `the second set at noon.`
- **Change (authored, replace the whole paragraph):**
  > A famous (and possibly apocryphal) cautionary tale makes the same point. As the
  > story goes, the US Army once tried to train a neural network to detect tanks
  > hidden among trees. They photographed a forest with no tanks, then drove tanks in
  > and photographed it again, and the classifier appeared to work *perfectly* on
  > held-out images---until it failed in the field. It had supposedly learned not to
  > find tanks but to tell early-morning photos (taken on the tank-free day) from
  > midday photos (taken with the tanks present): the two image sets differed in
  > lighting and shadow, not in their tanks. Whether or not it happened exactly this
  > way, the lesson is exact: a spurious feature correlated with the label in your
  > sample, but absent in deployment, is enough to fool a model that never saw the
  > distinction you actually care about.
- **Touches:** none.
- **Done when:** the paragraph no longer asserts the event as documented fact (contains
  a "possibly apocryphal" / "as the story goes" hedge) while retaining the spurious-
  feature lesson; `make html` clean.
- **Depends on:** none.

### DS-1 — Attribute the label-shift correction (Saerens 2002; BBSE 2018)  ·  [P1] · [S] · [authored]
- **Type:** coverage / currency (scholarship)
- **Where:** `chapter_linear-classification/environment-and-distribution-shift.md`,
  §4.7.3.2 Label Shift Correction. Two edits:
  (a) at the sentence introducing the confusion-matrix estimate — anchor
  `It turns out that under some mild conditions` (≈ line 501);
  (b) the bib file `d2l.bib`.
- **Change:**
  - (a) **authored** — append one sentence after the linear-system derivation
    (after `we get a solution $p(\mathbf{y}) = \mathbf{C}^{-1} \mu(\hat{\mathbf{y}})$.`,
    ≈ line 515):
    > This confusion-matrix estimator goes back to :citet:`Saerens.Latinne.Decaestecker.2002`;
    > :citet:`Lipton.Wang.Smola.2018` showed that, treating the trained classifier as a
    > black box, it yields *consistent* estimates of the target label distribution under
    > the label-shift assumption (an approach they call black-box shift estimation).
  - (b) **mechanical** — add to `d2l.bib`:
    ```
    @Article{Saerens.Latinne.Decaestecker.2002,
      title   = {Adjusting the outputs of a classifier to new a priori probabilities: A simple procedure},
      author  = {Saerens, Marco and Latinne, Patrice and Decaestecker, Christine},
      journal = {Neural Computation},
      volume  = {14},
      number  = {1},
      pages   = {21--41},
      year    = {2002},
      doi     = {10.1162/089976602753284446}
    }
    @InProceedings{Lipton.Wang.Smola.2018,
      title     = {Detecting and correcting for label shift with black box predictors},
      author    = {Lipton, Zachary and Wang, Yu-Xiang and Smola, Alexander},
      booktitle = {International Conference on Machine Learning (ICML)},
      pages     = {3122--3130},
      year      = {2018},
      organization = {PMLR},
      url       = {https://proceedings.mlr.press/v80/lipton18a.html}
    }
    ```
- **Touches:** `d2l.bib` (verified: neither key currently present).
- **Done when:** both citations render as textual citations in §4.7.3.2 with no
  "missing citation" warning; PDF + HTML build clean (`make html`, `make pdfs`).
- **Depends on:** none.

### DS-2 — One modern forward-pointer: shift in the foundation-model era (WILDS)  ·  [P1] · [S] · [authored]
- **Type:** currency
- **Where:** `chapter_linear-classification/environment-and-distribution-shift.md`,
  §4.7.6 Summary (preferred — keeps the body period-faithful and puts the "since
  then" note where summaries connect outward). Anchor — end of the Summary, after
  `…become entangled in unanticipated ways.` (≈ line 656). Plus a `d2l.bib` add.
- **Change:**
  - **authored** — append a short closing paragraph to the Summary:
    > Although the ideas above predate the current era of large pretrained models,
    > distribution shift has only become more central since: a foundation model is
    > deployed on domains, users, and time periods quite unlike its training corpus,
    > so covariate, label, and concept shift are now everyday operational realities
    > rather than corner cases. Curated benchmarks such as WILDS
    > :cite:`Koh.Sagawa.Marklund.ea.2021` collect real-world shifts---across hospitals,
    > cameras, countries, and time---and show that models with strong in-distribution
    > accuracy can still degrade sharply out of distribution. For deeper, methods-level
    > treatments of the corrections sketched here, see the references on the chapter's
    > cover page.
  - **mechanical** — add to `d2l.bib`:
    ```
    @InProceedings{Koh.Sagawa.Marklund.ea.2021,
      title     = {{WILDS}: A benchmark of in-the-wild distribution shifts},
      author    = {Koh, Pang Wei and Sagawa, Shiori and Marklund, Henrik and Xie, Sang Michael and Zhang, Marvin and Balsubramani, Akshay and Hu, Weihua and Yasunaga, Michihiro and Phillips, Richard Lanas and Gao, Irena and Lee, Tony and David, Etienne and Stavness, Ian and Guo, Wei and Earnshaw, Berton and Haque, Imran and Beery, Sara M and Leskovec, Jure and Kundaje, Anshul and Pierson, Emma and Levine, Sergey and Finn, Chelsea and Liang, Percy},
      booktitle = {International Conference on Machine Learning (ICML)},
      pages     = {5637--5664},
      year      = {2021},
      organization = {PMLR},
      url       = {https://proceedings.mlr.press/v139/koh21a.html}
    }
    ```
- **Touches:** `d2l.bib` (verified: `Koh.*` not present).
- **Done when:** the Summary contains exactly one foundation-model forward-pointer with
  the WILDS citation rendering correctly; no shift-methods survey content added; builds
  clean. Acceptance on scope: the addition is ≤ ~6 sentences and imports no methods.
- **Depends on:** none.

### DS-5 — Rebuild the exercises into a graded, content-connected set  ·  [P1] · [M] · [authored]
- **Type:** teaching
- **Where:** `chapter_linear-classification/environment-and-distribution-shift.md`,
  §4.7.7 Exercises (lines 658–664), replacing the four current items.
- **Change (authored, replace the list):**
  1. **(Conceptual, keep.)** If you change the behavior of a search engine, how might
     users respond? How might advertisers respond? Explain why this is an instance of
     the feedback loop described for the loan/footwear example.
  2. **(Recall/derive.)** Starting from the risk under the target distribution
     $p(\mathbf x, y)$, derive the covariate-shift reweighting identity
     :eqref:`eq_weighted-empirical-risk-min`, and state precisely the assumption on
     the supports of $p(\mathbf x)$ and $q(\mathbf x)$ under which the importance
     weights $\beta_i=p(\mathbf x_i)/q(\mathbf x_i)$ are finite.
  3. **(Implement — covariate-shift detector.)** Take any labeled dataset; create a
     shifted copy of the features (e.g., add Gaussian noise, or subsample by a feature
     threshold). Train a logistic-regression classifier to distinguish "original" from
     "shifted" inputs and report its accuracy. Relate the accuracy to how detectable
     the shift is, and to the classifier-two-sample-test idea in §4.7.3.1. *Hint: if the
     classifier cannot beat chance, the two distributions are indistinguishable from
     these features.*
  4. **(Implement — covariate-shift corrector.)** Using the classifier from Exercise 3,
     compute weights $\beta_i=\exp(h(\mathbf x_i))$, retrain your downstream model with
     weighted empirical risk minimization :eqref:`eq_weighted-empirical-risk-min`, and
     compare its target-domain accuracy with and without reweighting. What happens to
     the variance of $\beta_i$ as the shift grows, and how does clipping
     $\beta_i\leftarrow\min(\beta_i,c)$ help?
  5. **(Label shift.)** You have a $k$-class classifier and its validation confusion
     matrix $\mathbf C$. Show that the linear system $\mathbf C\, p(\mathbf y)=\mu(\hat{\mathbf y})$
     follows from the law of total probability under the label-shift assumption, and
     explain why $\mathbf C$ must be invertible for the estimate
     $p(\mathbf y)=\mathbf C^{-1}\mu(\hat{\mathbf y})$ to be usable.
  6. **(Conceptual, keep.)** Besides distribution shift, what else could make the
     empirical risk a poor approximation of the risk? *Hint: think about dependence
     between examples, and about the loss not matching the deployment objective.*
- **Touches:** none (`:eqref:` targets already exist in-file).
- **Done when:** ≥ 6 exercises spanning recall → implement → derive → conceptual; each
  implement-exercise names a concrete procedure and an acceptance/comparison criterion;
  `make html` clean.
- **Depends on:** none (DS-7 clipping clause makes Ex. 4's last sentence self-consistent;
  not a hard dependency).

### DS-4 — Name the concrete covariate-shift method (KMM / MMD)  ·  [P2] · [S] · [mechanical]
- **Type:** coverage / prose
- **Where:** `chapter_linear-classification/environment-and-distribution-shift.md`,
  §4.7.3.1, the sentence at ≈ lines 358–361.
- **Change (mechanical, old → new):**
  - old: `Many methods are available,\nincluding some fancy operator-theoretic approaches\nthat attempt to recalibrate the expectation operator directly\nusing a minimum-norm or a maximum entropy principle.`
  - new: `Many methods are available. One direct family matches the means of the\nreweighted source and the target features in a reproducing-kernel Hilbert\nspace---kernel mean matching, closely related to the maximum mean discrepancy\ntwo-sample test :cite:`Gretton.Borgwardt.Rasch.ea.2012`---solving for the weights\nwithout ever estimating the densities $p(\mathbf{x})$ and $q(\mathbf{x})$ separately.`
- **Touches:** none (`Gretton.Borgwardt.Rasch.ea.2012` **already in `d2l.bib`** — verified).
- **Done when:** the sentence names a concrete method and the existing Gretton citation
  renders; `make html` clean.
- **Depends on:** none.

### DS-6 — Cut/repair the ControlVAE non-sequitur in §Control  ·  [P2] · [S] · [judgment]
- **Type:** currency / prose
- **Where:** `chapter_linear-classification/environment-and-distribution-shift.md`,
  §4.7.4.4 Control, the sentence beginning `Recently, control theory (e.g., PID variants)
  has also been used` (≈ lines 573–577), ending `:cite:`Shao.Yao.Sun.ea.2020`.`
- **Change (judgment):** preferred — **delete** the sentence (and the now-orphan
  `Shao.Yao.Sun.ea.2020` use); the paragraph stands on its own as "the environment has
  memory; PID controllers are a classic response." If the author prefers to keep a
  forward nod, replace with a single neutral sentence that *control is itself a learning
  setting with memory* and drop the VAE-specific claim and citation. Do not retain the
  ControlVAE citation — it does not illuminate distribution shift.
- **Touches:** none (the bib entry may stay in `d2l.bib`; just remove the in-text use).
- **Done when:** the file's only off-topic citation is gone; §Control reads coherently;
  no dangling citation warning; `make html` clean.
- **Depends on:** none.

### DS-7 — Explain the weight-clipping `min(exp(h), c)`  ·  [P2] · [S] · [authored]
- **Type:** prose / coverage
- **Where:** `chapter_linear-classification/environment-and-distribution-shift.md`,
  §4.7.3.1, step 3 of the boxed algorithm — anchor
  `Weigh training data using $\beta_i = \exp(h(\mathbf{x}_i))$ or better $\beta_i = \min(\exp(h(\mathbf{x}_i)), c)$ for some constant $c$.` (≈ line 423).
- **Change (authored):** append one sentence immediately after the numbered list (before
  the "Note that the above algorithm relies on a crucial assumption" paragraph, ≈ line 426):
  > Clipping the weights at a ceiling $c$ trades a little bias for much lower variance:
  > when source and target barely overlap, a handful of examples get enormous weights
  > $\beta_i$ and would otherwise dominate---and destabilize---the weighted objective.
- **Touches:** none.
- **Done when:** the rationale for $c$ appears adjacent to the algorithm; reads
  consistently with the support caveat that follows; `make html` clean.
- **Depends on:** none.

### DS-8 — Name Goodhart's law at the loan/footwear hook  ·  [P2] · [S] · [mechanical]
- **Type:** prose
- **Where:** `chapter_linear-classification/environment-and-distribution-shift.md`,
  opening, the sentence at ≈ lines 42–45 ending
  `by introducing our model-based decisions to the environment,\nwe might break the model.`
- **Change (mechanical, old → new):**
  - old: `by introducing our model-based decisions to the environment,\nwe might break the model.`
  - new: `by introducing our model-based decisions to the environment,\nwe might break the model. This is a machine-learning incarnation of *Goodhart's law*:\nwhen a measure becomes a target, it ceases to be a good measure.`
- **Touches:** none (verified: "Goodhart" appears nowhere else in the book, so no
  duplicate-definition risk).
- **Done when:** Goodhart's law is named at the hook; `make html` clean.
- **Depends on:** none.

### DS-9 — (Optional) Schematic for covariate-shift reweighting  ·  [P2] · [L] · [authored/figure]
- **Type:** figure
- **Where:** `chapter_linear-classification/environment-and-distribution-shift.md`,
  §4.7.3.1, inserted after the reweighting identity / $\beta_i$ definition (≈ after line 351).
- **Change (figure spec):** one schematic SVG. Depicts a 1-D input axis $x$ with two
  overlapping density curves — source $q(x)$ and target $p(x)$ (e.g., two Gaussians with
  different means/variances) — plus a third curve, the importance weight
  $\beta(x)=p(x)/q(x)$, shown rising steeply (and annotated "→ ∞") in the region where
  $q(x)\to 0$ while $p(x)>0$. Caption ties the picture to (i) the reweighting identity
  and (ii) the nonzero-support caveat. Single consistent house style; this is a
  *schematic*, so it must be a committed generator SVG (e.g.
  `tools/gen_mdl_<chapter>_figures.py` → `img/<id>.svg` + `make figures`), **not** inline
  matplotlib in a notebook (the file has no notebook anyway). Reference with `:numref:`.
- **Touches:** a figure generator under `tools/` + `make figures` + `img/<id>.svg`.
- **Done when:** figure renders in HTML and PDF, is referenced by `:numref:`, SVG is
  byte-idempotent, `make html` clean. Skip if figure budget is tight — DS-1/DS-2/DS-3
  are the priorities.
- **Depends on:** none.

---

## 5. Keep — what is already excellent (do not lose this)

- **The whole conceptual arc and voice.** The opening loan/footwear feedback-loop hook,
  the impossibility-without-assumptions framing (lines 72–96, including the "God flips
  all the cat/dog labels" thought experiment), and the prose rhythm are top-tier. Do not
  sand these down while editing.
- **The covariate/label/concept-shift taxonomy tied to causal direction** ($\mathbf x$
  causes $y$ ⇒ covariate shift; $y$ causes $\mathbf x$ ⇒ label shift; lines 109–159) is
  crisp, correct, and better motivated than most textbook treatments. The
  deterministic-label degeneracy remark (lines 148–159) is a genuinely sophisticated
  point — keep it.
- **The cancer-startup and synthetic-roadside anecdotes** (§4.7.2.1–4.7.2.2) are
  verified-feeling, concrete, and land the covariate-shift lesson hard. Keep (only the
  *tanks* sub-anecdote needs the DS-3 caveat).
- **Both correction derivations.** The covariate-shift reweighting identity + the
  logistic-regression-gives-$\beta_i=\exp(h)$ trick (lines 332–410), and the label-shift
  confusion-matrix linear system (lines 454–523), are clean, honest about assumptions,
  and exactly the right depth — introductory-but-rigorous. The support caveat
  (lines 426–431) and the "labels are low-dimensional, so estimate in label space"
  insight (lines 468–475) are highlights. Add attribution (DS-1) but **do not** rewrite
  the math.
- **The "impatient reader may skip" signpost** before the technical section (lines
  302–304) — good pedagogy, keep.
- **The fairness/feedback-loop closing** (§4.7.5), especially the four-step predictive-
  policing runaway-feedback-loop walkthrough (lines 631–639) — vivid, correct, and a
  strong note to end the chapter on. Keep intact.
- **Markdown hygiene:** all `:numref:`/`:eqref:` cross-references resolve (verified
  against `generalization.md` and `sec_softmax`), figures render, single shared
  Discussions link is correct for a code-free section.
