# Chapter Overview — chapter_recommender-systems

Best sources by a wide margin: the original papers' own ablation studies (Rendle's
BPR and FM papers, Guo et al.'s DeepFM §3.3 hyperparameter study, Tang & Wang's
Caser ablation, He et al.'s NeuMF repo, Dacrema et al.'s reproducibility audit).
This chapter's topics are graduate-research-adjacent, not intro-course homework,
so a paper's own controlled comparison usually beats anything a university
problem set offers. Stanford CS246 (PS2, matrix factorization + CF derivations)
and MMDS ch. 9 were the strongest genuine *course*-homework sources, but both
stop at classical similarity-based CF and UV-decomposition — neither has any
ranking-loss, autoencoder, sequence-model, or CTR/FM/DeepFM content. Minnesota's
Coursera specialization and DataCamp supplied usable dataset-exploration and
grid-search assignments (movielens.md, mf.md) but nothing past classical MF.
Kaggle's Criteo/Avazu CTR challenges (plus their well-documented winning FFM
pipelines) are the strongest tradition for ctr.md/fm.md/deepfm.md. Verified
**no external exercise tradition at all** for: AutoRec, sequence-aware/session
recommendation as classroom homework (only the Caser paper's own ablation),
BPR/ranking losses as course homework, and DeepFM as a university problem set.
recsys-intro.md and ranking.md were already excellent (flagged "exemplary" in
the prior style review) and are kept almost intact; movielens.md, ctr.md, and
fm.md needed the most rework — two pure reading-prompt exercises in
movielens.md, a conspicuously thin 1-exercise ctr.md, and fm.md's grammar
defects plus zero-success-criterion filler (the weakest-written file in the
chapter).

---

## chapter_recommender-systems/recsys-intro.md — Overview of Recommender Systems

**Topic:** Conceptual framing of recommendation — explicit vs. implicit feedback,
collaborative/content-based/context-aware scoring, rating prediction vs. top-N
vs. next-item vs. CTR tasks, warm-start vs. cold-start evaluation.
**Current exercises:** 3; disposition: keep 3, rewrite 0, drop 0 — all three
(order-sensitivity construction, false-negative example, warm/cold-start split
design) already name a concrete deliverable and were rated exemplary by the
prior style review; kept verbatim and supplemented rather than replaced.

**External sources found:**
- Leskovec/Rajaraman/Ullman, *Mining of Massive Datasets*, ch. 9.2.8 Ex.
  9.2.1–9.2.3, mmds.org (http://infolab.stanford.edu/~ullman/mmds/ch9.pdf) —
  cosine similarity under feature scaling and normalized user profiles; a
  hand-computable content-based complement to this section's CF/content-based
  table.
- Leskovec/Rajaraman/Ullman, MMDS ch. 9.3.4 Ex. 9.3.1–9.3.2, same URL — a
  small (3-user × 8-item) toy utility matrix; compares Jaccard vs. cosine
  similarity under binarization/normalization, then reclusters.
- Stanford CS246, Winter 2018, PS2 Q4
  (http://snap.stanford.edu/class/cs246-2018/homeworks/hw2/hw2.pdf) — derive
  user-user vs. item-item CF scores algebraically in terms of the rating
  matrix and its factors (parts a–c are pure pencil derivation).
- FINDING: no external homework builds warm-start/item-cold-start split
  *design* as an exercise — the book's own exercise 3 is a rare, genuinely
  original task worth keeping as-is.

**Proposed problem set** (6 problems, our reference format):
1. [conceptual] **Order-Sensitivity Test.** Construct two interaction
   histories with the same items in different order; state which of the
   chapter's later models assign them identical scores and which can
   distinguish them.
   *Provenance:* original (kept from the book).
1. [conceptual] **False-Negative Construction.** Give a concrete example
   where an unobserved item is a false negative, and explain how uniform
   negative sampling changes the expected training objective relative to
   treating every unobserved pair as a true negative.
   *Provenance:* original (kept from the book).
1. [conceptual] **Warm/Cold-Start Split Design.** Design separate warm-start
   and item-cold-start evaluation splits, and state which input fields are
   required to produce a score in each split.
   *Provenance:* original (kept from the book).
1. [conceptual] **User-Item vs. Item-Item Score Derivation.** Adapted from
   CS246 PS2 Q4 (overlap high; cite on adoption). Derive, in terms of a
   ratings matrix $R$ and its later-introduced factors $P,Q$, the scoring
   formula a user-based CF method and an item-based CF method would each
   produce for the same $(u,i)$ pair, using a small worked numeric example
   (5 users × 4 items). State the one condition under which the two scores
   coincide.
1. [conceptual] **Similarity-Metric Sensitivity.** Inspired by MMDS Ex.
   9.2–9.3 (overlap low-med). Using a small toy user–item rating table (3
   users × 6 items, deliberately including one user who rates everything
   one point higher), compute both cosine similarity and Jaccard similarity
   (after binarizing at a threshold) between two users; show that the two
   metrics disagree on which pair is "more similar," and explain the
   disagreement using the rating-scale offset.
1. [conceptual] **Content-Based Complement.** Inspired by MMDS ch. 9.2
   (overlap low). Given a table of item attributes (e.g., genre indicators)
   in place of interaction history, sketch a content-based score for a new
   item with zero interactions; identify which cell of this section's
   input/task table (user–item only, item/user attributes, or context) it
   fills, and which cold-start case it still cannot address.

---

## chapter_recommender-systems/movielens.md — The MovieLens Dataset

**Topic:** Downloading/inspecting MovieLens-100K, sparsity and rating
distributions, random vs. sequence-aware splitting, explicit/implicit data
representations.
**Current exercises:** 2; disposition: keep 0, rewrite 1, drop 1 — both
current exercises are pure reading/browsing prompts with no artifact; ex. 1
("what other datasets can you find?") is rewritten into a concrete
comparison task, ex. 2 ("browse movielens.org") is dropped outright as it has
no salvageable deliverable.

**External sources found:**
- University of Minnesota, Coursera "Introduction to Recommender Systems,"
  Module 3 Assignment 1 (GroupLens) — computes top-rated-by-mean items,
  checks association between ratings and a specific title, and compares
  average ratings across a demographic split, using LensKit/spreadsheet
  tools (no stable public assignment URL located; course confirmed to
  exist).
- DataCamp, "Building Recommendation Engines in Python," matrix-sparsity
  exercise (verified content, no stable public exercise URL) — counts
  non-empty matrix cells to compute sparsity, the same quantity this
  section already prints.
- DataCamp, PySpark recommender-systems course, ch. 3 (verified from
  extracted PDF, no stable public URL) — computes ~99.8% sparsity on
  MovieLens-20M, groups ratings per user, and performs an 80/20
  `randomSplit` before configuring an ALS model.
- FINDING: no verified course exercise anywhere uses a leave-last-out /
  temporal split — every located source uses a random split. The
  seq-aware-vs-random comparison below has no external precedent and is
  original.

**Proposed problem set** (6 problems, our reference format):
1. [short-code] **Sparsity Across Subpopulations.** Adapted from the
   DataCamp PySpark sparsity exercise (overlap med; cite on adoption).
   Compute matrix sparsity for the full MovieLens-100K matrix, then
   separately for the subset of users with fewer than 20 ratings and the
   subset with more than 100; report all three sparsity numbers and explain
   why the "every user rated at least 20 movies" property already limits
   how sparse the low end can look.
1. [short-code] **Split Leakage Check.** Using `split_data_ml100k` in
   `seq-aware` mode, empirically verify that no training row for a user
   occurs later in time than that user's held-out test row; report the
   count of violations found (expected: zero), and describe one code change
   that would introduce a violation.
1. [conceptual] **Rating Count vs. Rating Quality.** Adapted from the
   Minnesota Coursera Module 3 assignment (overlap med; cite on adoption).
   Compute the top-5 items by mean rating and, separately, the top-5 by
   number of ratings; explain why the two lists differ, and connect the
   answer to the "aggregate rate is not itself a measure of quality"
   caution made later in ctr.md.
1. [short-code] **Demographic Rating Gap.** Adapted from the Minnesota
   Coursera Module 3 assignment (overlap med; cite on adoption). Using the
   demographic fields in `u.user` (not loaded by this section's own code,
   but present in the downloaded archive), compute the mean rating given by
   two demographic groups of your choice; report the numeric gap and one
   confound that could explain it besides a genuine taste difference.
1. [short-code] **Temporal vs. Random Split Divergence.** No verified
   external precedent (finding above); original. Train the same simple
   baseline (e.g., a per-item mean-rating predictor) under `random` mode and
   under `seq-aware` mode; report the RMSE under each split, and explain why
   a *lower* seq-aware RMSE would be a red flag rather than good news given
   what each split is testing.
1. [conceptual] **Alternative-Dataset Comparison.** Rewrite of the book's ex.
   1. Pick two other public recommendation datasets (e.g., Amazon reviews,
   Yelp, Last.fm); for each, report size, feedback type (explicit/implicit),
   and one respect in which its evaluation protocol would need to differ
   from MovieLens-100K's warm-start random split.
   *Provenance:* original (book prompt tightened with a required
   deliverable).

---

## chapter_recommender-systems/mf.md — Matrix Factorization

**Topic:** $\hat R = PQ^\top$ with user/item biases, RMSE evaluation, SGD/Adam
training on MovieLens-100K, effect of latent-factor dimension $k$.
**Current exercises:** 3; disposition: keep 0, rewrite 3, drop 0 — all three
"vary X and see impact" prompts have a salvageable core (factor size,
optimizer/lr/weight-decay, per-user prediction inspection) that only needed a
named metric, range, and success criterion.

**External sources found:**
- Stanford CS246, Winter 2018, PS2 Q3 "Latent Features"
  (http://snap.stanford.edu/class/cs246-2018/homeworks/hw2/hw2.pdf, verified)
  — derive SGD updates for $R\approx QP^\top$ with $\ell_2$ regularization,
  implement streaming updates from disk, and hit a checkable criterion
  ($E<65{,}000$ after 40 iterations at $k=20,\lambda=0.1$), plotting error vs.
  iteration. Exactly the well-specified pattern this section's exercises
  should emulate.
- Leskovec/Rajaraman/Ullman, MMDS ch. 9.4.6 Ex. 9.4.1–9.4.5
  (http://infolab.stanford.edu/~ullman/mmds/ch9.pdf, verified) — a sequence
  of UV-decomposition coordinate-descent derivations: closed-form single-entry
  updates and the effect of update order.
- University of Minnesota, Coursera "Matrix Factorization and Advanced
  Techniques," Module 3 — graded matrix-factorization and "Programming SVD"
  assignments (course/module title confirmed to exist; no stable public
  assignment-page URL located).
- DataCamp, PySpark recommender-systems course, ALS grid-search exercise
  (verified from extracted PDF) — sweeps rank $\in\{5,40,80,120\}$, `maxIter`
  $\in\{5,100,250,500\}$, and `regParam` $\in\{.05,.1,1.5\}$, reporting RMSE
  per configuration; turns "vary $k$" into a concrete grid with a named
  winner.
- NOT verified / not usable: Charu Aggarwal's *Recommender Systems: The
  Textbook* exercises are paywalled (no unauthorized mirror consulted); no
  course homework was found built directly around the Koren–Bell–Volinsky
  survey paper.

**Proposed problem set** (6 problems, our reference format):
1. [conceptual] **SGD Update Derivation.** Adapted from CS246 PS2 Q3
   (overlap high; cite on adoption). Derive, by hand, the gradient of the
   regularized squared-error objective with respect to $\mathbf p_u$,
   $\mathbf q_i$, $b_u$, $b_i$, and write the corresponding SGD update rule
   for each; check that setting $\lambda=0$ recovers plain unregularized
   least squares.
1. [short-code] **Latent-Dimension Sweep with a Stopping Point.** Rewrite of
   the book's ex. 1, inspired by CS246 PS2 Q3's checkable-criterion pattern
   (overlap med; cite on adoption). Train at $k\in\{5,10,20,30,50\}$ for a
   fixed 20 epochs each; plot test RMSE vs. $k$ and identify the value of
   $k$ beyond which RMSE stops improving.
1. [short-code] **Optimizer/LR/Weight-Decay Grid.** Rewrite of the book's ex.
   2, adapted from the DataCamp ALS grid-search exercise (overlap med; cite
   on adoption). Run a small grid over {SGD, Adam} × {two learning rates} ×
   {two weight-decay values} (8 configurations total); report a table of
   final test RMSE and name the single most sensitive hyperparameter.
1. [short-code] **Per-User Prediction Spread.** Rewrite of the book's ex. 3.
   For a fixed trained model and a fixed popular item, plot the distribution
   of predicted ratings across all users and compare it with the
   distribution of *actually observed* ratings for the subset of users who
   rated that item; report both standard deviations and state whether the
   predicted spread is narrower (a common MF shrinkage effect).
   *Provenance:* original (book prompt tightened with a required
   deliverable).
1. [conceptual] **Coordinate-Descent Alternative.** Adapted from MMDS ch.
   9.4 UV-decomposition exercises (overlap med; cite on adoption). Derive
   the closed-form update for a single entry of $\mathbf P$ (or
   $\mathbf Q$), holding every other entry fixed, that minimizes the same
   regularized squared-error objective; compare its per-update cost to one
   SGD step on the same entry.
1. [short-code] **Bias-Term Ablation.** Retrain the model with $b_u,b_i$
   removed (score $=\mathbf p_u^\top\mathbf q_i$ only); report the RMSE gap
   from the full model, and, using the rating-distribution histogram from
   movielens.md, explain why removing biases should hurt more for the
   heaviest raters.
   *Provenance:* original.

---

## chapter_recommender-systems/autorec.md — AutoRec: Rating Prediction with Autoencoders

**Topic:** Item-based AutoRec — a masked autoencoder that reconstructs a
partially-observed rating vector, trained only on observed coordinates.
**Current exercises:** 3; disposition: keep 0, rewrite 3, drop 0 — the "vary
hidden dimension," "add more layers," and "find a better activation
combination" prompts all have a salvageable experimental core; none had a
metric or range.

**External sources found:**
- **FINDING: verified negative — no external exercise tradition exists for
  AutoRec anywhere checked.** MMDS (via a public MMDS-exercises mirror; the
  Stanford infolab host itself returned a TLS error) covers only
  similarity-based CF and UV-decomposition, no autoencoder content. Stanford
  CS246 (2020 offering checked) has no recsys-specific homework at all.
  UC San Diego CSE 158/258 (McAuley; both assignment PDFs from a Fall 2025
  offering read directly) assign open-ended Goodreads-dataset projects with
  no AutoRec, BPR, or NeuMF component. University of Minnesota's Coursera
  specialization has no deep-learning module. BlueCourses (Baesens) presents
  an AutoRec notebook as delivered lecture content, not a graded assignment.
  Aggarwal's textbook chapter is paywalled (inconclusive, not counted).
  Dacrema et al.'s reproducibility study (below, cited under neumf.md)
  reproduces CDAE, Mult-VAE, and CVAE but explicitly not AutoRec.
- Because no course or paper ablation directly targets AutoRec's own
  hyperparameters, the proposed set below leans on the section's own
  disclaimed RMSE comparison with mf.md and the paper's stated user/item
  asymmetry rather than an external citation.

**Proposed problem set** (6 problems, our reference format):
1. [short-code] **Hidden-Dimension Sweep.** Rewrite of the book's ex. 1.
   Train at hidden width $h\in\{50,100,250,500,1000\}$ for a matched number
   of epochs; report test RMSE at each width and state whether RMSE turns
   upward past some $h$ (an overfitting signature, given only 943 users).
   *Provenance:* original (no external tradition found).
1. [short-code] **Depth Ablation at Matched Budget.** Rewrite of the book's
   ex. 2. Add one additional hidden layer (linear–sigmoid–linear–sigmoid),
   holding total parameter count roughly fixed relative to the one-layer
   baseline; report test RMSE and epochs-to-best-RMSE for both, and state
   whether the extra depth mainly adds capacity or mainly slows convergence.
   *Provenance:* original.
1. [short-code] **Activation-Combination Grid.** Rewrite of the book's ex. 3.
   Run all four combinations of {sigmoid, ReLU} encoder activation ×
   {identity, ReLU} decoder activation; report test RMSE for each of the
   four in one table and name the best combination.
   *Provenance:* original.
1. [conceptual] **User-Based vs. Item-Based Capacity.** Inspired by the
   AutoRec paper's own user/item variant comparison, which this section
   states without demonstrating (overlap low). Derive the user-based
   AutoRec's masked loss by swapping the roles of $\mathbf R$'s rows and
   columns in the objective given in this section; using $m=943$ users and
   $n=1682$ items on MovieLens-100K, state one structural reason the
   item-based encoder sees a different-sized input (and thus a different
   number of training examples per epoch) than the user-based one.
1. [short-code] **AutoRec vs. Matrix Factorization, Controlled.** The
   section's own text explicitly disclaims its AutoRec-vs-MF RMSE
   comparison as uncontrolled. Rerun AutoRec (this section) and MF (mf.md)
   with the same random seed, the same train/test split, and a matched
   total parameter count; report whether the RMSE gap already shown in the
   text survives this controlled rerun.
   *Provenance:* original.
1. [conceptual] **Why the Mask Is Necessary.** Show algebraically why
   removing the observation mask $M$ from the training loss (i.e., treating
   every unobserved entry as a target rating of 0) would bias the learned
   reconstruction toward under-predicting ratings for popular items with
   fewer zero-appearances; use the ~6% observed-entry figure from
   movielens.md in the argument.
   *Provenance:* original.

---

## chapter_recommender-systems/ranking.md — Personalized Ranking for Recommender Systems

**Topic:** Pairwise objectives for implicit feedback — Bayesian personalized
ranking (BPR) derived from a MAP argument, and a margin-based hinge
alternative.
**Current exercises:** 3; disposition: keep 3, rewrite 0, drop 0 — all three
(BPR/hinge gradient comparison, sampling-scheme expectations, label-
contradiction construction) were rated "exemplary, best-specified file in
the chapter" by the prior style review; kept verbatim and supplemented.

**External sources found:**
- Rendle, Freudenthaler, Gantner, Schmidt-Thieme, "BPR: Bayesian Personalized
  Ranking from Implicit Feedback" (UAI 2009), arXiv:1205.2618
  (https://arxiv.org/abs/1205.2618) — the paper's own "Analogies to AUC
  Optimization" section shows BPR-Opt as a smooth surrogate for a
  non-differentiable empirical-AUC objective (replacing a Heaviside step
  with $\ln\sigma(\cdot)$); not covered by any current exercise here.
- Rendle, Krichene, Zhang, Anderson, "Neural Collaborative Filtering vs.
  Matrix Factorization Revisited" (RecSys 2020), arXiv:2005.09683
  (https://arxiv.org/abs/2005.09683) — shows a properly tuned dot-product MF
  beats a learned-similarity model across embedding dimensions
  $d\in\{16,\ldots,192\}$; more directly relevant to neumf.md, referenced
  here only as background for the pairwise-comparison framing.
- Dacrema, Cremonesi, Jannach, "A Troubling Analysis of Reproducibility and
  Progress in Recommender Systems Research" (TOIS 2021), arXiv:1911.07698
  (https://arxiv.org/abs/1911.07698) — reports a carefully tuned BPR-MF
  baseline in its Tables 7–8, useful as an optional grounding point but not
  itself a pairwise-loss exercise.
- FINDING: no MMDS or CS246 exercise touches personalized ranking or BPR at
  all (verified absent); a public "RecSys summer school" exercise repo found
  during the search covers multi-armed bandits, not pairwise ranking losses,
  and is not relevant here.

**Proposed problem set** (6 problems, our reference format):
1. [conceptual] **BPR/Hinge Gradient Comparison.** Differentiate the BPR and
   hinge losses with respect to the score difference $d=\hat y_{ui}-\hat
   y_{uj}$; compare their gradients as $d\to-\infty$, at $d=0$, and once the
   hinge margin is satisfied.
   *Provenance:* original (kept from the book).
1. [conceptual] **Sampling-Scheme Expectations.** Write the expectation
   optimized under uniform negative sampling and under popularity-
   proportional sampling; state which proposal is more likely to sample an
   exposed-but-skipped item, and which one requires importance weights to
   estimate the uniform-item objective.
   *Provenance:* original (kept from the book).
1. [conceptual] **Label-Contradiction Construction.** Construct a user
   history in which a held-out positive item is sampled as a training
   negative; explain the effect on BPR and how a strict train/validation/test
   protocol avoids consulting test identities during fitting.
   *Provenance:* original (kept from the book).
1. [conceptual] **BPR as an AUC Surrogate.** Adapted from Rendle et al.,
   UAI 2009, "Analogies to AUC Optimization" (overlap high; cite on
   adoption). Show that averaging the indicator $\mathbf 1\{\hat
   y_{ui}>\hat y_{uj}\}$ over sampled pairs is an empirical AUC, and that
   replacing the indicator with $\ln\sigma(\hat y_{ui}-\hat y_{uj})$ gives a
   smooth surrogate with the same gradient sign at the decision boundary;
   state one case where the two objectives could rank a pair differently.
1. [short-code] **Margin Sensitivity of Hinge Loss.** Draw 200 synthetic
   score pairs $(\hat y_{ui},\hat y_{uj})$ from a chosen distribution (e.g.,
   both $\sim\mathcal N(0,1)$) and evaluate `HingeLossbRec` at margin
   $m\in\{0.5,1,2,4\}$; report, for each $m$, the fraction of pairs with
   zero loss (margin already satisfied), and connect the trend to the
   "different gradient behavior" claim in the section's Summary.
   *Provenance:* original.
1. [conceptual] **Tie-Handling in Pairwise Comparisons.** Given the pairwise
   indicator $\mathbf 1\{\hat y_{ui}>\hat y_{uj}\}$ implicit in both losses,
   define an AUC-style average of this indicator over many sampled $j$;
   work out what changes if exact ties are broken with half credit instead
   of zero, and give one practical situation (discretized or low-precision
   scores) where this choice changes the resulting average.
   *Provenance:* original.

---

## chapter_recommender-systems/neumf.md — Neural Collaborative Filtering for Personalized Ranking

**Topic:** NeuMF — a GMF (elementwise-product) branch plus an MLP
(concatenation) branch, jointly trained with BPR loss and negative sampling,
evaluated with leave-one-out Hit@$K$/AUC.
**Current exercises:** 4; disposition: keep 0, rewrite 4, drop 0 — all four
"vary X" / "try different Y" prompts (including the one valid cross-reference
to ranking.md's hinge loss) lacked a named metric even though this section
already defines Hit@50 and AUC.

**External sources found:**
- He, Liao, Zhang, Nie, Hu, Chua, "Neural Collaborative Filtering" (WWW
  2017); official repo hexiangnan/neural_collaborative_filtering
  (https://github.com/hexiangnan/neural_collaborative_filtering, verified) —
  ships exact hyperparameters (`num_factors=8`, `layers=[64,32,16,8]`,
  `num_neg=4`, Adam at lr 0.001) and an `evaluate.py` implementing
  leave-one-out Hit@$K$/NDCG@$K$ against 100 sampled negatives on
  ml-1m/pinterest-20 — the standard "reproduce these numbers" template,
  though not directly portable to this section's smaller ml-100k setup and
  simplified evaluator.
- Rendle, Krichene, Zhang, Anderson, "NCF vs. MF Revisited" (RecSys 2020),
  arXiv:2005.09683 (https://arxiv.org/abs/2005.09683, verified) — shows a
  properly tuned dot-product MF baseline outperforms NCF's learned
  similarity function across embedding sizes; a template for a controlled
  factor-size study.
- Dacrema, Cremonesi, Jannach, TOIS 2021, arXiv:1911.07698
  (https://arxiv.org/abs/1911.07698, verified), Appendix A.3 — documents
  that the original NCF reference code selects its final checkpoint by
  maximizing hit rate *on the test set itself* (an early-stopping leak), and
  reports that a tuned classical baseline (ItemKNN or BPR-MF) is
  competitive with or better than NeuMF once this and other protocol issues
  are fixed.
- FINDING: no MMDS/CS246/Minnesota-Coursera exercise touches NeuMF or
  learned-similarity CF at all.

**Proposed problem set** (6 problems, our reference format):
1. [short-code] **Latent-Factor and MLP-Depth Sweep.** Rewrite of the book's
   ex. 1–2, now with the metric already defined in this section. Train at
   `num_factors` $\in\{4,8,16,32\}$, and separately at MLP depth
   $\in\{1,2,3\}$ (fixed width 10); report Hit@50 and AUC for each config in
   one table and state whether returns diminish past a specific factor
   size.
1. [short-code] **Optimizer/Regularization Comparison.** Rewrite of the
   book's ex. 3. Compare {SGD, Adam} × {weight decay $0$, $10^{-5}$} (4
   configurations) using the section's own Hit@50/AUC evaluator; report the
   table and name the single most sensitive choice.
1. [short-code] **Hinge-Loss Substitution.** Rewrite of the book's ex. 4,
   now concrete. Replace `d2l.BPRLoss()` with `d2l.HingeLossbRec()` (margin
   1) in `train_ranking`, holding everything else fixed; report Hit@50/AUC
   for both losses and connect any gap to the gradient-behavior difference
   derived in ranking.md.
1. [conceptual] **GMF-Only vs. MLP-Only Reduction.** Inspired by the NeuMF
   paper's own ablation practice (overlap low). Derive what NeuMF's score
   reduces to if the MLP branch's output is fixed at zero (pure GMF), and
   separately if the GMF branch is fixed at zero (pure MLP); state which
   reduction is architecturally identical to a model already introduced
   earlier in the chapter, and why.
1. [short-code] **GMF/MLP Branch Ablation, Empirically.** Companion to the
   previous problem. Implement the two zeroed-branch variants by modifying
   `NeuMF.forward`; retrain each and report Hit@50/AUC against the full
   two-branch model to test the prediction made above.
   *Provenance:* original.
1. [conceptual] **Test-Set Peeking in Negative Sampling.** Adapted from
   Dacrema et al., TOIS 2021, Appendix A.3 (overlap med; cite on adoption).
   This section's own `PRDataset` already excludes the held-out test item
   from its negative-sampling pool, and the surrounding text flags this as
   "consulting the identity of the evaluation event during training."
   Explain concretely how this choice could inflate Hit@50 relative to a
   protocol that samples negatives without ever touching test labels, and
   propose a validation-set-based alternative that avoids it.

---

## chapter_recommender-systems/seqrec.md — Sequence-Aware Recommender Systems

**Topic:** Caser — a convolutional sequence recommender over the last $L$
interactions (horizontal + vertical filters) combined with a per-user
embedding, trained with BPR loss.
**Current exercises:** 3; disposition: keep 0, rewrite 3, drop 0 — the
horizontal/vertical ablation (ex. 1) had a good core but grammar issues and no
metric; the window-length sweep (ex. 2) had no metric; the session-vs-
sequence-aware question (ex. 3) was a bare "can you explain" filler.

**External sources found:**
- Tang & Wang, "Personalized Top-N Sequential Recommendation via
  Convolutional Sequence Embedding" (WSDM 2018), arXiv:1809.07426
  (https://arxiv.org/abs/1809.07426, verified) — the paper's own ablation
  compares variants Caser-p/h/v/vh/ph/pv/pvh by MAP on MovieLens and
  Gowalla (p worst; vh/pvh best), and separately sweeps window length $L$
  against a Markov-order baseline (its Fig. 6). This is the direct origin of
  the book's own ex. 1–2 and supplies both a comparison target and a
  citable "reproduce this figure" framing.
- Quadrana, Cremonesi, Jannach, sequence-aware recsys tutorial materials
  (RecSys 2018 / The Web Conf 2019), repo mquad/sars_tutorial
  (https://github.com/mquad/sars_tutorial, verified) — 8 runnable notebooks
  including a Markov-chain/FPMC sequence-aware model and a session-based RNN
  model side by side; a template for turning the book's "explain the
  difference" reading prompt into a hands-on comparison.
- FINDING: verified negative — no course homework tradition exists for
  sequence-aware or session-based recommendation. MMDS ch. 9's exercises are
  static-matrix only; CS246's HW2 is static MF/CF; Minnesota's Coursera
  specialization has no sequential-recommendation assignment; the official
  `caser_pytorch` repo's README ships no ablation scaffolding of its own.
  The Caser paper's own ablation is therefore the best available citable
  template, not a classroom exercise.

**Proposed problem set** (5 problems, our reference format):
1. [short-code] **Horizontal/Vertical Ablation with a Metric.** Rewrite of
   the book's ex. 1, adapted from Tang & Wang's own ablation (overlap high;
   cite on adoption). Retrain three variants — vertical-branch-only
   ($d{=}0$), horizontal-branch-only ($d'{=}0$), and the full model — using
   the Hit@50/AUC evaluator already defined in neumf.md; report which
   single branch, if either, recovers most of the full model's performance,
   and compare the qualitative ordering to the paper's own p/h/v/vh
   ranking.
1. [short-code] **Window-Length Sweep with a Metric.** Rewrite of the book's
   ex. 2, adapted from Tang & Wang's Fig. 6 window/Markov-order sweep
   (overlap med; cite on adoption). Train at $L\in\{3,5,7,9\}$; report
   Hit@50/AUC at each $L$ and state whether accuracy increases monotonically
   or peaks and declines, as it does for some datasets in the paper's own
   sweep.
1. [conceptual] **Session-Based vs. Sequence-Aware, Concretely.** Rewrite of
   the book's ex. 3 (was a bare "can you explain" filler). Given one user's
   interaction log split into two sessions by a multi-day gap, state what
   must change to score the second session's next item under (a) this
   section's sequence-aware framing (persistent user embedding plus a fixed
   window) versus (b) a session-based framing that discards user identity
   and treats each session independently; give one dataset property (e.g.,
   anonymous browsing with no persistent login) that forces framing (b).
1. [short-code] **Reproducing a Published Ordering at Smaller Scale.**
   Adapted from Tang & Wang's best-configuration settings (overlap low —
   the paper uses ml-1m/Gowalla, this section uses ml-100k, so absolute
   numbers will not match). Using the $L$ and $d/d'$ values the paper
   reports as its best configuration, retrain on MovieLens-100K and report
   whether the *direction* of the paper's vh > v > h > p ordering still
   holds at this much smaller scale; state explicitly that matching
   absolute MAP values is not expected.
1. [conceptual] **Negative-Pool Test Peeking, Here Too.** Parallel to the
   issue already flagged in this section's own code comments. `SeqDataset`,
   like `PRDataset` in neumf.md, excludes the held-out test item from its
   negative-sampling pool. Explain why this shortcut is arguably more
   consequential here than in neumf.md, given that the test item's identity
   is also used implicitly when this section's training windows are
   constructed from "earlier events only."
   *Provenance:* original.

---

## chapter_recommender-systems/ctr.md — Feature-Rich Recommender Systems

**Topic:** Click-through-rate prediction as binary classification over an
impression's categorical fields, via a `CTRDataset` wrapper mapping each
field's values to integer indices with a rare-value fallback.
**Current exercises:** 1; disposition: keep 1, rewrite 0, drop 0 — the sole
exercise (continuous-field extension with a leakage question) is well-posed;
the file was simply conspicuously thin, so all growth here is additive.

**External sources found:**
- Criteo Display Advertising Challenge (Kaggle, 2014)
  (https://www.kaggle.com/competitions/criteo-display-ad-challenge/overview,
  verified) — log-loss task over a label plus 13 integer and 26 categorical
  fields; the standard precedent for handling genuinely continuous fields
  and for the test-set-leakage pitfall this section's own exercise already
  raises.
- Avazu Click-Through Rate Prediction Challenge (Kaggle, 2014)
  (https://www.kaggle.com/competitions/avazu-ctr-prediction/overview,
  verified) — log-loss task over roughly 20 fields that are, with the
  exception of a temporal `hour` field, entirely categorical; a useful
  contrast to Criteo's numeric/categorical split.
- McMahan et al., "Ad Click Prediction: a View from the Trenches" (KDD
  2013) — the canonical systems reference for this section's citation; no
  course exercise built directly on it was found (verified absent).
- FINDING: MMDS ch. 8 covers AdWords matching/mechanism design, not CTR
  classification, and is not relevant despite the shared "advertising"
  keyword; CS246's HW4 (streaming count sketches) is a near-miss, not a CTR
  exercise.

**Proposed problem set** (5 problems, our reference format):
1. [short-code] **Continuous-Field Extension.** Extend `CTRDataset` with an
   explicit path for continuous fields; fit normalization or bin boundaries
   on the training split only, and verify no test-set statistic leaks into
   the fitted boundaries.
   *Provenance:* original (kept from the book).
1. [short-code] **Rare-Value Threshold Sensitivity.** Vary `min_threshold`
   over $\{1,4,10,50\}$; report the resulting total `field_dims` size and
   the fraction of test-set field values that fall back to the "unseen"
   bucket for each threshold, and identify the threshold beyond which that
   fallback fraction becomes large.
   *Provenance:* original.
1. [conceptual] **Criteo vs. Avazu Field Contrast.** Inspired by the Criteo
   and Avazu task definitions (overlap low; cite for framing). Criteo's
   public schema separates 13 numeric fields from 26 categorical fields,
   while Avazu's is (aside from `hour`) entirely categorical. Explain what
   changes about `CTRDataset`'s "reserve an index for rare/unseen values"
   strategy when a field is genuinely continuous rather than merely
   high-cardinality categorical.
1. [short-code] **Temporal Field Handling.** Inspired by Avazu's
   separately-flagged `hour` field (overlap low; cite for framing). Pick one
   of the 34 anonymous fields and, without knowing its semantics, build two
   minimal one-field logistic baselines directly from `CTRDataset`'s
   indices: one that treats the field as a categorical embedding (as the
   wrapper already encodes it) and one that treats its integer code as a
   single ordinal input to a one-parameter linear term; report the test log
   loss for both and state which encoding requires knowing the field is
   temporal/ordinal to justify.
1. [conceptual] **Vocabulary-Leakage Audit.** Extends the existing
   exercise's leakage question to the vocabulary-building step itself.
   `CTRDataset.__init__` builds `feat_mapper` from whichever file is passed
   to it; explain why calling `CTRDataset(test_path)` independently (rather
   than reusing `train_data.feat_mapper`, as fm.md and deepfm.md do) would
   leak test-set vocabulary into the encoding, and identify the exact line
   in the shown code that prevents this.
   *Provenance:* original.

---

## chapter_recommender-systems/fm.md — Factorization Machines

**Topic:** Second-order factorization machines — every feature pair's
coefficient is $\langle\mathbf v_i,\mathbf v_j\rangle$, computed in $O(kd)$ via
a square-of-sum identity.
**Current exercises:** 2; disposition: keep 0, rewrite 2, drop 0 — the
weakest-written file in the chapter (two grammar defects, both exercises
"Can you...?" filler with zero success criteria), but the underlying ideas
(cross-dataset test, embedding-size effect) are salvageable once a metric and
dataset are named.

**External sources found:**
- Rendle, libFM manual v1.4.2 (2014)
  (http://www.libfm.org/libfm-1.42.manual.pdf, verified) — a worked tutorial
  converting MovieLens-1M to libFM format and sweeping the embedding
  dimension across SGD/ALS/MCMC/adaptive-SGD learners, reporting RMSE for
  each; the direct template for fixing this section's "vary embedding size"
  exercise.
- Criteo Display Advertising Challenge (Kaggle, 2014)
  (https://www.kaggle.com/competitions/criteo-display-ad-challenge/overview,
  verified) and Avazu CTR Prediction Challenge (Kaggle, 2014)
  (https://www.kaggle.com/competitions/avazu-ctr-prediction/overview,
  verified) — the two datasets this section's own filler exercise names,
  both scored by log loss.
- Juan, Zhuang, Chin, Lin, "Field-aware Factorization Machines for CTR
  Prediction" (RecSys 2016)
  (https://www.csie.ntu.edu.tw/~cjlin/papers/ffm.pdf, verified) — a
  controlled comparison of a linear model, degree-2 polynomial model, plain
  FM, and field-aware FM on Criteo- and Avazu-style splits by log loss; the
  template for an FM-vs-FFM comparison exercise.
- "3 Idiots'" winning Criteo FFM pipeline, repo ycjuan/kaggle-2014-criteo
  (https://github.com/ycjuan/kaggle-2014-criteo, verified) and the
  companion Avazu solution, repo ycjuan/kaggle-avazu
  (https://github.com/ycjuan/kaggle-avazu, verified) — exact reproduction
  commands and reported log-loss scores for both competitions.
- LibFFM quick-start, repo ycjuan/libffm (https://github.com/ycjuan/libffm,
  verified) — ships a small GBDT-preprocessed Criteo subset for a fast local
  FM/FFM comparison.
- FINDING: Stanford CS246 was verified to have no FM/FFM homework at all
  (classical matrix factorization only).

**Proposed problem set** (6 problems, our reference format):
1. [short-code] **Embedding-Size Sweep with Log Loss.** Rewrite of the
   book's ex. 2, adapted from Rendle's libFM manual dim-sweep tutorial
   (overlap med; cite on adoption). Train at `num_factors` $k\in
   \{4,8,20,40\}$; report test log loss at each $k$ in a table, and state
   whether the pattern resembles mf.md's "diminishing returns past some
   $k$" or keeps improving.
1. [short-code] **Cross-Dataset Test, Concretely.** Rewrite of the book's
   ex. 1 (was a bare "can you test..." filler). Apply this section's
   `CTRDataset`/`FM` pipeline unchanged to MovieLens-100K ratings, binarized
   at a chosen threshold (e.g., rating $\ge 4\Rightarrow$ click $=1$) and
   encoded as (user, item) categorical fields only; report the test log
   loss, and state explicitly that no numeric comparison to mf.md's RMSE is
   valid since the two use different loss functions — only whether the
   model trains stably is comparable.
1. [conceptual] **FM Reduces to Matrix Factorization.** Grounded in this
   section's own claim that "this pairwise term contains the
   matrix-factorization score as a special case." Show algebraically that,
   when $\mathbf x$ is restricted to one one-hot user block and one one-hot
   item block, the FM pairwise sum collapses to exactly
   $\mathbf p_u^\top\mathbf q_i$ from mf.md; state what the FM's linear term
   $w_i$ contributes that mf.md's bias terms $b_u,b_i$ do not already cover.
   *Provenance:* original.
1. [conceptual] **Linear-Term Redundancy Check.** Companion to the previous
   problem. Under the same one-hot user/item restriction, determine whether
   the FM's intercept $w_0$ and linear weights $w_i$ are jointly
   identifiable from $b_u,b_i$ alone, or whether one parameterization
   strictly has more free parameters; state which term(s) would need to be
   fixed (e.g., $w_0=0$) to make the two models exactly equivalent.
   *Provenance:* original.
1. [short-code] **$O(kd)$ vs. Naive Pairwise Timing.** Grounded in this
   section's own complexity claim. Implement the naive $O(kd^2)$ double
   loop over feature pairs for a single training batch, and time it against
   this section's $O(kd)$ square-of-sum implementation while varying how
   many of the 34 fields are included; report the field count, if any,
   where the naive loop's wall-clock time exceeds the square-of-sum
   implementation's by an order of magnitude.
   *Provenance:* original.
1. [extended] **Field-Aware Extension.** Inspired by Juan et al., RecSys
   2016 (overlap low: field-aware embeddings are not introduced anywhere in
   this book; cite on adoption). Extend this section's `FM` class so each
   feature carries one latent vector *per other field* (field-aware
   embeddings) instead of one shared vector; retrain on this section's own
   CTR dataset, and report whether test log loss improves over plain FM and
   by how much the parameter count grows.

---

## chapter_recommender-systems/deepfm.md — Deep Factorization Machines

**Topic:** DeepFM — an FM branch and a deep MLP branch that share one field-
embedding table, summing their logits before the sigmoid.
**Current exercises:** 2; disposition: keep 0, rewrite 2, drop 0 — "vary MLP
structure" had no metric or range; "change dataset to Criteo" had a real
comparison target (FM) but no named metric.

**External sources found:**
- Guo, Tang, Ye, Ma, He, "DeepFM: A Factorization-Machine based Neural
  Network for CTR Prediction" (IJCAI 2017), arXiv:1703.04247
  (https://arxiv.org/abs/1703.04247, verified), §3.3 — the paper's own
  hyperparameter study varies activation function (ReLU vs. tanh), dropout
  rate, neurons per layer, number of layers, and network *shape* (constant,
  increasing, decreasing, diamond — constant wins in their Table 2/Figure);
  the direct template for fixing this section's "vary MLP structure"
  exercise.
- DeepCTR, repo shenweichen/DeepCTR
  (https://github.com/shenweichen/DeepCTR, verified),
  `examples/run_classification_criteo.py` — trains a DeepFM-style model on a
  13-dense/26-sparse Criteo sample with binary cross-entropy on an 80/20
  split. Correction: this library has no standalone class literally named
  `FM`; its comparable single-branch baselines are NFM/AFM/PNN/FNN/WDL/FwFM,
  so any FM-vs-something comparison exercise here should use this book's
  own `FM` class from fm.md rather than assume a DeepCTR class of the same
  name.
- FINDING: no university course was found with a documented DeepFM problem
  set anywhere — the paper's own ablation and library examples are the only
  citable material.

**Proposed problem set** (6 problems, our reference format):
1. [short-code] **Network-Shape Ablation.** Rewrite of the book's ex. 1,
   adapted from Guo et al.'s §3.3 hyperparameter study (overlap high; cite
   on adoption). Compare four `mlp_dims` shapes at roughly matched total
   width — constant $[20,20,20]$, increasing $[10,20,30]$, decreasing
   $[30,20,10]$, diamond $[10,30,10]$; report test log loss for each shape
   and state whether "constant" wins here as it does in the paper.
1. [short-code] **Depth and Dropout Sweep.** Adapted from Guo et al. §3.3
   (overlap med; cite on adoption). Vary number of layers $\in\{1,2,3,4\}$
   at fixed width 20, and separately vary dropout rate $\in
   \{0,0.1,0.3,0.5\}$ at the best depth found; report log loss for each and
   identify the point of diminishing or negative returns.
1. [short-code] **Criteo Cross-Dataset Comparison.** Rewrite of the book's
   ex. 2, now with a named metric. Change the dataset to a public Criteo
   sample and retrain both `FM` (fm.md) and `DeepFM` unchanged; report test
   log loss for both and state whether the FM-vs-DeepFM gap on Criteo has
   the same sign as the gap already observed on this section's anonymous
   advertising dataset.
1. [conceptual] **Branch Contribution Isolation.** Mirrors the GMF/MLP
   ablation pattern used in neumf.md, applied here. Derive what DeepFM's
   prediction reduces to if the MLP branch's logit is held at zero (pure
   FM) and if the FM branch's logit is held at zero (pure deep model);
   state which reduction is architecturally identical to a model already
   trained earlier in the chapter.
   *Provenance:* original.
1. [short-code] **Branch Contribution, Empirically.** Companion to the
   previous problem. Implement the two zeroed-branch variants by modifying
   `DeepFM.forward`; retrain each on this section's dataset and report log
   loss for FM-only, deep-only, and the full model, testing whether summing
   both branches' logits actually beats either branch alone.
   *Provenance:* original.
1. [conceptual] **Shared vs. Separate Embeddings.** Grounded in this
   section's own framing ("shares the embedding table between them").
   Argue what could go wrong — in terms of gradient conflict or a
   compromised shared representation — if the FM branch and the deep branch
   instead used two independently-learned embedding tables of the same
   size; state one experiment, implementable by adding a second
   `nn.Embedding` to this section's own code, that would test whether
   sharing helps or hurts.
   *Provenance:* original.
