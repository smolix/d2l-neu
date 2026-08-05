# Chapter overview: chapter_mdl-information-theory

Best source by far was MIT 6.441 (Polyanskiy, MIT OCW Spring 2016; the same lineage
as Polyanskiy & Wu's newer book) — its problem sets 1, 3, and 6 map almost
one-to-one onto our Gaussian-KL, data-processing/Pinsker, and hypothesis-testing
material. Cover & Thomas ch. 2 (48 exercises, verified via a course-notes mirror)
was the deepest well of short precise drills, and its Exercise 2.25 directly
undercuts our mutual-information section's own Venn-diagram figure — corroborated
independently by MacKay's Exercise 8.8, which uses the identical XOR construction.
Stanford EE376A/Stats376A homeworks (official site's PDFs are gone; used a former
TA's mirror) mostly duplicate Cover & Thomas but contributed one sharp AEP
strengthening. Confirmed gap, not a failure: none of the four sources has any
exercise tradition for optimal transport, MMD, kernel Stein discrepancy, f-GAN
duality, or InfoNCE/variational MI bounds — all post-2015 ML constructs invented
after these sources were written. Format finding: MacKay rates every exercise
1–3(+) for difficulty in its own header (e.g. "Exercise 2.26.[3,p.44]": 3 =
moderately hard, p.44 = worked solution), with a leading "." marking especially
recommended items — a real precedent for a difficulty marker distinct from our
[conceptual]/[short-code]/[extended] type tag. All three sections were already
reviewed defect-free, so dispositions lean "keep"; totals below.

Totals: 3 sections, 28 existing exercises audited, 24 proposed (8 per section);
keep 17, rewrite 5 (2 of those merge a pair into one slot), drop 6, new 4 adopted
(plus 2 more sourced-but-not-adopted, noted per section for completeness).

---

## chapter_mdl-information-theory/mdl-information-theory.md — Entropy, Cross-Entropy, and KL Divergence

**Topic:** Self-information, Shannon/differential entropy, KL divergence and Gibbs'
inequality, cross-entropy as a strictly proper scoring rule, the coding view (Kraft,
Shannon codes, arithmetic coding, AEP), rate–distortion and channel capacity
previews, perplexity, MDL, label smoothing, and knowledge distillation.

**Current exercises:** 9; disposition: keep 7, rewrite 0, drop 2 — the file was
reviewed defect-free ("the monkey/typesetter/language-model progression ... is a
clean worked comparison"; "no clarity issues found"), so drops are about density,
not quality: the perplexity base-invariance exercise is a fully mechanical restatement
of $\textrm{PPL}=\exp(\textrm{CE})$ already stated twice in the prose, and the
distillation-gradient exercise asks to redo a derivation the text has just walked
through step by step immediately above it.

**External sources found:**
- MacKay, *Information Theory, Inference, and Learning Algorithms* (free PDF,
  inference.org.uk), Ch. 2, Exercise 2.26 [3, p.44] — prove $D_{\textrm{KL}}(P\|Q)\ge0$
  (Gibbs' inequality), the same result our section proves via Jensen in-text —
  https://www.inference.org.uk/itprnn/book.pdf
- MacKay, *ITILA* — the book's own difficulty-rating convention: every exercise is
  headed "Exercise N.M.[d, p.xx]" where $d\in\{1,2,3,\ldots\}$ rates difficulty
  (1 = one minute, 2 = quarter hour, 3 = moderately hard) and "p.xx" points to a
  worked solution later in the book; a leading "." marks an especially recommended
  exercise — a structural finding for our own format, not a specific problem.
- Cover & Thomas, *Elements of Information Theory* 2nd ed., Ch. 2, Exercise 2.19
  "Infinite entropy" — exhibits a distribution on the positive integers with
  $H=\infty$, verified via a course-notes mirror:
  https://samfinlayson.com/files/notes/Cover_and_Thomas_ch2_entropy.pdf
- Cover & Thomas, Ex. 2.26 — a second, elementary proof of Gibbs' inequality via
  $\ln x \le x - 1$, a genuinely different route from our section's Jensen proof.
- MIT 6.441 (Polyanskiy), Problem Set 3, Problem 6 "Elias coding" — builds a
  universal (no-known-distribution-needed) prefix code for the integers and bounds
  its expected length by $2H(X)+2$ bits —
  https://ocw.mit.edu/courses/6-441-information-theory-spring-2016/2c357583ca114e1896369a6cdc209148_MIT6_441S16_problem_set3.pdf
- Stanford EE376A/Stats376A (Weissman), Homework 2 (Winter 2016–17 offering),
  Problem 6 "The Typical Set vs. High Probability Sets" — strengthens the AEP to
  show *any* set of probability $>1-\delta$ has $\gtrsim 2^{n(H-\epsilon)}$ elements,
  not only the typical set — https://web.stanford.edu/~kedart/ee376a_winter1617/hw2.pdf

**Proposed problem set** (8 problems, our reference format):

1. [short-code] **Card-Deck Surprise, Verified.** Compute the self-information of
   each of the four card-deck reports ($0$, $\ln 4$, $\ln 52$, $\ln 52!$ nats) with
   the section's `self_information` helper, then estimate $\ln 52!$ with Stirling's
   approximation and report the relative error against the exact value.
   *Provenance:* original (existing Exercise 1, kept)
1. [conceptual] **Uniform Maximizes Entropy, via Gibbs.** Show
   $D_{\textrm{KL}}(P\|U) = \log k - H(P)$ for the uniform $U$ on $k$ outcomes, and
   conclude $H(P)\le\log k$ with equality iff $P=U$ — a second, one-line proof of
   the proposition already proved via Jensen in the text.
   *Provenance:* original (existing Exercise 2, kept)
1. [short-code] **Entropy of Three Data Sources.** Compute the per-character entropy
   (nats and bits) of (a) a monkey typing uniformly among 44 keys, (b) a typesetter
   drawing uniformly from a 2,000-word vocabulary of average length 4.5 letters, and
   (c) a language model with per-word perplexity 15; explain in one sentence why
   the three numbers differ.
   *Provenance:* original (existing Exercise 3, kept; review flagged this as "a
   clean worked comparison")
1. [conceptual] **Deriving the Gaussian KL Closed Form.** Derive
   :eqref:`eq_mdl-gaussian_kl` from $E_{x\sim P}[\log p(x)-\log q(x)]$ using only
   $E_P[x]=\mu_1$ and $E_P[(x-\mu_1)^2]=\sigma_1^2$, and check it against the
   `gaussian_kl` code cell for one numeric pair.
   *Provenance:* original (existing Exercise 4, kept; this closed form is reused
   in the mutual-information section's Gaussian-MI proof, so the derivation is
   load-bearing)
1. [conceptual] **Self-Information Is Log, Uniquely.** Show that any continuous,
   decreasing $I(p)$ on $(0,1]$ with $I(1)=0$ and $I(p_1p_2)=I(p_1)+I(p_2)$ must
   equal $-c\log p$ for some constant $c>0$.
   *Provenance:* original (existing Exercise 5, kept)
1. [conceptual] **Two Strictly Proper Scoring Rules.** Prove from Gibbs' inequality
   that the log score is strictly proper, then show the Brier score is too, by
   proving its expected penalty exceeds the truthful reporter's by exactly
   $\|\mathbf q - \mathbf p\|^2$.
   *Provenance:* original (existing Exercise 6, kept; already cites
   :cite:`Gneiting.Raftery.2007`)
1. [conceptual] **A Source with Infinite Entropy.** Exhibit a distribution on the
   positive integers with $p(n)\propto 1/(n\log^2 n)$, verify numerically that a
   truncated-sum estimate of $H$ keeps growing rather than converging as the
   truncation point increases, and explain why this does not contradict the
   section's bound $H(P)\le\log k$.
   *Provenance:* adapted from Cover & Thomas, Exercise 2.19 "Infinite entropy"
   (overlap: medium; cite on adoption)
1. [short-code] **Universal Codes for the Integers.** Construct the Elias
   $\gamma$-code (prepend the length of a positive integer's binary representation,
   in unary), verify it is prefix-free by encoding and decoding a short stream by
   hand or in code, then show $E[\log X]\le H(X)$ for decreasing pmfs on the
   positive integers and use it to bound the $\gamma$-code's expected length by
   $2H(X)+2$ bits.
   *Provenance:* adapted from MIT 6.441 (Polyanskiy), Problem Set 3, Problem 6
   "Elias coding" (overlap: medium; cite on adoption)

---

## chapter_mdl-information-theory/mdl-divergences-distances.md — Divergences and Distances Between Distributions

**Topic:** f-divergences and their Fenchel-dual (f-GAN) view, forward vs. reverse
KL as mode-covering vs. mode-seeking, total variation and Pinsker's inequality,
integral probability metrics and MMD, optimal transport and Wasserstein-1, and the
score/Fisher-divergence/Stein-discrepancy family, closing with a divergence-to-objective map.

**Current exercises:** 9; disposition: keep 6, rewrite 2 (merged into 1 slot),
drop 1 — review found no defects ("ex6's two-sided Hellinger/TV sandwich is
unusually well hinted and closed with a numerical verification"), so this is a
density and topical-gap fix, not a quality fix: the two-point-masses exercise
(dropped) restates the disjoint-support/vanishing-gradient point the prose already
makes right next to the same figure, and the f-divergence-template exercise and the
Fenchel-conjugate exercise are natural halves of one multi-part problem.

**External sources found:**
- MIT 6.441 (Polyanskiy), Problem Set 6, Problems 1 and 3 — defines the
  hypothesis-testing region $R(P,Q)$ and proves
  $\textrm{TV}(P,Q)=\sup_\alpha(\alpha-\beta_\alpha(P,Q))$, formalizing the "best
  possible single-sample test" reading our section already states informally —
  https://ocw.mit.edu/courses/6-441-information-theory-spring-2016/149011ceca32c34a103e15a4176415fc_MIT6_441S16_problem_set6.pdf
- MIT 6.441, Problem Set 6, Problem 6 — a second-order (CLT) refinement of Stein's
  lemma, tying the exponential decay rate $-nD(P\|Q)$ of hypothesis-testing error
  to a Gaussian correction term (noted; not adopted, to keep the set at 8 — a good
  candidate if the book later wants a second [extended] alternative here).
- MIT 6.441, Problem Set 1, Problem 3 — proves the Pinsker–Csiszár inequality via
  the *data-processing inequality*, a genuinely different route from our section's
  log-sum-inequality proof —
  https://ocw.mit.edu/courses/6-441-information-theory-spring-2016/a914bdb7fc9444a78b55e9715e6e6279_MIT6_441S16_problem_set1.pdf
- Cover & Thomas, *Elements of Information Theory* 2nd ed. — checked chapter 2 and
  the hypothesis-testing/large-deviations material in later chapters; found no
  f-divergence, IPM, or optimal-transport exercise tradition at all. **Finding, not
  a failure:** that material postdates the book by decades.
- MacKay, *ITILA* — same absence; the book's only overlap is the relative-entropy
  exercise (2.26), already used under the sibling entropy section.

**Proposed problem set** (8 problems, our reference format):

1. [conceptual] **f-Divergence Generators and Their Duals.**
    1. Recover KL and reverse KL from the f-divergence template with
       $f(u)=u\log u$ and $f(u)=-\log u$; show $f(u)$ and $f(u)+c(u-1)$ define the
       same divergence for any constant $c$, and use that freedom to exhibit a
       nonnegative-everywhere generator for reverse KL.
    1. Derive the convex conjugate $f^*(t)=t+t^2/4$ of the $\chi^2$ generator over
       $u\in\mathbb R$, then redo it over the true domain $u\in(0,\infty)$ to get
       $f^*(t)=-1$ for $t\le -2$; explain why the larger $\mathbb R$-conjugate still
       gives a valid (if weaker) f-GAN bound, and verify numerically that the
       critic $T^\star=2(p/q-1)$ attains the exact divergence.
   *Provenance:* original (merges existing Exercises 1 and 2; both are
   cross-referenced by number elsewhere in the prose, so the merge should carry
   those references forward if adopted)
1. [conceptual] **Total Variation Is a Metric.** Prove the triangle inequality for
   TV, and classify which of KL, reverse KL, squared Hellinger, and $W_1$ are
   metrics; relate the Hellinger distance $H=\sqrt{H^2}$ to an $\ell_2$ norm of
   $\sqrt p-\sqrt q$.
   *Provenance:* original (existing Exercise 4, kept)
1. [conceptual] **Sharpness of Pinsker's Constant.** Expand $D_{\textrm{KL}}$ to
   second order in $\epsilon$ for coins with bias $\tfrac12$ vs. $\tfrac12+\epsilon$
   to show $D_{\textrm{KL}}=2\epsilon^2+O(\epsilon^4)$, concluding
   $\textrm{TV}/\sqrt{D_{\textrm{KL}}/2}\to1$; explain why the bound is loose for
   two distant unit-variance Gaussians.
   *Provenance:* original (existing Exercise 5, kept)
1. [short-code] **The Hellinger–TV Sandwich.** Prove
   $\tfrac12 H^2(P,Q)\le\textrm{TV}(P,Q)\le H(P,Q)\sqrt{1-\tfrac14H^2(P,Q)}$, then
   verify both sides over 10,000 random Dirichlet pairs on five outcomes, reporting
   how close each side comes to equality.
   *Provenance:* original (existing Exercise 6, kept; review singled this out as
   "unusually well hinted")
1. [short-code] **Wasserstein-1 as Sorted Differences.** Derive the quantile form
   $W_1=\int_0^1|F_P^{-1}(u)-F_Q^{-1}(u)|\,du$ from the CDF formula, show it equals
   the mean absolute difference of sorted samples for two equal-size empirical
   distributions, and verify numerically against the linear program on a small
   example.
   *Provenance:* original (existing Exercise 7, kept)
1. [short-code] **MMD in Closed Form.** For $P=\mathcal N(0,1)$,
   $Q=\mathcal N(\delta,1)$ and the RBF kernel with $\ell=1$, derive
   $\textrm{MMD}^2=\tfrac{2}{\sqrt3}(1-e^{-\delta^2/6})$ via the Gaussian
   moment-generating-function integral, evaluate at $\delta=0.5$, and compare
   against the section's Monte Carlo estimate.
   *Provenance:* original (existing Exercise 8, kept; cross-referenced by the
   prose, which quotes this exercise's $\approx0.047$ population value)
1. [conceptual] **Fisher Divergence and Stein's Identity by Hand.** Compute the
   Fisher divergence between two univariate Gaussians in closed form, confirm it
   reduces to $(\mu_1-\mu_2)^2/(2\sigma^4)$ for equal variances, then verify
   Stein's identity for $\mathcal N(0,1)$ with $f(x)=x$ by hand and name the
   classical fact about the standard Gaussian it recovers.
   *Provenance:* original (existing Exercise 9, kept)
1. [conceptual] **Total Variation as a Testing Advantage.** For a single sample
   drawn from $P$ or $Q$ with equal prior, define the Neyman–Pearson testing
   region $R(P,Q)=\{(P[Z{=}0],Q[Z{=}0]):P_{Z\mid X}\}$ and prove
   $\textrm{TV}(P,Q)=\sup_\alpha(\alpha-\beta_\alpha(P,Q))$, where $\beta_\alpha$
   is the smallest achievable false-accept rate at true-accept rate $\alpha$. Show
   this reduces to the Bayes-error identity $P_e=\tfrac12(1-\textrm{TV}(P,Q))$ the
   section already states informally, and use both to explain why Pinsker's
   inequality is the right tool for certifying indistinguishability but the wrong
   one for computing an exact error probability.
   *Provenance:* adapted from MIT 6.441 (Polyanskiy), Problem Set 6, Problems 1 and
   3 (overlap: medium; cite on adoption)

---

## chapter_mdl-information-theory/mdl-mutual-information.md — Mutual Information and Representation Learning

**Topic:** Mutual information as KL-from-independence, the data-processing
inequality, why MI is statistically hard to estimate (the $O(\log N)$ ceiling and
the InfoNCE cap), variational lower bounds (Barber–Agakov, Donsker–Varadhan/MINE,
NWJ, InfoNCE), the information bottleneck, and Fano's inequality.

**Current exercises:** 10; disposition: keep 4, rewrite 3 (one absorbs a dropped
exercise, one is retagged as the section's extended problem), drop 3 — review
found no defects here either ("ex8's 'what does the optimal critic look like?' is
a bounded follow-up ... not open-ended"), so again this is about making room for a
genuine gap (the Venn-diagram caution) rather than fixing anything broken.

**External sources found:**
- Cover & Thomas, *Elements of Information Theory* 2nd ed., Ch. 2, Exercise 2.25
  "Venn diagrams" — proves 3-variable mutual information (interaction information)
  can be *negative* and gives two symmetric entropy identities for it — a direct
  critique of the section's own two-circle Venn-diagram figure
  (:numref:`fig_mdl-mi-overlap`) —
  https://samfinlayson.com/files/notes/Cover_and_Thomas_ch2_entropy.pdf
- MacKay, *ITILA*, Ch. 8, Exercise 8.8 — independently makes the identical "the
  Venn diagram is misleading" argument, using the identical XOR ensemble as our
  own existing Synergy exercise and as Cover & Thomas's 2.25 —
  https://www.inference.org.uk/itprnn/book.pdf
- MacKay, *ITILA*, Ch. 8, Exercise 8.9 — proves the data-processing theorem via a
  $w\to d\to r$ Markov chain, the same structure (and nearly the same proof) as
  our own existing Data-processing exercise; corroboration for the proof strategy,
  not a new citation.
- MIT 6.441 (Polyanskiy), Problem Set 3, Problem 2 — for jointly Gaussian
  $(A,B,C)$, $I(A;C)=I(B;C)=0\Rightarrow I(A,B;C)=0$, plus a discrete
  counterexample under a positivity condition —
  https://ocw.mit.edu/courses/6-441-information-theory-spring-2016/2c357583ca114e1896369a6cdc209148_MIT6_441S16_problem_set3.pdf
- MIT 6.441, Problem Set 1, Problem 6 — mutual information of a uniform
  distribution on a disk, a genuinely non-Gaussian continuous worked example
  (noted; not adopted, to hold the proposed set at 8) —
  https://ocw.mit.edu/courses/6-441-information-theory-spring-2016/a914bdb7fc9444a78b55e9715e6e6279_MIT6_441S16_problem_set1.pdf
- Stanford EE376A, Homework 0, Problem 4(d) — construct a random vector whose
  every proper subset of coordinates is jointly Gaussian but the whole vector is
  not, a continuous-case cousin of the zero-correlation-but-dependent exercise
  (noted; not adopted) — https://web.stanford.edu/~kedart/ee376a/WWW/files/homework/hw0/hw0.pdf

**Proposed problem set** (8 problems, our reference format):

1. [short-code] **Mutual Information by Hand.** Compute $I(X;Y)$ for the joint pmf
   $\left(\begin{smallmatrix}0.3&0.2\\0.1&0.4\end{smallmatrix}\right)$: form the
   marginals, evaluate pointwise mutual information at each cell, average, and
   check against the section's `mutual_information` function.
   *Provenance:* original (existing Exercise 2, kept)
1. [conceptual] **Zero Correlation Is Not Independence.** Construct a pair of
   discrete random variables with zero correlation but positive mutual
   information, verify both numerically, then prove that for jointly Gaussian
   variables zero correlation *does* imply $I=0$.
   *Provenance:* original (existing Exercise 4, kept)
1. [conceptual] **The Data-Processing Inequality, Both Directions.** For a Markov
   chain $X\to Y\to Z$ with a deterministic $Z=g(Y)$, show $I(X;g(Y))\le I(X;Y)$
   with equality when $g$ is invertible; give a noninvertible $g$ for which
   equality still holds; and decide whether a randomized $g$ can ever increase
   mutual information.
   *Provenance:* original (existing Exercise 5, kept; MacKay, *ITILA*, Exercise
   8.9 proves the same theorem via the same Markov-chain technique for a
   "world/data/processed-data" chain — corroboration, not a new citation)
1. [conceptual] **Synergy: When Conditioning Creates Dependence.** For independent
   fair bits $X,Y$ and $Z=X\oplus Y$, show $I(X;Y)=0$ but $I(X;Y\mid Z)=\ln2$, and
   explain why this does not contradict the data-processing inequality.
   *Provenance:* original (existing Exercise 6, kept; MacKay, *ITILA*, Exercise 8.7
   poses the identical XOR construction independently — good corroborating
   evidence this is the canonical example; cite MacKay on adoption if a second
   reference is wanted)
1. [short-code] **The InfoNCE Ceiling, Quantitatively.** Show
   $\mathcal L_{\textrm{NCE}}\ge0$ from the definition, conclude
   $\hat I_{\textrm{NCE}}\le\log N$, and explain in one paragraph why increasing
   the batch size tightens the bound. Then compute the smallest $\rho$ for which a
   Gaussian pair's true mutual information could exceed a batch ceiling of
   $\log 256$.
   *Provenance:* original (rewrites existing Exercise 8, absorbing the "does
   $I(X;Y)$ exceed the $N=256$ ceiling" question from existing Exercise 3, whose
   other half — deriving the Gaussian MI formula by direct integration — duplicated
   the entropy-identity proof already given in-text)
1. [extended] **Design a Mutual-Information Sanity Check.** Using Fano's
   inequality and the $\log N$ ceiling, estimate the smallest contrastive batch
   size that could in principle certify enough mutual information for 1% error on
   a balanced 10,000-class problem. Then design and run a numerical sanity check
   for any MI estimator of your choice (histogram, KSG, or InfoNCE) against a
   known-MI Gaussian pair, reporting where the estimate starts to diverge from the
   truth.
   *Provenance:* original (existing Exercise 10, kept and re-tagged as the
   section's one extended/project-scale problem — its "design and run a sanity
   check" clause already asks for exactly that scope)
1. [conceptual] **Mutual Information Among Three Variables Can Be Negative.**
   Define the interaction information $I(X;Y;Z)=I(X;Y)-I(X;Y\mid Z)$, prove the
   two identities expressing it symmetrically in joint and pairwise entropies, and
   exhibit a triple $(X,Y,Z)$ for which it is negative (check whether the XOR
   triple from the Synergy exercise above is one such triple). Conclude that the
   two-circle Venn-diagram picture used earlier in this section
   (:numref:`fig_mdl-mi-overlap`) has no valid three-variable extension.
   *Provenance:* adapted from Cover & Thomas, Exercise 2.25 "Venn diagrams"
   (overlap: medium; cite on adoption); MacKay, *ITILA*, Exercise 8.8 makes the
   same point independently with the same XOR ensemble — two textbooks converging
   on one warning is a strong signal this belongs in the book
1. [short-code] **Pairwise-Independent Information Need Not Add.** For jointly
   Gaussian $(A,B,C)$, show $I(A;C)=I(B;C)=0$ implies $I(A,B;C)=0$; then find a
   discrete counterexample where $I(A;C)=I(B;C)=0$ but $I(A,B;C)>0$, under the
   positivity condition $P_{ABC}(a,b,c)>0$ for all $a,b,c$. Deliverable: the
   Gaussian proof plus an explicit discrete joint distribution violating it,
   checked numerically.
   *Provenance:* adapted from MIT 6.441 (Polyanskiy), Problem Set 3, Problem 2
   (overlap: medium; cite on adoption)

**Dropped, with reasons** (not carried into the proposed set above):
- Existing Exercise 1 (discrete joint/conditional entropy bounds
  $\max\{H(X),H(Y)\}\le H(X,Y)\le H(X)+H(Y)$, and where the analogue fails for
  differential entropy) — a generic drill whose content is already stated as fact
  in this section's and the entropy section's prose; low incremental value next to
  the new interaction-information problem, which uses the same machinery to make a
  sharper, less obvious point.
- Existing Exercise 7 (derive the Barber–Agakov bound from Gibbs) — the full proof
  is given step by step in the text immediately above the exercise; the new
  pairwise-independence problem exercises the same "chain rule + Gibbs" toolkit on
  a genuinely open question instead.
- Existing Exercise 9 (derive the Gaussian information-bottleneck critical
  $\beta^*=1/\rho^2$) — the in-text experiment already plots the frontier and
  visibly shows this collapse; narrow next to the section's one extended slot,
  which already asks the reader to engage with the estimation-limits material at
  project scale.
