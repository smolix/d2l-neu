# Review — chapter_linear-classification/softmax-regression.md  (§4.1 "Softmax Regression")

**Role in the chapter:** The conceptual opener of chapter 4. It defines the
classification setting, the linear (logit) model, the softmax map, the
cross-entropy / negative-log-likelihood loss, the clean softmax gradient, and
the information-theoretic reading of cross-entropy. It is pure theory — there
are **no code cells** (the implementations live in the scratch/concise
siblings), so review dimension 3 is largely N/A here.

**Verdict:** Strong, well-written foundations prose that is close to the bar:
the probabilistic motivation, the softmax-as-Gibbs-measure history, and the
`softmax−y` gradient identity are genuinely excellent and at CS229 quality. Two
things hold it back from "best-in-class." First, a **scope-discipline
violation**: the `## Information Theory Basics` section reteaches entropy,
surprisal, and cross-entropy — material this book now owns, and treats far
better, in `chapter_mdl-information-theory` (`sec_mdl-information_theory`),
which the chapter cover already references. Second, two **in-scope gaps the task
explicitly flags**: the softmax→logistic special case is never shown, and the
numerical-stability point is hand-waved to "frameworks take care of this" with
no `:numref:` to the log-sum-exp derivation in the concise sibling. The single
highest-value change is to **compress the information-theory section to one
paragraph + a forward pointer**, reclaiming the space for the logistic special
case and a tightened gradient derivation.

**Grade:** **B+.** Assignable today and better than most treatments, but the
duplicated information-theory tutorial reads as pre-`chapter_mdl-*` legacy, and
the two flagged in-scope omissions (logistic special case; log-sum-exp pointer)
are exactly the elegant touches a top course expects.

**Top priorities (ranked):**
1. **[P1, coverage/scope]** Collapse `## Information Theory Basics` (entropy,
   surprisal, cross-entropy revisited, ll. 395–465) into ~1 paragraph that
   states the cross-entropy = expected-surprisal / code-length reading and
   **forward-points to `:numref:sec_mdl-information_theory`**. That chapter has
   the superior version (card-deck hook, nats-throughout convention, Kraft/
   Shannon coding argument, perplexity, label smoothing). Duplicating a weaker
   copy here is the scope-map's textbook anti-pattern.
2. **[P1, coverage]** Add the **softmax→logistic special case** (the task names
   this as in-scope and "worth getting elegant"): show that for $q=2$,
   $\hat y_1 = \sigma(o_1 - o_2)$ with $\sigma$ the logistic sigmoid, so binary
   logistic regression is softmax regression with one redundant logit removed —
   tying back to the "slightly redundant parametrization" remark at ll. 119–123.
3. **[P1, correctness/coordination]** Replace the vague "Deep learning
   frameworks take care of this automatically" (l. 262) with a one-sentence
   statement of the overflow/underflow risk **and a `:numref:` to
   `subsec_softmax-implementation-revisited`** (concise sibling), which derives
   the log-sum-exp / fused-loss fix. Coordinate, don't duplicate — but do point.
4. **[P1, currency]** Add one sentence + forward pointer on **calibration**:
   softmax outputs are probabilities by construction but are *not* well-
   calibrated in practice (Guo et al. 2017, already cited on the cover);
   temperature scaling — the same $T$ from the Boltzmann paragraph — fixes it.
5. **[P2, teaching]** Tighten the gradient derivation (ll. 343–369) to make the
   log-partition / exponential-family thread explicit *where it is used*, not
   only in the exercises.

---

## 1. Coverage

### Add

- **Softmax → logistic (binary) special case — MISSING, in-scope (P1).** The
  task brief calls this out specifically, and it is a glaring omission for a
  foundations chapter: nowhere does the text connect softmax to the logistic
  sigmoid that every reader has met. It also pays off the unresolved promissory
  note at ll. 119–123 ("Strictly speaking, we only need one fewer … but for
  reasons of symmetry we use a slightly redundant parametrization"). Proposed
  insertion after :eqref:`eq_softmax_y_and_o` (~l. 215), 4–5 lines:

  > For two classes the redundancy is explicit. Writing $o = o_1 - o_2$,
  > $$\hat y_1 = \frac{\exp o_1}{\exp o_1 + \exp o_2} = \frac{1}{1+\exp(-o)} = \sigma(o),$$
  > the **logistic sigmoid**. Binary logistic regression is thus softmax
  > regression with the redundant logit removed; only differences of logits are
  > identifiable, which is why fixing $o_q\equiv 0$ (or any constant shift) leaves
  > $\hat{\mathbf y}$ unchanged — the translation-invariance the exercises revisit.

  This single addition unifies §4.1 with the binary classifier every reader
  already knows, and motivates the identifiability remark with a concrete
  picture. (Forward-point only; full logistic-regression development isn't
  needed here.)

- **A numerical-stability forward pointer (P1).** Lines 259–262 raise the
  overflow/underflow caveat and then dismiss it ("Deep learning frameworks take
  care of this automatically"). The actual fix — subtracting $\max_k o_k$ and
  fusing softmax+log into log-sum-exp — is derived beautifully in the concise
  sibling's `subsec_softmax-implementation-revisited`. Per the scope watch-point
  the *implementation* belongs there, but this file should not leave the reader
  with magic; add: "…can cause numerical overflow or underflow; the standard
  log-sum-exp fix and the fused softmax-cross-entropy loss are derived in
  :numref:`subsec_softmax-implementation-revisited`." This is the in-scope
  correctness hook the task asks for, done by reference.

- **Calibration, one sentence + pointer (P1, currency).** Written in 2021, the
  text presents $\hat{\mathbf y}$ as "the (estimated) conditional probabilities"
  (l. 279) with no caveat. The modern, well-established result (Guo, Pleiss,
  Sun & Weinberger, *On Calibration of Modern Neural Networks*, 2017 — **already
  cited on the chapter cover**) is that softmax confidences are systematically
  **miscalibrated**, and temperature scaling (literally the $T$ in the Boltzmann
  paragraph at ll. 226–233) recalibrates them. Add one sentence near l. 281 and
  a forward pointer. This also turns the otherwise-decorative temperature
  digression into something load-bearing.

- **Make the exponential-family/log-partition thread explicit (P2).** Line 368
  asserts "In any exponential family model, the gradients of the log-likelihood
  are given by precisely this term," but the supporting fact —
  $\partial_{o_j}\log\sum_k\exp o_k=\mathrm{softmax}(\mathbf o)_j$, i.e. the
  softmax is the gradient of the log-partition function $g(\mathbf o)=\log\sum_k
  \exp o_k$ — is left to Exercise 6. Since the loss was just rewritten as
  $l = g(\mathbf o) - \mathbf y^\top\mathbf o$ (ll. 343–349), one extra line
  noting "$\nabla_{\mathbf o}g=\mathrm{softmax}(\mathbf o)$, so $\nabla_{\mathbf o}l
  =\mathrm{softmax}(\mathbf o)-\mathbf y$" makes the derivation self-contained and
  *more* elegant than CS229's index-pushing version. Forward-point the convexity
  (Hessian = covariance) to the Optimization part / Exercise 6.

### Remove / trim

- **`## Information Theory Basics` (ll. 395–465) — trim hard (P1).** This is the
  central scope problem. The three subsections (Entropy, Surprisal,
  Cross-Entropy Revisited) are a self-described "survival guide" to information
  theory. But the book now has a **dedicated, superior** treatment at
  `chapter_mdl-information-theory/mdl-information-theory.md`
  (`sec_mdl-information_theory`), which the chapter cover already links and which
  covers exactly this (self-information/surprisal, entropy, cross-entropy, KL),
  *plus* the coding argument (Kraft/Shannon), the nats-vs-bits convention,
  perplexity, label smoothing, and distillation — and opens with a better hook
  (the shuffled-deck thought experiment) than this file's "boring data stream."
  Keeping a thinner duplicate here is precisely what the scope map warns against.
  **Proposed replacement** for ll. 386–465 (everything from "We can demystify
  the name…" through the end of the IT section): a single paragraph —

  > **Why "cross-entropy"?** The name comes from information theory. The
  > *entropy* $H[P]=\sum_j -P(j)\log P(j)$ is the expected surprisal — the
  > average number of nats needed to encode draws from $P$ when you know $P$;
  > the *cross-entropy* $H(P,Q)=\sum_j -P(j)\log Q(j)$ is the cost when you
  > instead encode them under your model $Q$, and it is minimized exactly when
  > $Q=P$. Minimizing :eqref:`eq_l_cross_entropy` therefore does two equivalent
  > things: it maximizes the likelihood of the labels, and it minimizes the
  > extra bits your predictions waste relative to the truth. We develop entropy,
  > cross-entropy, and the Kullback–Leibler divergence — and the coding argument
  > behind the "bits" language — in :numref:`sec_mdl-information_theory`.

  This keeps the MLE↔code-length duality that makes §4.1 sing (the genuinely
  good payload of the current ll. 462–465), cites the right place, and removes
  ~55 lines of out-of-scope tutorial. Net: tighter file, no lost content
  (it moves to / already exists in the chapter that owns it).

- **The probit-model paragraph (ll. 177–188)** is a nice "softmax isn't the only
  choice" aside and can **stay**, but trim the Fechner-1860 historical clause if
  space is tight — it's lower-value than the Boltzmann paragraph that follows.

### Reorder / restructure

- **Top-level spine.** After trimming, the `##` sections become **Classification
  → Loss Function → Summary and Discussion** (3 sections), with the
  information-theory material folded into Loss Function as a closing paragraph.
  That lands inside the guide's 3–5 ideal and removes the slightly odd promotion
  of "Information Theory Basics" to a peer of "Loss Function." (Currently the
  file has 4 `##`; the IT section is the weakest peer.)
- The softmax→logistic addition belongs in `### The Softmax`
  (`subsec_softmax_operation`), right after the softmax definition, where the
  redundant-parametrization remark is still fresh.

## 2. Teaching quality

### Structure & flow

The within-section flow is good: one-hot encoding → linear/logit model →
softmax (with the "argmax is order-preserving" observation, l. 206–215, which is
a nice operational point) → vectorization → MLE → cross-entropy → gradient. The
hook (regression's *how much?* vs classification's *which?*, ll. 9–36) is
solid and on-brand, though it runs long; the bulleted examples (ll. 40–67) plus
the multi-label aside (ll. 58–67) could lose a few lines. The single weak join
is the abrupt promotion of information theory to a top-level section — fixed by
the trim above.

### Figures

- **`:numref:fig_softmaxreg`** ("Softmax regression is a single-layer neural
  network," l. 145, `img/softmaxreg.svg`) — the only figure. It is a clean,
  hand-authored SVG (the fully-connected $x_{1:4}\!\to\!o_{1:3}$ diagram),
  renders correctly (confirmed on the served page as Fig. 4.1.1), is properly
  captioned and `:numref:`-referenced, and earns its place. No inline
  figure-drawing matplotlib (there is no code in this file at all), so no
  house-style violation. **Keep.**
- **Missing figure that would unlock an idea (P2):** a small schematic of the
  softmax map from logit space onto the **probability simplex** — three logits
  $\mathbf o\in\mathbb R^3$ mapping into the 2-simplex, with a temperature slider
  showing the distribution sharpening toward a vertex as $T\to0$ and flattening
  toward the centroid as $T\to\infty$. This would make the "squishing" language
  (l. 175), the argmax-preservation point, and the Boltzmann-temperature
  paragraph concrete in one picture. If added, it must be a committed generator
  (`tools/gen_mdl_*`) → `img/`, not inline. (Lower priority than the prose/scope
  fixes; flag as a nice-to-have.)

### Prose & clarity

Generally excellent — this is among the better-written theory files. Specific
points:

- **ll. 164–174 (the two softmax-motivation bullets).** The second bullet —
  "There is no guarantee that the outputs $o_i$ are even nonnegative, even if
  their outputs sum up to 1, or that they do not exceed 1" — is tangled ("their
  outputs" is ambiguous; the "even if … sum up to 1" qualifier muddies the point).
  Tighten to: "There is no guarantee the $o_i$ are nonnegative, nor that they lie
  in $[0,1]$." The mansion example that follows (ll. 169–174) is vivid; keep it.
- **l. 262 "Deep learning frameworks take care of this automatically."** Reads as
  hand-waving in a foundations text; replace with the forward pointer (Add §3
  above).
- **ll. 312–331 (loss is bounded below by 0).** This is a careful, correct, and
  pedagogically valuable passage (the "can never reach 0 for finite weights"
  observation is exactly the kind of rigor the bar wants). **Keep**, but it can
  shed ~2 lines.
- **Notation nit:** the linear-model block (ll. 129–135) writes weights as
  $w_{11},\dots$ giving a $3\times4$ matrix (l. 151), while the vectorization
  block (ll. 244–250) switches to $\mathbf W\in\mathbb R^{d\times q}$ (so the
  *transpose* layout) and $\mathbf O=\mathbf X\mathbf W$. Both are internally
  consistent, but a reader tracking shapes hits a silent transpose. One
  half-sentence acknowledging "(now with examples in rows, so $\mathbf W$ is
  $d\times q$)" would prevent confusion.

### Exercises

The exercise set is a genuine strength — among the best in these chapters and
worthy of a top problem set. Ex. 1 (2nd derivative = variance), Ex. 5
(`RealSoftMax` bounds and the $\lambda\to\infty\Rightarrow\max$ limit), and Ex. 6
(log-partition convexity + translation equivariance + numerical stability) are
excellent and directly support the in-text claims. Suggestions:

- **Add a softmax→logistic exercise** to lock in the new in-text special case:
  "Show that for $q=2$ classes the softmax reduces to the logistic sigmoid of
  the logit difference, and that adding a constant to both logits leaves
  $\hat{\mathbf y}$ unchanged. Conclude that softmax regression has one redundant
  degree of freedom per example." (Mechanical, builds directly on the new
  paragraph.)
- **Ex. 7 (temperature, ll. 544–547)** is good; consider adding a sub-part
  connecting it to calibration: "Argue that scaling all logits by $1/T$
  (temperature scaling) cannot change the argmax, hence not the accuracy — yet
  it changes the cross-entropy. Why is this the basis of post-hoc calibration?"
- Exercises 2–3 (binary/ternary codes, PAM-3) are tied to the information-theory
  section being trimmed. They are good standalone puzzles and can **stay** even
  after the in-text IT trim — but consider prefacing the block with a pointer to
  `:numref:sec_mdl-information_theory` so the coding background is sourced.

## 3. Code & examples

**Not applicable — this file contains no code cells.** Confirmed: no
`outputs/<fw>/.../softmax-regression.json` manifests exist for any framework,
and the served page shows no executable cells. All four frameworks' softmax-
regression *code* lives in `softmax-regression-scratch.md` and
`softmax-regression-concise.md`; review those separately. The one code-adjacent
correctness point that touches this file — numerical stability of
softmax+cross-entropy — is handled by the forward-pointer recommendation in
§1 (Add), and the actual log-sum-exp implementation is correctly located in
`subsec_softmax-implementation-revisited` (concise sibling), which I verified
derives it cleanly (subtract $\max_k o_k$; fuse into the loss).

### Cross-framework consistency & d2l conventions

N/A for code. One **cross-file convention issue** worth surfacing to the
overview: the **chapter cover** (`chapter_linear-classification/index.md`,
l. 36) links the information-theory background to the **retired legacy URL**
`d2l.ai/chapter_appendix-mathematics-for-deep-learning/information-theory.html`.
Per the project's current status (the legacy math appendix was retired and
replaced by `chapter_mdl-information-theory`), that link is stale and should
point at this book's own `sec_mdl-information_theory`. Not strictly this file,
but it is the same dangling reference this file should be establishing — fix
both together.

## 4. Prioritized change list

| # | Sev | Dimension | Change (specific, actionable) | Effort |
|---|-----|-----------|-------------------------------|--------|
| 1 | P1 | coverage/scope | Trim `## Information Theory Basics` (ll. 395–465) to one paragraph + `:numref:sec_mdl-information_theory`; keep the MLE↔code-length duality (draft provided in §1 Remove). | M |
| 2 | P1 | coverage | Add softmax→logistic special case after :eqref:`eq_softmax_y_and_o` (draft provided); resolves the redundant-parametrization remark. | S |
| 3 | P1 | correctness | Replace "frameworks take care of this automatically" (l. 262) with the overflow/underflow statement + `:numref:subsec_softmax-implementation-revisited`. | S |
| 4 | P1 | currency | Add calibration caveat + pointer (Guo et al. 2017, on cover) near l. 281; connect temperature scaling to the Boltzmann $T$. | S |
| 5 | P1 | cross-file | Fix the chapter cover's stale info-theory link (`index.md` l. 36) → `sec_mdl-information_theory`. | S |
| 6 | P2 | teaching | Make $\mathrm{softmax}=\nabla g$ (log-partition) explicit in the gradient derivation (ll. 343–369); one line. | S |
| 7 | P2 | teaching | Rewrite the tangled second softmax-motivation bullet (ll. 164–166); add a half-sentence flagging the $\mathbf W$ shape transpose (ll. 244–250). | S |
| 8 | P2 | exercises | Add softmax→logistic exercise and a temperature↔calibration sub-part. | S |
| 9 | P2 | teaching | Optional: add a probability-simplex / temperature schematic figure (committed generator, not inline). | L |

## 5. Keep — what is already excellent (do not lose this)

- **The softmax gradient identity** $\partial_{o_j} l = \mathrm{softmax}(\mathbf o)_j - y_j$
  and its framing as "the difference between what the model predicted and what
  happened, just like regression" (ll. 351–370). This is exactly as clean as
  CS229's derivation and is the conceptual heart of the section — protect it.
- **The MLE derivation of cross-entropy** (ll. 276–330): factorization under
  independence → negative log-likelihood → per-example cross-entropy, plus the
  careful "loss is bounded below by 0 and unattainable for finite weights"
  discussion. Rigorous and well-paced.
- **The Gibbs/Boltzmann history of the softmax** (ll. 218–237): a memorable,
  accurate hook ($\exp(-E/kT)$, "energy equates to error," energy-based models)
  that most textbooks omit. Keep it — and make it earn its keep by tying $T$ to
  calibration (change #4).
- **The exercise set** — Ex. 1, 5, 6 in particular are top-course quality and
  reinforce the in-text claims (variance = 2nd derivative; log-sum-exp convexity
  and stability). Keep nearly all of it.
- **`fig_softmaxreg`** — clean, correctly referenced, no inline drawing code.
- **The two-readings summary** (ll. 462–465: MLE *and* minimum-bits) — the one
  sentence of the information-theory section that must survive the trim.
