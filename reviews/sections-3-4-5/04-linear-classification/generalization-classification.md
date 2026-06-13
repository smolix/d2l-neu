# Review — chapter_linear-classification/generalization-classification.md  (§4.6 "Generalization in Classification")

**Role in the chapter:** This is the **learning-theory** capstone of the linear-classification chapter. It is the book's dedicated home for (a) test-set statistics — how trustworthy a held-out error estimate is and how big a test set must be (CLT + Bernoulli variance + Hoeffding); (b) the danger of *reusing* a test set (multiple testing, adaptive overfitting); and (c) the *a priori* generalization question — uniform convergence, the union-bound intuition, and VC dimension — closing with the honest admission that classical capacity bounds are vacuous for deep nets, handing off to §5.5 `generalization-deep.md`.

**Verdict:** This is genuinely good, well-paced expository prose — intuition-led, honest about what theory can and cannot deliver, and it carves a clean division of labor with the regression `generalization.md` (which owns bias–variance/model-selection) and `generalization-deep.md` (which owns double descent/Zhang-2021/Rademacher). It is **not yet best-in-class**, for four concrete reasons: (1) the central VC-bound equation silently switches to a *different notation system* ($R[p,f]$/$R_\textrm{emp}$) from the $\epsilon(f)$ notation the whole rest of the file uses; (2) a factual non-sequitur in the variance argument ("cannot be greater than 1"); (3) the Hoeffding numeric example conflates a one-sided bound with a two-sided 95% confidence claim; (4) it is **pure prose with not a single figure or worked numerical cell** — top programs (CS229, CMU 11-785) anchor exactly this material with a picture (a shattering diagram) and a concrete number you can recompute. The single highest-value change is **fixing the notation break in eq. 4.6.4** so the file's keystone result is consistent. Currency lift: a one-paragraph Rademacher mention + a Zhang-2021 citation in the vacuous-bounds passage.

**Grade:** **B+.** Assignable today as a reading with a TA's errata note; the notation break and the one-sided/two-sided slip are exactly the kind of thing a careful student at a top program flags, and the absence of *any* figure or recomputable number leaves it below the chapter's own gold-standard (calculus/linear-algebra) bar. Fixing the four P0/P1 items moves it to A−/A.

**Top priorities (ranked):**
1. **[P1] GC-1** — Fix the notation break in the VC bound (eq. 4.6.4): re-express in the file's own $\epsilon(f)$/$\epsilon_\mathcal{S}$ notation (mechanical, but load-bearing — it's the chapter's keystone equation).
2. **[P1] GC-2** — Add a *shattering* figure for VC dimension (3 points shattered by a line in 2D, 4 cannot), referenced by `:numref:`; the VC paragraph currently *describes* this picture in words ("a line can assign any possible labeling to three points… but not to four") without showing it.
3. **[P1] GC-3** — Fix the one-sided/two-sided Hoeffding inconsistency: either state the two-sided bound (factor of 2) to match the two-sided "$\pm 0.01$, 95%" prose, or relabel the claim as one-sided. Currently the "15,000" figure is the one-sided number under a two-sided framing.
4. **[P1] GC-4** — Sharpen the forward-pointer: §73 and §566 point to the whole MLP *part* (`:numref:`chap_perceptrons``); point instead to the specific section `:numref:`sec_generalization_deep`` that actually owns "why classical bounds fail / double descent / Rademacher," and add the Zhang-2021 citation (`:cite:`zhang2021understanding``) to the vacuous-bounds sentence.
5. **[P2] GC-5** — Fix the factual non-sequitur "we know that it cannot be greater than $1$" → "$\tfrac12$"-based maximization argument.
6. **[P2] GC-6** — One-paragraph mention of **Rademacher complexity** as the data-dependent modern complement to VC (in-scope-light per the scope map), forward-pointing for depth.
7. **[P2] GC-7** — Add a short worked numerical demo (the recomputable Hoeffding/CLT sample-size table) — either as the one teaching code cell this file lacks, or as an inline worked example — so the "$\mathcal O(1/\sqrt n)$" claim becomes something the reader can verify.
8. **[P2] GC-8** — Tighten the adaptive-overfitting passage with the *empirical* counterpoint (Recht et al. reproductions), balancing Dwork's worst-case theory with the reassuring practice.

---

## 1. Coverage

### Add

- **(GC-2, P1) A figure for VC dimension / shattering.** The file *talks through* the canonical picture — "It is easy to see that a line can assign any possible labeling to three points in two dimensions, but not to four" (lines 483–485) — but shows nothing. This is the one place in the file where a picture is worth a thousand words, and it is exactly the figure CS229's notes, Mohri, and Shalev-Shwartz & Ben-David all draw. A clean schematic (three points in general position with the $2^3=8$ labelings all realizable by a halfplane, contrasted with the XOR-style 4-point configuration that no line can shatter) makes "shattering" click instantly. Per house rules this must be a *generated* SVG (`img/mdl-…svg`), not inline matplotlib. See §4 for the spec.

- **(GC-6, P2) Rademacher complexity — one paragraph + forward-point.** The scope map lists Rademacher as explicitly *in-scope-light* here ("introductory-but-rigorous; forward-point external refs for full proofs"). The file currently mentions only VC, then waves at "numerous alternative complexity measures" (line 551) citing `boucheron2005theory`. A single paragraph naming Rademacher complexity — *data-dependent* (it measures how well the class can fit random sign noise *on your actual sample*), hence often far tighter than the distribution-free, combinatorial VC dimension, and the standard modern vehicle for margin/norm-based bounds — would modernize the section without importing a proof. It should forward-point to `sec_generalization_deep` (which already references Rademacher, line 163 there) for the deep-learning angle. This is the "modern complement to VC" the currency lens calls for.

- **(GC-4, P1 / currency) Cite Zhang et al. 2021 in the vacuous-bounds passage.** The Summary's honest punchline — "they turn out to be powerless… for explaining why deep neural networks generalize… can easily assign random labels to large collections of points… generalize better when larger and deeper, despite incurring higher VC dimensions" (lines 555–565) — is *precisely* the thesis of "Understanding deep learning (still) requires rethinking generalization." The key `zhang2021understanding` is **already in `d2l.bib`** (line 3959) and is already cited by `generalization-deep.md`. Citing it here is a one-token mechanical add that turns an unsourced claim into a grounded one and links the two generalization sections. (This is the single most important currency fix; the rest of the section has aged well because it teaches classics honestly.)

- **(GC-8, P2 / currency) Empirical counterpoint to adaptive overfitting.** Lines 318–320 say the worst-case adaptive-overfitting analyses "may be too conservative," but cite nothing on the empirical side. The landmark result here is Recht, Roelofs, Schmidt & Shankar, *Do ImageNet Classifiers Generalize to ImageNet?* / *Do CIFAR-10 Classifiers Generalize to CIFAR-10?* (2018/2019), which built *fresh* test sets for benchmarks the whole field had hammered for years and found accuracies dropped but rankings were largely preserved — i.e., a decade of test-set reuse had *not* produced catastrophic adaptive overfitting. That is the perfect, reassuring, *2026-current* empirical bookend to Dwork's bleak theory and belongs in one sentence here. (Requires a new bib entry — see §4; stub provided.)

- **(GC-7, P2) A recomputable number / worked example.** This file makes several crisp quantitative claims — 2500 samples for ±0.01 at one SD, 10,000 for 95% via CLT, ~15,000 via Hoeffding (line 215), the $\mathcal O(1/\sqrt n)$ rate — but the reader cannot *check* any of them; the file has zero code cells (confirmed: 0 fences, no executed-output manifests). The gold-standard chapters (calculus, linear-algebra) always pair a claim with something computable. A tiny worked cell (or an inline worked derivation) that solves Hoeffding for $n$ and prints the CLT-vs-Hoeffding comparison would make the asymptotic-vs-finite-sample point land. (Low priority — the prose is self-contained — but it is the cheapest way to lift this above a pure-essay section.)

### Remove / trim

- Nothing substantial to cut — the file is already lean (no sibling overlap to trim; the regression `generalization.md` owns bias–variance and model selection, this file owns test-set statistics + VC, and they barely intersect). The prose is occasionally a touch chatty (the "3am brilliant idea" narrative, lines 261–269, is charming and earns its place pedagogically — *keep*).
- The Bernoulli-variance paragraph (lines 162–175) has one redundant/garbled sentence to **fix not cut** (GC-5).

### Reorder / restructure

- Structure is sound: four `##` sections (Test Set → Test Set Reuse → Statistical Learning Theory → Summary) trace exactly the three "burning questions" posed in the intro (lines 32–35). No reorder needed. The only structural weakness is that **Statistical Learning Theory is a single long `##` with no `###` subsections** — at ~165 lines it runs past the "3–5 `##`, each with `###`" shape the guide wants. Recommend two `###` subheadings inside it: `### Uniform Convergence and the Union Bound` (lines 389–448) and `### VC Dimension` (lines 450–498). Light, mechanical, improves navigability and the rendered ToC. (Folded into GC-2's edit region; flagged in §4 as optional within GC-2 or as its own trivial change.)

## 2. Teaching quality

### Structure & flow

Strong. The three-question framing up front (lines 30–35) and the deliberate "test sets are all that we really have, and yet this fact seems strangely unsatisfying" pivot (line 338) into learning theory is exactly the intuition-first → formalism arc the gold-standard chapters use. The honest, recurring acknowledgment that the *a priori* guarantees are impractical for deep nets is a real strength — it sets up §5.5 rather than overselling VC. Only gap: the long unsubdivided third section (see Reorder).

### Figures

**Zero figures in the file** (confirmed against the rendered page — "No figures are present"). For most of the file that is fine (it's statistics prose), but the **VC/shattering paragraph is crying out for the one figure that would carry the idea** (GC-2). This is the single biggest teaching-quality gap and the reason the section reads as below the chapter's figure-rich bar. No inline figure-drawing matplotlib exists to flag (there's no code at all) — so the fix is purely additive: a committed generator SVG.

### Prose & clarity

Mostly excellent and breathes well. Three specific blemishes:

- **Line 166 — factual non-sequitur (GC-5).** "While $\epsilon(f)$ is initially unknown, we know that it cannot be greater than $1$." In context (bounding the Bernoulli variance $\epsilon(1-\epsilon)$ by $0.25$), the *fact that matters* is that $\epsilon\in[0,1]$ and the parabola $\epsilon(1-\epsilon)$ is maximized at $\epsilon=\tfrac12$ with value $\tfrac14$ — stating only "$\le 1$" is true but irrelevant to the variance bound and reads as a confused leftover. The very next sentence already makes the correct max-at-0.5 argument, so this sentence is redundant *and* misleading. Tighten — see §4.
- **Lines 207–217 — one-sided vs two-sided Hoeffding (GC-3).** Eq. 4.6.3 is stated one-sided: $P(\epsilon_\mathcal{D}(f)-\epsilon(f)\ge t) < e^{-2nt^2}$. The prose then asks for "the distance $t$ between our estimate… and the true error rate… does not exceed 0.01" with "95% confidence" — that is a *two-sided* statement ($|\epsilon_\mathcal{D}-\epsilon|\le t$), which needs the two-sided bound $2e^{-2nt^2}$. Solving $e^{-2nt^2}=0.05$ (one-sided) gives $n\approx 14{,}979$ ≈ "15,000" (the number in the text); the two-sided $2e^{-2nt^2}=0.05$ gives $n\approx 18{,}445$. So the displayed inequality and the quoted number are *one-sided*, but the surrounding confidence language is *two-sided*. A top-program reader will catch this. Fix by either (a) writing the two-sided bound and updating to ~18,400, or (b) keeping one-sided and rephrasing the claim as a one-sided overestimate guarantee. Recommend (a) for consistency with the CLT paragraph above it, which is explicitly two-sided ("two standard deviations… 95% confident… $\pm 0.01$"). See §4 for both options.
- **Lines 465–466 — notation break (GC-1, P1, the keystone issue).** Eq. 4.6.4 reads $P(R[p,f]-R_\textrm{emp}[\mathbf{X},\mathbf{Y},f] < \alpha)\ge 1-\delta$. But this file has, up to this point, exclusively used $\epsilon(f)$ (population error), $\epsilon_\mathcal{D}(f)$ (test error), and $\epsilon_\mathcal{S}(f_\mathcal{S})$ (training error). The symbols $R[p,f]$ and $R_\textrm{emp}[\mathbf{X},\mathbf{y},f]$ belong to the *regression* `generalization.md` (lines 141/147 there), and are dropped in here verbatim — *including* the capital $\mathbf{Y}$ (vs. lowercase $\mathbf{y}$ used in the regression file). The reader hits the chapter's central theoretical result in a notation they've never seen in this file. This is the highest-value clarity fix: restate the bound in the file's own notation, e.g. $P\!\left(\epsilon(f_\mathcal{S})-\epsilon_\mathcal{S}(f_\mathcal{S}) < \alpha\right)\ge 1-\delta$, $\alpha\ge c\sqrt{(\mathrm{VC}-\log\delta)/n}$, which also makes the immediately-following sentence ("fixing the model class and $\delta$, our error rate again decays at $\mathcal O(1/\sqrt n)$") visibly about the *same* $\epsilon$ quantities. See §4.

### Exercises

Four exercises, well-graded from mechanical to conceptual:
1. Sample-size-for-precision (mechanical application of the CLT/Hoeffding machinery) — good, but **it has no worked answer slot** and its intended method (CLT $\sqrt{0.25/n}$? Hoeffding? one- or two-sided?) is ambiguous given the GC-3 looseness above; tightening GC-3 makes this exercise crisp.
2. Test-set leakage / "how many models to leak the whole label set" — excellent, genuinely thought-provoking, the standout. Keep.
3. VC dimension of fifth-order polynomials — good, but slightly under-specified: "fifth-order polynomials" *in one variable, thresholded* (sign of $\sum_{k=0}^5 a_k x^k$) has VC dim 6; the answer depends on the (unstated) input dimension and that we threshold to get a classifier. Add a one-clause clarification ("…as binary classifiers on the real line, i.e. $\operatorname{sign}(p(x))$"). 
4. VC of axis-aligned rectangles in 2D (answer: 4) — classic and clean. Keep.

**Add one exercise (P2):** a Rademacher-flavored or *empirical* exercise to match the two new currency items — e.g. "A model class can shatter your $n$-point training set (VC $\ge n$). What does this imply about the Hoeffding/union-bound guarantee of §4.6.3, and why does this *not* contradict the fact that overparameterized nets generalize? (Forward-point: §5.5.)" — which directly exercises the file's own honest punchline.

## 3. Code & examples

### Does the code teach?

**There is no code in this file** (0 fences; no `outputs/*/…/generalization-classification.json` manifests — confirmed). That is *defensible* for a learning-theory section, and per the project's "code teaches, it does not draw" rule it would be wrong to bolt on illustrative plotting code. The *only* code that would earn its place is a tiny cell that **computes** a teaching result the prose asserts — the Hoeffding/CLT sample-size comparison (GC-7). Even that is optional (P2); the section can remain code-free and excellent if GC-1/2/3/4 land. If a cell is added it must (per house style) be one compact, framework-agnostic numeric cell (this is pure NumPy-style arithmetic — no framework divergence), not a matplotlib figure generator.

### PyTorch / JAX / TensorFlow / MXNet

N/A — no per-framework code cells exist. **MXNet note (book-wide):** since this file has no code and no framework tabs, the MXNet-retirement question does not arise here; nothing to de-emphasize. (Flagging for the overview only so it can confirm this file needs no MXNet action.)

### Cross-framework consistency & d2l conventions

- No imports cell, no `#@save`, no cell IDs needed (no code). Conventions N/A.
- The **one cross-file consistency defect** is mathematical-notation, not code: the $R/\epsilon$ clash (GC-1). The fix should align this file *internally* on $\epsilon$ (its dominant choice), and the overview should note that across the two generalization files the book uses *both* an $R[p,f]$/$R_\textrm{emp}$ system (regression) and an $\epsilon(f)$ system (classification) — a deliberate, defensible split (risk-of-a-loss vs. 0/1-error-rate), but the VC bound here was lifted from the wrong one. No need to unify the two files globally; just fix the imported equation.

## 4. Implementation spec (downstream agents act on THIS)

### GC-1 — Re-express the VC bound (eq. 4.6.4) in the file's own ε-notation  ·  [P1] · [S] · [mechanical]
- **Type:** prose / correctness (notation consistency)
- **Where:** `chapter_linear-classification/generalization-classification.md`, the display equation at lines 465–466. Verbatim anchor:
  `$$P\left(R[p, f] - R_\textrm{emp}[\mathbf{X}, \mathbf{Y}, f] < \alpha\right) \geq 1-\delta`
  `\ \textrm{ for }\ \alpha \geq c \sqrt{(\textrm{VC} - \log \delta)/n}.$$`
- **Change:** replace those two lines verbatim with:
  `$$P\left(\epsilon(f_\mathcal{S}) - \epsilon_\mathcal{S}(f_\mathcal{S}) < \alpha\right) \geq 1-\delta`
  `\ \textrm{ for }\ \alpha \geq c \sqrt{(\textrm{VC} - \log \delta)/n}.$$`
  This uses the symbols already defined earlier in this file ($\epsilon(f_\mathcal{S})$ = population error of the learned classifier, line 373; $\epsilon_\mathcal{S}(f_\mathcal{S})$ = its training/empirical error, line 369). Do **not** introduce $R$/$R_\textrm{emp}$/$\mathbf{Y}$ here.
- **Touches:** none (no `:eqlabel:`/`:numref:` points at this equation; it's referenced only in prose).
- **Done when:** eq. 4.6.4 uses only $\epsilon(\cdot)$/$\epsilon_\mathcal{S}(\cdot)$ symbols; the sentence at lines 491–493 ("fixing the model class and $\delta$, our error rate again decays… $\mathcal O(1/\sqrt n)$") now visibly refers to the same $\epsilon$ quantities; `make html` renders the equation without `??`; no other occurrence of `R[p, f]` or `\mathbf{Y}` remains in this file.
- **Depends on:** none.

### GC-2 — Add a shattering / VC-dimension figure  ·  [P1] · [M] · [authored]
- **Type:** figure (+ optional sub-structuring)
- **Where:** `chapter_linear-classification/generalization-classification.md`, immediately after the VC-explanation sentence ending at line 485 ("…but not to four."). Verbatim anchor for insertion point — after:
  `It is easy to see that a line can assign`
  `any possible labeling to three points in two dimensions,`
  `but not to four.`
- **Change:** Insert a generated figure and a one-line caption + label, and rewrite the sentence to reference it. Author the figure spec; the figure is produced by a committed matplotlib generator (house rule — **no drawing code in the notebook**, and this file has no notebook anyway).
  - **Figure depicts:** two side-by-side panels in the chapter's shared house style.
    - *Left ("shattered — VC ≥ 3"):* three points in general position (a triangle) in 2D; show that a straight line (halfplane) can realize a representative subset of the $2^3=8$ ± labelings — draw 3–4 small sub-panels or one panel annotated with the separating line for the "hardest" split (one point vs. the other two, for each of the three choices), making clear all 8 are achievable.
    - *Right ("cannot shatter 4 — VC = 3"):* four points in the XOR configuration (a square with alternating +/− at opposite corners); show that **no** single line separates the +'s from the −'s. Annotate "no linear separator."
  - **Caption (must contain no `]`):** `A linear classifier in two dimensions shatters any 3 points in general position (all $2^3$ labelings are realizable) but cannot shatter 4 (the XOR labeling has no linear separator). Hence the VC dimension of lines in $\mathbb{R}^2$ is 3.` then on the next line `:label:`fig_vc_shattering``.
  - **Rewrite the lead-in sentence** (lines 482–485) to reference it: replace `It is easy to see that a line can assign any possible labeling to three points in two dimensions, but not to four.` → `As :numref:`fig_vc_shattering` illustrates, a line in the plane can realize *every* labeling of three points in general position, but no line can realize the XOR labeling of four points—so the VC dimension of two-dimensional linear classifiers is exactly 3 (matching $d+1$ with $d=2$).`
  - *(Optional, fold in here:)* add `### Uniform Convergence and the Union Bound` before line 389 and `### VC Dimension` before line 450 to subdivide the over-long Statistical Learning Theory section.
- **Touches:** create generator `tools/gen_mdl_classification_figures.py` (importing the shared style from `tools/gen_mdl_figures.py`, per the `mdl-figure` skill) **or** add a function to the existing classification figure generator if one exists; output `img/mdl-classification-vc-shattering.svg` (name per the `mdl-<chapter>-<id>` convention; reconcile the exact stem with whatever the chapter already uses); run `make figures`; commit the SVG (LFS). Use the **`mdl-figure`** skill to scaffold and **`figure-style-audit`** to verify.
- **Done when:** the figure renders in HTML and PDF, `:numref:`fig_vc_shattering`` resolves (no `??`), the SVG is byte-idempotent under a second `make figures`, no figure-drawing code lives in any notebook, and `figure-style-audit` passes for the chapter. The caption contains no `]`.
- **Depends on:** none (independent of GC-1/3/4).

### GC-3 — Reconcile one-sided Hoeffding with the two-sided confidence claim  ·  [P1] · [S] · [judgment]
- **Type:** correctness / prose
- **Where:** `chapter_linear-classification/generalization-classification.md`, eq. 4.6.3 (lines 205–207) and the paragraph lines 209–217. Anchors:
  display `$$P(\epsilon_\mathcal{D}(f) - \epsilon(f) \geq t) < \exp\left( - 2n t^2 \right).$$`
  prose `roughly 15,000 examples are required`.
- **Change:** make the bound and the quoted number consistent. **Preferred (two-sided, to match the CLT paragraph and the "$\pm 0.01$, 95%" framing):**
  - Replace the display with the two-sided form: `$$P\left(|\epsilon_\mathcal{D}(f) - \epsilon(f)| \geq t\right) < 2\exp\left( - 2n t^2 \right).$$`
  - Update the prose `roughly 15,000 examples are required as compared to the 10,000 examples suggested by the asymptotic analysis above` → `roughly 18,500 examples are required, as compared with the 10,000 suggested by the asymptotic analysis above` (solving $2e^{-2nt^2}=0.05$ at $t=0.01$ gives $n\approx 18{,}445$).
  - **Alternative (keep one-sided):** leave the display as-is but change the claim at lines 211–214 from a two-sided distance ("the distance $t$ between our estimate… and the true error rate… does not exceed 0.01") to a one-sided overestimate guarantee ("…that our estimate overstates the true error rate by no more than 0.01"), and keep "roughly 15,000." Pick one; do not leave a one-sided inequality under a two-sided claim.
- **Touches:** none. (If GC-7's worked cell is added, its printed numbers must match whichever convention is chosen here.)
- **Done when:** the inequality (one- vs two-sided) and the surrounding "$\pm 0.01$/95%" language agree, and the stated sample count is the correct solution for that convention (one-sided ⇒ ~15,000; two-sided ⇒ ~18,500). Exercise 1's expected method is now unambiguous.
- **Depends on:** none (but coordinate with GC-7 if that is implemented).

### GC-4 — Sharpen the deep-generalization forward-pointer + cite Zhang 2021  ·  [P1] · [S] · [mechanical]
- **Type:** currency / cross-reference
- **Where:** `chapter_linear-classification/generalization-classification.md`, two spots.
  - (a) Lines 73–78, anchor `When we get to :numref:`chap_perceptrons`, / we will revisit generalization`.
  - (b) Lines 555–567, anchor `they turn out to be powerless` … and the final `In the next chapter, we will revisit generalization / in the context of deep learning.`
- **Change:**
  - (a) Replace `When we get to :numref:`chap_perceptrons`,` → `When we get to :numref:`sec_generalization_deep`,` (point to the section that owns this, not the whole part).
  - (b) In the vacuous-bounds sentence, add the citation: replace `they turn out to be powerless` `(as straightforwardly applied)` `for explaining why deep neural networks generalize.` → `they turn out to be powerless` `(as straightforwardly applied)` `for explaining why deep neural networks generalize :cite:`zhang2021understanding`.` And replace the closing `In the next chapter, we will revisit generalization in the context of deep learning.` → `We revisit generalization in the context of deep learning in :numref:`sec_generalization_deep`.`
- **Touches:** none — `zhang2021understanding` already exists in `d2l.bib` (line 3959) and `sec_generalization_deep` is a valid label (`chapter_multilayer-perceptrons/generalization-deep.md:2`). Both verified.
- **Done when:** both `:numref:`s resolve to §5.5 (not the part index), the Zhang citation renders as "(Zhang et al., 2021)" in HTML and appears in the bibliography, `make html` clean.
- **Depends on:** none.

### GC-5 — Fix the Bernoulli-variance non-sequitur  ·  [P2] · [S] · [mechanical]
- **Type:** correctness / prose
- **Where:** `chapter_linear-classification/generalization-classification.md`, lines 165–166. Verbatim anchor:
  `While $\epsilon(f)$ is initially unknown,`
  `we know that it cannot be greater than $1$.`
- **Change:** replace those two lines with:
  `While $\epsilon(f)$ is initially unknown,`
  `it is a probability and so lies in $[0, 1]$,`
  `and the function $\epsilon(1-\epsilon)$ is maximized at $\epsilon = \tfrac{1}{2}$.`
  (This makes the *next* sentence — "our variance is highest when the true error rate is close to 0.5" — a direct consequence rather than a restatement, and removes the irrelevant "$\le 1$" claim.)
- **Touches:** none.
- **Done when:** the paragraph's logic reads as $\epsilon\in[0,1]\Rightarrow \epsilon(1-\epsilon)\le\tfrac14\Rightarrow$ SD $\le\sqrt{0.25/n}$ with no dangling "$\le 1$" statement; renders clean.
- **Depends on:** none.

### GC-6 — Add a one-paragraph Rademacher-complexity mention  ·  [P2] · [M] · [authored]
- **Type:** coverage / currency (in-scope-light per scope map)
- **Where:** `chapter_linear-classification/generalization-classification.md`, in the Summary, immediately after the sentence ending at line 554 (`See :citet:`boucheron2005theory` for a detailed discussion of several advanced ways of measuring function complexity.`). Insert before the "Unfortunately, while these complexity measures…" sentence (line 555).
- **Change:** Insert this drafted paragraph (keep it to ~5 sentences; do **not** introduce a proof):
  > `The most influential of these modern measures is *Rademacher complexity*, which asks how well a model class can correlate with random $\pm 1$ noise *on your actual sample*. Because it is measured on the data at hand rather than over all conceivable inputs, it is *data-dependent* and typically far tighter than the distribution-free, purely combinatorial VC dimension; it is also the natural vehicle for *margin*- and *norm*-based bounds, which can be meaningful even when the parameter count is enormous. We give it only a name here; :numref:`sec_generalization_deep` returns to norm-based and Rademacher-style arguments when we ask why heavily overparameterized networks generalize. For a rigorous development, see :citet:`boucheron2005theory`.`
- **Touches:** none (reuses the existing `boucheron2005theory` key and the `sec_generalization_deep` label; both verified present). *Optional upgrade:* if a downstream agent wants a dedicated Rademacher textbook citation (Mohri/Rostamizadeh/Talwalkar or Shalev-Shwartz & Ben-David), add a bib entry — but the conservative version above needs no new key.
- **Done when:** the paragraph renders, `:numref:`sec_generalization_deep`` and `:citet:`boucheron2005theory`` resolve, and the Summary still flows into the existing "Unfortunately, while these complexity measures…" sentence (which now reads as the VC-and-Rademacher caveat). `make html` clean.
- **Depends on:** none (complements GC-4; both forward-point to §5.5).

### GC-7 — Add the recomputable Hoeffding/CLT sample-size worked example  ·  [P2] · [S] · [judgment]
- **Type:** code / teaching
- **Where:** `chapter_linear-classification/generalization-classification.md`, after the Hoeffding paragraph (after line 227, before `## Test Set Reuse`).
- **Change:** add **one compact, framework-agnostic** numeric cell (pure Python/`numpy`/`scipy`-style arithmetic; this is *not* a plotting cell, so it complies with "code teaches, it does not draw") that recomputes the three numbers the prose claims, e.g.: solve the CLT condition $\sqrt{0.25/n}=0.01$ for one and two SDs (→ 2500, 10000) and the Hoeffding condition for the chosen one-/two-sided convention (→ 15000 or 18500 per GC-3), and `print` them in a small labeled table. Untagged (applies to all frameworks). Give it a stable cell id via `tools/add_cell_ids.py` after insertion. Keep it to <12 lines. *This is optional;* if the maintainers prefer the file to stay prose-only, satisfy the intent of this item by adding a one-line inline derivation instead and mark GC-7 done.
- **Touches:** if a cell is added: `tools/add_cell_ids.py` (assign id), and the notebook/output pipeline (`make notebooks-<fw>` / capture) so an executed manifest exists; the printed integers must match GC-3's chosen convention. If done as inline prose, none.
- **Done when:** either (a) a single cell prints sample sizes that equal the values stated in the surrounding prose (2500 / 10000 / [15000|18500]) with no warnings, *or* (b) an inline one-line derivation makes the Hoeffding "≈15,000/18,500" number explicitly recomputable. No matplotlib/plotting code introduced.
- **Depends on:** GC-3 (numbers must agree with the chosen one-/two-sided convention).

### GC-8 — Add the empirical counterpoint (Recht et al. reproductions) to adaptive overfitting  ·  [P2] · [S] · [authored]
- **Type:** currency / coverage
- **Where:** `chapter_linear-classification/generalization-classification.md`, lines 318–326 (the "while it is possible to leak all information… these analyses may be too conservative… In practice, take care…" passage). Insert one sentence right after `these analyses may be too conservative.` (line 320), before `In practice, take care…`.
- **Change:** Insert:
  > `Encouragingly, when researchers constructed entirely *fresh* test sets for benchmarks the community had reused for years—CIFAR-10 and ImageNet—model accuracies dropped on the new data but their relative *rankings* were largely preserved :cite:`recht2019imagenet`, suggesting that, at least on these benchmarks, years of test-set reuse had not produced the catastrophic adaptive overfitting the worst-case theory permits.`
- **Touches:** **new bib entry required** — add to `d2l.bib`:
  ```
  @inproceedings{recht2019imagenet,
    title={Do {ImageNet} classifiers generalize to {ImageNet}?},
    author={Recht, Benjamin and Roelofs, Rebecca and Schmidt, Ludwig and Shankar, Vaishaal},
    booktitle={International Conference on Machine Learning},
    pages={5389--5400},
    year={2019}
  }
  ```
  (Confirmed: no existing `recht*` key in `d2l.bib` for this paper — must be added.)
- **Done when:** the sentence renders, `:cite:`recht2019imagenet`` resolves to "(Recht et al., 2019)" and appears in the bibliography, `make html` clean.
- **Depends on:** none.

## 5. Keep — what is already excellent (do not lose this)

- **The three-question opening (lines 30–35)** and the way the four sections answer them in order — a textbook-perfect "here's what we'll resolve" frame. Don't touch the framing.
- **The honest, repeated message that *a priori* guarantees are impractical for deep nets** (lines 53–72, and the Summary 555–565). This intellectual honesty — refusing to oversell VC — is exactly the gold-standard tone and the perfect setup for §5.5. Preserve every bit of it; GC-4 only *strengthens* it with a citation.
- **The "3am brilliant idea / you do not have a test set!" narrative (lines 261–269)** and the whole Test Set Reuse section — memorable, correct, and it teaches multiple-testing + adaptive overfitting better than a dry treatment. Keep the voice.
- **The CLT → "2500 / 10,000 samples" concretization (lines 177–196)** and the wry observation that "thousands of applied deep learning papers… make a big deal out of error rate improvements of 0.01 or less." This is the kind of grounded, slightly irreverent insight that makes a section memorable. Keep (just make the Hoeffding sibling number consistent — GC-3).
- **Exercise 2** (test-set leakage via repeated model evaluation) — a genuinely thought-provoking problem worthy of any top course. Keep verbatim.
- **The clean division of labor with the two sibling generalization files** — this file does *not* re-teach bias–variance (regression `generalization.md`) or double descent (`generalization-deep.md`); it owns test-set statistics + VC. That restraint is correct; do not import downstream content (the scope map agrees). All `:numref:`/`:cite:` keys in the file currently resolve and render correctly — only the *targets'* precision (GC-4) needs tuning.
