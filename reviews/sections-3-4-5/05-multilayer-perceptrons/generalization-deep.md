# Review — chapter_multilayer-perceptrons/generalization-deep.md  (§5.5 "Generalization in Deep Learning")

**Role in the chapter:** The conceptual capstone on *why* deep nets generalize. After the chapter has built MLPs, trained them, and shown they can interpolate, this section confronts the central puzzle: classical complexity control (VC/Rademacher) cannot explain deep generalization, yet the networks generalize anyway. It is a prose-only essay (no code, no figures) that connects forward to weight decay, dropout, and the modern theory literature, and back to §3.6 (`generalization.md`).

**Verdict:** This is one of the better introductory treatments of modern generalization in any textbook — it is honest, it cites Zhang et al. and Nakkiran, and it frames the field as unresolved rather than overclaiming. But it was written ~2021 and reads that way: double descent — the single most clarifying idea here, and explicitly in-scope per the scope map — is mentioned in *one subordinate clause* with no figure and no interpolation-threshold explanation; grokking (2022) is absent; the implicit-bias-of-SGD story is gestured at but never named; and the section carries **zero figures** in a place where one figure (the double-descent curve) would unlock the whole idea. The single highest-value change is **DD-1: add a pre-generated double-descent figure with the interpolation threshold and a tight paragraph that actually explains the curve.** That alone moves the section from "good summary" to "best-in-class."

**Grade:** B+. Assignable today at a top program — it is accurate and refreshingly honest — but it underdelivers on its own central, in-scope topic (double descent) and shows its 2021 vintage (no grokking, no named implicit regularization, no figure). The fixes are surgical and well-scoped; with DD-1 + DD-2 + DD-3 it is a clear A.

**Top priorities (ranked):**
1. [P0] **DD-1** — Add the double-descent figure (pre-generated SVG) + a paragraph explaining the interpolation threshold and the second descent. This is the section's marquee in-scope idea and it currently has neither a figure nor a real explanation. `authored`.
2. [P1] **DD-2** — Add a short "Implicit Regularization" treatment (name the flat-minima / small-norm SGD story; cite `Soudry.Hoffer.Nacson.ea.2018`, already in the bib) and a one-line **grokking** forward-mention (Power et al. 2022, new bib entry). `authored`.
3. [P1] **DD-3** — Cite `Belkin.Hsu.Ma.ea.2019` (already in the bib!) at the double-descent mention; it is the canonical reference and is currently missing. `mechanical`.
4. [P1] **DD-4** — Fix three genuine prose/grammar defects that hurt a flagship conceptual section (lines 81, 307, the "Machine learning researchers are consumers of optimization algorithms" non-sequitur at 17–19). `mechanical`/`authored`.
5. [P2] **DD-5** — Upgrade the exercises: the current five are recall questions about early stopping. Add 3–4 that engage double descent, random-label fitting, and implicit bias. `authored`.
6. [P2] **DD-6** — Add a "Discussion and Further Reading" closer with `:citet:` pointers (Zhang, Belkin, Nakkiran, Bartlett benign overfitting, the NTK) and a forward-pointer to §4.6 for VC/Rademacher mechanics. `authored`.
7. [P2] **DD-7** — `:cite:` → `:citet:` cleanup where the citation is used as a sentence subject (lines 118, 147, 246). `mechanical`.

---

## 1. Coverage

### Add

**(P0) Double descent — the figure and the actual explanation.** This is the heart of the matter and the scope map names it explicitly as *in scope, develop here*. The current treatment (lines 142–147) is one clause:

> Stranger yet, the pattern relating the generalization gap to the *complexity* of the model … can be non-monotonic, with greater complexity hurting at first but subsequently helping in a so-called "double-descent" pattern :cite:`nakkiran2021deep`.

That names the phenomenon but never explains it: no **interpolation threshold**, no picture, no statement of *what* descends twice. A reader who does not already know double descent learns nothing actionable. Top treatments (Belkin et al. 2019 PNAS; Nakkiran et al. 2021; the MLU-Explain visual essay) all lead with the curve. **The single most valuable addition to this file is a pre-generated double-descent figure** (test error vs. model capacity, classical U-curve on the left, the interpolation threshold spike where #params ≈ #samples, and the second monotone descent into the overparametrized regime) plus a paragraph that walks it. Drafted figure spec + prose in **DD-1** below. The chapter has *zero* figures; this is the one that earns its place.

**(P1) Name the implicit regularization of SGD.** Lines 310–319 circle the right idea — that weight decay and early stopping may help "not because they meaningfully constrain the power of the neural network but rather because they … encode inductive biases" — but the section never states the cleanest modern version: **SGD itself is an implicit regularizer**, biasing solutions toward small-norm / flat minima even with no explicit penalty. For separable data this is a theorem: gradient descent on logistic loss converges to the max-margin (minimum-norm) solution (`Soudry.Hoffer.Nacson.ea.2018`, already in the bib). This is the modern complement to "why do overparametrized nets generalize," it is in scope (it *is* deep generalization), and it is a one-paragraph add. See **DD-2**.

**(P1) Grokking — at least a forward-mention.** The watch-points and the research digest both flag this. Power et al. 2022 showed networks can generalize *long after* they have memorized the training set (sudden jump from chance to perfect, well past the overfitting point) — which directly extends the section's own theme that "training models until they interpolate noisy data is typically a bad idea" is *not* the whole story. A footnote/one-line mention with a citation is the right scope (not a subsection). Drafted in **DD-2**; needs a new bib entry (**DD-2a**).

**(P1) Cite Belkin et al. 2019 at the double-descent claim.** `Belkin.Hsu.Ma.ea.2019` ("Reconciling modern machine-learning practice and the classical bias–variance trade-off", PNAS) **is already in `d2l.bib`** (line 5269) but is **not cited anywhere in this file**. It is *the* paper that introduced and named double descent; citing only Nakkiran (which extends it) and omitting Belkin is a citation gap a reviewer at a top program would notice immediately. Pure mechanical fix — **DD-3**.

**(P2) A "Discussion / Further Reading" closer.** The section ends on the Summary with no curated pointers. A best-in-class foundations section closes by connecting outward. Add a short closer that forward-points to §4.6 (`generalization-classification.md`) for the VC/Rademacher mechanics (honouring the scope map — those proofs live there, not here) and lists the key reads (Zhang et al.; Belkin et al.; Nakkiran et al.; Bartlett et al. 2020 on benign overfitting; the NTK paper already cited). See **DD-6**.

### Remove / trim

**Nothing should be cut wholesale — the file is lean.** Two micro-trims:

- Lines 17–19 ("Machine learning researchers are *consumers* of optimization algorithms. Sometimes, we must even develop new optimization algorithms. But at the end of the day, optimization is merely a means to an end.") is a three-sentence aside that interrupts the otherwise strong opening flow and is partly redundant with the surrounding "optimization is a means to a statistical end" point. Tighten to one sentence (**DD-4**).
- The §3.6 sibling already carries the full IID/empirical-risk/model-selection apparatus; §5.5 correctly does *not* repeat it. Good — keep that discipline. No overlap to cut.

### Reorder / restructure

The spine is sound and well-sequenced: *the puzzle* (intro) → *classical view breaks* (Revisiting Overfitting) → *a better mental model* (Nonparametrics/NTK) → *what still helps in practice* (Early Stopping, Classical Regularization) → Summary. One structural recommendation:

- **Promote double descent out of the "Revisiting Overfitting" subordinate clause into its own short `###` subsection** (or a clearly delimited paragraph block) under "Revisiting Overfitting and Regularization," co-located with the figure. Right now the single most important modern result is buried mid-paragraph. Giving it a heading signals its importance and is where students will look. This is folded into **DD-1**.

The file has 4 content `##` sections + Summary + Exercises — within the 3–5 guideline. No flattening needed.

## 2. Teaching quality

### Structure & flow

Strong overall. The intro is genuinely good — the "wild west on both fronts," the optimization-vs-generalization split, and the honest "pour yourself a drink" tone are exactly the right register for a topic that *is* unresolved. The Nonparametrics → NTK move is a highlight (see §5 Keep). The one structural weakness is that the marquee idea (double descent) is structurally demoted; **DD-1** fixes this.

### Figures

**The section has zero figures (confirmed against the rendered page).** For a conceptual essay this is mostly fine — but double descent is the canonical exception: it is a *picture-first* idea. Every leading treatment leads with the curve.

- **Missing (P0):** the double-descent curve. Spec in **DD-1**. It must be **pre-generated** (committed `img/mdl-mlp-double-descent.svg` via a generator, per the house convention in CLAUDE.md "Content authoring"), **not** drawn by matplotlib in a notebook cell. The MLP chapter has **no figure generator yet** (`tools/gen_mdl_mlp_figures.py` does not exist; the chapter's other art — `mlp.svg`, `forward.svg`, `dropout2.svg` — are legacy hand-drawn SVGs), so this change also *creates* that generator, importing the shared style from `tools/gen_mdl_figures.py` exactly as `gen_mdl_calculus_figures.py` does. `make figures` auto-discovers any `tools/gen_mdl_*_figures.py`, so no Makefile edit is needed.
- No figure-drawing code currently lives in any cell of this file (there are no cells), so there is nothing to flag for removal.

### Prose & clarity

Mostly clean and well-paced. Three genuine defects to fix (**DD-4**):

1. **Line 81 — broken construction.** "a model relies on certain assumptions: to achieve human-level performance it may be useful to identify *inductive biases*…" — the colon promises a definition of the assumptions but delivers a different thought (how to *choose* good ones). Rewrite: "a model relies on certain assumptions, or *inductive biases*. To achieve human-level performance it may help to choose inductive biases that reflect how humans think about the world; these encode a preference for solutions with certain properties."
2. **Line 307 — garbled clause + typo.** "are insufficient to prevent the networks from interpolating the data :cite:`zhang2021understanding` and thus the benefits if interpreted as regularization might only make sense in combination with the early stopping criterion." Reads as a run-on with a misplaced citation and "the benefits if interpreted". Rewrite: "…are insufficient to prevent the networks from interpolating the data :cite:`zhang2021understanding`. Their benefit, *if* interpreted as classical regularization, may therefore make sense only in combination with early stopping."
3. **Lines 17–19 — the "consumers of optimization" aside** (see Remove/trim) — tighten.

Smaller: line 79 "generalizes better on data with certain distributions, and worse with other distributions" is wordy ("better on some data distributions and worse on others"); line 333-area uses "datasets of interests" (typo: "interest"). These are folded into the mechanical fix in **DD-4**.

### Exercises

The current five are all narrow recall questions about **early stopping** (Q2–Q5 are *all* early stopping) plus one on classical-bound failure (Q1). For a section whose headline ideas are interpolation, double descent, the NTK, and implicit regularization, the exercise set badly under-samples the content. **DD-5** adds problems that engage the actual marquee ideas (predict-the-shape-of-the-double-descent-curve; the random-label thought experiment from Zhang et al.; why 1-NN is consistent despite zero training error; an implicit-bias question). Keep Q1; trim the early-stopping cluster to two.

## 3. Code & examples

**This file contains no code cells** (confirmed: the source has no ```` ```{.python .input} ```` blocks, and there is no `outputs/<fw>/chapter_multilayer-perceptrons/generalization-deep.json` manifest — it is a prose-only conceptual section). Sections "Does the code teach?", "PyTorch", "JAX", "TensorFlow", "MXNet", and "Cross-framework consistency" are therefore **N/A**.

Two consequences worth recording for the orchestrator:

- **The book-wide MXNet-tab question does not arise here.** There are no framework tabs in this file. (The discussion link at the bottom is a single bare URL, not a per-framework `:begin_tab:` block, unlike the sibling `generalization.md` which has four. No action.)
- **No `:cite:`/`:citet:` correctness issues block rendering**, but there is a stylistic inconsistency (`:cite:` used where the citation is a grammatical subject) — see **DD-7**.

## 4. Implementation spec (the executable part — downstream agents act on THIS)

### DD-1 — Double-descent: figure + explanation (the marquee fix)  ·  [P0] · [L] · [authored]
- **Type:** coverage + figure + teaching
- **Where:** `chapter_multilayer-perceptrons/generalization-deep.md`. Anchor the prose insertion at the double-descent clause inside "Revisiting Overfitting and Regularization", lines 142–147, verbatim: *"Stranger yet, the pattern relating the generalization gap to the *complexity* of the model (as captured, for example, in the depth or width of the networks) can be non-monotonic, with greater complexity hurting at first but subsequently helping in a so-called "double-descent" pattern :cite:`nakkiran2021deep`."* Promote this into a short `###` subsection and attach the figure.
- **Change:** Replace that single sentence with a `### Double Descent` subsection (authored prose below) and insert the figure immediately before its explanatory paragraph.

  Drafted prose (integrate verbatim; keep `:cite:` keys exact):

  > ### Double Descent
  >
  > The strangest of these phenomena deserves its own picture. Classical theory predicts a *U-shaped* test-error curve: as we add capacity, error first falls (we stop underfitting) and then rises (we begin overfitting), with a sweet spot in between (:numref:`fig_capacity_vs_error`). Deep networks do not obey this. As we keep adding capacity *past* the point where the model can exactly fit the training set --- the **interpolation threshold**, where the number of parameters is roughly the number of training examples --- the test error, after spiking sharply at the threshold, *descends a second time*, often dropping below the classical sweet spot. This non-monotone, two-valley shape is called **double descent** :cite:`Belkin.Hsu.Ma.ea.2019`, and it has been observed not only as we grow the model but also as we train for more epochs or add more data :cite:`nakkiran2021deep` (:numref:`fig_double_descent`).
  >
  > ![Classical bias--variance theory predicts the U-shaped curve (left): test error falls, then rises as capacity grows, with a sweet spot in between. Deep networks follow it only up to the *interpolation threshold*, where the model can just fit the training set (here #parameters $\approx$ #examples) and test error spikes. Beyond it lies the *overparametrized regime*, where adding still more capacity makes the test error *descend a second time*, often below the classical optimum. Training error (gray) falls monotonically to zero at the threshold and stays there.](../img/mdl-mlp-double-descent.svg)
  > :label:`fig_double_descent`
  >
  > Why should bigger be better past the point of perfectly fitting the data? The interpolation threshold is where a model has *just barely* enough capacity to fit the training set: it is forced into a single, often jagged, high-variance solution. Past the threshold there are *many* parameter settings that interpolate the data, and the training procedure --- gradient descent from a small initialization --- selects among them, favouring smooth, small-norm solutions that happen to generalize. The phenomenon thus reframes "more capacity" not as "more overfitting" but as "more room for the optimizer to find a simple interpolant." It is the cleanest single picture of why the classical complexity intuition fails for deep networks, and we return to *why* gradient descent prefers such solutions below.

- **Touches:**
  - **New file** `tools/gen_mdl_mlp_figures.py` — create it modeled on `tools/gen_mdl_calculus_figures.py` (`sys.path.insert` + `import gen_mdl_figures as fl`, reuse `fl.np, fl.plt`, `BLUE/ORANGE/GRAY`, `fl.clean_axes`, byte-idempotent `save()`). It must draw `img/mdl-mlp-double-descent.svg`.
  - **Figure content spec:** a single axes, x = "Model capacity" (no ticks needed, or a single tick labelled "interpolation threshold"), y = "Error". Plot (a) **training error** in gray: a smooth curve decreasing from moderate to 0 at the threshold, flat at 0 thereafter; (b) **test error** in BLUE: the classical U on the left (down then up), a sharp **peak exactly at the threshold**, then a second monotone descent to a level *below* the left-hand minimum. Mark the threshold with a light vertical dashed line + label "interpolation threshold"; annotate the two regions "classical regime" (left) and "overparametrized regime" (right). Curves can be synthetic/analytic (no real training run needed) — e.g. test = a downward parabola blended with a `1/(capacity-threshold)`-style spike then a slow decay; keep it clean and schematic in the house style. Single panel, ~6×3.5 in, matching the calculus figures' weight.
  - Run `make figures`; commit `img/mdl-mlp-double-descent.svg`. No Makefile edit (the `figures` target globs `tools/gen_mdl_*_figures.py`).
- **Done when:** `img/mdl-mlp-double-descent.svg` exists and is byte-idempotent (`make figures` twice → clean `git diff img/`); `make html` is clean; the page renders the figure with a resolved `:numref:`fig_double_descent`` and the new `### Double Descent` heading appears in the §5.5 ToC; `Belkin.Hsu.Ma.ea.2019` resolves in the bibliography; PDF build is clean (caption contains no `]` and no `$`-immediately-before-digit — the drafted caption is already compliant).
- **Depends on:** DD-3 (the `Belkin` citation key is used in this block — they can also be applied together).

### DD-2 — Implicit regularization of SGD + grokking forward-mention  ·  [P1] · [M] · [authored]
- **Type:** coverage + currency
- **Where:** `chapter_multilayer-perceptrons/generalization-deep.md`. Best home is a new short paragraph at the end of "Classical Regularization Methods for Deep Networks" (after line 323, the paragraph ending "…even if the theoretical rationale for their efficacy may be radically different.") *and* it pays off the forward-promise made in DD-1 ("we return to *why* gradient descent prefers such solutions below").
- **Change:** Append the following authored paragraph (keep `:cite:` keys exact; `Soudry.Hoffer.Nacson.ea.2018` already exists in the bib, `Power.Burda.Edwards.ea.2022` is added by DD-2a):

  > **Implicit regularization.** A growing body of work suggests that the strongest regularizer in deep learning may be one we never write down: the optimizer itself. Among the many parameter settings that interpolate the training data, stochastic gradient descent does not pick one at random --- starting from a small initialization, it is biased toward solutions with small norm and "flat" minima, which tend to generalize. This is not merely empirical: for linearly separable data, gradient descent on the logistic loss provably converges to the *maximum-margin* (minimum-norm) separator, even with no explicit penalty :cite:`Soudry.Hoffer.Nacson.ea.2018`. This *implicit bias* helps explain why overparametrized networks --- which classical theory says should overfit catastrophically --- instead find simple, generalizing solutions, and why the interventions above (weight decay, early stopping) help by *nudging* an already-benign bias rather than by brute-force capacity control. A striking illustration is *grokking*: on small algorithmic tasks, networks first memorize the training set and only much later, after many further steps of training, suddenly generalize --- a reminder that optimization *dynamics*, not just architecture, govern generalization :cite:`Power.Burda.Edwards.ea.2022`.

- **Touches:** DD-2a (new bib entry). No figure, no build target beyond `make html`.
- **Done when:** `make html` clean; both `Soudry.Hoffer.Nacson.ea.2018` and `Power.Burda.Edwards.ea.2022` resolve in the rendered bibliography (no `[?]`); PDF builds.
- **Depends on:** DD-2a.

### DD-2a — Add the grokking bib entry  ·  [P1] · [S] · [mechanical]
- **Type:** currency (citation infrastructure)
- **Where:** `d2l.bib`. Add a new entry (place near `nakkiran2021deep`, ~line 3982, alphabetical/topical neighborhood is fine).
- **Change:** insert verbatim:
  ```
  @article{Power.Burda.Edwards.ea.2022,
    title		= {Grokking: Generalization Beyond Overfitting on Small
  		  Algorithmic Datasets},
    author	= {Power, Alethea and Burda, Yuri and Edwards, Harrison and
  		  Babuschkin, Igor and Misra, Vedant},
    journal	= {arXiv preprint arXiv:2201.02177},
    year		= {2022}
  }
  ```
- **Touches:** none beyond the bib.
- **Done when:** `:cite:`Power.Burda.Edwards.ea.2022`` resolves in `make html` and the PDF without a `[?]`.
- **Depends on:** none.

### DD-3 — Cite Belkin et al. 2019 at the double-descent mention  ·  [P1] · [S] · [mechanical]
- **Type:** coverage (missing canonical citation already in the bib)
- **Where:** `chapter_multilayer-perceptrons/generalization-deep.md`, line ~147. *If DD-1 is applied, this is already handled inside DD-1's drafted prose and this entry is a no-op.* If DD-1 is **not** applied, apply this standalone edit.
- **Change (standalone):**
  - old: `subsequently helping in a so-called "double-descent" pattern :cite:`nakkiran2021deep`.`
  - new: `subsequently helping in a so-called "double-descent" pattern :cite:`Belkin.Hsu.Ma.ea.2019,nakkiran2021deep`.`
- **Touches:** none (`Belkin.Hsu.Ma.ea.2019` already in `d2l.bib` line 5269).
- **Done when:** both citations resolve in `make html`; PDF builds.
- **Depends on:** none.

### DD-4 — Prose/grammar fixes (three defects)  ·  [P1] · [S] · [mechanical]
- **Type:** prose
- **Where:** `chapter_multilayer-perceptrons/generalization-deep.md`, lines 17–19, 78–81, 307.
- **Change:** apply each exact replacement:
  1. **Lines 17–19 (opening aside).**
     - old: `Machine learning researchers are *consumers* of optimization algorithms.\nSometimes, we must even develop new optimization algorithms.\nBut at the end of the day, optimization is merely a means to an end.`
     - new: `Optimization is merely a means to an end: machine learning researchers consume optimization algorithms, and sometimes invent new ones, but always in service of a statistical goal.`
  2. **Line 79 (No-free-lunch wording).**
     - old: `any learning algorithm generalizes better on data with certain distributions, and worse with other distributions.`
     - new: `any learning algorithm generalizes better on some data distributions and worse on others.`
  3. **Line 81 (broken colon construction).**
     - old: `Thus, given a finite training set,\na model relies on certain assumptions: \nto achieve human-level performance\nit may be useful to identify *inductive biases* \nthat reflect how humans think about the world.`
     - new: `Thus, given a finite training set, a model must rely on assumptions, or *inductive biases*. To achieve human-level performance it can help to choose inductive biases that reflect how humans think about the world.`
  4. **Line 307 (garbled run-on).**
     - old: `are insufficient to prevent the networks\nfrom interpolating the data :cite:`zhang2021understanding` and thus the benefits if interpreted\nas regularization might only make sense\nin combination with the early stopping criterion.`
     - new: `are insufficient to prevent the networks from interpolating the data :cite:`zhang2021understanding`. Their benefit, *if* interpreted as classical regularization, may therefore make sense only in combination with early stopping.`
  5. **Line ~319 typo.**
     - old: `found in datasets of interests.`
     - new: `found in datasets of interest.`
- **Touches:** none.
- **Done when:** `make html` clean; the five strings above no longer appear; reading the paragraphs aloud, each is a single clean sentence. (Line numbers are hints — match on the quoted text.)
- **Depends on:** none.

### DD-5 — Upgrade exercises to engage the marquee ideas  ·  [P2] · [M] · [authored]
- **Type:** teaching
- **Where:** `chapter_multilayer-perceptrons/generalization-deep.md`, the `## Exercises` block (lines 365–371). Keep Q1; collapse the four early-stopping recall questions to two; add four conceptual problems.
- **Change:** replace the current list with:
  > 1. In what sense do traditional complexity-based measures (VC dimension, Rademacher complexity) fail to account for the generalization of deep neural networks? What experiment of :citet:`zhang2021understanding` makes this failure concrete?
  > 1. Sketch the test-error-vs-capacity curve predicted by classical bias--variance theory and the curve actually observed for deep networks. Mark the interpolation threshold. In which regime do modern over-parametrized networks operate, and why does adding capacity there *help*?
  > 1. The 1-nearest-neighbor classifier achieves zero training error yet is consistent. Explain how zero training error and good generalization can coexist, and connect this to why over-parametrized networks are better understood as nonparametric models.
  > 1. Why might *early stopping* be considered a regularization technique, and how is the stopping criterion (the *patience* criterion) typically determined? Why does early stopping help much more under label noise than on a cleanly separable task?
  > 1. Beyond generalization, what practical benefit does early stopping provide?
  > 1. *Implicit bias.* Two networks with identical architecture are trained to zero training error, one with plain gradient descent and one with a very large batch / different optimizer; they reach different test accuracies. Since both interpolate the data, classical capacity arguments predict identical generalization. What is missing from that argument? (Hint: among the many interpolating solutions, which one does the optimizer select?)
- **Touches:** none.
- **Done when:** `make html` clean; exercise list renders with the new problems; at least Q2 and Q6 explicitly reference double descent / implicit bias.
- **Depends on:** DD-1 (Q2 references the figure's interpolation threshold), DD-2 (Q6 references implicit bias) — apply after those, or the exercises stand alone if reviewers prefer.

### DD-6 — Add "Discussion and Further Reading" closer  ·  [P2] · [S] · [authored]
- **Type:** coverage (resource curation + forward-pointer)
- **Where:** `chapter_multilayer-perceptrons/generalization-deep.md`, insert a short section between `## Summary` (ends line 362) and `## Exercises` (line 365).
- **Change:** add:
  > ## Discussion and Further Reading
  >
  > The honest summary is that deep generalization remains an open problem with several promising, partial accounts and no settled theory. The starting point is :citet:`zhang2021understanding`, which shows that deep networks fit random labels and so cannot be explained by uniform-convergence bounds; the formal apparatus of those classical bounds (VC dimension, Rademacher complexity) is developed in :numref:`sec_generalization_basics` and, more rigorously, in the classification chapter. The reconciliation of classical bias--variance theory with interpolation is double descent :cite:`Belkin.Hsu.Ma.ea.2019,nakkiran2021deep`; the kernel/nonparametric view runs through the neural tangent kernel :cite:`Jacot.Gabriel.Hongler.2018`; and the implicit-bias view runs through :citet:`Soudry.Hoffer.Nacson.ea.2018`. For the over-parametrized regression analogue --- where a minimum-norm interpolant can be near-optimal --- see the literature on *benign overfitting*.
- **Touches:** Optionally add a `Bartlett.Long.Lugosi.ea.2020` ("Benign overfitting in linear regression", PNAS) bib entry if the agent wants the benign-overfitting pointer to be a live `:citet:` rather than prose; otherwise leave it as the plain-prose phrase above (no new key needed). If adding: that is a small extra mechanical step, not required for `make html` to pass.
- **Done when:** `make html` clean; all `:citet:`/`:cite:` keys in the new section resolve; the `:numref:`sec_generalization_basics`` link works; PDF builds.
- **Depends on:** DD-2a is *not* needed here; DD-3/DD-1 share the `Belkin` key (already in bib).

### DD-7 — `:cite:` → `:citet:` where citation is a sentence subject  ·  [P2] · [S] · [mechanical]
- **Type:** prose (citation style consistency)
- **Where:** `chapter_multilayer-perceptrons/generalization-deep.md`, lines 116–118, 245–246. (Line 147 is handled by DD-1/DD-3.) These are spots where the citation functions grammatically as a noun ("…even in datasets consisting of millions [Zhang et al.]") and reads better as a textual citation; this matches the file's own style at lines 78, 226, 331 which use `:citet:`.
- **Change:**
  1. old: `even in datasets consisting of millions\n:cite:`zhang2021understanding`.`
     new: `even in datasets consisting of millions of examples :citet:`zhang2021understanding`.`
  2. old: `even when labels are assigned incorrectly or randomly\n:cite:`zhang2021understanding`,\nthis capability only emerges over many iterations of training.\nA new line of work :cite:`Rolnick.Veit.Belongie.Shavit.2017`\nhas revealed`
     new: `even when labels are assigned incorrectly or randomly :cite:`zhang2021understanding`, this capability only emerges over many iterations of training. :citet:`Rolnick.Veit.Belongie.Shavit.2017` revealed`
- **Touches:** none.
- **Done when:** `make html` clean; citations render as author-year text where changed; no `[?]`.
- **Depends on:** none. (Low priority / optional — purely stylistic; skip if churn-averse.)

## 5. Keep — what is already excellent (do not lose this)

- **The intro's honesty and register (lines 26–73).** "Our understanding of deep learning still resembles the wild west on both fronts," the optimization-vs-generalization split with the observation that "(i) is seldom a problem … understanding generalization is by far the bigger problem," and the refusal to overclaim. This is exactly the tone a top program wants on an unresolved topic. Preserve verbatim (apart from the DD-4 micro-trim of the lines 17–19 aside).
- **The "Inspiration from Nonparametrics" section (166–237).** The reframing of overparametrized nets as *nonparametric* (complexity grows with data), the 1-NN-is-consistent-despite-zero-training-error example, and the NTK landing (`Jacot.Gabriel.Hongler.2018`) form the strongest conceptual arc in the file and a genuinely excellent piece of teaching. Do not touch except to let DD-2's implicit-bias paragraph build on it.
- **The early-stopping section (240–283).** The clean-data-first / noisy-data-later finding (`Rolnick…2017`), the generalization guarantee (`Garg…2021`), the patience criterion, and the realizable-vs-noisy distinction ("Training models until they interpolate noisy data is typically a bad idea") are accurate, current, and well-motivated. Keep intact.
- **The honest treatment of classical regularizers (286–334)** — that weight decay's *theoretical* rationale in deep nets may be "radically different" from its classical one, and that its benefit may be inseparable from early stopping. This is correct and refreshingly non-dogmatic; DD-2 extends rather than replaces it.
- **The Summary's framing of the interpolation regime** and the paradox that interventions sometimes *increase* and sometimes *decrease* apparent complexity (337–362). Strong. Keep.
