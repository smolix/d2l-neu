# Review — chapter_multilayer-perceptrons/mlp.md  (§5.1 "Multilayer Perceptrons")

**Role in the chapter:** The conceptual on-ramp to deep learning: it motivates *why* a hidden layer is needed (linear models collapse under composition), defines the one-hidden-layer MLP, states universal approximation, and surveys the three classical activation functions (ReLU/sigmoid/tanh) with their derivatives. It owns the labels `sec_mlp`, `fig_mlp`, `subsec_activation-functions`, and `subsec_tanh`, which are cited by `dropout.md`, `alexnet.md`, `optimization-intro.md`, and `rnn-scratch.md` — so it is load-bearing for the whole book.

**Verdict:** Strong, well-written, and close to the bar — the motivation prose is genuinely excellent and the activation/derivative reference is the clearest single page in any intro text. Two things hold it back from "best-in-class as written in 2026": (1) it never *shows* nonlinear separation — the XOR demonstration that every top treatment (Goodfellow §6.1, CMU 11-785 Lec 2) opens with is present *only in the slides*, not the prose, so the central claim "a hidden layer lets you fit nonlinear functions" is asserted but never witnessed; and (2) the six activation plots are drawn by inline matplotlib in teaching cells, which violates the repo's own figure convention (illustrative function plots should be pre-generated `img/*.svg`). The single highest-value change is **adding a small worked/constructed XOR example with a 2-D figure** — it is in-scope, it is the canonical wedge, and the asset is already half-built in the slide deck.

**Grade:** B+. Assignable at a top program today on prose and figures, but the "code that only draws" convention break, the missing XOR demonstration, and a live co-equal MXNet tab keep it from an A.

**Top priorities (ranked):**
1. [P1] **Add the XOR demonstration to the prose** (figure + a hand-constructed 2-neuron solution, or a tiny trained net). The book's signature move — *show* linear inseparability and a 2-neuron fix — is currently slide-only. This is the strongest single demonstration of the section's thesis and is explicitly in scope (the "cat/dog" and "body-temperature" examples motivate but never *exhibit* a solvable nonlinear case).
2. [P1] **Move the six activation/derivative plots out of inline matplotlib teaching cells** into pre-generated `img/mdl-mlp-*.svg` (committed generator), per CLAUDE.md and the review guide. These cells *draw* a fixed function; they compute nothing the prose discusses. (See §3 for the nuance — the autograd-derivative cells do have minor teaching value, but the SVG output is a plain function plot.)
3. [P1] **Currency: de-emphasize/tombstone the MXNet tab** (Apache MXNet archived to the Attic, 2023–24) and **upgrade the activation forward-pointer** — the GELU/Swish mention is a good instinct but should also name SwiGLU (LLaMA/PaLM/Mistral FFN default) and frame these explicitly as forward-pointers to the architectures part, not as content to add here.
4. [P2] **Sharpen "Universal Approximators"** with the Goodfellow three-way caveat (existence ≠ learnability ≠ generalization) and a one-line depth-vs-width statement (Hornik 1991 / Leshno 1993; one hidden layer can need exponentially many units). Keep it to a paragraph — full theory is downstream.
5. [P2] **Tighten the Summary** — the "practitioner circa 1990 / coded in C, Fortran, Lisp" anecdote is charming but long; compress and replace the slightly dated "over the past decade" framing.

---

## 1. Coverage

### Add

- **An XOR (linear-inseparability) demonstration — the top gap.** The motivation section (lines 73–151) argues *by appeal* that linear models fail (loan repayment, body temperature, cat/dog pixels) but never exhibits a function a hidden layer provably fixes. Every leading treatment uses XOR as the opening wedge: **Goodfellow et al.** §6.1 constructs the 2-neuron ReLU solution by hand; **CMU 11-785** Lec 2 ("Neural Nets as Universal Approximators") does the same; the digest (lines 15, 106, 138) flags this as the single clearest missing pedagogical device. It is squarely *in scope* (it is the foundational "why hidden layers" demonstration, not downstream material) and the asset is **already drafted in the slide deck** (line 674 lists XOR as the canonical case; the "MLPs add nonlinear hidden layers" slide already names it). Two viable forms, in order of preference:
  - **(a) A pre-generated figure** (`img/mdl-mlp-xor.svg`): four points at the corners of the unit square colored by XOR label, with a single straight line shown failing to separate them, beside the same four points after the hidden-layer feature map, now linearly separable. One `:numref:` and two sentences. This is the cleanest, matches the house figure style, and needs no notebook code.
  - **(b) A short *computed* cell** that constructs the classic 2-hidden-ReLU-unit network with explicit weights and prints the truth table it realizes — this *teaches* (it computes a result the prose discusses), so unlike the activation plots it would legitimately belong in the notebook. Pair with figure (a).
  Suggested placement: a new `#### The XOR Problem` (or a short `### Why One Layer Is Not Enough`) at the end of "From Linear to Nonlinear" (after line 264), immediately *demonstrating* the nonlinearity the prose has just introduced.

- **Universal-approximation caveats (sharpen, don't expand).** Lines 266–294 state the theorem and the nice "C programming language" analogy, but undersell the caveats. Add, in one paragraph, Goodfellow's three-way separation: UAT guarantees a function *exists* that approximates the target; it does **not** guarantee gradient descent will *find* it, nor that it will *generalize* from finite data (digest lines 33, 144). Add one sentence on **depth vs width**: a single hidden layer suffices but may require exponentially many units, whereas depth can represent the same functions far more compactly (Hornik 1991 generalized Cybenko to any bounded non-constant activation; the depth-efficiency intuition — the line already half-makes it at 292–293 with `Simonyan.Zisserman.2014`, which is an odd citation for a *theory* claim; Telgarsky 2016 / Eldan-Shamir 2016 are the on-point depth-separation references). This stays a paragraph; full rigor is correctly deferred ("subsequent chapters", line 294).

- **Activation forward-pointers, modernized (one sentence, not a subsection).** The Summary (lines 619–626) already nods to GELU and Swish — good. Per the scope map, modern activations are forward-pointers, *not* in-scope content, so do **not** add a GELU/SwiGLU subsection. But the pointer should be current: name **SwiGLU** (Shazeer 2020; the FFN default in LLaMA/PaLM/Mistral — digest lines 100, 226) alongside GELU, and frame it as "the activations you will meet in the Transformer chapters," with a `:numref:` once those chapters are settled. Note that **GELU and Swish are already in the slide "cheat sheet"** (lines 828, 833) but the cheat-sheet table never appears in the prose body — see §2.

### Remove / trim

- **The pReLU formula placement** (lines 404–410) sits oddly mid-ReLU; it is fine to keep (it motivates "dead ReLU"), but the section never actually says *why* you'd want it — i.e., it never states the dead-ReLU problem the slides cover well (line 764–772). Either add the one-sentence dead-ReLU motivation to the prose (it currently lives only in the slide deck) or trim the pReLU formula to a footnote. Preference: **add the dead-ReLU sentence** — it is the reason pReLU exists and is in scope.
- **Summary anecdote** (lines 602–612): the "toolkit of a practitioner circa 1990 … code up layers and derivatives explicitly in C, Fortran, or even Lisp (in the case of LeNet)" is delightful but overlong and slightly off-topic for a section whose thesis is nonlinearity. Trim to one sentence.
- **"over the past decade"** (line 617) is a 2021 framing; in 2026 the ReLU resurgence is ~15 years old. Reword to "in the early 2010s" or similar.

### Reorder / restructure

- Structure is **already sound**: 4 `##` sections (Hidden Layers / Activation Functions / Summary / Exercises) with 7 `###` subsections — matches the exemplar `calculus.md` (which runs 6 `##`). No re-spine needed. The only structural add is the XOR subsection above.
- **Do not** import initialization / vanishing-gradient content. The sibling `numerical-stability-and-init.md` owns "Vanishing and Exploding Gradients," "Xavier Initialization," and a "Beyond" subsection — that is exactly where the digest's **He/Kaiming-init** suggestion (digest lines 98, 238) belongs, *not* here. The current forward-pointer in this file ("the well-documented problem of vanishing gradients … more on this later", lines 400–402) is correct and should stay. **Cross-file note for the overview:** He-init is genuinely missing *from the chapter* — flag it against `numerical-stability-and-init.md`, not against this file.

---

## 2. Teaching quality

### Structure & flow

Logical and well-paced: limitations → stacking → the collapse argument → the nonlinearity fix → universal approximation → the activation zoo. The collapse derivation (lines 205–228) is a model of "motivate then interpret" — the "*we gain nothing for our troubles*" beat lands well. The rowwise/elementwise nuance (lines 249–258) is a nice precision most texts skip.

### Figures

- **`fig_mlp`** (line 168, `../img/mlp.svg`): the standard MLP schematic, well-captioned, correctly `:numref:`-referenced (and reused by `dropout.md:142`). Good — keep. It is a pre-generated SVG, the right pattern.
- **The six activation/derivative plots** (`mlp-relu-function-1/2`, `mlp-sigmoid-function-1/2`, `mlp-tanh-function-1/2`): these are the clearest single-page activation reference around (digest lines 21, 136 praise exactly this), but they are **drawn by inline matplotlib in teaching cells** (e.g. lines 332–337: `x = torch.arange(...); y = torch.relu(x); d2l.plot(...)`). I inspected the executed output (`outputs/pytorch/.../mlp/mlp-relu-function-1-1.svg`): it is a plain function plot — a single blue ReLU line — and the manifests confirm **all six cells emit only an SVG, no text output and no warnings**, across all four frameworks. Per CLAUDE.md ("Code teaches; it does not draw … Illustrative figures are pre-generated, never drawn inline") and the review guide ("Flag any figure-drawing matplotlib code living in a teaching cell"), the *function* plots (`-function-1` cells) should become pre-generated `img/mdl-mlp-relu.svg` etc. **Nuance:** the *derivative* cells (`-function-2`) do compute something pedagogically real — they demonstrate `autograd`/`vmap(grad(...))` recovering the analytic derivative — so there is a teaching argument for keeping a derivative cell. The cleanest resolution that honors the convention: pre-generate all six as a clean two-panel SVG per activation (function + derivative side by side, the layout the digest praises), and **optionally** retain *one* tiny autograd cell elsewhere if the chapter wants to show grad-checking. At minimum, the three `-function-1` plots must move to generated SVGs.
- **Missing figure (high value):** the XOR before/after figure described in §1. This is the one figure that would "unlock the idea."
- **Cheat-sheet table** (slide lines 823–836): an excellent comparison table (Range / Saturates? / Use case across ReLU/LeakyReLU/GELU/Sigmoid/Tanh/Softmax) exists *only in the slides*. A trimmed version (the three in-scope activations, with GELU as a forward-pointer row) would be a strong addition to the prose Summary.

### Prose & clarity

Generally excellent — this is among the better-written sections in the chapter. Specific points:

- **Loan-repayment `\$` escaping** (lines 91–94): the source uses `\$0`, `\$50,000`, `\$1 million`, `\$1.05 million`. These are correctly backslash-escaped for the prose, and the rendered page shows them fine. Worth a one-line note to the author per the CLAUDE.md tripwire ("a closing `$` must not be followed by a digit") — these are escaped dollar *signs*, not math, so they are safe, but it is exactly the construct that trips the PDF build if anyone later unescapes them. No change needed; flagging for awareness.
- **"the sigmoid as a special case of the softmax"** (line 442): true and a nice connection, but stated without the one-line reason. Add: "(softmax over the two logits $\{x, 0\}$)" so the reader sees *why*.
- **"piecewise linear"** (line 321): correct for ReLU; the prose could note the payoff that an all-ReLU MLP is therefore a *continuous piecewise-linear* function (Exercise 4 already asks this — line 635) — a one-clause forward nod would tie the exercise to the text.
- **`micchelli1984interpolation` / RBF aside** (lines 271–272): the parenthetical "in a way that could be seen as radial basis function (RBF) networks with a single hidden layer" is dense and tangential for an intro reader. Consider trimming to just the Cybenko statement, with the RBF/RKHS connection as a citation only.

### Exercises

The set (lines 630–641) is good and builds reasonably: collapse-of-linear-stacks (1), three derivative computations (2–3, mechanical), the piecewise-linear structure result (4, genuinely nice), the sigmoid/tanh equivalence (5, the two-parter with the affine-bias hint is excellent), a batch-norm "what could go wrong" prompt (6), and a vanishing-gradient example (7). Suggested improvements:

- **Add an XOR exercise** to pair with the new demonstration: "Find weights for a 2-hidden-unit ReLU network that computes XOR; verify on the four inputs." Mechanical-but-illuminating, and it cements the section's thesis.
- **Add a depth-vs-width exercise:** "Give a function that a depth-2 network represents with $O(k)$ units but a depth-1 network needs $\Omega(2^k)$ units to approximate" (sawtooth/Telgarsky construction) — or, gentler, "explain intuitively why composing ReLUs can double the number of linear pieces per layer." Connects to the new UAT caveat.
- Exercise 6 references `Ioffe.Szegedy.2015` (batch norm) — fine as a forward-pointer, but ensure the reader is told batch norm is covered later (`:numref:` to `chapter_convolutional-modern/batch-norm.md`) so the exercise is not orphaned.

---

## 3. Code & examples

### Does the code teach?

This is the section's weakest dimension against the bar. **All six code cells exist solely to draw a figure.** The `-function-1` cells (`relu`, `sigmoid`, `tanh`) are pure function plots — they compute nothing the prose is discussing beyond "here is the function," which is exactly the "wall of matplotlib plumbing that teaches nothing" the conventions warn against. The `-function-2` derivative cells are *slightly* better — they exercise each framework's autodiff to recover the analytic derivative — but their rendered output is still just a function plot, and the analytic derivative is already given in closed form in the prose (lines 485, 561). Recommendation as in §2: pre-generate the plots; if the chapter wants to retain an autograd demonstration, keep exactly one small, clearly-labeled cell, not six.

Verified against the executed manifests (all four frameworks): every cell's only output is the SVG asset — no stream text, no display text, no warnings or errors. So there is nothing *broken*; the issue is purely the convention break (teaching cell that only draws).

### PyTorch

Idiomatic and modern (torch 2.x); `torch.relu` / `torch.sigmoid` / `torch.tanh` are correct. Two small points:

- The derivative cells use `y.backward(torch.ones_like(x), retain_graph=True)` then re-zero with `x.grad.zero_()` between activations (lines 504, 579). This is correct but is the kind of autograd-bookkeeping that the prose never motivates — another argument for pre-generating the plots. If kept, a one-line comment ("sum-of-outputs trick: `ones_like` makes `backward` compute the elementwise derivative") would help, since vector-Jacobian-product semantics are non-obvious to a learner here.
- `requires_grad=True` on the input (line 334) is only needed because of the manual `.backward()`; with pre-generated plots it disappears entirely.

### JAX

The cleanest of the four: `vmap(grad(jax.nn.relu))` (lines 392, 519, 594) is genuinely elegant and idiomatic — it teaches functional autodiff in one line, far better than the imperative tape/`backward` patterns. If any derivative cell is retained as a teaching artifact, **this is the one to feature**. No Flax/Linen API is used in this file (the imports at lines 48–55 pull only `jax`, `jnp`, `grad`, `vmap`), so the digest's Flax-NNX currency concern (digest line 290) does **not** apply here — it applies to `mlp-implementation.md`.

### TensorFlow

`tf.nn.relu/sigmoid/tanh` with `tf.GradientTape` (lines 384–387 etc.) is correct and reasonably modern. One inconsistency: the function-value cells use `tf.nn.sigmoid` while the input is built with `tf.Variable(tf.range(...))` (line 341) — fine, but the `-function-2` cells **recompute `y` inside the tape** (e.g. lines 511–512) whereas pytorch/mxnet reuse the `y` from the value cell. This is a necessary TF/eager difference (the tape must observe the op), not gratuitous, but it is worth a comment if the cells survive. No Keras-3 concern in this file (no `tf.keras` usage).

### MXNet

`npx.relu` / `npx.sigmoid` / `np.tanh` with `autograd.record()` + `y.backward()` (lines 324–329, 371–373, etc.) is the legacy pattern. The **currency issue dominates the per-framework critique**: Apache MXNet was archived by the ASF (retired 2023-09, repo archived 2023-11, moved to the Apache Attic 2024-02). Presenting it as one of four co-equal selectable tabs in 2026 is misleading to students (digest lines 122, 288). Recommendation (cross-file, for the overview to set policy): de-emphasize or tombstone MXNet book-wide; if kept for now, this file is a clean place to demonstrate the pattern since it has no training loop. Note also that with the plots pre-generated, the MXNet *code* here largely disappears anyway.

### Cross-framework consistency & d2l conventions

- **Imports cell** (lines 26–55): one per-framework imports cell near the top, no later re-imports — compliant. Good.
- **Stable cell IDs**: present and consistent across tabs (`#mlp-relu-function-1`, etc.) — compliant.
- **`d2l.plot` over raw matplotlib**: used (good) — but the cells still *are* figure-drawing cells, which is the deeper convention the plotting helper does not excuse.
- **Gratuitous divergence:** the four derivative implementations diverge about as much as the frameworks force (tape vs `backward` vs `vmap(grad)`). The one avoidable inconsistency is the mxnet/pytorch `y.backward()` reuse vs tf/jax recomputation — but this is minor and mostly evaporates if the plots are pre-generated. JAX's `vmap(grad(f))` is the idiom worth elevating if a teaching cell is kept.

---

## 4. Prioritized change list

| # | Sev | Dimension | Change (specific, actionable) | Effort |
|---|-----|-----------|-------------------------------|--------|
| 1 | P1 | coverage / figures | Add an **XOR demonstration to the prose** (new `#### The XOR Problem` after line 264): a pre-generated `img/mdl-mlp-xor.svg` showing the 4 points + a failing separator, then the same points linearly separable after a 2-ReLU hidden map; optionally a small *computed* cell printing the truth table the constructed network realizes. The asset is already half-built in the slide deck (line 674). | M |
| 2 | P1 | code / figures | **Pre-generate the activation plots** as `img/mdl-mlp-{relu,sigmoid,tanh}.svg` (function + derivative two-panel) via a committed generator; remove the inline figure-drawing teaching cells. Optionally retain one JAX `vmap(grad(...))` cell as an autodiff demonstration. | M |
| 3 | P1 | currency | **De-emphasize/tombstone the MXNet tab** (ASF-archived 2023–24); coordinate book-wide via the overview. | S (this file) |
| 4 | P2 | coverage | **Sharpen UAT** (lines 266–294): add Goodfellow's existence≠learnability≠generalization caveat + one depth-vs-width sentence (Hornik 1991; replace the off-point `Simonyan.Zisserman.2014` theory citation with Telgarsky 2016 / Leshno 1993). | S |
| 5 | P2 | currency | **Modernize the activation forward-pointer** (lines 619–626): name SwiGLU alongside GELU, frame as forward-pointers to the Transformer chapters (`:numref:` once settled). Do **not** add a modern-activation subsection. | S |
| 6 | P2 | teaching | **Add the dead-ReLU sentence** to the prose (currently slide-only, lines 764–772) to motivate pReLU at lines 404–410; consider porting a trimmed activation **cheat-sheet table** (slide lines 823–836) into the Summary. | S |
| 7 | P2 | prose | **Trim the Summary anecdote** (lines 602–612) and fix the dated "over the past decade" (line 617 → "in the early 2010s"). | S |
| 8 | P2 | exercises | Add an **XOR-by-hand** exercise and a **depth-vs-width** exercise; add a `:numref:` to batch-norm for Exercise 6 so it is not orphaned. | S |
| 9 | P2 | prose | Line 442: explain "sigmoid as special case of softmax" in one clause `(softmax over {x, 0})`; line 271–272: trim the dense RBF/RKHS parenthetical to a citation. | S |

---

## 5. Keep — what is already excellent (do not lose this)

- **The motivation prose (lines 73–151).** The monotonicity → body-temperature (non-monotone) → cat/dog-pixels escalation is one of the clearest "why linear models fail" arguments in any text, and the "we don't know how to compute the representation by hand → so we learn it" pivot (lines 132–139) is exactly the right framing for the whole book. The closing historical sweep (decision trees → kernels → splines → biological neurons, lines 141–151) is rich and well-cited. Preserve verbatim.
- **The collapse derivation (lines 205–228)** with the "*we gain nothing for our troubles*" beat — motivates the nonlinearity viscerally before introducing it. Keep.
- **The activation + derivative reference itself** (the *content* and captions, lines 305–596) — the digest singles this out as the clearest single-page activation reference anywhere. The fix in #2 is about *how the plots are produced*, not about cutting them; keep the pedagogy, the derivative formulas (lines 485, 561), and the "max gradient 0.25 at 0" / "approaches linear near 0" annotations.
- **The rowwise-vs-elementwise precision** (lines 249–258) — a subtlety most intro texts gloss; keep.
- **Exercise 5** (sigmoid/tanh equivalence with the affine-bias hint) and **Exercise 4** (all-ReLU ⇒ continuous piecewise-linear) — both genuinely thought-provoking. Keep.
- **The slide deck** is excellent (dead-ReLU, the vanishing-gradient $4^{-10}$ arithmetic, the cheat sheet, XOR) — and is the source from which several recommended prose additions can be lifted. Out of scope to review, but worth noting the prose is *behind* the slides on exactly the points flagged above.
