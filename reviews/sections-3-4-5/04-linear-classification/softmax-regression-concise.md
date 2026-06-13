# Review — chapter_linear-classification/softmax-regression-concise.md  (§4.5 "Concise Implementation of Softmax Regression")

**Role in the chapter:** The "concise" counterpart to §4.4 (scratch): re-implements softmax regression on Fashion-MNIST with framework primitives (one linear layer + the built-in fused softmax-cross-entropy loss). Its load-bearing job — and the reason it exists separately from the regression `*-concise` file — is to **own the numerical-stability story**: derive the log-sum-exp trick and explain *why* frameworks fuse softmax+cross-entropy and consume logits. The sibling `softmax-regression.md` review (already filed) explicitly forward-points to **this file's** `subsec_softmax-implementation-revisited` as the canonical derivation site, so this section is the single source of truth for log-sum-exp in the chapter.

**Verdict:** Close to the bar and clearly the right design, but the centerpiece derivation is **not yet airtight or elegant**. It develops the max-subtraction fix and the `log 0` failure mode well, then stops at `log ŷ_j` and never writes the one object the whole section is about: the cross-entropy loss in stable logit form, `ℓ = log Σ_k exp(o_k − ō) + ō − o_y`. The clean LogSumExp identity exists only in the *slide*, not the prose. The single highest-value change is to finish the derivation in the body (one display + two sentences) so the fused loss is *derived*, not merely *invoked*. Per-framework loss code is correct in all four tabs (logits handled properly everywhere). A secondary issue is cross-file coherence: §4.4 was revised to add a clamp, and this section's prose still describes §4.4 as plainly "risky" without acknowledging it or explaining why fusion beats clamping.

**Grade:** B. Correct, well-organized, code is right — but the marquee derivation is incomplete (P0 to finish it) and the prose has a cross-file seam (P1). With SMC-1 and SMC-2 done this is a solid A− foundations section.

**Top priorities (ranked):**
1. [P0] **SMC-1** — Finish the log-sum-exp derivation in the body: write the cross-entropy loss in stable logit form and state the identity the loss actually computes. Right now the math stops one line short of the point.
2. [P1] **SMC-2** — Reconcile with the revised §4.4: acknowledge the scratch clamp and explain *why* the fused loss is strictly better (no unstable softmax is ever formed; gradient is exact), instead of implying §4.4 is simply broken.
3. [P1] **SMC-3** — Tie the derivation explicitly to the four code tabs: a single sentence mapping "pass logits, not probabilities" to `from_logits=True` / `CrossEntropyLoss` / `softmax_cross_entropy_with_integer_labels` / `SoftmaxCrossEntropyLoss`, so the derivation and the code are visibly the same thing.
4. [P2] **SMC-4** — Fix the FP32 range statement (the underflow/overflow bounds are stated loosely/asymmetrically) and quantify the "reasonably accurate" training claim.
5. [P2] **SMC-5** — Replace the two generic hyperparameter exercises (which duplicate §4.4) with a hands-on "trigger and fix the instability" exercise that exercises *this* section's content.
6. [P2] **SMC-6** — Currency: flag the pre-Keras-3 `tf.keras.losses.Reduction` enum and the legacy `flax.linen` pattern (book-wide; defer to overview). MXNet tab presence is a book-wide decision (defer).

---

## 1. Coverage

### Add

- **The loss in logit form — the missing climax (P0, SMC-1).** §4.5 derives, beautifully, the stabilized softmax
  $$\hat y_j = \frac{\exp(o_j-\bar o)}{\sum_k \exp(o_k-\bar o)} \quad\Rightarrow\quad \log\hat y_j = o_j - \bar o - \log\textstyle\sum_k \exp(o_k-\bar o)$$
  (ll. 157–183) and then says "instead of passing softmax probabilities … we just pass the logits and compute the softmax and its log all at once inside the cross-entropy loss" (ll. 186–192). But it never writes **the loss**. Cross-entropy on an integer label $y$ is $\ell = -\log\hat y_y$, so substituting the line above gives the single clean object the entire section is building toward:
  $$\ell(y, \mathbf{o}) = \log\sum_k \exp(o_k) - o_y = \Big(\bar o + \log\sum_k \exp(o_k-\bar o)\Big) - o_y.$$
  The first form is exactly the **log-sum-exp** function (named on the slide, l. 320, but absent from the body); the second is its numerically stable evaluation. *This* is what `cross_entropy(logits, y)` computes. Leaving it implicit is the one thing keeping this section below the Stanford/CMU bar — the reader is shown the stabilized $\log\hat y_j$ and then asked to take the fused loss on faith. Drafted prose in §4 (SMC-1). The sibling §4.1 review *forward-points here for exactly this derivation*, so it must be complete here.

- **Why fusion beats clamping — the §4.4 bridge (P1, SMC-2).** §4.4 (scratch) was revised (see `softmax-regression-scratch.md` ll. 261–300) to clamp $\hat y$ away from $0$ (`clip(min=1e-12)`) and now explicitly says "Production code typically uses a log-softmax layer that fuses the softmax and log into a single numerically stable operation; the explicit clamp here is the minimal change…". This file (ll. 140–143) still opens "In :numref:`sec_softmax_scratch` we calculated our model's output and applied the cross-entropy loss. While this is perfectly reasonable mathematically, it is risky computationally" — which now slightly misreads its own sibling (the sibling already mitigated the risk and pointed *here* for the real fix). The fix is not to delete the framing but to *complete the arc*: the clamp prevents `log 0` but **still forms the overflow-prone softmax first**, and it silently zeroes the gradient on clamped entries; the fused loss never materializes the softmax at all and keeps the gradient exact ($\partial_{o_j}\ell = \mathrm{softmax}(\mathbf o)_j - y_j$, the identity §4.1 already proved). Drafted in §4 (SMC-2). This closes the loop scratch→concise cleanly.

- **One sentence binding derivation ↔ code (P1, SMC-3).** The four `loss` tabs (ll. 194–242) are the payoff of the derivation but the prose never names the mechanism in each framework. Add one sentence after the loss cells listing how each framework spells "I am giving you logits, not probabilities": PyTorch `F.cross_entropy` (logits by definition), TensorFlow `SparseCategoricalCrossentropy(from_logits=True)`, JAX `optax.softmax_cross_entropy_with_integer_labels`, MXNet `gluon.loss.SoftmaxCrossEntropyLoss` (`from_logits=False` default ⇒ it applies the stable softmax internally). This is the single highest-leverage way to make "pass logits" concrete and to inoculate readers against the classic double-softmax bug. Drafted in §4 (SMC-3).

### Remove / trim

- **Nothing major to cut.** The section is already lean (5 `##` sections, fits on a screen-and-a-half). Do **not** import softmax theory (§4.1 owns it) or re-teach cross-entropy from probability (§4.1/§4.4 own it) — the section correctly stays in its lane and should keep doing so.
- **Trim the Summary's editorializing (P2, optional).** The Summary (ll. 262–265) is a good "blessing and a curse" essay about high-level APIs, but it is generic to *any* concise chapter and says nothing specific about the numerical-stability lesson this section just taught. Consider adding one closing sentence that names the takeaway ("the framework's fused loss is not just fewer lines — it is the *correct* implementation you should not hand-roll"), so the Summary lands the section's actual point rather than a generic API homily. Low priority.

### Reorder / restructure

- **Promote "Softmax Revisited" conceptually, keep it positionally.** The order (Model → Softmax Revisited → Training) is right: define the model, then explain why the loss is built the way it is, then train. No structural reorder needed. The `### subsec` is currently a flat `## Softmax Revisited`; that is fine for a short file. (If SMC-1+SMC-2 expand it, consider two `###` subheads: "The overflow/underflow problem" and "Fusing softmax and cross-entropy" — judgment call, not required.)

## 2. Teaching quality

### Structure & flow

Clean 5-section spine matching the regression `*-concise` sibling, which is the right parallel to draw. The "Softmax Revisited" interlude between Model and Training is well-placed pedagogically. The one flaw is that the section's intellectual core (the derivation) currently *trails off* into a prose hand-wave ("does smart things like the LogSumExp trick", l. 192) exactly where it should *culminate* in an equation — see SMC-1.

### Figures

- **No illustrative figures, and none needed.** The only visual is the training loss/accuracy curve (`softmax-regression-concise-training-1.svg`, present for all four frameworks, 22,395 bytes each — confirmed in `outputs/<fw>/.../softmax-regression-concise/`). That is a legitimate *computed* data plot from `d2l.plot`, not a schematic, so it does not fall under the "pre-generate illustrative figures" rule. No inline figure-drawing matplotlib in any cell. Clean.
- **No missing figure rises to P1.** One could imagine a tiny schematic contrasting "naive: softmax → log → NLL (three ops, overflow then `log 0`)" vs "fused: logits → log-sum-exp (one op, stable)", but this is comfortably carried by the equations in SMC-1 and is not worth a generator. Skip.

### Prose & clarity

- **ll. 152–153, FP32 bounds are loose (P2, SMC-4).** "single precision floating point numbers approximately cover the range of $10^{-38}$ to $10^{38}$. As such, if the largest term in $\mathbf{o}$ lies outside roughly $[-88, 88]$, the result will not be representable in FP32." Two issues: (i) $\log(10^{38})\approx 87.5$ governs *overflow* (upper bound ~88 ✓), but the *underflow* threshold is $\log(10^{-38})\approx -87.3$, so the symmetric "$[-88,88]$" elides that a *single* very-negative logit underflows its own $\exp$ to 0 long before the sum does — which is fine for the sum but is the exact mechanism behind the `log 0` problem two sentences later. (ii) "if every argument is a very large negative number, we will get underflow" (l. 150) is the imprecise version of this. Tighten to separate the two regimes cleanly (drafted in SMC-4).
- **l. 192 "does smart things like the LogSumExp trick" is the weakest sentence in the section.** It is the moment the prose should be most precise and is instead most vague. Fixed by SMC-1 (replace the hand-wave with the actual identity).
- **ll. 255–257 "reasonably accurate" is unquantified (P2, SMC-4).** The training cell prints nothing (manifests show only the plot asset, no stream/text output in any framework), so the reader cannot check the number. Either state the expected validation accuracy (~0.83–0.84 on Fashion-MNIST, matching §4.4) or point to the plot. Minor.
- The overflow/underflow derivation itself (ll. 145–183) is genuinely good — clear, well-motivated, correct algebra. Protect it (see §5).

### Exercises

The four exercises (ll. 270–274):
1. **Number formats / smallest-largest safe exponent argument** — *excellent and on-theme*; directly exercises this section's content (the $[-88,88]$ bound generalized across FP64/FP32/BF16/FP16/TF32/INT8). Keep. (Consider tabulating the answer per format in a solution slot.)
2. **INT8 dynamic-range extension** — good, stretches into quantization (a real 2026 concern); keep. Slightly tangential but rewards curiosity.
3. **"Increase epochs, why does val accuracy decrease, how to fix"** — generic overfitting prompt; **duplicates** §4.4 exercise 5 and the regression-concise exercises. Low value here.
4. **"Increase the learning rate, compare loss curves"** — also generic and **duplicates** §4.4's hyperparameter exercise. Low value here.

**Recommendation (SMC-5):** keep 1–2; replace 3–4 with one exercise that makes the *numerical* point hands-on (e.g., "construct logits with a $+1000$ entry; verify the scratch softmax from §4.4 returns `nan`/`inf` while the framework's `cross_entropy` on the same logits returns a finite loss; then verify the two agree to floating-point precision on benign logits"). That is the exercise a top course would set for *this* section, and there is currently none.

## 3. Code & examples

### Does the code teach?

Yes — every cell pulls weight: imports, the one-linear-layer model, the fused-loss override, training. No boilerplate-for-boilerplate. The `loss` override is the teaching centerpiece and the four tabs are appropriately terse. One subtlety worth a prose note (SMC-3): the `loss` is `@d2l.add_to_class(d2l.Classifier)` with `#@save`, i.e. it **overrides the base-class `loss`** with the stable fused version for all downstream classifiers — a nice design point currently left silent.

### PyTorch

- `F.cross_entropy(Y_hat, Y, reduction='mean' if averaged else 'none')` (ll. 196–201) — **correct and idiomatic.** `F.cross_entropy` consumes raw logits and internally does `log_softmax` + `nll_loss` (the stable path). No double softmax. `import torch.nn.functional as F` (l. 29) is used and standard. Modern (this is unchanged 1.x→2.x idiom). Matches the `d2l/torch.py` library copy (l. 541). 
- Model: `nn.Sequential(nn.Flatten(), nn.LazyLinear(num_outputs))` (ll. 89–90) — fine; `LazyLinear` is the modern choice (consistent with the regression-concise sibling). No issue.

### JAX

- `optax.softmax_cross_entropy_with_integer_labels(Y_hat, Y)` (l. 238) — **correct and the modern Optax idiom**: takes logits + integer labels, fused and stable, no double softmax. The `@partial(jax.jit, static_argnums=(0,5))` and the `(loss, {})` aux-dict tuple are consistent with the `Classifier` base (`classification.md`) and the BatchNorm-forward-compat convention. Forward pass (`@nn.compact`, ll. 130–134) returns logits (no softmax) — correct.
- Currency (book-wide, SMC-6): uses `flax.linen` (`@nn.compact`), the pre-`flax.nnx` API. This is consistent across the whole book; flag for the overview, do not change in isolation.

### TensorFlow

- `SparseCategoricalCrossentropy(from_logits=True, reduction=...)` (ll. 223–225) — **correct**: `from_logits=True` is precisely the "pass logits" contract; with integer labels `SparseCategorical*` is the right loss; the model outputs logits (no softmax, ll. 116–118). No double softmax. Good.
- **Currency (P2, SMC-6):** `tf.keras.losses.Reduction.SUM_OVER_BATCH_SIZE` / `.NONE` (ll. 221–222) is the **pre-Keras-3** enum. Keras 3 (the default in TF ≥ 2.16) moved/renamed this surface (string reductions like `"sum_over_batch_size"`; the `Reduction` class relocated). The committed store executed under `tensorflow==2.21.0` (provenance), so it currently runs, but this is the kind of dated-API tripwire the currency lens asks us to flag. The same enum is used in `gan.md`/`dcgan.md`, so this is a book-wide consistency item — flag for the overview rather than patch one file. (Weight: TF is a secondary framework; do not over-invest.)

### MXNet

- `gluon.loss.SoftmaxCrossEntropyLoss()` (l. 210) then `fn(Y_hat, Y)` — **correct.** Its default `from_logits=False` means it expects **logits** and applies the stable `log_softmax` internally (the MXNet analogue of `from_logits=True`); the model's `Dense` outputs raw logits (no softmax, ll. 103–106). No double softmax. The naming is the easiest of the four to misread, which is exactly why SMC-3's one-line mapping is valuable.
- Book-wide: **Apache MXNet was archived by the ASF (2023).** Presenting it as a co-equal first tab is the standard currency concern across these chapters; this is a book-wide decision (de-emphasize vs. drop), not something to resolve in this file. Defer to the overview. No file-local action.

### Cross-framework consistency & d2l conventions

- **Single imports cell per framework** (ll. 16–46), no re-imports later — convention satisfied.
- **Stable cell IDs** present and shared across tabs; `#@save` on `SoftmaxRegression` and on the `loss` override — hygiene OK.
- **Divergence is framework-mandated, not gratuitous.** The four `loss` bodies differ only as much as the four loss APIs require (reduction handling, JAX purity/aux-tuple). Acceptable.
- **One real consistency seam is cross-file, not cross-framework:** the scratch↔concise narrative (SMC-2). The four frameworks here are internally consistent.

## 4. Implementation spec (downstream agents act on THIS)

### SMC-1 — Finish the log-sum-exp derivation: write the loss in logit form  ·  [P0] · [M] · [authored]
- **Type:** coverage / teaching
- **Where:** `chapter_linear-classification/softmax-regression-concise.md`, in the "## Softmax Revisited" section. Anchor: the paragraph beginning "Fortunately, we are saved by the fact that even though we are computing exponential functions" (l. 172) through the sentence ending "…the ["LogSumExp trick"](https://en.wikipedia.org/wiki/LogSumExp)." (l. 192), and the display equation at ll. 179–183.
- **Change (authored):** Keep the existing display `log ŷ_j = … = o_j - ō - log Σ_k exp(o_k - ō)` (ll. 179–183). Immediately *after* it, before the "Fortunately…" paragraph, the derivation must reach the loss. Replace the paragraph at ll. 186–192 ("We will want to keep the conventional softmax function handy … the ["LogSumExp trick"]…") with:

  > For an example with true class $y$, the cross-entropy loss is $\ell = -\log \hat y_y$. Substituting the stabilized expression above turns the loss into a function of the **logits alone**:
  >
  > $$\ell(y, \mathbf{o}) = \log \sum_{k} \exp(o_k) - o_y = \underbrace{\bar o + \log \sum_{k} \exp(o_k - \bar o)}_{\textrm{numerically stable}} - o_y, \qquad \bar o = \max_k o_k.$$
  >
  > The first term, $\log\sum_k \exp(o_k)$, is the **log-sum-exp** function (a smooth upper bound on $\max_k o_k$); the second equality is the only safe way to evaluate it, since every exponent $o_k - \bar o \le 0$. This is precisely what the built-in cross-entropy loss computes when handed raw logits: it never forms the softmax probabilities at all, so neither $\exp$ of a large number nor $\log$ of a zero ever occurs. We keep the explicit softmax (:numref:`sec_softmax_scratch`) only for *reading off* predicted probabilities at inference time; for the loss we pass logits and let the fused operation do the rest.

  (Mind the gotcha rules: `$$` alone on its own lines; no `$`-immediately-before-digit. The `\underbrace` renders in both HTML and the PDF preamble.)
- **Touches:** none (prose + one display equation).
- **Done when:** the body contains the display `ℓ(y,𝐨) = log Σ exp(o_k) − o_y`; the term "log-sum-exp" appears in the *prose body* (not only the slide); `make html` renders the new equation with no raw LaTeX; PDF builds clean.
- **Depends on:** none.

### SMC-2 — Reconcile with the revised scratch section (clamp vs. fusion)  ·  [P1] · [S] · [authored]
- **Type:** coverage / prose / correctness
- **Where:** `chapter_linear-classification/softmax-regression-concise.md`, opening paragraph of "## Softmax Revisited". Anchor: "In :numref:`sec_softmax_scratch` we calculated our model's output and applied the cross-entropy loss. While this is perfectly reasonable mathematically, it is risky computationally, because of numerical underflow and overflow in the exponentiation." (ll. 140–143).
- **Change (authored):** Replace that sentence with:

  > In :numref:`sec_softmax_scratch` we computed the softmax explicitly and then took its logarithm in the cross-entropy loss. To keep that version usable we had to *clamp* the probabilities away from zero — a band-aid that prevents $\log 0$ but still forms the overflow-prone softmax first and silently kills the gradient on any clamped entry. Here we remove the problem at its source rather than patch its symptom.

  Then, after the new loss display from SMC-1, add one sentence tying the gradient back to §4.1:

  > Because the fused loss differentiates the exact expression above, its gradient is the clean $\partial_{o_j}\ell = \mathrm{softmax}(\mathbf o)_j - y_j$ derived in :numref:`subsec_softmax_and_derivatives` — with no clamp to perturb it.

- **Touches:** none. (Cross-file *coordination* only: this matches `softmax-regression-scratch.md` ll. 261–300, which already point here; no edit to that file required.)
- **Done when:** the opening paragraph no longer implies §4.4 is simply broken; it names the clamp and explains why fusion is strictly better; the `:numref:` to `subsec_softmax_and_derivatives` resolves (that label exists in `softmax-regression.md` l. 334).
- **Depends on:** SMC-1 (the gradient sentence references SMC-1's display).

### SMC-3 — One sentence mapping "pass logits" to each framework's loss API  ·  [P1] · [S] · [authored]
- **Type:** code / teaching
- **Where:** `chapter_linear-classification/softmax-regression-concise.md`, immediately *after* the four `loss` code cells (after l. 242, before "## Training" at l. 244).
- **Change (authored):** Insert:

  > Each framework exposes this fused loss under a slightly different name, but all four take **logits**, not probabilities — passing softmax outputs would apply the softmax twice. PyTorch's `F.cross_entropy` consumes logits by definition; TensorFlow uses `SparseCategoricalCrossentropy(from_logits=True)`; JAX/Optax provides `softmax_cross_entropy_with_integer_labels`; and MXNet's `SoftmaxCrossEntropyLoss` (with its default `from_logits=False`) applies the stable softmax internally. Correspondingly, the model's `forward` returns raw logits and contains **no** softmax — the loss owns that step. We defined this `loss` on the base `Classifier` (note the `#@save`), so every classifier in the rest of the book inherits the numerically stable version.

- **Touches:** none.
- **Done when:** the four API names appear with the correct logits contract; the "no softmax in forward" point is stated; `make html` clean. (If SMC-6 changes the TF reduction surface, keep the API name in sync.)
- **Depends on:** none (complements SMC-1).

### SMC-4 — Tighten the FP32 range statement and quantify "reasonably accurate"  ·  [P2] · [S] · [mechanical]
- **Type:** prose / correctness
- **Where:** `chapter_linear-classification/softmax-regression-concise.md`, ll. 149–153 and ll. 255–257.
- **Change (mechanical):**
  - Replace (ll. 149–153) `Likewise, if every argument is a very large negative number, we will get *underflow*. For instance, single precision floating point numbers approximately cover the range of $10^{-38}$ to $10^{38}$. As such, if the largest term in $\mathbf{o}$ lies outside roughly $[-88, 88]$, the result will not be representable in FP32.`
    → `Conversely, a strongly negative $o_k$ makes $\exp(o_k)$ *underflow* to $0$. Single-precision floats span roughly $10^{-38}$ to $10^{38}$, i.e. $\exp$ overflows once an argument exceeds about $88$ and underflows to $0$ once it drops below about $-88$. So a single large positive logit overflows the numerator, while strongly negative logits underflow individual terms to $0$ — harmless in the sum, but fatal once we take a logarithm.`
  - Replace (ll. 255–257) `As before, this algorithm converges to a solution that is reasonably accurate, albeit this time with fewer lines of code than before.`
    → `As before, training converges to about 83--84% validation accuracy — the same solution as the from-scratch version of :numref:`sec_softmax_scratch`, now in far fewer lines of code.`
- **Touches:** none.
- **Done when:** both passages updated verbatim; renders clean; the accuracy figure matches the §4.4 result (cross-check the scratch training curve, ~0.83).
- **Depends on:** none.

### SMC-5 — Replace two generic exercises with a hands-on instability exercise  ·  [P2] · [S] · [authored]
- **Type:** teaching (exercises)
- **Where:** `chapter_linear-classification/softmax-regression-concise.md`, "## Exercises", items 3 and 4 (ll. 273–274: "Increase the number of epochs…" and "What happens as you increase the learning rate…").
- **Change (authored):** Remove items 3 and 4 (they duplicate §4.4 exercise 5 and the regression-concise hyperparameter exercises) and replace with:

  > 3. Take the from-scratch `softmax` of :numref:`sec_softmax_scratch` and feed it the logits $\mathbf o = (1000, 0, 0)$. What do you get, and why? Now compute the loss for the same logits with the framework's `cross_entropy` (passing the logits directly). Why is it finite? Verify that on *benign* logits, e.g. $\mathbf o = (2, 1, 0)$, the two routes agree to floating-point precision.
  > 4. Show, using the identity $\ell = \log\sum_k \exp(o_k) - o_y$, that adding the same constant $c$ to every logit leaves the loss unchanged. Why does this make subtracting $\bar o = \max_k o_k$ a free and safe choice?

- **Touches:** none.
- **Done when:** the two generic items are gone; the two new items are present and reference this section's own content (the log-sum-exp identity, shift-invariance); no duplication with §4.4 exercise 5.
- **Depends on:** SMC-1 (item 4 references the SMC-1 identity).

### SMC-6 — Currency flags: pre-Keras-3 Reduction enum; legacy linen; MXNet tab  ·  [P2] · [S] · [judgment]
- **Type:** currency
- **Where:** `chapter_linear-classification/softmax-regression-concise.md` — TF `loss` cell ll. 221–222 (`tf.keras.losses.Reduction.SUM_OVER_BATCH_SIZE` / `.NONE`); JAX imports/model (`flax.linen`, ll. 41, 130); MXNet tab throughout.
- **Change (judgment):** These are **book-wide** decisions, not file-local edits. Do **not** patch them in isolation. (a) The `tf.keras.losses.Reduction` enum is pre-Keras-3 and also appears in `gan.md`/`dcgan.md` (ll. 463/787) — if/when the book migrates TF to Keras 3, update all together (string reductions / relocated `Reduction`). (b) `flax.linen` is the pre-`nnx` API used book-wide. (c) MXNet was archived by the ASF (2023); whether to de-emphasize or drop the tab is a book-wide call. **Action for this review:** record all three in the cross-cutting overview; take no isolated edit here.
- **Touches:** would touch many files + the committed `outputs/` store + framework venvs if acted on — hence book-wide only.
- **Done when:** the three items are listed in the overview's book-wide-decisions section. (No file edit expected from this entry alone.)
- **Depends on:** the cross-cutting overview.

## 5. Keep — what is already excellent (do not lose this)

- **The overflow/underflow → max-subtraction derivation (ll. 145–183).** Genuinely clear and correct: it motivates the problem from FP range, derives the shift-invariant stabilized softmax, identifies the residual `log 0` failure mode, and sets up the fusion. This is the spine SMC-1/SMC-2 build *on top of* — do not rewrite it, only finish it.
- **All four `loss` implementations are correct** (logits handled properly; no double softmax in any tab). Protect them; SMC-3 only adds explanatory prose around them, no code change.
- **The lean 5-section structure** mirroring the regression-concise sibling — the right parallel; keep it.
- **Exercises 1–2** (number formats; INT8 dynamic range) — on-theme, modern, and the kind of question that rewards a strong student. Keep.
- **The Summary's "blessing and a curse" framing** about high-level APIs hiding sharp edges — it is the honest, distinctive voice the book is known for; keep it (SMC, optional, only *adds* a closing line).
