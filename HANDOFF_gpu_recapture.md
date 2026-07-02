# Handoff: re-running notebooks and slides on the GPU server

*Branch `math-appendix-polish` @ `fb47218` (2026-07-02). Written for the 4×RTX 4090 box (or any
≥2-GPU host with the framework venvs). Everything below is also summarized in the commit history
`64e4e39..fb47218`; the driving documents are `REVIEW_math_appendix.md` and `REVIEW_ch2-5.md`.*

## 1. What this branch contains

Two completed arcs, both executed and verified **pytorch-only** on a GPU-less Mac:

1. **Math appendix polish** (chapters 22–27, `chapter_mdl-*/`): full review-driven rework, two new
   sections (§24.2 `mdl-adaptive-stochastic-methods`, §25.5 `mdl-concentration-generalization`),
   ch. 25 reordered, all 25 mdl slide decks rebuilt to Northstar quality.
2. **Chapters 2–5** (`chapter_preliminaries`, `chapter_linear-regression`,
   `chapter_linear-classification`, `chapter_multilayer-perceptrons`): 14 confirmed errors fixed,
   ~20 new teaching cells, ~17 new/promoted figures, ~20 appendix hand-offs, heading restructures,
   and all 28 slide decks rebuilt to Northstar quality (plus a new `lookup-api` deck).

The **pytorch** output store (`outputs/pytorch/...`) is fully re-captured and fresh for every
edited file. The **tensorflow / jax / mxnet** stores are the debt this handoff pays off.

## 2. What needs re-execution here

- **tf / jax / mxnet outputs for all 53 edited chapter files** (list: `git diff --name-only
  main..math-appendix-polish -- 'chapter_*' | grep '\.md$'`). Until then, non-pytorch site tabs
  and slide decks show the new cells without outputs; the decks degrade gracefully via
  `only="pytorch"` evidence slides with text fallbacks.
- **Two files have NO tf/jax/mxnet store JSON at all** (new files / newly scaffolded):
  `chapter_mdl-optimization/mdl-adaptive-stochastic-methods.md`,
  `chapter_mdl-probability-statistics/mdl-concentration-generalization.md`, and the ch4 files that
  gained their first code scaffold: `chapter_linear-classification/generalization-classification.md`,
  `chapter_linear-classification/environment-and-distribution-shift.md`. **The `--stale` audit may
  not list files whose manifests are missing entirely — force these explicitly (step 3b).**
- **`chapter_computational-performance/multiple-gpus.md` (pytorch)** — HARD-stale on the d2l lib
  fingerprint (an `eps` rename in `SyntheticRegressionData` regenerated `d2l/{jax,mxnet,tensorflow}.py`).
  Needs ≥2 GPUs; this box has them.
- **mxnet `chapter_linear-classification/image-classification-dataset.md`** — its stored outputs
  are buried in oneDNN storage-fallback warnings. Re-capture with
  `MXNET_STORAGE_FALLBACK_LOG_VERBOSE=0` exported (recommend exporting it for the whole run).
- Note: `kaggle-house-price.md`'s tf/jax/mxnet tabs were *edited* (100-epoch competent baseline,
  `model_fn` interface, log-space ensemble) but never executed — their old 10-epoch outputs
  contradict the new code until re-captured.

## 3. Recipe

```bash
git fetch && git checkout math-appendix-polish
gmake detect                      # sanity: GPUs + slot plan
export MXNET_STORAGE_FALLBACK_LOG_VERBOSE=0

# (a) regenerate lib + all four frameworks' notebooks, then let the audit drive:
gmake lib
gmake notebooks                   # staleness across tf/jax/mxnet is only computable after this
gmake refresh-stale               # re-executes every stale notebook (all fw), re-captures per file

# (b) force the files whose non-pytorch manifests don't exist (audit may skip them):
for f in chapter_mdl-optimization/mdl-adaptive-stochastic-methods \
         chapter_mdl-probability-statistics/mdl-concentration-generalization \
         chapter_linear-classification/generalization-classification \
         chapter_linear-classification/environment-and-distribution-shift; do
  for fw in tensorflow jax mxnet; do
    gmake -B "_notebooks/$fw/$f.executed" || true    # skip if no notebook for that fw
  done
  gmake capture-outputs FILES="$f.md"
done

# (c) verify + rebuild everything:
gmake verify-outputs-fresh        # must be strict-clean on a >=2-GPU host
gmake slides                      # all four frameworks, CPU-only, parallel-safe (-j4 ok)
gmake html
gmake -j4 pdfs                    # known benign warning: TeX '4/\sqrt{3}' in a bib title
```

Alternative pristine path (~2.5–3 h): `gmake clean && gmake all`, then bless everything with
`gmake capture-outputs` (no `FILES=` captures every executed notebook), then step (c). Use this if
`refresh-stale`'s serial per-file loop is too slow for the stale set — `gmake all` runs the
scheduler with full GPU parallelism.

Commit the refreshed `outputs/` (JSON manifests + LFS images) when the gate is clean.

## 4. After the recapture: optional slide cleanups

The ch3/ch4/ch5 decks carry `only="pytorch"` evidence slides paired with `except="pytorch"`
text/formula fallbacks, solely because the other stores lacked the new cells. Once the stores are
fresh, those fallback slides can be retired (delete the `except=` twin, drop the `only=` from the
evidence slide) in: `linear-regression.md` (MAE demo), `generalization.md` (bias²/variance),
`weight-decay.md` (spectral shrinkage), `softmax-regression-scratch.md` (accuracy sweep, confusion
matrix), `softmax-regression-concise.md` (lse-vs-max), `generalization-classification.md` (both
simulations), `environment-and-distribution-shift.md` (correction demo), `mlp.md` (XOR/region-count
fallbacks), `numerical-stability-and-init.md` (depth-sweep verdict), `kaggle-house-price.md`
(baseline verdict). Keep the fallbacks where the cell is genuinely pytorch-only in source
(`backprop-verify`, `mlp-region-count`). After edits: re-render + `tools/check_slide_overflow.py`
(needs playwright — run under a venv that has it, as the slide agents did).

Quoted "receipts" in slide titles/captions are framework-invariant (deterministic or seeded), so
no caption should go wrong after recapture; two soft spots were written run-robust on purpose
(Fashion-MNIST "82–83%", kaggle linear-vs-MLP "dead heat").

## 5. Known non-blockers

- `:numref:` does not resolve inside slide decks — all decks use plain `§x.y` literals. Flagged
  tooling idea: a `@sec:` resolver in `tools/gen_slides.py`.
- `mdl-clf-decision-regions.svg` labels render small at slide scale (fine on book pages); a
  slide-scale variant would be a `tools/gen_mdl_classification_figures.py` change.
- Legacy-style figures slated for eventual house-style redraws (explicitly deferred L-items):
  `fit-linreg`/`singleneuron`/`capacity-vs-error` (ch3, one already swapped), `softmaxreg.svg`
  (ch4), `mlp.svg`/`dropout2.svg` (ch5).
- Four pre-existing same-title bib key pairs in old chapters — left alone.
- Branch not merged to `main` — pending review.

## 6. Verification bar that this branch already passes (pytorch, on the Mac)

`gmake html` green (0 errors, all crossrefs resolve); freshness gate clean modulo the items in §2;
figures byte-idempotent (`gmake figures` twice); all 28 ch2–5 decks + 25 mdl decks render green
with 0 overflows at 720 px; `tools/audit_slides.py` clean; `tools/lint_source.py` clean. The same
bar should hold here for all four frameworks after §3.
