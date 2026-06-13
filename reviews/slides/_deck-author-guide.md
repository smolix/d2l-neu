# Deck-author guide — generate one north-star slide deck per section

You are an **Opus slide-deck author**. Produce an **outstanding, top-5-university**
reveal.js deck for ONE section, matching the **§2.1 reference's visual + teaching
bar**. The author reviews the finished deck, so **confirm visual quality, do not
assume it**.

## Study the bar FIRST (do not skip)
- **`docs/slides/north-star.html`** — the §2.1 visual exemplar. **This is the bar.**
  Open it (read the file / screenshot it) so you know what "excellent" looks like.
- **`chapter_preliminaries/ndarray.md`** `<!-- slides -->` block — the same deck in
  the repo's authoring format (titles, `@id`, layouts, fragments, bookends).
- **`docs/slides-northstar-design.md`** — §3 (conceptual style: *one idea per
  slide, lead with the diagram, code illustrative not exhaustive*), the grammar
  (§2), and the **§8 acceptance checklist**. **`docs/slides.md`** — visual
  vocabulary + the **overflow sweep / quality rules**. These govern everything.

## Author the deck
1. Read your section `.md` (prose + code cells and their `#id`s) and the executed
   notebooks `_notebooks/<fw>/<chapter>/<file>.ipynb` for **all four frameworks**
   (note the outputs you will show).
2. **REPLACE** the existing `<!-- slides -->` block (disregard it) with a new one
   that is genuinely excellent:
   - **One idea per slide; lead with a diagram or the intuition.** Code is
     support: `@id` (code + output) or `@!id` (output only); **curate** — drop
     cells that don't teach. No walls of code or prose.
   - **Reuse the existing polished house-style figures** `img/mdl-<slug>-*.svg`
     (`linreg`/`clf`/`mlp`) via a plain markdown image line inside the slide, for
     structural/geometric ideas (they were just refined to a high bar). Only
     **flag** (do not author) a genuinely new diagram if one is truly needed.
   - **Concise ORIGINAL captions** (1–2 lines; never paste chapter prose).
     Fragments (`. . .`, top-level only) to stage a build. **Framework-specific
     framing** via `only="fw"` / `except="fw"` where the concept differs (JAX
     immutability/PRNG, TF `Variable`/Keras, MXNet). **Bookends:** a cover, a
     why/what opener, section dividers, a recap.
   - **PyTorch-only cells** (a cell with no jax/tf/mxnet sibling) can only render
     on the pytorch tab — scope those slides `only="pytorch"` and **list them** in
     your report.
3. **Triage staleness:** if a notebook errors / uses dead APIs / teaches something
   obsolete, **flag it** and keep the deck honest — do not invent outputs.

## Build + VERIFY (this is the "confirm, don't assume" step)
- Build your deck: `make -B slides-pytorch SLIDES_FILTER=<chapter>/<file>.md`
  (slide builds are CPU-only + parallel-safe; if it flakes under concurrency,
  retry once).
- `.venv-build/bin/python tools/audit_slides.py` — slide↔cell integrity must be clean.
- Run the **overflow sweep** (`docs/slides.md` → Quality rules): **no slide may
  overflow 720 px.** Fix by trimming / `@-` code-only / splitting / a wider
  layout — never a per-slide scrollbar.
- **View it.** Render + screenshot your built deck and look at it (per
  `docs/slides-northstar-design.md` §10: serve the built HTML, Playwright with
  `Reveal.configure({transition:'none',fragments:false})`, navigate
  `Reveal.slide(0, v)`, a `?v=N` cache-buster). Inspect the cover, a code+figure
  slide, and the densest slide. **Iterate until it matches the §2 bar** — clean,
  uncluttered, no overflow, diagram-first. If you genuinely cannot screenshot,
  pass the automated checks and say so explicitly so the main thread does the
  visual pass.

## Rules
- Edit **ONLY your section `.md`'s `<!-- slides -->` block.** Do not touch other
  files, other sections, the diagram-engine modules, or `_quarto.yml`. Source
  `.md` is truth — never edit `.qmd`.
- No `---` em-dashes in any new prose/captions.

## Report (concise)
Slide count; which `mdl-*` figures you reused; framework-specific slides
(`only=`/`except=`); PyTorch-only cells you scoped + flagged; audit_slides +
overflow status; and a **one-line visual self-assessment vs the §2 bar** (with
"screenshotted ✓" or "needs main-thread visual pass").
