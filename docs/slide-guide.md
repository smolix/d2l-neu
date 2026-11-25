# D2L Slide Style & Authoring Guide

Scope: the `<!-- slides -->` decks across **Basics** (Chapters 1–8), **Advanced** (Chapters 9–17), and **Mathematics for Deep Learning** (Chapters 33–38). Pair this guide with [`docs/slides.md`](slides.md) for the technical Reveal.js / Quarto syntax and pipeline mechanics. Reference exemplars: Chapter 1 (`chapter_preliminaries/*.md`).

---

## 1. Pedagogical Standard: University-Grade Teaching Decks

A deck must teach on its own at the level of an upper-division or graduate course at a top university (Stanford, MIT, CMU, Berkeley). A lecturer can teach the section directly from the slides without reading from the textbook, and a student reviewing the deck after lecture can follow the complete logical story, mathematical derivations, and systems pitfalls.

### What to Avoid
1. **No "Claude staccato" fragments:** Never write telegraphic 3-word sentence fragments or disconnected keyword lists that leave out the *why* and *how*. Write complete, natural, information-dense clauses that connect cause, mechanism, and effect.
2. **No walls of prose or pasted book paragraphs:** Never copy-paste multi-sentence textbook paragraphs onto a slide. Synthesize prose into structured bullets, side-by-side comparisons, compact equations, or summary tables.
3. **No bureaucratic meta-slides:** Never include filler slides such as *"By the End of This Deck / Uses / Skips"* or *"Agenda / Outline"*. Start teaching immediately after the cover slide.
4. **No walls of boilerplate code:** Show only the core 2–8 lines of code that express the concept. Omit imports, plotting plumbing, or repetitive setup (`@-id` suppresses noisy output; `@!id` shows the resulting plot or output only).

---

## 2. Deck Structure & Sizing

- **Scale slide count to the section's conceptual depth:**
  - **Chapter opener (`chapter_*/index.md`):** **5–7 slides** framing the chapter's central problem, mathematical/architectural roadmap table, concrete running benchmark, and study guide.
  - **Standard section notebook:** **15–22 slides**.
  - **Long or mathematically dense section notebook (e.g., *Mathematics for Deep Learning* or multi-stage architectures):** **20–30+ slides**. Never cram two distinct concepts or a long derivation onto one crowded slide to save slide count—add slides so every slide breathes and fits the 14" MacBook viewport cleanly.
- **Narrative arc of every section deck:**
  1. **Cover slide (`::: {.cover}`):** Section kicker (`[Dive into Deep Learning · §N.M]{.kicker}`, automatically updated by `gen_slides.py`), 1-line descriptive subtitle, and bold dot-separated keyword bar.
  2. **Motivation & Core Mental Model (Slide 2):** Why does this concept exist? Anchor immediately with a conceptual/systems diagram (`@fig:...`), concrete tensor shapes, parameter counts, computational costs, or failure modes of naive approaches.
  3. **Core Concepts, Mechanics & Visual/Code Evidence (Body Slides):** Develop one concept per slide, pairing mathematical definitions or algorithms with geometric/structural diagrams (`@fig:...`), chapter SVG figures (`![](../img/...)`), or live code/output cards (`@cell-id`).
  4. **Practical Systems / Statistical Pitfalls:** Highlight what goes wrong in real deep learning workflows (e.g., memory aliasing, silent broadcasting into `(B, B)`, data leakage from fitting preprocessing on the test set, numerical underflow/overflow, vanishing/exploding gradients) and illustrate the failure mode visually with a diagram.
  5. **Interactive Concept Check (Second-to-last slide):** 2–3 concrete design or calculation questions the class can work out in 60 seconds, with answers revealed in a `::: {.d2l-note .fragment .fade-up}` callout.
  6. **Summary / Key Takeaways Table (Final slide):** A structured markdown table or synthesis contrasting the section's operations, mathematical forms, complexity/scaling, and deep learning roles.

### Visual Density & De-Densification Standard ($\ge 65\%$ Visual Slides)
- **Visual-first teaching:** At least **$65\text{--}75\%$ of the slides in every section deck** (and at least **3 of 5 slides in every chapter opener `index.md`**) must carry a visual anchor: an inline diagram (`@fig:<id>`), a chapter figure (`![](../img/<name>.svg)`), or a plot output (`@!<cell-id>`).
- **De-densify when adding a diagram:** Never squeeze an existing full-width wall of text, wide markdown table, or multi-line algebraic derivation into a narrower column next to a diagram. Instead, let the diagram carry the spatial, architectural, or worked-example story and trim the paired left column (`::: {.col}`) down to **2–3 concise bullets** (plus at most one compact display equation or one compact `@cell-id` card).
- **Fullscreen & Mobile/iPad View Mode:** `_d2l-slides-overlay.html` and `_d2l-slides.scss` provide a **Fullscreen toggle button** in both the top navigation bar (`#d2l-nav-fullscreen`) and the bottom-left control cluster (`#d2l-deck-fullscreen`, shortcut `F`). Toggling fullscreen invokes the browser's native Fullscreen API (on desktop, iPad, and Android) AND applies a CSS viewport-fullscreen fallback (`body.d2l-deck-fullscreen`, supporting iPhone iOS Safari and embedded iframes) that hides `.d2l-deck-nav`, expands `.reveal-viewport` to `100dvh`, and triggers `Reveal.layout()`.

---

## 3. Slide Anatomy & Visual Vocabulary (`_d2l-slides.scss`)

Every slide uses the classes defined in `_d2l-slides.scss`:

```markdown
<!-- slides -->

::: {.slide}
::: {.cover}
[Dive into Deep Learning · §1.1]{.kicker}

Tensors: The Core Data Structure of Deep Learning<br>**storage buffers · shapes & strides · views · broadcasting · memory semantics**
:::
:::

::: {.slide title="A Tensor Is a Storage Buffer Plus Metadata"}
[The Unifying Mental Model]{.kicker}

Every operation in this section—reshaping, slicing, broadcasting, and in-place mutation—follows from one systems distinction:

::: {.cols .vc}
::: {.col}
- **Contiguous 1D storage buffer:** A flat block of typed bytes (`float32`, `bfloat16`, `int64`) living in CPU RAM or GPU HBM.
- **Lightweight metadata header:** Small integers (`shape`, `strides`, `dtype`, `device`) describing how multi-axis indices map to byte offsets:
  $$\text{offset}(i, j) = \text{base} + i \cdot s_0 + j \cdot s_1$$
- **Zero-copy view vs. allocation:** Changing metadata (`reshape`, `T`, slice) is $O(1)$; allocating a new buffer costs $O(n)$ memory traffic.
:::

::: {.col .fig}
@fig:ndarray-rank-ladder
:::
:::
:::
```

### Mandatory Markup Rules
- **Single-line slide titles (`title="..."`):** Every content slide must have a descriptive `title="..."` attribute (≤ 58 characters) that fits on **one `H2` line** at `1280×720`, paired with a `[Category]{.kicker}` eyebrow on the first body line.
- **Two-column layout (`::: {.cols}` / `::: {.cols .vc}`):**
  - Always use `::: {.cols}` (top-aligned) or `::: {.cols .vc}` (vertically centered) with child `::: {.col}` divs.
  - Right-column width modifiers:
    - `::: {.col .fig}` (44% width) — **default** whenever the left column contains display math (`$$...$$`), code cells (`@id`), or detailed bullets.
    - `::: {.col .fig .big}` (54% width) — use **only** when the left column has short prose/bullets and no wide display equations.
    - `::: {.col .narrow}` (36% width) — for a narrow text/table column paired with a wider main column.
  - **NEVER** use Quarto's reserved `::: {.columns}` / `::: {.column}` (which lacks our flex sizing and causes vertical stacking or overflow).
- **Callout boxes (`::: {.d2l-note}`):**
  - Use `::: {.d2l-note}` (blue info/takeaway), `::: {.d2l-note .warn}` (amber pitfall/hazard), or `::: {.d2l-note .rule}` (purple mathematical rule/invariant).
  - **NEVER** use `::: {.callout}` (reserved by Quarto).
- **Fragments:**
  - Top-level `. . .` pauses work **only** at the top level of a `::: {.slide}`, **never inside a `::: {.col}`** (where Pandoc renders them as literal `"..."`). Inside columns or for concept-check answers, use `::: {.d2l-note .fragment .fade-up}`.
- **No book-only directives or hardcoded section numbers on slides:**
  - Never put `:numref:`, `:eqref:`, `:cite:`, `:exref:`, or `:label:` inside `<!-- slides -->`.
  - **Never hardcode section or chapter numbers** (`"§9.3"`, `"Chapter 9"`). Always use `§{sec_label}` (`→ §9.3`), `§{chap_label}` (`→ §9`), or `§{num:sec_label}` (`→ 9.3`) tokens (`tools/section_refs.py`). Cover kickers (`[Dive into Deep Learning · §N.M]{.kicker}`) are rewritten automatically by `gen_slides.py`.
- **Zero framework leaks:**
  - Shared slides (`::: {.slide}` without `only=`) and shared diagrams (`@fig:<id>`) must be 100% framework-neutral. When APIs, tensor layouts (`NCHW` vs. `NHWC`), or semantics (e.g., JAX immutability / PRNG keys) differ per framework, emit `only="pytorch"` / `only="jax"` / `except="..."` sibling slides and `@fig:<id>@<fw>` diagram variants, respecting each part's framework coverage (**Basics Ch. 1–8, Image Models Ch. 32, Math Ch. 33–38, Tools Ch. 39** = all 4 frameworks; **Advanced Ch. 9–17** = `pytorch` + `jax` only; **Ch. 14 RL** = `pytorch` only).
- **Output-verified numbers only:**
  - Every metric, loss, timing, or shape quoted on a slide or diagram card must be verified against `outputs/<fw>/...` across **all** frameworks the page carries, rounded to run-to-run-stable precision (ranges/approximations rather than brittle single-seed decimals).

---

## 4. Active Linebreak, `<= 720 px` Fit, & Export Compatibility (HTML / PPTX / Beamer)

All decks render onto a `1280×720` Reveal.js canvas **and** export to editable PowerPoint (`.pptx`) and Beamer PDF/LaTeX (`.pdf` / `-latex.zip`). Every slide must pass five strict layout invariants:

### Invariant 1 — Zero Vertical Overflow (`scrollHeight <= 720`) & Unclipped Outputs
- Never rely on vertical scrollbars (`scrollHeight <= 720` in `tools/check_slide_overflow.py`, zero scrolled `.cell-output` boxes):
  - **Full-width text/table slide:** 1 intro line + 5–7 table rows (or 4–5 bullets) + at most 1 `.d2l-note` (do not place trailing prose/notes *after* a Markdown table on the same slide, as Pandoc's PPTX writer splits or overlaps post-table elements).
  - **Full-width code slide:** 1 intro sentence + 1 `@cell-id` card + 1 `.d2l-note` (or 2 compact `@cell-id` cards without a note). Never create an empty Setup slide holding only a hidden preamble cell.
  - **Two-column slide:** 2–4 concise bullets on the left + 1 figure (`@fig:...` or `@!cell-id`) or 1 compact `@cell-id` on the right.
  - **Slice long cells or stacked outputs** across `(cont.)` slides with `@<id>[lines, out=items]` selectors (`@-id[1-18]` then `@id[19-]`, or `@!id[out=1]` then `@!id[out=2-]`; inspect numbering via `python3 tools/gen_slides.py . _slides --files <ch/file.md> --frameworks pytorch jax --show-cell <id>`).
  - **Output row caps:** Default output box fits ~10 rows; use `layout="tall-output"` (13–15 rows) for shape/result tables, `layout="tight"` (6–7 rows) when prose dominates, or `layout="code"` (14–16 rows) for dense logs. Split multi-part Concept Checks so questions + answers never exceed `720 px`.

### Invariant 2 — Zero Horizontal Column / Equation Overlap (`scrollWidth <= clientWidth`)
- Inside a two-column `.col` (~520–620 px wide), display math (`$$...$$`) renders at `0.84em`.
- **Never** place two display equations side-by-side with `\quad` or `\qquad` inside a `.col`.
- **Never** write a 5-term horizontal equality chain in a single `$$...$$` block inside a `.col`. Keep display equations in columns to a single compact relation (`A = B`), and move secondary definitions inline or onto a second line.
- When a left column has display math, pair it with `::: {.col .fig}` (44%), not `::: {.col .fig .big}` (54%).

### Invariant 3 — Atomic Inline Math & Multi-Step Chains
- `_d2l-slides.scss` sets `.reveal .math.inline, .reveal .math.inline .katex { white-space: nowrap; }` so atomic inline formulas (`$B = 256$`, `$\approx 100$`, `$P(\mathcal{A} \cup \mathcal{B})$`) never snap in half across a line break.
- `tools/gen_slides.py` automatically wraps parenthesized inline math and code (`($p = 0.5$)`, `(` + `<code>...</code>` + `)`) in `<span class="nowrap">...</span>` so an opening parenthesis `(` never dangles at the end of a line.
- **Authoring rule for multi-step inline chains:** Because each `$ ... $` span is non-breaking, **never wrap a long multi-clause derivation in a single `$ ... $` block** (e.g., avoid `$u = x^2 = 4 \to y = 3u = 12 \to dy/dx = 12$`). Instead, split multi-step inline chains into separate `$ ... $` tokens at arrow or clause boundaries:
  - Good: `$u = x^2 = 4$ $\to$ $y = 3u = 12$ $\to$ $dy/dx = 12$`
  - This allows the browser to wrap cleanly *between* steps while keeping every individual equation intact on one line.

### Invariant 4 — Single-Line Slide Titles (`H2`)
- `_d2l-slides.scss` applies `text-wrap: balance` to headings, but every `title="..."` should be authored concisely (≤ 55–58 characters) so that it renders on a **single line**.

### Invariant 5 — Unwrapped Code Lines Inside Two-Column Cells
- In two-column slides, `.reveal .cols .cell pre` renders at `0.52em`, fitting up to ~58 characters per line. If a referenced code cell sits inside a `.col` and has long trailing comments, prefer wider `.col` pairing (`.col.fig` at 44%) or full-width layout so code lines do not soft-wrap mid-comment.

---

## 5. Automated Verification Workflow (SVG Defect Audit + Lint + Render + Overflow Check)

Before committing or opening a PR for any slide deck or slide diagram module, run all automated checks:

### Step 1 — Build, Visually Audit, and Outline Slide Diagrams (`tools/build_module_diagrams.mjs`)
Every slide diagram module (`diagrams/<module>.mjs`) must obey `diagrams/README.md`:
- **11 px font-size floor (`fs >= 11`):** Never shrink card labels below `11 px` to fit a box; wrap onto 2 lines (measuring post-script widths via `diagrams/mdl-textwidth.mjs`).
- **Section labels & TeX scripts:** Use `sec('sec_x')` / `num('chap_x')` from `diagrams/sections.mjs` (never literal `"§N.M"` or `"Chapter N"`), brace non-trivial TeX scripts (`max_{a}`, `ℝ^{d}`, `∑_{t=1}^{T}`), escape literal code underscores (`'y\\_hat'`), and pass explicit colors to `arrow()`.

`tools/build_module_diagrams.mjs` runs `scriptSvg`, verifies **0 geometry defects** in headless Chromium (`*-intersects-text`, `rect-*-edge-cuts-text`, `text-overlap`, `text-outside-viewbox`), outlines `<text>` to bundled font paths via `tools/figstyle/outline_svg.py`, and syncs any book copy in `img/`:

```bash
node tools/test_diagram_scripts.mjs
node tools/build_module_diagrams.mjs diagrams/<module>.mjs
# Visually inspect modified SVGs as PNGs:
rsvg-convert -z 2 img/auto/<id>.svg -o /tmp/<id>.png
```

### Step 2 — Lint Corpus Sources and Render Slides Across All Supported Frameworks

```bash
# 1. Corpus-wide source, cross-reference, section-token, & figure-pair lint
python3 tools/lint_source.py --corpus
python3 tools/sync_auto_figures.py --check

# 2. Render modified decks for all frameworks the chapter carries
#    (pytorch jax for Ch. 9–17; pytorch jax tensorflow mxnet for Ch. 1–8, 32, 33–39)
python3 tools/gen_slides.py . _slides --frameworks pytorch jax --render --files chapter_<name>/*.md
```

### Step 3 — Verify `<= 720 px` Slide Fit (`tools/check_slide_overflow.py`) and Exports
Serve `_slides` locally and verify every rendered deck in `tools/check_slide_overflow.py` (must exit `0` with zero slides `> 720 px`):

```bash
python3 -m http.server 8027 -d _slides &
SERVER_PID=$!
for f in _slides/pytorch/chapter_<name>/*.html _slides/jax/chapter_<name>/*.html; do
  rel="${f#_slides/}"
  .venv-build/bin/python tools/check_slide_overflow.py "http://localhost:8027/$rel"
done
kill $SERVER_PID
```

When touching complex layouts (tables, display math, or multi-column cards), also spot-check PowerPoint and Beamer generation (`python3 tools/slides_to_pptx.py` / `python3 tools/slides_to_beamer.py`, or `gmake slides-pptx` / `gmake slides-beamer`).
