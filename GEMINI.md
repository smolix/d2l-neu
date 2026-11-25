# D2L-Neu — Gemini Operating Guide & Guardrails

This file is automatically loaded into every Gemini session and subagent in this repository (keep it **< 24,000 bytes** so it is never truncated). Read this alongside `CLAUDE.md` and the domain guides before editing anything:

- **Build & execution:** `CLAUDE.md`, `docs/build-guide.md`, `docs/build-system.md`, `docs/notebook-scheduler.md`
- **Prose & code:** `docs/writing-guide.md`, `docs/code-guide.md`, `docs/syntax.md`
- **Slides & exports (HTML / PPTX / Beamer):** `docs/slide-guide.md`, `docs/slides.md`
- **Figures & slide diagrams:** `diagrams/README.md`, `docs/figure-style-guide.md`, `.claude/skills/mdl-figure/SKILL.md`, `.claude/skills/figure-style-audit/SKILL.md`

Every guardrail below addresses a real class of defects previously introduced by Gemini PRs (#268–#275) and fixed in #276–#316. Treat every rule here as a **hard gate**.

---

## 1. Source of Truth, Part/Chapter Map, & Build Rules

1. **Edit only source `.md` files (and generators/tools).**
   - **Never** edit `.qmd` files (`chapter_*/*.qmd`, `index.qmd`) or anything under `_notebooks/`, `_book/`, `_pdf/`, or `_slides/`. They are generated build outputs.
2. **Canonical Part & Chapter Numbering (`CHAPTER_NUMBERING` == `_quarto.yml` order):**
   - **Front matter:** Preface, Installation, Notation, Introduction (unnumbered).
   - **Basics (Ch. 1–8 — all 4 frameworks: `pytorch`, `jax`, `tensorflow`, `mxnet`):**
     `1` Preliminaries · `2` Linear Regression · `3` Linear Classification · `4` Multilayer Perceptron · `5` Builder's Guide · `6` Convolutional Networks · `7` Modern Convnets · `8` Sequence Models.
   - **Advanced (Ch. 9–17 — `pytorch` + `jax` ONLY; Ch. 14 RL is `pytorch` ONLY):**
     `9` Optimization · `10` Attention · `11` Transformers · `12` State Space Models (`chapter_recurrent-modern`) · `13` Computational Performance · `14` Reinforcement Learning · `15` Deep Reinforcement Learning · `16` GANs · `17` Diffusion Models.
   - **Language Models (Ch. 18–31 — `pytorch` via `.venv-lm`):** `chapter_lm-*`.
   - **Image Models (Ch. 32 — all 4 frameworks):** `chapter_computer-vision`.
   - **Mathematics for Deep Learning (Ch. 33–38 — all 4 frameworks):**
     `33` Linear Algebra (`ch33-mdl-la`) · `34` Calculus (`ch34-mdl-calc`) · **`35` Optimization (`ch35-mdl-opt`)** · **`36` Probability & Statistical Learning (`ch36-mdl-prob`)** · `37` Information Theory (`ch37-mdl-info`) · `38` Dynamics (`ch38-mdl-dyn`). *(Do NOT swap Ch. 35 Optimization and Ch. 36 Probability!)*
   - **Tools for Deep Learning (Ch. 39 — all 4 frameworks):** `chapter_appendix-tools-for-deep-learning` (`ch39-tools`).
   - **Attic (Ch. 40–42):** `40` Gaussian Processes · `41` Hyperparameter Optimization · `42` Recommender Systems.
3. **Immutable Cell IDs & Output Freshness:**
   - Every Python code fence has a stable `#<id>` assigned by `tools/add_cell_ids.py`. **Never** rename or reuse an existing cell ID.
   - Modifying any `{.python .input}` cell (or `#@save` helper) invalidates stored outputs in `outputs/<fw>/...`. Run `python3 tools/audit_outputs.py` (`gmake audit-outputs`) to inspect freshness; re-execute via the scheduler (`docs/build-guide.md` §4.3–4.4) and run `capture_outputs.py` **only** after a `0 failed` run.
   - Never touch **locked notebooks** (`<!-- execution: locked; ... -->`) unless explicitly instructed (`gmake rerun-locked`).
   - On macOS, `make` means **GNU Make ≥ 4.3** (`gmake`). Never wrap `gmake html` or scheduler targets in `-j`.

---

## 2. Rule 1 — Zero Framework Leaks (Prose, Code, Exercises, Slides, & Diagrams)

A reader viewing the JAX, TensorFlow, or MXNet tab/deck/PDF must **never** encounter untabbed PyTorch API names, NCHW-only tensor shapes, or claims that hold only in PyTorch.

1. **Untabbed book prose must be 100% framework-neutral:**
   - State the shared mathematical or systems concept in untabbed prose.
   - Put any framework-specific API name (`nn.Module`, `torch.save`, `torch.load`, `nnx.Optimizer`, `optax`, `GradientTape`, `init_cnn`, etc.), tensor layout (`NCHW` `(B, C, H, W)` vs. `NHWC` `(B, H, W, C)`), initialization default, or state-allocation detail inside single-framework `:begin_tab:` / `:end_tab:` blocks (one block per supported framework, `:begin_tab:` and `:end_tab:` each alone on its own line).
   - Never cross-compare tabs inside a single framework's tab note or code comments (e.g., never write `# same as PyTorch above` in a JAX cell).
2. **Respect each part's framework set:**
   - Never add TensorFlow or MXNet tabs, exercise variants, or slide branches to **Advanced (Ch. 9–17)** or **Language Models (Ch. 18–31)**.
   - In Basics (Ch. 1–8), Image Models (Ch. 32), Math (Ch. 33–38), and Tools (Ch. 39), verify claims and code against **all four** frameworks (`pytorch`, `jax`, `tensorflow`, `mxnet`).
3. **Framework-clean slides and SVG diagrams:**
   - Shared slides (`::: {.slide}` without `only=`) and shared diagrams (`@fig:<id>`) must use framework-neutral terminology (e.g., *"in-place checkpoint save"*, *"parameter tree"*, *"framework PRNG"* rather than `torch.save` or `state_dict` on an all-framework deck).
   - When the mechanism, API, shape layout, or code structure genuinely differs by framework (e.g., JAX immutable array updates, functional PRNG splitting, `nnx` state vs. `nn.Module`, NCHW vs. NHWC), emit `only="pytorch"` / `only="jax"` / `except="..."` sibling slides and, when the diagram itself names framework specifics, expose per-framework diagram variants (`<id>-pytorch`, `<id>-jax`, `<id>-tensorflow`, `<id>-mxnet`) referenced as `@fig:<id>@<fw>`.
4. **Exercises:**
   - If an exercise references a framework-specific API or behavior, split it into per-framework `:begin_tab:` variants matching the page's framework coverage (`expand_exercise_tabs` emits panes only for the page's frameworks).

---

## 3. Rule 2 — Zero Hallucinated Numbers, Output Drift, or Unverified Citations

Never invent, extrapolate, or guess quantitative results, benchmark figures, hardware specs, or paper claims.

1. **Verify every number against `outputs/<fw>/...` across ALL supported frameworks:**
   - Before writing any loss, accuracy, perplexity, timing, parameter count, shape, or error figure into prose, tables, slide bullets, or SVG diagram cards, read `outputs/<fw>/<chapter>/<section>.json` for **every** framework the page carries (`pytorch`, `jax`, `tensorflow`, `mxnet`).
   - **Quote only precision that survives re-execution** (`CLAUDE.md` → *Experimental results in prose*): seed-level metrics fluctuate across frameworks and runs. Never quote brittle per-seed last digits (`"90.21%"`) or single-framework figures when another framework prints `86.8%` — widen to a range supported by all stored outputs (`"87–90%"`, `"about 92%"`, `"roughly 1.5×"`), and never build a conceptual conclusion on a difference smaller than run-to-run noise.
   - Double-check back-of-the-envelope systems arithmetic (FLOPs, KV-cache bytes, roofline ridge points, collective bandwidth `2(W-1)/W`, checkpoint intervals) with Python before putting numbers on cards.
2. **Verify citations and paper attributions:**
   - Every `:cite:` key **must** exist in `d2l.bib` (a missing key breaks PDF builds). Check `grep -F "<Key>" d2l.bib` before using any citation.
   - Attribute to a paper **only** what that paper actually proves or measures (e.g., RegNet's speedup over EfficientNet is attributed to activation memory at comparable FLOPs, not grouped convolutions; Goyal et al. is cited for linear scaling + warmup, not our notebook's divergence count). Check exact publication years, page ranges, and published errata.

---

## 4. Rule 3 — Never Hardcode Section, Chapter, or Exercise Numbers

Chapter and section numbers change when parts are restructured. Hardcoded `"§14.3"`, `"Chapter 13"`, or `"Exercise 4"` strings silently rot and are rejected by `lint_source.py --corpus`. Use the exact reference syntax for each surface:

| Surface | Correct Syntax | Forbidden |
| :--- | :--- | :--- |
| **Book prose & captions** (above `<!-- slides -->`) | `:numref:\`sec_sgd\``, `:numref:\`chap_optimization\``, `:eqref:\`eq_...\``, `:exref:\`ex_...\`` | `"Section 9.2"`, `"Chapter 9"`, `"§9.2"`, `"Exercise 4"`, or `§{...}` tokens |
| **Slide decks** (below `<!-- slides -->`) | `§{sec_sgd}` (`→ §9.2`), `§{chap_optimization}` (`→ §9`), `§{num:sec_sgd}` (`→ 9.2`), `§{sec_a}–§{num:sec_b}` | `:numref:`, `:eqref:`, `:cite:`, `:exref:`, or literal `"§9.2"` / `"Chapter 9"` |
| **Diagram modules** (`diagrams/*.mjs`) | `import { sec, num } from './sections.mjs';` then `${sec('sec_sgd')}` or `Chapter ${num('chap_optimization')}` | Literal `"§9.2"`, `"§§11.1–11.2"`, or `"Chapter 13"` in JS strings |

- **Cover kickers (`[Dive into Deep Learning · §N.M]{.kicker}`):** Leave as-is; `gen_slides.py` rewrites cover kickers automatically from `CHAPTER_NUMBERING`.
- **Target labels for `§{...}` and `sec()`/`num()`:** Must be the H1 `:label:` of a numbered chapter (`index.md`) or section page — never a `##` subsection label or unnumbered page.
- **Exercise labels (`:exlabel:`):** Must be **indented inside the numbered list item** of the exercise so `tools/exercise_refs.py` attaches them (an unindented `:exlabel:` detaches and renders `"??"`).

---

## 5. Rule 4 — Slide Authoring, `<= 720 px` Fit, & HTML / PPTX / Beamer Compatibility

Every slide deck builds **three** deliverables (`make slides`): Reveal.js HTML, editable PowerPoint (`.pptx` via `tools/slides_to_pptx.py` + `tools/pptx_fit.py`), and Beamer PDF + LaTeX source (`.pdf` / `-latex.zip` via `tools/slides_to_beamer.py`). Never declare slide edits done without rendering and checking overflow.

1. **Strict `<= 720 px` Slide Height & Unclipped Outputs:**
   - Every slide in **every** supported framework deck must measure `scrollHeight <= 720` in `tools/check_slide_overflow.py` with zero scrolled `.cell-output` boxes.
   - **Slice long cells or multi-item outputs** across `(cont.)` slides using `@<id>[lines, out=items]` selectors (`docs/slides.md` → *Code-cell placeholders*):
     - `@-my-cell[1-18]` (code lines 1–18 only, no output) followed by `@my-cell[19-]` on the next slide.
     - `@!my-cell[out=1]` and `@!my-cell[out=2-]` to separate stacked plots or plot + printed table.
     - Inspect exact slide-numbered lines and output items via:
       `python3 tools/gen_slides.py . _slides --files <chapter/file.md> --frameworks pytorch jax --show-cell <id>`
   - **Output-box row caps (`inject_outputs.py`):** Default output box holds ~10 rows. Use `layout="tall-output"` on `::: {.slide ...}` for 11–13-row shape/result tables that are the point of the slide, `layout="tight"` (6–7 rows) when prose dominates, or `layout="code"` (14–16 rows) for dense logs.
   - **Split Concept Checks:** Never cram 3–4 questions plus a multi-bullet answer box onto one overflowing slide; put answers (or Q3–Q4) on a `(cont.)` slide.
   - **Drop empty Setup slides:** Never create a slide whose only cell is a hidden preamble/imports cell.
2. **De-Densification & Visual Anchor Standard (`>= 65%` visual slides):**
   - Use `::: {.cols}` / `::: {.cols .vc}` with `::: {.col}` + `::: {.col .fig}` (`44%`, default when left column has display math or code) or `::: {.col .fig .big}` (`54%`, short bullets only). **Never** use Quarto's reserved `.columns` / `.column` or `.callout`.
   - Keep left-column text paired with a diagram to **2–3 concise, complete clauses** (never telegraphic fragments, never pasted book paragraphs).
   - Keep `title="..."` concise (`<= 55–58` chars, single H2 line; no unquoted YAML-breaking characters).
   - Break multi-step inline math chains *between* `$...$` spans (`$u = x^2$ $\to$ $y = 3u$`) because `.reveal .math.inline` is `white-space: nowrap`.
3. **PowerPoint & Beamer Layout Traps:**
   - Avoid placing trailing paragraphs or callout boxes *after* a Markdown table on the same slide unless intended as a single stacked column (Pandoc's default PPTX table layouts otherwise split or overlap).
   - In display/inline math on slides, brace subscripts/superscripts cleanly (e.g., `x_{\mathrm{in}}`, not bare multi-token scripts that break `unicode-math` in Beamer XeLaTeX).

---

## 6. Rule 5 — Slide Diagrams (`diagrams/*.mjs`) & Book Figures (`tools/gen_mdl_*_figures.py`)

1. **Typography & 11 px Minimum Font-Size Floor in `diagrams/*.mjs`:**
   - **NEVER shrink text below `11 px` (`fs >= 11`)** to squeeze long strings into boxes or pass `rect-*-edge-cuts-text`. Measure post-script label widths (see `diagrams/mdl-textwidth.mjs`) and **wrap onto two lines** or shorten the wording while keeping `fs >= 11` (typically `12–16 px`).
   - **Never emit `dominant-baseline`** (`librsvg` ignores it in PDF builds); always use `tx()` or `baselineY()` from `diagrams/engine.mjs`.
   - Always pass an explicit color to `arrow(x1, y1, x2, y2, color)` (and spread fan-in arrowheads onto distinct target coordinates rather than piling them on one pixel). Use `C` / `TOKENS` palette constants, never raw ad-hoc hex.
2. **TeX Subscripts & Superscripts (`diagrams/scripts.mjs`):**
   - `scriptSvg` automatically converts math scripts (`x_{k+1}`, `ℝ^{B×T}`, `Σ_{i=1}^{n}`, `max_{a}`, `‖x‖_{F}`) into 70% `<tspan baseline-shift="sub|super">` runs.
   - **Brace non-trivial bases/scripts:** Always brace scripts after operators/delimiters/symbols (`max_{a}`, `argmin_{θ}`, `∑_{t=1}^{T}`, `∂_{t}`, `ℝ^{d}`, `‖·‖_{2}`, `X^{(L)}`). Keep each braced script inside a **single** `tx()` call (never split `{...}` across two wrapped `tx()` lines).
   - **Escape literal code underscores** in labels that would otherwise convert as math (`'y\\_hat'`, `'c\\_attn'`, `'W\\_qkv'`). Run `node tools/test_diagram_scripts.mjs` after any script-heavy diagram work.
   - Do **not** mix Unicode superscript `⁻` with full-size letters (`2⁻ℓ` — write `2^{-ℓ}` so the entire exponent shifts together).
3. **Mandatory Build, Defect Check, Sync, & Visual Inspection:**
   - Build one module at a time via:
     `node tools/build_module_diagrams.mjs diagrams/<module>.mjs`
     This runs `scriptSvg`, checks all 4 geometry defect classes in headless Chromium (`*-intersects-text`, `rect-*-edge-cuts-text`, `text-overlap`, `text-outside-viewbox`), outlines `<text>` to bundled font paths via `tools/figstyle/outline_svg.py` (preserving legacy viewBox crop state), and refreshes any matching `img/<id>.svg` book copy.
   - **Visually inspect rendered PNGs** (`rsvg-convert -z 2 img/auto/<id>.svg -o /tmp/<id>.png` + `read_file` on `/tmp/<id>.png`) — geometry checks catch bounding-box collisions, not awkward arrows, cramped layouts, or ugly line breaks.
   - **Keep `img/` and `img/auto/` in sync and orphan-free:** Every matplotlib figure present in both `img/<id>.svg` and `img/auto/<id>.svg` must be byte-identical (`python3 tools/sync_auto_figures.py --check`). Never leave unreferenced SVGs in `img/` or `img/auto/`.
4. **Book Figures (`tools/gen_mdl_*_figures.py`):**
   - Illustrative/schematic chapter figures are pre-generated by committed matplotlib scripts (`make figures`), **never** drawn with matplotlib patch/quiver plumbing inside notebook cells (*code teaches; it does not draw*).
   - Follow the 7-point house-style checklist in `CLAUDE.md` (black axes & labels, 13–16 pt legible fonts, zero text↔curve intersections, tight limits, balanced panel heights, no redundant supercaptions, byte-idempotent `fl.save()`), render to PNG with `rsvg-convert`, inspect visually, and run `python3 tools/lint_figures.py --strict <chapter/*.md>`.

---

## 7. Rule 6 — Markdown, Pandoc, LaTeX/PDF, & CSS Tripwires

1. **Multi-line display math with `:eqlabel:`:** Put `$$` **alone** on its opening and closing lines, with `:eqlabel:` immediately on the next line (otherwise the label detaches). Multi-line `\\` requires `\begin{aligned}...\end{aligned}`.
2. **Image captions:** Never put unbalanced `]` or nested `[[...]]` inside `![caption](...)` — Pandoc truncates the alt text and prints `{#fig-...}` literally. Write matrices in captions as `\left(\begin{smallmatrix}a & b \\ c & d\end{smallmatrix}\right)`. Always follow `![...](../img/...)` immediately with `:label:\`fig_...\``.
3. **Closing `$` before a digit:** Never write `$x$2` or `$\mathcal{F}$1` — Pandoc's currency heuristic refuses to close the math span and breaks XeLaTeX (`\symcal allowed only in math mode`). Keep digits inside math or `\text{...}`.
4. **Block boundaries:** Keep `:begin_tab:` and `:end_tab:` alone on their own lines, leave a blank line before every bullet/numbered list and ATX heading, and never start a wrapped list continuation line with `<n>. ` or `(@ref)`.
5. **CSS & Chromium hazards:**
   - Never add `vertical-align: middle` to `.math.inline` in `_d2l-style.css` (shifts inline math off the text baseline across the whole book).
   - Never set `text-wrap: pretty` on `p, li` in `_d2l-slides.scss` (hangs Chromium layout on list items containing inline KaTeX).

---

## 8. Rule 7 — Subagent Delegation & Git / PR Hygiene

1. **Briefing subagents:** Subagents do not know facts you haven't verified. Whenever delegating chapter, slide, or diagram work to subagents:
   - Specify the chapter's **exact number** (`Ch. 35` = Optimization, `Ch. 36` = Probability!) and **exact framework set** (`pytorch,jax` vs. all 4).
   - Require subagents to read `outputs/<fw>/...` for every number they write, use `§{...}` / `sec()` / `:numref:` (zero hardcoded numbers), keep diagram font sizes `>= 11 px`, and run the verification commands below before reporting back.
   - **Never trust subagent self-reports blindly:** always run corpus-wide verification (`lint_source.py --corpus`, `sync_auto_figures.py --check`, rendered `check_slide_overflow.py`) in the coordinator before committing.
2. **Git & PR checks:**
   - Before pushing to an existing branch/PR, run `gh pr view <N> --json state` to confirm the PR is still `OPEN` (never push new work onto an already-merged PR branch).
   - Never stage untracked user scratch files; pass `--no-verify` only if explicitly appropriate, and verify `git status` is clean after committing.

---

## 9. Mandatory Pre-Commit Verification Checklist

Run every applicable check (all must exit `0`) before committing:

```bash
# 1. Corpus-wide source, cross-reference, section-token, & figure-pair lint
python3 tools/lint_source.py --corpus
python3 tools/sync_auto_figures.py --check

# 2. Output store freshness audit (whenever book prose/code or #@save touched)
python3 tools/audit_outputs.py

# 3. Book figures (whenever tools/gen_mdl_*_figures.py or img/mdl-*.svg touched)
python3 tools/lint_figures.py --strict chapter_<name>/*.md

# 4. Slide diagrams (whenever diagrams/*.mjs touched)
node tools/test_diagram_scripts.mjs
node tools/build_module_diagrams.mjs diagrams/<module>.mjs

# 5. Slides: render ALL supported frameworks for touched files & verify <= 720 px
python3 tools/gen_slides.py . _slides --frameworks pytorch jax --render --files chapter_<name>/*.md
# (add tensorflow mxnet for Basics Ch. 1–8, Image Models Ch. 32, Math Ch. 33–38, Tools Ch. 39)
# Serve _slides on a local port and run check_slide_overflow.py on every rendered deck:
# .venv-build/bin/python tools/check_slide_overflow.py http://localhost:8027/<fw>/chapter_<name>/<file>.html
```
