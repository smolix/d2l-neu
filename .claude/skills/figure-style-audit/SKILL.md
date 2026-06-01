---
name: figure-style-audit
description: >-
  Audit a chapter's illustrative figures against the d2l house conventions
  before committing or rendering — caption integrity, attached :label:, no
  dangling/orphan figures, byte-idempotent SVGs, no figure-drawing code in
  notebooks, one style per chapter. Use after adding/editing figures (typically
  alongside the mdl-figure skill), when reviewing a chapter for the 5/5/5 bar,
  or to find what regressed the figure conventions.
---

# figure-style-audit — guard the figure conventions

Catches the ways figures silently drift from the house style (CLAUDE.md →
"Content authoring", and the `mdl-figure` skill). Part mechanical linter, part
two checks only a human/model eye can make.

## 1. Mechanical lint — `tools/lint_figures.py`

Run it on the chapter you're working on (NOT the whole legacy tree — the
pre-existing d2l chapters predate these conventions and will be noisy):

```bash
python3 tools/lint_figures.py chapter_mdl-<topic>/*.md     # one chapter
python3 tools/lint_figures.py chapter_mdl-*/*.md           # all new-bar chapters
```

Output is `path:line:col: error|warning: msg` (same format as
`tools/lint_source.py`, so it shows in the Problems pane). It checks:

- **ERROR** — image references `../img/<id>.svg` that doesn't exist.
- **ERROR** — caption contains `[` or `]` (truncates the markdown alt-text and
  detaches the `:label:`); write matrices as `\begin{smallmatrix}…\end{smallmatrix}`.
- **ERROR** — an image not immediately followed by a `:label:`fig_…`` line.
- **WARN** — a referenced SVG carries a date/timestamp (non-idempotent → noisy
  git diffs).
- **WARN** — a code cell contains figure-DRAWING primitives (`add_patch`,
  `Polygon(`, `FancyArrow`, `quiver(`, `.savefig(`, empty-string `annotate`, …)
  — that figure should be pre-generated, not drawn inline.
- **WARN** — the chapter mixes generated figures with inline drawing code.
- **WARN** (full-tree scan only) — a generated `img/mdl-*.svg` referenced by no
  chapter (orphan).

`--strict` makes warnings fail too (use in pre-commit/CI). A clean new-bar
chapter reports `0 error(s), 0 warning(s)`.

## 2. Idempotence re-run (the linter can't prove this alone)

The committed SVGs must be byte-stable, or every regen churns the repo. Confirm:

```bash
.venv-pytorch/bin/python tools/gen_mdl_figures.py
git diff --stat img/        # MUST be empty
```

A non-empty diff means a figure picked up a timestamp/random id — check it uses
`save()` (not a bare `fig.savefig`) and didn't drop `svg.hashsalt` /
`metadata={'Date': None}` from the generator's style block.

## 3. Judgment checks (read the rendered chapter)

The linter can't see *taste*. After `make html`, look at the chapter and confirm:

- **One visual family.** All figures share the palette, line weights, fonts, and
  arrow/right-angle style — no figure looks imported from elsewhere. Reproduce
  any straggler in `tools/gen_mdl_figures.py` rather than mixing styles.
- **Every figure earns its place.** Each illustrative figure clarifies something
  the prose discusses; each `:numref:` resolves and points at the right picture;
  captions are accurate and self-contained.
- **No teaching plot got demoted.** A plot of a *computed* result (loss curve,
  fitted line) belongs as a `d2l.plot(...)` cell in the notebook, not as a
  pre-generated SVG — the linter won't tell you if you pre-generated something
  that should have stayed live code.

## See also

- `mdl-figure` skill — how to add/edit a figure correctly (this audits the result).
- `tools/lint_figures.py` — the checker; `tools/gen_mdl_figures.py` — the generator.
- `CLAUDE.md` → "Content authoring" — the source rules.
