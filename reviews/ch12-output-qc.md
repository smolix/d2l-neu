# Chapter 12 "State Space Models" — Output QC (HTML + PDF)

Date: 2026-07-21. Scope: **formatting/rendering** defects in the built output only
(not prose/content). Read-only review — no files edited. Math-heavy chapter (many
display equations, matrices) reviewed with extra care per the task brief.

Artifacts reviewed:
- HTML: `_book/chapter_recurrent-modern/{index,lstm,ssm,mamba,matrix-state,deltanet,test-time-regression,hybrids}.html`
- PDF: `_book/pdf/Dive-into-Deep-Learning-{pytorch,jax}.pdf` (ch12 = printed pp.
  741–860 / PDF pages 808–927 pytorch; printed pp. 749–872 / PDF pages 816–939 jax).
  This chapter is PyTorch+JAX only; tensorflow/mxnet PDFs not in scope.

Method: rasterized all 120 (pytorch) + 124 (jax) ch12 pages at 110 dpi, scanned via
16-page contact sheets, then zoomed to full resolution on every figure (all 14,
rendered from source SVG), the 6 tables, 3 representative multi-line/matrix
equations, the chapter-opening page, and every page flagged by targeted greps
(leaked markup, unresolved refs, code-output artifacts, overfull/underfull LaTeX
boxes cross-referenced against the xelatex log). HTML was audited with targeted
regex passes (all 8 files) for every defect class in the brief, plus a systematic
extraction of every `quarto-xref`/Figure/Table/Equation reference text.

## Verdict

| Severity | Count |
|---|---|
| Blocking | 0 |
| Should-fix | 3 |
| Cosmetic | 1 |

No blocking defects — the chapter compiles cleanly (0 notebook failures, the
freshness gate is clean) and the overwhelming majority of the chapter renders
correctly in both editions: all 14 figures, all 6 tables, every display equation
and matrix I checked at full resolution, every code cell, and 95%+ of
cross-references. One should-fix defect (a PDF-only cross-reference bug) is the
same *class* of bug already documented and triaged as "should-fix" in
`reviews/ch13-output-qc.md` (crossref-scan flake), so it is rated consistently
here rather than escalated, even though it is more pervasive in this chapter.

---

## Should-fix defects

### 1. `:numref:`sec_mamba`` resolves to a bare, wrong, framework-inconsistent number — **PDF only** (HTML fully clean)

Every cross-reference to Section 12.3 ("Selective State Space Models", the
Mamba section) that uses `:numref:`sec_mamba`` renders in the PDF as a **bare
integer with no "Section"/"Chapter" prefix and no subsection digit** — and the
integer **differs between the pytorch and jax editions** (12 vs. 13). The
chapter's own opening paragraph is hit:

> *(printed p. 741 / PDF p. 808, pytorch)* "…where a transformer's cache would
> have grown a thousandfold. **12** restores what linearization gave up."
> *(jax, printed p. 749 / PDF p. 816)*: "…**13** restores what linearization
> gave up."

It should read "**Section 12.3** restores…". At least 13 further instances
recur through the chapter, e.g.:

| Location (pytorch printed p. / jax printed p.) | Rendered (pytorch / jax) | Should read |
|---|---|---|
| index.md "map" table, "Selective copying" row (744 / 752) | "Selective copying (**12**)" / "(**13**)" | "(Section 12.3)" |
| index.md Resources, "closest companions to…" (744 / 752) | "Chapter 12.2 and **12**." | "…and Section 12.3." |
| index.md Resources, "the architecture of…" (744 / 752) | "the architecture of **12** in one readable PyTorch file" | "…of Section 12.3…" |
| lstm.md §12.1.5 (756 / —) | "selective in **12**, reconciled with attention in Chapter 12.4" | "selective in Section 12.3" |
| ssm.md §12.2.6 (765 / 765) | "first-order simplification we meet in **12**)" | "…in Section 12.3)" |
| matrix-state.md opener (796 / 806) | "the selective state space models of **12** exact" | "…of Section 12.3 exact" |
| matrix-state.md §12.4.1 (800 / 810) | "The selective SSM of **12** lives on the diagonal rung" | "Section 12.3 lives on…" |
| test-time-regression.md (836 / 847) | "state transition of **12**, whose input-dependent gate…" | "Section 12.3, whose…" |
| hybrids.md Table 12.5 caption (856 / 866) | "For the recurrence column see **12** and Chapter 12.5" | "see Section 12.3 and…" |
| hybrids.md §12.7.6.1 (857 / 867) | "Mamba-1/2 from **12**, gated DeltaNet…" | "Mamba-1/2 from Section 12.3…" |

**Confirmed HTML-clean by direct A/B on the identical source sentence** (e.g.
`index.html`): `<a href="mamba.html" class="quarto-xref"><span>Section
12.3</span></a> restores what linearization gave up.` — correct, hyperlinked,
every time. A systematic extraction of **every** `quarto-xref` span (all 8
files) plus every Figure/Table/Equation reference text turned up zero other
anomalies — this is an isolated, single-label defect, not a general crossref
problem in this chapter.

**Root cause / diagnosis.** `grep -c '\ref{sec-mamba}'` on the compiled
`_pdf/{pytorch,jax}/Dive-into-Deep-Learning.tex` returns **0** in both editions,
despite ~19 source `:numref:`sec_mamba`` usages — every one was frozen into a
literal (wrong) number instead of a live `\ref{}` macro, e.g. the raw tex shows
`the closest companions to Chapter~\ref{sec-ssm} and 12.` (live ref for
`sec-ssm`, dead literal for `sec-mamba`, same sentence). Sibling section labels
in this same chapter — `sec-lstm`, `sec-ssm`, `sec-matrix-state`,
`sec-deltanet`, `sec-test-time-regression`, `sec-hybrids` — all have healthy
live `\ref{}` counts (10–29 each); only `sec-mamba` is affected, and no
duplicate `:label:` or citation-key collision exists anywhere in the repo. This
is the same **signature** as `reviews/ch13-output-qc.md` finding #1 (a single
`:numref:` target frozen into a bare, build-unstable number while every sibling
resolves correctly) — attributed there to the crossref-scan flake documented in
`docs/build-system.md` §6.8. **Fix:** re-render `make pdfs` from a clean tree
(single, non-concurrent pass) and re-verify `sec_mamba`; if it persists, it's a
gap in `tools/fix_latex.py`, which (unlike `tools/fix_crossref_numbers.py`'s
HTML equivalent) has no general fallback for a bare frozen pandoc position
number.

### 2. Long generated-text code output bleeds past the right margin — **PDF only** (HTML wraps normally)

`mamba.md` §12.3.3.1 "The Three Answers, Measured on One Task" prints three
`d2l.generate(...)` samples in a `verbatim` block (no wrapping/hyphenation).
When a sample is long, it overflows the text width:

- **pytorch**, printed p. 792 / PDF p. 859: the `LSTM:` line runs to x≈851 of a
  910 px page (110 dpi) against a normal right margin of x≈800 — **≈0.46 in
  (36.5 pt) past the margin**; the `minGRU:` line ≈0.19 in over. Confirmed by
  xelatex's own `Overfull \hbox (36.4931pt too wide)` / `(16.78326pt too wide)`
  warnings at the corresponding `.tex` lines.
- **jax**, printed p. 802 / PDF p. 869: the `minGRU:` line is worse — it runs
  to x≈906 of 910 px, i.e. **≈0.96 in past the margin**, landing 4 px from the
  physical page trim (still fully on the page, not clipped, but visibly
  crowding the edge — the most severe instance found).
- A smaller instance: `lstm.md` §12.1.2.3 "Training", printed p. 751 / PDF
  p. 818 (pytorch) — the `perplexity 90.7, "..."` line overflows only ≈0.1 in
  (10.2 pt), barely perceptible.

Nothing is clipped or lost (verbatim text isn't hard-truncated by the page
edge, just rendered into the margin whitespace), so this is should-fix, not
blocking. HTML is unaffected — the browser wraps the same text normally inside
the cell-output block. **Fix:** wrap/`textwrap.fill()` the generated samples
before printing, reduce the font size for that specific captured output, or
shorten the printed continuation length.

### 3. Figure 12.4 (`ssm-views.svg`): `$\bar K_5$` label sits over the wrong stem — HTML **and** PDF (shared static SVG)

In `tools/gen_mdl_modernrnn_figures.py`, `fig_ssm_views()` draws the
convolution-kernel stem plot with `k = Tc - 1 - i` mapping stem `i` to kernel
index `k` (so `xs[2]` ↔ `k=4`), but line 667 places the label at:

```python
ax.text(xs[2], y_ker + 0.62, r"$\bar{K}_5$", ...)
```

— i.e., the text says "$\bar K_5$" but sits above the stem that is actually
$\bar K_4$ (the true $\bar K_5$ stem is at `xs[1]`, one position to the left).
Unlike the neighboring `$\bar K_0$` label, which has a connecting arrow down to
its stem, `$\bar K_5$` is a bare floating label with no connector, so the
mismatch isn't visually self-evident but is confirmed by tracing the code.
Both stems are short (k=4 and k=5 differ little in height at this decay rate),
which is likely why it wasn't caught by eye. Affects both editions identically
since it's one static image (`img/mdl-modernrnn-ssm-views.svg`, Figure 12.4 /
"Equation 12.2.x" region, printed p. 768–769 pytorch / same content jax).
**Fix:** change `xs[2]` → `xs[1]` on line 667 (or relabel to `$\bar K_4$` to
match the current position), and consider adding a connecting arrow to match
`$\bar K_0$`'s treatment.

---

## Cosmetic defect

### 4. `:citet:` immediately before a sentence-ending period gets a stray leading space — HTML **and** PDF

Two narrative citations in this chapter are the very last token of their
sentence, and both render with an extra space before the final period:

- `ssm.md:793` — `:citet:`Astrom.Murray.2021`.` → "…see Åström and Murray
  (2021) ." (note the space before the period). Confirmed identical in HTML
  (`</a></span> .`) and PDF (both frameworks, printed p. 768/9 pytorch).
- `deltanet.md:413` — `:citet:`Yang.Wang.Zhang.ea.2024`.` → "…by Yang, Wang,
  Zhang, et al. (2024) . Multiplying out…" — same pattern.

Only these two instances exist in ch12 (systematically checked: zero other
`</a></span> [.,;:]` patterns across all 8 files). Purely typographic; doesn't
affect meaning. **Fix:** trim the trailing space the citation macro emits
before terminal punctuation, or rephrase so the citation isn't sentence-final.

---

## Classes verified CLEAN

- **Leaked source markup — clean.** Zero occurrences of `:numref:`, `:eqref:`,
  `:cite:`, `:label:`, `:begin_tab:`/`:end_tab:` anywhere in the 8 rendered
  HTML files (including inside code comments/docstrings — ch13's cosmetic #6
  class does not recur here).
- **Unresolved crossrefs — clean.** Zero `??` or `?@label` markers in HTML or
  in either PDF's extracted text for the ch12 page range. The build log's one
  `?@tbl-gpu-specs` warning is a ch13 (Computational Performance) reference,
  unrelated to this chapter.
- **Raw math markup — clean.** No bare `$…$`/`$$` leaking into visible prose
  (the only `$` occurrences in ch12 HTML are inside syntax-highlighted Python
  string literals — an explicitly-accepted false positive per the brief). No
  `\mathbf`/`\blacksquare`/`\begin{…}` leaking outside a `class="math
  inline/display"` span anywhere in the 8 files (checked by stripping all
  math/code/script blocks and searching the remainder for 16 LaTeX-command
  patterns — zero hits). No instance of the documented "`$` immediately before
  a digit" pandoc mis-parse tripwire (checked with correct $-pair parity
  tracking across all 8 source files — zero genuine hits).
- **Matrices and multi-line equations — clean.** Checked at full PDF
  resolution: the LSTM gates `aligned` block (Eq. 12.2–12.4, printed p. 747),
  the HiPPO `cases` piecewise matrix (Eq. 12.18, printed p. 769), and the
  2×2 `pmatrix` diagonal system (printed p. 768) all render correctly —
  properly sized brackets, aligned `&=`, no clipping, no overflow. Several
  other equation-dense pages (DeltaNet's generalized delta rule, Eq. 12.37;
  Table 12.4's per-model math cells) spot-checked equally clean.
- **Cross-references — clean except #1.** A full extraction of every
  `quarto-xref` span and every Figure/Table/Equation reference text across all
  8 HTML files (dozens of distinct values) turned up zero anomalies beyond the
  isolated `sec_mamba` PDF bug: every other reference resolves to a well-formed
  "Section 12.x[.y[.z]]", "Chapter N", "Figure 12.x.y", "Table 12.x.y", or
  "Equation 12.x.y" with plausible book-wide chapter numbers (3, 4, 6–13, 23).
- **Figures — clean (except #3's mislabel).** All 14 figures (12.1–12.14)
  resolve on disk, all 71 unique `<img>` sources in the chapter resolve to real
  files, captions are present, substantive, and numbered sequentially with no
  gaps. Rendered each figure natively from its source SVG at 2× and inspected
  individually: no label/line collisions, no dead whitespace, no missing
  glyphs, balanced panels, black axes — the chapter's figure quality is high.
- **Tables — clean.** All 10 tables (2 in index.md, 1 each in lstm/ssm/mamba/
  deltanet/test-time-regression, 2 in hybrids) have a single, consistent
  column count across every row, zero empty cells, zero raw-pipe leakage
  (checked programmatically across all 8 files). No auto-generated slug
  title-case captions (ch13's #2 class): every caption (including the two
  un-labeled "map" tables in index.md, which are deliberately caption-less
  pipe tables, not broken ones) carries real, substantive text. The two
  widest/densest tables (12.5 "Shipped hybrid recipes", 6 cols; 12.6 "Per-layer
  contracts", 6 cols with inline math) were checked at full PDF resolution on
  both frameworks: fit entirely within margins, no clipping, no bad page break
  (Table 12.6 doesn't spill onto the next page).
- **Code cells + outputs — clean except #2.** Syntax highlighting renders
  correctly; every cell-output is populated; zero tracebacks, zero
  `Warning`/`DeprecationWarning`/`FutureWarning`, zero tqdm-style progress
  spew, zero `cell-output-stderr` divs, in either HTML or the PDF text, across
  the whole chapter (only false-positive matches on ordinary prose words like
  "Error linear in..." and code comments like `raise max_len`).
- **Framework tabset — clean, PyTorch+JAX only as required.** Every per-cell
  code tabset in ch12 is `(PyTorch, JAX)` or, in three intentional
  framework-specific asides (`jax.lax.scan` for the LSTM, `jax.lax.
  associative_scan`, and a JAX-only one-hot-embedding note), a solo `(JAX,)`
  pane — never a stray TensorFlow/MXNet tab. The only "TensorFlow"/"MXNet"
  text found anywhere in the 8 files is the book-wide navbar's PDF/Notebook
  download dropdowns (present on every page of the whole site, all four
  editions of the whole-book download), not per-cell content.
- **Unicode — clean.** A source-level scan (body prose only, excluding math
  spans, code fences, and `<!-- slides -->` sections) found **zero** non-ASCII
  special characters (arrows, ≲, ⊙, etc.) anywhere in ch12's book body — every
  arrow/⇒/≠/≈ instance in the source `.md` files is inside the slide section
  (out of scope for the book render) or inside `$…$` math (typeset via the
  math font, confirmed rendering correctly, e.g. ⊙ in Eq. 12.4). No tofu boxes
  observed anywhere across the full 244-page contact-sheet visual scan of
  both PDFs.

## Framework PDFs

- **JAX PDF** — same content, correct JAX code (`nnx.view`, `jnp.array`,
  `d2l.numpy(...)`), same figures/tables/math. Shares defects #1 (worse here —
  every `sec_mamba` ref shows "13" instead of "12") and #3; #2's worst instance
  is in this edition. Table 12.5/12.6 render identically well.
- **TensorFlow / MXNet** — out of scope per the Advanced-part PyTorch+JAX-only
  policy; ch12 is not expected to (and does not) appear in those PDFs.

## Note: HTML/PDF figure-table-equation numbering depth (book-wide, NOT a ch12 defect)

Not counted above because it is a pre-existing, book-wide characteristic
confirmed outside this chapter, not something ch12 introduced. The HTML
numbers figures/tables/equations **3-part, resetting per section**
("Figure 12.1.1", "Equation 12.2.11", "Table 12.7.1"), while the PDF numbers
the identical items **2-part, continuous per chapter** ("Figure 12.5",
"Equation (12.18)", "Table 12.6"). Verified pairs: HTML "Figure 12.1.1" =
PDF "Figure 12.1" (LSTM cell); HTML "Equation 12.2.11" = PDF "(12.18)"
(HiPPO matrix); HTML "Table 12.7.2" = PDF "Table 12.6". Each edition is
internally consistent and unambiguous on its own; the only cost is that a
figure/table/equation number quoted from one edition (e.g. an errata note
citing "Figure 12.1") won't match the other edition's numbering for the same
item. Confirmed present in Chapter 10 (Attention) too — e.g. HTML
"Equation 10.5.1" = PDF "(10.19)" for the identical FLOPs equation — so this
is a long-standing LaTeX-vs-HTML numbering-depth gap in the build pipeline,
out of scope to fix as part of a ch12-specific pass.
