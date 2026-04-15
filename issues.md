# Comparison Report: d2l-neu vs d2l.ai

Automated scan of all 192 HTML pages plus targeted comparison of 7 key
pages against the live d2l.ai site.

## Automated Scan Results

192 pages checked for structural issues:
- Unresolved cross-references: **0**
- Wrong section numbers (file-position >23): **0**
- Wrong figure numbers (file-position >23): **0**
- Wrong equation tags (file-position >23): **0**
- Wrong sidebar numbers: **0**
- "Chapter X.Y" instead of "Section X.Y": **0**
- Mixed figure chapter prefixes: **2** (both are legitimate cross-chapter
  references — bahdanau-attention refs seq2seq's Fig 10.7.2, mlp refs
  classification's Fig 4.1.1)

## Targeted d2l.ai Comparison

Compared 7 pages that had issues in the previous round:

| Page | Status |
|------|--------|
| chapter_preliminaries/calculus.html | **CLEAN** — figure numbered 2.4.1, equations numbered |
| chapter_linear-regression/linear-regression.html | **CLEAN** — figures 3.1.1–3.1.3, equation cross-refs correct |
| chapter_convolutional-neural-networks/lenet.html | **CLEAN** — "Section" prefix used correctly |
| chapter_recurrent-neural-networks/rnn.html | **CLEAN** — equations numbered |
| chapter_optimization/sgd.html | **CLEAN** — equation cross-refs correct |
| chapter_preface/index.html | **CLEAN** — figure unnumbered, cross-refs show correct chapter numbers |
| chapter_introduction/index.html | **CLEAN** — figures 1.1–1.10, chapter number "1" shown |

## Remaining Differences from d2l.ai (cosmetic, non-blocking)

1. **Equation count mismatch**: Some pages have slightly fewer numbered
   equations than d2l.ai because the auto-numbering assigns one label per
   `$$...$$` block, while d2l.ai/Sphinx numbers each `\begin{aligned}`
   row separately in some cases. Example: calculus has 9 numbered equations
   vs d2l.ai's 10.

2. **"Figure N.M" vs "Fig. N.M.K"**: Quarto uses "Figure" prefix, d2l.ai
   uses "Fig." prefix. d2l.ai uses 3-level numbering (chapter.section.seq),
   Quarto uses 2-level within each file (chapter.seq) which happens to match
   for section-level files.

3. **Citation style**: "Author et al. 2016" (no comma) vs d2l.ai
   "Author et al., 2016" (with comma). CSL style difference.

## Build Status

Zero errors. Zero warnings. All 2,007 artifacts built successfully.
