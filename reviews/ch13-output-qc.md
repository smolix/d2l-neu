# Chapter 13 "Computational Performance" — Output QC (HTML + PDF)

Date: 2026-07-21. Scope: **formatting/rendering** defects in the built output only
(not prose/content). Read-only review — no files edited.

Artifacts reviewed:
- HTML: `_book/chapter_computational-performance/{index,performance-model,hardware,compilation,memory-precision,multiple-gpus,multi-gpu-practice,fast-transformer}.html`
- PDF: `_book/pdf/Dive-into-Deep-Learning-{pytorch,jax}.pdf` (ch13 = printed pp. 861–926 pytorch / 873–938 jax; physical offset +67). Spot-check of `-tensorflow`/`-mxnet` PDFs.

## Verdict

| Severity | Count |
|---|---|
| Blocking | 0 |
| Should-fix | 5 |
| Cosmetic | 1 |

No blocking defects. The chapter renders well overall — figures, tables, math,
code cells, and cross-refs are almost entirely clean; the layout quality in the
PDF is high (no margin bleed, no overlaps, no bad page breaks across 66 pages).
Five should-fix defects and one cosmetic remain, listed below.

---

## Should-fix defects

### 1. Broken cross-references in Exercises (dropped refs + corrupted list structure) — HTML **and** PDF, 3 pages
Highest-priority defect. Cross-file `:numref:` refs in some Exercises sections
fail to resolve, rendering as **bare, unstable numbers** and — worse —
**corrupting the ordered-list structure** (phantom empty bullets, split items,
truncated sentences).

- **memory-precision §13.4.8:**
  - Ex 1: `:numref:`sec_gpt`` (memory-precision.md:484) → "Take the GPT of **1**" (HTML) / "GPT of **9**" (pytorch PDF) / "GPT of **10**" (jax PDF). Should be "Section 11.2".
  - Ex 2: `(:numref:`sec_gpt`)` (memory-precision.md:490) → ref **dropped**; the `<li>` **splits** into "…one `TransformerBlock`" + a phantom bullet "as a function of width…".
  - Ex 4: `:numref:`fig_float_formats`` (memory-precision.md:~502) → ref **dropped** (HTML: "…in terms of Why does a full training run fail…"); PDF renders "10.".
- **multiple-gpus §13.5.6 Ex 2:** `:numref:`subsec_hw-interconnects`` (multiple-gpus.md:662) → sentence **truncated** at "explain why in terms of" + an **empty bullet**; PDF shows "11.".
- **fast-transformer §13.7.7 Ex 3:** `:numref:`sec_mamba`` (fast-transformer.md:1294) → **dropped**; item **splits** into "…Mamba capstone" + phantom bullet "or ch. 11's ViT."; PDF shows "(12)".

**This is not a source typo.** The identical `@sec-gpt` construct resolves
correctly to "Section 11.2" in `hardware.html` §13.2.9 (×2), `fast-transformer.html`
body (×4), and in the memory-precision *body* (e.g. `@sec-adam` → "Section 9.6").
The fallback number is **unstable across builds** (1 / 9 / 10), which is the
signature of `@ref` being swept into **citation numbering** when Quarto's
cross-reference scan does not resolve it in time — i.e. the crossref-scan flake
documented in `docs/build-system.md` §6.8 (subset/concurrent renders don't
amortize the scan). The `sec_gpt` target exists (`chapter_transformers/gpt.md:2`,
`id="sec-gpt"` present in the built `gpt.html`).

**Fix:** re-render clean and single-pass (`make html`, `make pdfs` from a clean
tree) and re-verify these four exercises; the affected refs then likely resolve.
If it persists, reformat the offending refs so `(:numref:...)` does not sit at
the start of a wrapped continuation line inside a list item.

### 2. Table captions are auto-generated slug title-case — HTML **and** PDF, all 4 tables
Every ch13 table carries a bare `:label:` with **no caption text**, so the caption
is synthesized by title-casing the slug:

| Table | Source | Rendered caption |
|---|---|---|
| `tab_gpu_specs` | hardware.md:248 | "Table 13.2.1: **Gpu Specs**" (HTML) / "Table 13.1: Gpu Specs" (PDF) |
| `tab_rules_of_thumb` | hardware.md:465 | "**Rules Of Thumb**" |
| `tab_collectives` | multi-gpu-practice.md:471 | "**Collectives**" |
| `tab_pt_jax_parallel` | multi-gpu-practice.md:702 | "**Pt Jax Parallel**" |

"Pt Jax Parallel" and "Gpu Specs" read as broken. **Fix:** add real caption text
before each `:label:` (d2l `:Caption text :label:`tab_x`` form). Present in both
HTML and PDF, both pytorch and jax.

### 3. `host↔device` renders as a tofu box in the PDF — PDF only (HTML fine)
`hardware.md:460` (tab_rules_of_thumb, "PCIe per direction" row) uses `↔`
(U+2194 + U+FE0E variation selector). The PDF serif font lacks the glyph → the
cell renders **"host□device"** (missing-glyph box). HTML renders `host↔device`
correctly. Confirmed at 200 dpi on printed p. 882 (pytorch) and present in jax PDF.
No `↔` appears in any SVG figure, so figures are unaffected. **Fix:** use
`$\leftrightarrow$` or plain text ("host/device", "host-to-device").

### 4. `4090$→$5090` / `4090$$5090` — stray literal dollar signs — HTML **and** PDF
`hardware.md:176` `4090$\to$5090` — the closing `$` immediately precedes digit `5`,
triggering pandoc's "don't make `$5` money" rule (the exact tripwire in
CLAUDE.md / `docs/build-system.md` §6.6). Renders:
- HTML: "the consumer **4090$$5090** step" (arrow lost, two `$`).
- PDF: "the consumer **4090$→$5090** step" (literal `$` around the arrow).

Only literal-`$` leak in the chapter (all other `$…$` spans and `$NP$` byte-count
math render correctly). **Fix:** `$4090\to5090$`, or `4090 $\to$ 5090` (spaces),
or `\(4090\to5090\)`.

### 5. Profiler output table clipped at the right margin — PDF only (HTML scrolls)
`performance-model` §13.1.5 "The Profiler" (printed p. 872 pytorch / 878 jax).
The `prof.key_averages().table(...)` output is **11 columns** wide
(…CPU time avg, **Self CUDA, Self CUDA %, CUDA total, CUDA time avg, # of Calls**).
The PDF code block fits only ~6 columns (through "CPU time avg") and **clips the
5 CUDA columns off the right edge** — the data cells end mid-value ("207.", "0.",
"180.0"). These are exactly the columns the prose tells the reader to read
("two columns matter most: device (CUDA) time…"). HTML shows all 11 columns in a
horizontally-scrollable `cell-output-stdout` block. **Fix:** narrow the captured
output (fewer columns, smaller font for that block, or truncate the table in the
source cell so it fits the PDF text width).

---

## Cosmetic defects

### 6. `:numref:` literal markup leaks inside code comments/docstrings
The preprocessor does not expand directives inside code cells, so `:numref:`…``
written in a comment renders verbatim:
- `performance-model.html`: `# … (see :numref:`sec_memory_precision`).`
- `fast-transformer.html` (×2): `"""Tokens/s with warmup + device sync (see :numref:`sec_perf_model`)."""`

Reader sees literal d2l markup in the code. **Fix:** reword the comments in prose
("see the memory/precision section") rather than using `:numref:` in code.

---

## Classes verified CLEAN

- **Figures — clean.** All 18 `mdl-perf-*.svg` + 5 computed output plots resolve on
  disk (`_book/img/…`); captions present and **numbered sequentially** (13.1.1 →
  13.7.x) with no orphans or gaps. Render beautifully in the PDF (roofline, memory/
  bandwidth/latency ladders, shoreline, float-formats, PCIe topology, energy ladder,
  compute-graph, capture-pipelines, memory-anatomy, three-ways-to-split, data-parallel,
  ring-allreduce, DDP-bucketing, FSDP-lifecycle, async-dispatch, three-regime, waterfall)
  — no overflow, overlap, missing boxes, or detached captions.
  *Note (not a ch13 defect):* figure `<img>` tags carry no `alt` attribute, but this is
  book-wide Quarto behavior (accessible name via `<figcaption>` + `aria-describedby`;
  identical in mdl-linear-algebra, preliminaries, etc.).
- **Tables — clean (structure/content).** All 4 render as real `<table>`s, fully
  populated, correct header rows, em-dashes ("—") for N/A cells, no raw-pipe leakage on
  any page, no empty cells beyond the intended blank top-left corner. In the PDF they sit
  within margins, un-clipped, no lost headers across page breaks. (Only the *captions*
  are defective — see #2 — and the one tofu glyph — see #3.)
- **Cross-references — clean except the §-Exercises flake (#1).** Within-chapter and
  most cross-chapter refs resolve to numbers: "Section 13.4.1", "Equation 13.5.1/13.5.2",
  "Figure 13.2.5", "Section 9.6", "Section 11.2", "Section 12.3", "Chapter G.6". No `??`,
  no `?@`, no literal `[sec_…]`.
- **Math — clean.** Display equations render as MathJax (HTML) / proper display (PDF):
  roofline `eq_roofline` (13.1.1), ring-traffic `eq_ring_traffic` (13.5.1), cost-model
  `eq_dp_cost` (13.5.2), tokens/s, and the counting/intensity equations. `:eqref:` all
  resolve ("Equation 13.5.1", etc.). No raw `$$`, `\blacksquare`, or `\mathbf` leaks.
  Slides-only equations (past `<!-- slides -->`) correctly excluded from the book body.
- **Code cells + outputs — clean.** Every output block is populated (text or image; the
  apparently "empty" blocks are `<img>` plot outputs). No Python tracebacks, no
  `UserWarning`/`FutureWarning`, no `RESOURCE_EXHAUSTED`/NCCL warnings, no download-progress
  or bare-integer spew. ("Error:" on every page = KaTeX `throwOnError:false` config;
  "exception" in hardware = prose "the exception that proves the byte rule".) Long code
  blocks (e.g. the DDP launcher script) fit within the PDF margins with no truncation.
- **Unicode symbols — clean except `↔` (#3).** `→` (×36 in prose), `≲`, `×`, `≈`, `≤`,
  `−`, `§`, `µ` all render in both HTML and PDF. `≡`, `∝`, `¼`, and the `⇒` cluster are
  all slide-only (do not affect the book).
- **Framework tabsets — present** with PyTorch + JAX controls on all content pages.

## Framework PDFs

- **JAX PDF — good.** Renders ch13 with correct JAX code (`jax.random.PRNGKey`,
  `jnp.dot(...).block_until_ready()`), same figures/tables/math. Shares every
  framework-independent defect (#2 auto-captions, #3 tofu, #4 `4090$→$5090`, #5 profiler
  clip); the Exercises flake (#1) is worse here — *all* `sec_gpt` refs render "10",
  including the hardware exercise that resolved correctly in the pytorch build.
- **TensorFlow / MXNet PDFs — present, not broken/empty, likely acceptable.** ch13
  appears in both (tf printed 727–768, mxnet 723–764; 42 pp each vs 66 for pytorch/jax).
  Full prose, figures, equations, tables, summaries, and exercises render correctly and
  the chapter ends cleanly (13.7.6 Summary / 13.7.7 Exercises before ch14). The
  PyTorch/JAX-only **code cells are essentially absent** (≈6 code lines vs 157 pytorch /
  120 jax), so tf/mxnet readers get the prose skeleton without the code examples. This is
  consistent with the Advanced-part PyTorch+JAX-only policy and the RL-chapter precedent
  (present but code-less). Minor editorial oddity (not a rendering defect): the shared
  prose still says "on the PyTorch side… on the JAX side" in books whose reader picked
  tf/mxnet. If undesired, ch13 could be excluded from the tf/mxnet books entirely.
