# Chapter 11 "Transformers" — Output QC (HTML + PDF)

Date: 2026-07-21. Scope: **formatting/rendering** defects in the built output only
(not prose/content). Read-only review — no files edited, nothing committed.

Artifacts reviewed:
- HTML: `_book/chapter_transformers/{index,transformer-block,gpt,kv-cache,encoders-decoders,vision-transformer,moe,scaling-laws}.html`
- PDF: `_book/pdf/Dive-into-Deep-Learning-{pytorch,jax}.pdf` (ch11 = printed pp.
  661–740 pytorch / 669–748 jax; physical PDF pages 728–807 / 736–815; offset
  +67 in both, same as the ch13 review). This chapter is PyTorch+JAX only —
  tensorflow/mxnet PDFs were out of scope and not checked.

Method: rasterized every ch11 page in both PDFs at 110 dpi (`pdftoppm`), tiled
into low-res contact sheets for a full visual pass, then re-rendered
full-resolution close-ups of every figure, both tables, and every flagged
location; cross-checked `pdftotext` extractions against the HTML and source
`.md` for every `:numref:`/`:eqref:` target used in the chapter (37 distinct
labels; all but 3 confirmed correct). Scratch work at
`/tmp/claude-4002/-home-smola-d2l-neu/319ce40a-90af-4f30-bdd8-cfe83f878de0/scratchpad/oqc11/`.

## Verdict

| Severity | Count |
|---|---|
| Blocking | 2 |
| Should-fix | 3 |
| Cosmetic | 0 |

The chapter's figures, math, framework tabsets, and the large majority of
cross-references and tables are clean and render well in both HTML and PDF.
Two blocking defects recur pervasively enough in the PDF to need a fix before
this chapter should ship in print: a small set of cross-chapter `:numref:`
targets resolve to bare, wrong integers instead of "Chapter/Section N.M"
(~44 occurrences across the chapter), and several wide captured-output lines
are silently clipped (not wrapped) at the page's right margin, permanently
losing part of the printed content. Three should-fix defects (a table-header
text collision, a table missing its number in HTML, and a leaked debug
warning with a local filesystem path) round out the findings.

---

## Blocking defects

### 1. Three cross-chapter `:numref:` targets resolve to bare wrong integers in the PDF — pervasive, ~44 occurrences, both frameworks

`:numref:`sec_gpt`` (33 uses), `:numref:`sec_multihead-attention`` (8 uses), and
`:numref:`sec_attention-scoring-functions`` (3 uses) all render correctly as
"Section 11.2" / "Section 10.3" / "Section 10.2" **everywhere in the HTML**
(confirmed at every occurrence across all 6 files that use `sec_gpt`, plus both
files using the other two). In **both PDF editions**, every single occurrence
of these three specific labels instead renders as a bare, contextless integer —
never "Chapter"/"Section", never a decimal:

| Label | Should render | PyTorch PDF shows | JAX PDF shows |
|---|---|---|---|
| `sec_gpt` (×33) | Section 11.2 | **9** | **10** |
| `sec_multihead-attention` (×8) | Section 10.3 | **5** | **6** |
| `sec_attention-scoring-functions` (×3) | Section 10.2 | **4** | **5** |

Representative locations (all confirmed by direct text extraction, pytorch
printed-page / jax printed-page):
- Chapter-opening roadmap paragraph: "...ending with the configurable block.
  **9** assembles the blocks into a language model..." (p.661 / p.669).
- `transformer-block.md`: "the two normalization arrangements... Each token
  queries the others as in **5** and adds the retrieved mixture..." (p.664/672);
  "will matter when we load its weights in **9**.)" (p.670/678, further down
  the same file).
- `kv-cache.md`: "The generate method of **9** was left deliberately naive"
  (p.682/690).
- `encoders-decoders.md`: "The GPT of **9** made one architectural commitment"
  (p.697/705); "...each composed from the two primitives of **4**: a padding
  mask..." (p.703/712).
- `moe.md`: "the dense `d2l.GPT` of **9** at width 256..." and "Since **9**
  showed this corpus drives models into memorization" (p.727/736).
- `scaling-laws.md`: "**9** promised that these rows would come back as
  argument lists" (p.737/746); Table 11.2's own gloss text: "Every row is the
  block of Chapter 11.1 in the causal wiring of **9**" (p.736/746).

Every *other* checked `:numref:` target in the chapter — same-chapter section
refs (11.1, 11.3–11.7), cross-chapter refs into ch. 6–9, 17, 19, and several
*other* chapter-10 targets (`sec_positional-information`→10.4,
`sec_attention-at-scale`→10.5, `sec_what-attention-computes`→10.6) — resolves
correctly and identically in pytorch and jax. Only these three labels are
affected, and the wrong value is consistently 1 higher in the jax build than
the pytorch build (9→10, 5→6, 4→5), which is a useful diagnostic signature.

**Worse in one place: dropped ref + corrupted list markup, HTML.**
`encoders-decoders.md` Exercise 3 ends with `:numref:`sec_multihead-attention``
alone on its own wrapped continuation line inside a numbered list item
(line 1060, indented 3 spaces — a markdown list continuation, not a new
paragraph). There, instead of a wrong number, the HTML **drops the reference
entirely** and injects a phantom empty nested list:
```html
...Reconcile this with the warnings about reading attention maps in
<ol class="example" type="1">
<li></li>
</ol></li>
<li>Change one line of ...
```
The sentence trails off with no reference and no punctuation; a stray empty
`<ol><li></li></ol>` is left behind before Exercise 4. The PDF equivalent at
this same spot (pytorch p.710, jax p.719) degrades only to the milder "bare
integer" defect (renders "...in **5**." / "**6**.", list numbering itself
stays intact) — so the corruption is HTML-specific and position-specific
(list-continuation-line only; two other *working* labels — `sec_scaling`,
`sec_positional-information` — sit in the exact same "alone on a wrapped
exercise continuation line" position elsewhere in the chapter and render
perfectly, which isolates the trigger to "broken label" × "alone on a list
continuation line," not either condition alone).

**This is a known, recurring pipeline issue, not a one-off.** The sister
same-day review `reviews/ch13-output-qc.md` (defect #1) independently found
`sec_gpt` resolving to the identical unstable fallback pattern ("1" HTML-list-
context / "9" pytorch / "10" jax) and traced it to the crossref-scan flake
documented in `docs/build-system.md` §6.8 ("subset/concurrent renders don't
amortize the crossref scan... falls back to standalone mode"). This ch11 scan
shows the same flake reproducing again in a separate, later build, and shows
it is **not confined to Exercises/list-item contexts** as the ch13 finding's
fix hint speculated — most of the 44 ch11 occurrences are in plain body prose,
and HTML resolves those correctly while the PDF does not, so whatever
non-amortized/subset code path affects the PDF target evidently runs
separately from (and is not fixed by) the HTML target's `fix_crossref_numbers.py`
pass. **Fix:** re-render the PDF from a clean tree in one single-pass invocation
and re-verify these three labels; if it persists, the PDF assembly path
(`gen_pdf.py`/`fix_latex.py`) needs its own crossref-amortization fix analogous
to whatever protects `make html`. Separately, reformat
`encoders-decoders.md:1059–1060` so the ref does not sit alone on a wrapped
list-continuation line (matches the ch13 fix hint).

### 2. Wide captured-output lines are silently clipped at the PDF's right margin — content lost, not wrapped — both frameworks

Any single output/print line longer than ~110–113 monospace characters is cut
off exactly at the page's printable margin with **no wrapping, no ellipsis,
and no indication text is missing** — the rest of the line simply never
appears anywhere in the PDF. Confirmed at three distinct locations (visually,
110 dpi renders) and via text-extraction diffing against the (complete) HTML:

- **`gpt.md` §11.2.3/11.2.4 "Sampling"/"Loading GPT-2"** (pytorch p.678, p.680;
  jax p.687 confirmed, same pattern) — **7 of 7** generated-text sample lines
  across this pair of sections exceed the safe width and are clipped, e.g.:
  - Pytorch p.678: `T=1.0, top_k=None: 'the time traveller put forth his finger towards the lever no he said suddenly lend me your ha` — cut off mid-word; HTML has 45 more characters ("nd and turning to the psychologist he took tha'"). Two more temperature/top-k samples on the same page are clipped identically.
  - Pytorch p.680: `The secret of a good deep learning textbook is that you don't know what you're supposed to write. That's not quite` — cut off mid-word ("quite" should continue "right, but it's...").
  - Jax p.687: `T=1.0, top_k=None: 'the time traveller holding the lamp aloft i intend to explore time is that plain i was never m` — cut off mid-word; two more samples on the same page clipped identically.
  This is the section whose entire pedagogical point is "look at what the
  model generated" — the clipping actively undermines the content being taught.
- **`kv-cache.md` §11.3.4.2 "A Window Needs a Sink"** (pytorch p.694, jax
  p.703) — the per-layer attention-sink measurement is real data, not noise,
  and is truncated after 8 of 12 layers:
  `layer 0: 0.00 layer 1: 0.01 layer 2: 0.11 layer 3: 0.30 layer 4: 0.32 layer 5: 0.51 layer 6: 0.46 layer 7: 0.5` — cuts off mid-number (should be "0.57") and layers 8–11 (0.48, 0.48, 0.45, 0.34) are entirely absent from the printed book. Identical clip point in both pytorch and jax editions.
- **`scaling-laws.md` §11.7.1.3 "Checking the Arithmetic"** (pytorch p.731,
  pytorch-only — jax's code path doesn't hit this warning) — see should-fix
  defect #5 below; the same clipping mechanism cuts the leaked warning off
  after "UserWarning:", dropping the rest of the (unwanted) message text.

The clipped-vs-full boundary is consistent across all three independently
confirmed instances (107, 110, 113 characters shown before the cut), which
points to a fixed-width verbatim/code-output LaTeX environment with no
`breaklines`/`breakanywhere` for cell-output blocks. This is the same defect
class flagged in the ch13 review (#5, a profiler table clipped at 198 chars)
recurring here at a lower, more easily-triggered threshold (~110 chars) and
hitting ordinary `print()` output rather than only wide tabular dumps — i.e.
it is easy for any future notebook cell with a longish output line to trigger.
**Fix:** the code-output LaTeX template needs line-wrapping (e.g. `fvextra`/
`breakanywhere`, or Quarto's PDF code-block line-wrap option) so wide output
wraps instead of vanishing; as a stopgap, any notebook cell whose output
already exceeds ~100 characters per line should be reformatted to print
shorter lines (e.g. one item per line) before the next capture.

---

## Should-fix defects

### 3. Table 11.2 header cells collide — "normalizationpositions" — PDF only, both frameworks, both header instances

`scaling-laws.md`'s recipe table (`tab_modern-recipe`) renders correctly in
HTML, but in the PDF the "normalization" and "positions" column headers print
with **no separating space**, reading as one merged word
"**normalizationpositions**" — confirmed at 200% zoom on both the table's
first page (pytorch p.736, jax p.745) and its page-break continuation (jax
p.746, where the header row correctly repeats but with the same collision).
Data rows are unaffected — every data cell wraps correctly within its own
column (e.g. "RMSNorm," / "pre" stack cleanly) — only the header text overflows
into the neighboring column with no gap. **Fix:** the header cells need the
same wrapping treatment as the body cells (e.g. force a `\parbox`/wrap on the
header row, or shorten "normalization" to fit, or add explicit `\hspace` /
column-width padding between the two header columns).

### 4. `moe.md`'s table has a caption but no `:label:` — unnumbered in HTML, silently auto-numbered in the PDF

Every other captioned table/figure in ch11 uses the `:Caption text\n:label:`tag`` pairing (confirmed: `scaling-laws.md:541-542` has both). `moe.md:661` has
only the caption line (`:Routed-expert configurations of deployed MoE language
models`) with **no `:label:` line following it**. Effect:
- **HTML** (`moe.html`): renders as a bare, un-numbered `<table><caption>Routed-expert
  configurations of deployed MoE language models</caption>` — no "Table 11.X:"
  prefix, no crossref anchor, cannot be `:numref:`'d from the web edition.
- **PDF**: LaTeX auto-numbers *any* captioned table via its native counter
  regardless of `\label`, so both pytorch and jax print it correctly as
  "**Table 11.1:** Routed-expert configurations of deployed MoE language
  models" (confirmed, p.726 pytorch / p.736 jax).

This is the inverse of the classic "bare label, no caption → slug title-case
caption" failure mode (checked for explicitly across all 8 figures + both
labeled tables in this chapter — none found; every other caption is real,
hand-written prose, correctly numbered in both formats). Here the caption text
is fine but the crossref plumbing is incomplete, so only the *web* edition
loses the number. **Fix:** add `:label:`tbl_moe-experts`` (or similar)
immediately after `moe.md:661`, matching the `tab_modern-recipe` pattern.

### 5. Leaked PyTorch profiler `UserWarning` with a local absolute filesystem path — HTML and PDF, pytorch only

`scaling-laws.md` §11.7.1.3's profiler-check cell captures and renders a raw
dependency warning, absolute path included, in the middle of otherwise clean
output:
```
/home/smola/d2l-neu/.venv-pytorch/lib/python3.12/site-packages/torch/profiler/profiler.py:224: UserWarning: Warning: Profiler clears events at the end of each cycle.Only events from the current cycle will be reported.To keep events across cycles, set acc_events=True.
  _warn_once(
```
Confirmed in `scaling-laws.html`'s `cell-output-stdout` block and in the
pytorch PDF (p.731; also clipped there per defect #2). Not present in the jax
build (jax's cost-accounting code path doesn't invoke `torch.profiler`). This
is the only warning/traceback/absolute-path leak found anywhere in the chapter
(swept all 8 HTML files and both PDF page ranges for `Warning`, `Traceback`,
`site-packages`, `/home/smola`, `ipykernel` — this is the single hit). **Fix:**
suppress the warning at the source — either `warnings.filterwarnings('ignore',
category=UserWarning)` around the `profile(...)` call, or pass
`acc_events=True` to `profile(...)` as the warning itself suggests — then
`make -B _notebooks/pytorch/chapter_transformers/scaling-laws.executed && make
capture-outputs FILES=chapter_transformers/scaling-laws.md` to re-bless.

---

## Classes verified CLEAN

- **Figures — clean.** All 8 labeled figures (`fig_transformer-block`,
  `fig_norm-taxonomy`, `fig_kv-cache`, `fig_gqa`, `fig_three-wirings`,
  `fig_latent-bottleneck`, `fig_vit`, `fig_moe-layer`) plus every computed
  output plot resolve on disk under `_book/img/...`; every `<img>` checked
  across all 8 HTML files resolves (no missing images). All carry full,
  hand-written captions (never a slug/title-cased placeholder) and are
  numbered correctly and consistently in both formats. Visually inspected at
  full 110 dpi resolution — the transformer-block anatomy, 4-way norm-taxonomy
  comparison, vision-transformer architecture (including embedded raster patch
  photos), and mixture-of-experts diagrams all render with no clipping,
  overlap, distortion, or dead whitespace; colors, arrows, and math labels are
  crisp. The full 80-page contact-sheet pass over both PDFs found no blank
  figure boxes or detached captions anywhere in the chapter.
- **Tables — clean on structure/population**, aside from defects #3/#4 above.
  All 3 tables (`tab_modern-recipe`, `moe.md`'s MoE-configs table, and the
  intentionally-uncaptioned "Which Wiring When" table in `encoders-decoders.md`)
  render fully populated with correct headers and zero empty cells in both
  HTML and PDF; no raw-pipe leakage anywhere (the only `|` characters found in
  visible HTML text are LaTeX norm-bar math `\|\mathbf{H}\|`, not unparsed
  table syntax). No page-break orphans — Table 11.2's continuation onto a
  second PDF page correctly repeats the header row.
- **Cross-references — clean except defect #1.** All 34 other distinct
  `:numref:` targets checked (of 37 total) resolve to real, correct numbers
  identically in HTML and both PDF editions: same-chapter sections 11.1/
  11.3–11.7, cross-chapter chapters/sections 6.1, 7.3, 7.4, 8.2, 8.7, 9.11,
  10.4–10.6, 12, 13, 13.2, 17, 17.4, 19, and all in-chapter figure/table
  self-references. No `??`, no `?@`, no literal `[sec_…]` found anywhere.
  *(Observation, not a defect: HTML numbers figures/tables hierarchically per
  file — e.g. "Figure 11.3.1/11.3.2" — while the PDF numbers them flat per
  chapter — "Figure 11.3/11.4." Both schemes are internally self-consistent
  in every instance checked; this is evidently a deliberate difference between
  the two numbering pipelines, not a defect.)*
- **Math — clean.** MathJax loads (`tex-chtml-full.js`) and every inline/
  display math span is present and delimiter-balanced across all 8 files
  (`\(`/`\)` and `\[`/`\]` counts match exactly, 0 raw `$…$` found outside code
  blocks in any file). Display equations (11.1 block-anatomy, 11.5 SwiGLU, the
  MoE router equations, the scaling-law power laws) render fully in the PDF
  with no clipping and correct numbering. No literal `\mathbf`/`\blacksquare`/
  `\mathcal` leaking outside proper math delimiters.
- **Code cells + outputs — clean** aside from defects #2 and #5. Every output
  block is populated; no Python tracebacks anywhere in either PDF's ch11 page
  range or any HTML file; no download/training progress-bar spew; no
  accidental bare-integer-only debug print lines. Framework tabsets are
  present and correct on every code-bearing page — PyTorch + JAX only, no
  stray TensorFlow/MXNet tabs anywhere (consistent with the Advanced-part
  framework policy).
- **Unicode — clean.** No U+FFFD replacement characters (no encoding
  corruption). The only non-ASCII characters in any of the 8 HTML files are
  em/en dashes, curly quotes, and two accented Latin letters (é, ó) — all
  fine in any font. The specific risk characters called out for this class of
  check (↔, ⇄, ≲) do not appear anywhere in this chapter's content, so that
  risk is simply not present here. Extensive visual sampling of the rasterized
  PDF (all 160 pages at low-res, ~10 pages at full 110 dpi) found no tofu/
  missing-glyph boxes — θ, α, β, ×, √, †, —, ² all render as proper glyphs.

---

## Cross-check with the concurrent ch13 review

`reviews/ch13-output-qc.md` (same day) independently found the identical
`sec_gpt` crossref-fallback signature and attributed it to the crossref-scan
flake in `docs/build-system.md` §6.8. This review's findings corroborate that
diagnosis and extend it: the flake also hits `sec_multihead-attention` and
`sec_attention-scoring-functions`, is not confined to Exercises/list-item
contexts, and reproduces on a separate, later build — i.e. it is a persistent
pipeline weakness, not a one-off fluke, and re-rendering alone may not be
sufficient to clear it.
