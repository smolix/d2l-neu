# PDF cross-reference "frozen number" bug — root cause & proposed fix

**Date:** 2026-07-21
**Scope:** the PDF-only `:numref:` freeze reported in `reviews/ch11-output-qc.md`,
`ch12-output-qc.md`, `ch13-output-qc.md` (e.g. `sec_mamba` → bare "12"/"13").
**Method:** static analysis of the four pipeline tools + the last-render
`.tex` (snapshotted to scratch) + **isolated scratch Quarto renders** that
reproduce and bisect the bug. No repo build products were touched.

---

## 1. TL;DR

* **Confirmed, reproduced, deterministic, PDF-only.** It is a bug in **Quarto's
  LaTeX cross-reference resolution** (Quarto `1.9.37`), *downstream of the
  correct input the pipeline produces*. HTML is always correct.
* For an affected section label, **every** `@sec-X` reference is emitted into the
  `.tex` as a **bare, static, parenthesis-free parent-chapter number** (e.g.
  `selective models of 12.`) instead of a live `\ref{sec-X}` — dropping the
  section digit (`.3`), the "Section"/"Chapter" prefix word, and the hyperlink.
* The affected set is **not** a property of any individual label. It is an
  **emergent, input-sensitive** function of the whole document's cross-reference
  index: it is **stable for a fixed input** (re-rendering identical sources
  reproduces the identical freeze — so *"re-render from a clean tree" does NOT
  fix it*), but flips unpredictably on small content edits, and **differs
  between editions**: **8 labels freeze in the pytorch PDF, 10 in the jax PDF**
  (jax adds `sec-lazy-init`, `sec-read-write`).
* Because the frozen output is an **unmarkable bare integer**, `fix_latex.py`
  **cannot** detect or repair it post-hoc (unlike the HTML path, which never
  sees the bug). The fix must **prevent** the freeze or **guard** against it.
* The QC docs' hypothesis ("crossref-scan flake, re-render — `docs/build-system.md`
  §6.8") is **wrong**: §6.8 is about *HTML* concurrency/subset-render flakes;
  this PDF bug is deterministic and re-rendering the same input will reproduce it
  every time.

---

## 2. Confirmed frozen sets (from the last-render `.tex`)

A label is "frozen" if it is defined (`\label{sec-X}`) and referenced in source
(`:numref:`/`:ref:`sec_X``) yet has **zero** `\ref{sec-X}` in the compiled `.tex`.

| edition | frozen labels |
|---|---|
| **pytorch** (8) | `sec-attention-scoring-functions`, `sec-gpt`, `sec-mamba`, `sec-mdl-continuous-normalizing-flows`, `sec-mdl-euler-runge-kutta`, `sec-mdl-maximum-likelihood`, `sec-multihead-attention`, `sec-numerical-stability` |
| **jax** (10) | the 8 above **+** `sec-lazy-init`, `sec-read-write` |

`sec_mamba` freezes to **12** (pytorch) / **13** (jax) — i.e. Quarto's positional
chapter counter for the *containing* chapter, one higher in jax because jax has
one extra chapter before it. Sibling labels in the very same files resolve
correctly (`sec-lstm` 29 live `\ref`, `sec-ssm` 18, `sec-deltanet` 18, …).

The definition sites are **byte-identical in structure** between frozen and
working labels (`\section{…}\label{sec-mamba}` vs `\section{…}\label{sec-ssm}`),
the reference sites are **identical in form** (frozen and live refs occur in the
*same sentence*: `… closest companions to Chapter~\ref{sec-ssm} and 12.`), and
Quarto's per-file index (`.quarto/idx/*.json`, `headingAttr.id`) is identical.
So the discriminator is **not** in the sources.

---

## 3. What it is NOT (hypotheses tested and falsified)

Each was checked against the full frozen set and/or in isolated renders:

* **Heading level** — frozen set mixes file-top `#` (mamba, gpt, …) and `##`
  subsections (euler-runge-kutta, cnf). Within `mdl-odes-solvers.md` the four
  same-level `##` siblings freeze in an **alternating** pattern
  (linear-odes-stability ok / euler-runge-kutta FROZEN / neural-odes ok /
  continuous-normalizing-flows FROZEN).
* **Duplicate anchor / stale `_pdf` dir** — `sec-attention-scoring-functions`
  and `sec-multihead-attention` *do* have a duplicate `{#…}` in the stale
  `_pdf/<fw>/chapter_attention-mechanisms-and-transformers/` tree, **but** that
  dir is not in `_quarto.yml`, and an isolated test proved Quarto **ignores
  files not listed in the book** (adding an unlisted duplicate did not poison
  resolution). Red herring — the other 6 labels are single-def.
* **Source label collision** (`subsec_X`→`sec-X` vs `sec_X`) — none of the 8
  frozen ids has a second source `:label:`.
* **Forward reference** (referenced before defined) — falsified: working
  `sec-lstm` first-ref/def gap = 69 files; frozen `sec-mdl-euler-runge-kutta`
  gap = 0.
* **Per-file / total reference count** — falsified: working `sec-lstm` (29 refs)
  and `sec-kv-cache` (35) beat every frozen label; 100 synthetic refs before a
  target did not freeze it.
* **Reference inside an ordered list / Exercises** — falsified in the full book:
  working `sec-adam` has 3 exercise-list refs, frozen labels have 1.
* **Unresolved crossrefs elsewhere** — falsified: up to 5 synthetic unresolved
  refs (any type) did not freeze a resolved target.

---

## 4. Decisive reproduction (isolated scratch Quarto renders)

All in `…/scratchpad/xref-rootcause/`, using the repo venv Quarto and the real
`_pdf/pytorch/chapter_mdl-dynamics/*.qmd` files (`--to latex`, `execute:
enabled:false`, same `crossref: {chapters:true}` / `number-sections:true` as
`gen_pdf.py`).

1. **Clean 3-chapter synthetic** → all refs `\ref{}` (baseline works).
2. **Single real file (`mdl-odes-solvers.qmd`) + a simple referrer** →
   `euler-runge-kutta` resolves (`\ref` present). **No freeze in isolation.**
3. **The whole real `chapter_mdl-dynamics/`** → `euler-runge-kutta` and
   `continuous-normalizing-flows` **FREEZE** (0 `\ref`); `neural-odes`,
   `linear-odes-stability` stay live. **Reproduced from unmodified sources.**
4. **Determinism:** identical inputs rendered twice → identical freeze both
   times (refutes the "flake / re-render" theory).
5. **HTML control (same freezing project):** the euler refs render as correct
   `class="quarto-xref"` links with "Section N.M" — **HTML clean even when the
   PDF freezes.** So the bug is in the **LaTeX** path only.
6. **Bisection** showed the freeze is chaotically sensitive: adding/removing a
   single reference or a block of the referring file flips the whole label
   between live and frozen. No single reducible markdown construct triggers it
   in a clean synthetic — it only appears at real-document crossref-index scale.
   This, plus the **edition-dependent frozen set** (§2), is the signature of an
   **emergent Quarto-internal index/ordering bug**, deterministic per fixed
   input.

### Why the pipeline is not at fault, and where Quarto breaks

* The intermediate markdown Quarto feeds pandoc (`keep-md`) still contains
  `@sec-X` — so the number is injected **inside** Quarto's pandoc/Lua render.
* Quarto's shared `resolveRefs` filter
  (`…/quarto_cli/share/filters/main.lua:14800`) **always emits `\ref{label}` for
  LaTeX** for a `sec` ref (there is no `sec` custom-ref category), and emits the
  number+hyperlink only on the **non-LaTeX** branch — which is exactly why HTML
  is correct and LaTeX *should* be too. The bare number is therefore produced by
  a **separate Quarto book/crossref resolution step that runs for LaTeX output,
  downstream of `resolveRefs`**, and pre-resolves some references to a static
  chapter number instead of leaving the `\ref`. That step is the defect.
* Neither `crossref-resolve-refs: false` nor `crossref: {chapters: false}`
  suppresses it (both tested — still frozen).

---

## 5. Is the HTML "bare 1" flake the same bug? — **No, distinct.**

* This PDF freeze never occurs in HTML: `resolveRefs`' HTML branch produces the
  correct `quarto-xref` link+number, confirmed even in the freezing repro (§4.5).
* The HTML "cross-chapter exercise ref occasionally drops to a bare 1"
  (*nondeterministic across builds*) matches the **§6.8 concurrency/subset-render
  flake** (transient `main.lua` filter errors / standalone fallback when renders
  share a project dir). The production HTML build is a single `quarto render`,
  which resolves these correctly; the flake only surfaces under concurrent/subset
  renders. Different mechanism, different determinism class.

---

## 6. A separate, adjacent bug found: `?@tbl-gpu-specs`

The compiled pytorch `.tex` contains **`?@tbl-gpu-specs` ×4** — an *unresolved*
crossref. The target **is** defined (`chapter_computational-performance/hardware.md`
`:label:`tab_gpu_specs`` → table-caption `{#tbl-gpu-specs}` in `hardware.qmd:282`)
yet `@tbl-gpu-specs` (referenced from `hardware`, `fast-transformer`,
`multiple-gpus`) fails to resolve. This is a **table-caption label-attachment**
failure (a different crossref symptom, ch13), and unlike the section freeze it
**is** already caught by `fix_latex.py`'s residual-`?@` warning — but the warning
is non-fatal, so it currently ships as a literal `?@tbl-gpu-specs` in the PDF.
Worth fixing alongside (verify the table caption/label attaches — see the
table-caption gotchas in `CLAUDE.md`), and worth promoting the warning to a hard
failure (see §7B).

---

## 7. Proposed fix (ranked; implement + validate cheaply)

**Validation shortcut (important):** the bug is deterministic per input, so any
candidate fix can be validated in **minutes** by rendering only the affected
chapter(s) in an isolated scratch project (as in §4) — you do **not** need the
~45-min full PDF render until final sign-off. `chapter_mdl-dynamics/` is a
reliable one-chapter reproducer.

### A. (cleanest potential real fix) Bump/loosen the Quarto pin, validate on the repro
Since this is a Quarto-internal LaTeX-crossref defect in **1.9.37**, first test
whether a different Quarto version resolves it: render the `chapter_mdl-dynamics/`
repro under a newer (or older) Quarto and check `grep -c 'ref{sec-mdl-euler-runge-kutta}'`.
If a version fixes it, that is the lowest-blast-radius fix (Quarto lives in
`.venv-build`; nothing in the committed store changes). File a minimal upstream
report either way (the repro in `scratchpad/xref-rootcause/repro3` is ready).

### B. (safe, do now regardless) Add a detection guard to `tools/fix_latex.py`
`fix_latex.py` already warns on residual `?@` (`fix_html_only_refs`, lines 30-41)
but has **no** guard for the *silent* frozen-number case. Add a check in
`fix_all()` (after the content is loaded, before returning): for every
`\label{sec-X}` present in the `.tex`, if `sec-X` is used as `@sec-X` anywhere in
the sibling `_pdf/<fw>/**/*.qmd` sources **and** the `.tex` contains **zero**
`\ref{sec-X}`, that label is frozen → **`sys.exit` with a loud error listing the
labels**. This converts a silently-wrong PDF into a hard build failure and also
catches the `?@tbl-gpu-specs` class if you extend it to `\label{(fig|tbl|eq)-…}`.
Blast radius: none on output; it only fails builds that were already shipping
wrong numbers. ~20 lines.

### C. (prevention, validated to work) Emit raw LaTeX refs for section crossrefs in the PDF path
**Validated in scratch:** rewriting `@sec-X` → `` `\ref{sec-X}`{=latex} `` in the
freezing repro produced **8/8 live `\ref`** where the `@`-form gave 0. Bypassing
Quarto's crossref resolution for section refs eliminates the bug class.
Implement **PDF-only** (do not touch the HTML path in
`d2l_preprocess.translate_directives`): add a post-pass in
`tools/gen_pdf.py::convert_file_pdf` (after `emit_pdf_qmd`, alongside the existing
`_pdf_excluded_ref_replacements()` rewrite at lines 197-200) that rewrites each
`@sec-X` to a raw-LaTeX reference.
* **Prefix word:** to preserve current wording you must emit `Chapter~\ref{sec-X}`
  for file-top (`#`) section labels and `Section~\ref{sec-X}` for `##`+
  subsection labels — so the pass needs a global *label→heading-level* map (build
  it once by scanning the generated `.qmd`s for `^#{1,N} … {#sec-X}`).
  Alternatively emit `\Cref{sec-X}` and add `\usepackage{cleveref}` to
  `static/d2l-preamble.tex` so cleveref supplies the prefix automatically
  (needs a render test against the `fix_latex.py` `\setcounter{chapter}`
  manipulation).
* **Scope to `sec-` only.** The frozen class is entirely section refs; leave
  `@eq-`/`@fig-`/`@tbl-` to Quarto (and fix `tbl-gpu-specs` separately, §6).
* **Blast radius:** touches **every** section cross-reference in **both** PDFs
  (thousands). Requires a full PDF re-render of pytorch **and** jax plus a visual
  spot-check of ref wording/numbers. It also *aligns* the PDF with the HTML
  (which shows "Section 12.3" for file-top section refs, where the current PDF
  shows "Chapter 12.1"). **No `outputs/` recapture needed** — this is render-only,
  no notebook execution changes.

**Recommendation:** ship **B immediately** (safety net, tiny, zero output risk),
try **A** (may fix it outright with the least churn), and fall back to **C** if A
does not resolve it.

---

## 8. Blast-radius summary

* Nothing here requires re-executing notebooks or re-capturing `outputs/`.
* Fix B: build-guard only, no output change.
* Fix A: `.venv-build` Quarto version only.
* Fix C: full PDF re-render (both editions) + wording review; HTML unaffected.
* The frozen set is edition-specific and input-sensitive — after any fix, re-run
  the §7 detection check (Fix B) on **both** pytorch and jax `.tex` to confirm
  zero frozen labels, not just the pytorch ones originally reported.
