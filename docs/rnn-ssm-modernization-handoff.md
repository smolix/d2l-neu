# Handoff: Implement the Chapter 9–10 Modernization

*Written 2026-07-11 on Alex's laptop for execution by a Fable agent on the 4×RTX 4090
Linux box. This document is the operational spec; the content spec lives in two
companion files that MUST be present and read first (in this order):*

1. `rnn-ssm-modernization-overview.md` — the approved design (diagnosis, chapter
   shapes, what's cut and why). Approved by Alex; do not re-litigate its decisions.
2. `rnn-ssm-modernization-outline.md` — the per-section content outline: learning
   goals, subsection structure, what every code cell computes, figures, exercises,
   labels, word budgets. **This is the source of truth for content.** Where this
   handoff and the outline conflict on content, the outline wins; where they
   conflict on repo mechanics, this handoff wins.

Also read before starting: repo `CLAUDE.md` (all of it, especially "Content
authoring", "Rules", "Source conventions"), `docs/build-system.md` §3 (outputs
store, capture/bless, freshness gate) and §6.6–6.8 (PDF tripwires, render
concurrency), `docs/notebook-scheduler.md` (skim).

---

## 1. Decisions already made by Alex (do not reopen)

- SSMs are taught **before** attention, at the end of ch. 10, as the continuation
  of the recurrence arc. Hybrid/architecture-comparison details are deferred to the
  attention/LLM chapters (later work, not yours).
- One or more LLM chapters will exist later; ch. 9–10 plant forward hooks to them
  but you do **not** create any LLM-chapter files.
- MT survives only as the compact example inside 10.2. Keep it compact.
- `rnn-scratch` + `rnn-concise` are merged into one section (9.5).
- MXNet is **omitted** in 10.3 and 10.4 (no scan primitive). Everywhere else all
  four frameworks are kept.
- `tiktoken` becomes a real shared dependency, used in 9.2 (see outline §9.2 as
  amended: text-as-bytes primer, pre-tokenization subsection, rank-table
  verification against tiktoken).

## 2. Hard rules (some of these live only in Alex's local memory — they do NOT
   appear in CLAUDE.md; treat them as if they did)

1. **No em-dashes in book prose.** Alex's standing request: never use "---" (or
   the em-dash character) in chapter text. Use commas, parentheses, colons, or
   reword. (Planning docs exempt; the book is not.)
2. **`make capture-outputs` blesses ALL frameworks from whatever sits in
   `_notebooks/`.** If you executed only one framework (or a run partially
   failed), an unscoped capture will WIPE the other frameworks' committed outputs
   for those files. Always pass `--frameworks`/use the documented scoping when you
   haven't just executed all frameworks for the captured files. Check
   `tools/capture_outputs.py --help` for the exact flag syntax before first use.
3. **Verify staging before every commit.** Never suppress `git add` stderr; run
   `git status` and confirm the staged set is exactly the session's intended edits.
   Never `git add -A` while subagents are mid-edit.
4. **Commit after each verified milestone** (phase gates below). Do not push
   unless Alex asks.
5. **Grep before renaming/deleting files**: renames must be reflected in
   `tools/runtime_env.py` (`MULTI_GPU_NOTEBOOKS`, `CPU_ONLY_NOTEBOOKS` — verified
   2026-07-11: none of the ch. 9–10 files appear in either, but re-verify),
   `tools/d2l_preprocess.py` `CHAPTER_NUMBERING`, `_quarto.yml`, and the outputs
   store. New `#@save` symbols that other chapters import must be added to the
   preface imports cell if the existing pattern requires it (check how current
   `#@save` utilities are exposed).
6. **Never edit `.qmd` files** (generated). Source `.md` only.
7. **Never run concurrent `quarto render`s**, and don't wrap `make html` in `-j`.
8. Use `make` targets for everything (logging goes to `logs/`). GNU make ≥ 4.3
   (`make` on the Linux box; it's only macOS that needs `gmake`).
9. New JAX code is written **NNX-style** (see §3).
10. Figure generators must be **byte-idempotent** (no timestamps, no unseeded
    randomness); follow the CLAUDE.md figure house-style checklist and the
    render-and-inspect loop. The project skills `mdl-figure` and
    `figure-style-audit` are checked into `.claude/skills/` and available to you.

## 3. Preflight (do all of this before editing anything)

```bash
git pull && git status                  # clean tree, note HEAD
./bootstrap.sh                          # if fresh clone; then venvs:
make venv-pytorch venv-tensorflow venv-jax venv-mxnet   # as needed
make detect                             # confirm 4 GPUs / slot plan
grep -rln "nnx" chapter_recurrent-neural-networks/ chapter_recurrent-modern/
```

**NNX gate (blocking):** as of 2026-07-11 the JAX→NNX refactor has NOT landed in
ch. 9–10 (`grep` above returns nothing; jax cells still use flax linen). A separate
agent is converting the whole book. **Do not start Phase 2+ until either (a) the
NNX refactor has landed on these chapters (grep finds `nnx` in them / git log shows
the conversion commit), or (b) Alex explicitly tells you to proceed anyway.** If
(b), write all JAX cells NNX-style yourself and say so in your report. Working
branch: create `rnn-ssm-modernization` off current main; rebase if the NNX work
lands on main mid-flight.

## 4. File-operations manifest

Both directory names are kept. Section slots after the change:

| File | Op | `CHAPTER_NUMBERING` entry | Label(s) it must carry |
|---|---|---|---|
| `chapter_recurrent-neural-networks/index.md` | rewrite | `[9]` | `chap_rnn` |
| `…/sequence.md` | rewrite | `[9, 1]` | `sec_sequence` |
| `…/text-sequence.md` | rewrite | `[9, 2]` | `sec_text-sequence` |
| `…/language-model.md` | rewrite | `[9, 3]` | `sec_language-model` |
| `…/rnn.md` | revise | `[9, 4]` | `sec_rnn` |
| `…/rnn-implementation.md` | **new** (merge of rnn-scratch.md + rnn-concise.md) | `[9, 5]` | `sec_rnn-scratch` (primary, for external refs) + `sec_rnn-concise` on the concise subsection |
| `…/bptt.md` | revise | `[9, 6]` | `sec_bptt` |
| `…/decoding.md` | **new** | `[9, 7]` | new `sec_decoding`; `sec_beam-search` on the beam-search subsection |
| `…/rnn-scratch.md`, `…/rnn-concise.md` | **delete** | remove entries | — |
| `chapter_recurrent-modern/index.md` | rewrite | `[10]` | `chap_modern_rnn` |
| `…/lstm.md` | rewrite (absorbs gru/deep-rnn/bi-rnn) | `[10, 1]` | `sec_lstm`; `sec_gru`, `sec_deep_rnn`, `sec_bi_rnn` on their subsections |
| `…/seq2seq.md` | rewrite (absorbs machine-translation-and-dataset + encoder-decoder) | `[10, 2]` | `sec_seq2seq`; `sec_machine_translation`, `sec_encoder-decoder` on their subsections |
| `…/ssm.md` | **new** | `[10, 3]` | new `sec_ssm` |
| `…/mamba.md` | **new** | `[10, 4]` | new `sec_mamba` |
| `…/gru.md`, `…/deep-rnn.md`, `…/bi-rnn.md`, `…/machine-translation-and-dataset.md`, `…/encoder-decoder.md`, `…/beam-search.md` | **delete** | remove entries | — |

Subsection-level `:label:`s attach fine to `##`/`###` headings; keeping the old
names on the new anchors means external `:numref:` references keep resolving
(they'll render as "Section 10.2.3"-style numbers, which is correct).

**Verified external crossref sites** (files outside ch. 9–10 that reference these
labels; re-run the grep yourself, but as of 2026-07-11):

- `chap_rnn`, `chap_modern_rnn` ← `chapter_preface/index.md`, `chapter_natural-language-processing-applications/index.md`
- `sec_language-model` ← `chapter_attention-mechanisms-and-transformers/large-pretraining-transformers.md`, `chapter_mdl-information-theory/mdl-information-theory.md`, `chapter_natural-language-processing-pretraining/{bert.md,index.md}`
- `sec_rnn-scratch` ← `chapter_attention-mechanisms-and-transformers/large-pretraining-transformers.md`, `chapter_natural-language-processing-pretraining/word2vec.md`, `chapter_optimization/convexity.md`
- `sec_lstm` ← `chapter_attention-mechanisms-and-transformers/index.md`, `chapter_mdl-calculus/mdl-multivariable-calculus.md`, `chapter_multilayer-perceptrons/mlp.md`
- `sec_machine_translation` ← `chapter_attention-mechanisms-and-transformers/attention-scoring-functions.md`, `chapter_natural-language-processing-applications/sentiment-analysis-and-dataset.md`, `chapter_natural-language-processing-pretraining/index.md`
- `sec_seq2seq` ← `chapter_attention-mechanisms-and-transformers/{queries-keys-values.md,bahdanau-attention.md}`, `chapter_mdl-probability-statistics/mdl-statistics.md`, `chapter_natural-language-processing-pretraining/word2vec-pretraining.md`
- No external references found for: `sec_sequence`, `sec_text-sequence`, `sec_rnn`,
  `sec_rnn-concise`, `sec_bptt`, `sec_gru`, `sec_deep_rnn`, `sec_bi_rnn`,
  `sec_encoder-decoder`, `sec_beam-search`.

With the label-preservation scheme above, **no external file needs editing for
crossrefs**. Still, after Phase 4 run a full sweep:
`grep -rn ":numref:\`sec_\|:ref:\`sec_" chapter_* | grep -f <(labels you removed)`
and `make html` will hard-fail on dangling `:numref:`s anyway. Also grep for
literal *prose* references to removed sections ("as described in the beam search
section" etc.) in the attention and NLP chapters and fix the handful you find.
**Do not otherwise edit chapters outside 9–10**; the attention-chapter rewrite is
a separate later project.

## 5. Config and infrastructure edits

1. **`_quarto.yml`** (currently lines ~127–143): replace the two chapter file
   lists with the new `.qmd` names in order
   (index, sequence, text-sequence, language-model, rnn, rnn-implementation,
   bptt, decoding | index, lstm, seq2seq, ssm, mamba).
2. **`tools/d2l_preprocess.py` `CHAPTER_NUMBERING`** (~line 978): update per the
   manifest table. Files absent from the dict render unnumbered, so every new
   file must be added.
3. **`pyproject.toml`**: add `"tiktoken"` to the main `[project] dependencies`
   list (next to `requests`). Re-lock/sync per the repo's uv flow (`make
   venv-<fw>` targets re-sync). Verify `import tiktoken` works in all four
   framework venvs. Encoding files download+cache on first use at notebook
   execution time (network is available; consistent with DATA_HUB). If the
   scheduler environment blocks network, pre-seed `TIKTOKEN_CACHE_DIR`.
4. **`d2l.bib`** (repo root): add the ~20 entries listed in outline §0
   ("Citations added"). Follow existing key style. Add them all up front so
   `:cite:` tags never dangle.
5. **Figure generators**: create `tools/gen_mdl_rnn_figures.py` (ch. 9) and
   `tools/gen_mdl_modernrnn_figures.py` (ch. 10), importing the shared style from
   `tools/gen_mdl_figures.py`. The `make figures` target auto-discovers
   `tools/gen_mdl_*_figures.py`, so no Makefile edit is needed. Output naming:
   `img/mdl-rnn-<id>.svg`, `img/mdl-modernrnn-<id>.svg`. The ~10 carryover SVGs
   being restyled (unfolded-rnn, lstm/gru cells, deep/bi-rnn, encoder-decoder,
   seq2seq, beam tree, partitioning diagram) are *reproduced* in the generators
   under the new names and the old hand-drawn SVGs left untouched in `img/` (other
   chapters may share them; grep `img/<name>.svg` across chapters before assuming
   an old SVG is orphaned — delete only true orphans in a final cleanup commit).
6. **New `d2l` library components** (`#@save` in their host sections; `make lib`
   rebuilds `d2l/*.py`): BPE tokenizer class (9.2), decoding helpers (9.7),
   log-depth associative scan for pytorch (10.3), chrF scorer (10.2). Keep APIs
   small; they're used cross-section (9.2's tokenizer in 9.3/9.5/10.2/10.4; 9.7's
   decoders in 10.2/10.4). Remember rule 2.5: `make lib` changes can affect other
   chapters' notebooks — keep these symbols NEW (no collisions with existing
   `d2l` names; grep `d2l/` for name clashes before choosing names).

## 6. Authoring mechanics per section (applies to every rewritten/new file)

- Follow the outline section-by-section: structure (3–5 `##` sections with `###`
  subsections), learning goals realized in prose, every listed code cell present
  and *computing what the outline says*, exercises replaced/pruned as listed.
- One imports cell per framework near the top; `%%tab <fw>` / `:begin_tab:`
  conventions as in existing sections. Untagged Python cells = all frameworks.
  For 10.3/10.4, tag every cell explicitly with `%%tab pytorch` /
  `%%tab tensorflow` / `%%tab jax` (no mxnet). Precedent for a no-mxnet section:
  `chapter_attention-mechanisms-and-transformers/vision-transformer.md`. After
  regenerating notebooks, confirm no
  `_notebooks/mxnet/chapter_recurrent-modern/{ssm,mamba}.ipynb` is produced (or
  that the pipeline tolerates its absence) before running mxnet execution.
- Run `tools/add_cell_ids.py` after content lands (idempotent; never edit
  existing IDs). Then rebuild each file's `<!-- slides -->` block against the new
  cell IDs: every section needs a slide deck (model the density on the current
  ch. 9–10 slide blocks before you delete them — extract them first for
  reference). Slide render check: `make -B slides-pytorch
  SLIDES_FILTER=chapter_recurrent-neural-networks/decoding.md` etc.
- `tools/lint_source.py <file>` clean before each commit (it's also pre-commit).
- Math/PDF tripwires (CLAUDE.md "Gotchas"): `$$` fencing for labeled display
  math, no `]` in figure captions, no digit immediately after a closing `$`,
  `\left(\begin{smallmatrix}…` for parenthesized small matrices.
- Citations: `:cite:` keys must exist in `d2l.bib` (added in Phase 1).
- Prose style: match the book's voice; intuition-first; no em-dashes (rule 2.1);
  forward references by `:numref:` to labels, never by hardcoded numbers.

## 7. Compute budgets and datasets (4×4090 box)

- Keep every notebook's end-to-end runtime in line with the existing corpus
  (minutes, not tens of minutes; the scheduler runs ~130 notebooks/framework).
  Guidance: RNN/LSTM/GRU LMs at the current sections' scale; seq2seq at the
  current section's scale; 10.3 sMNIST S4D ≤ ~10 epochs with a model small
  enough to finish in a few minutes; 10.4 Mamba LM 2–4 blocks, d_model 128–256,
  Time Machine, a few minutes.
- Datasets: Time Machine (exists), fra-eng (exists in DATA_HUB), sequential
  image classification in 10.3: default to the existing FashionMNIST pipeline
  read as 784-step pixel sequences ("sequential FashionMNIST") to avoid new data
  infrastructure; if the LSTM-vs-S4D contrast is unconvincing there, fall back
  to true MNIST via a DATA_HUB addition and note it in your report. No
  TinyStories download (exercise pointer only).
- The GPU box executes everything including mxnet (wheel pinned per CLAUDE.md;
  mxnet capped at `MXNET_GPU_SLOTS=2` by the Makefile — don't override).

## 8. Phased execution plan (commit at each gate; suggested messages inline)

**Phase 0 — preflight** (§3). Branch created, NNX gate resolved, `make detect`
sane, baseline `make html` passes on the untouched tree (catches pre-existing
breakage so you don't own it later).

**Phase 1 — shared infrastructure.**
Bib entries; pyproject + tiktoken into all venvs; figure-generator skeletons
(style import + `save()` wiring, no figures yet); decide and stub the four
`#@save` APIs (signatures + docstrings in a scratch note, not yet in sections).
Gate: `make lib` clean, venvs import tiktoken. Commit: "ch9-10 modernization:
infra (bib, tiktoken dep, figure generator skeletons)".

**Phase 2 — chapter 9, in dependency order: 9.2 → 9.3 → 9.5 → 9.4 → 9.6 → 9.1 →
9.7 → index.** (9.2 first because its tokenizer/`Vocab` API is imported
everywhere; 9.4 is prose-heavy and can go anytime; 9.7 last because it samples
from 9.5's model class.) For each section: write content per outline → cell IDs
→ figures for that section rendered and *visually inspected* (rsvg-convert to
PNG, look at it; montage grids for batches) → per-framework smoke execution of
just that notebook (`make -B _notebooks/<fw>/chapter_recurrent-neural-networks/<f>.executed`)
→ lint. Delete `rnn-scratch.md`/`rnn-concise.md` when 9.5 lands, updating
`_quarto.yml` + `CHAPTER_NUMBERING` in the same commit so the tree always
renders. Gate: all ch. 9 notebooks execute in all 4 frameworks; `make html`
renders the chapter with no dangling refs. Commit per section or per 2–3
sections: "ch9: rewrite <section> (…)".

**Phase 3 — chapter 10: 10.1 → 10.2 → 10.3 → 10.4 → index.** Same per-section
loop. Absorb-and-delete in the same commit as the absorbing section (10.1 lands
together with deletion of gru/deep-rnn/bi-rnn.md; 10.2 with
machine-translation-and-dataset/encoder-decoder.md; beam-search.md was already
deleted in Phase 2 with 9.7). For 10.3/10.4 execute pytorch+jax+tf only; confirm
the mxnet pipeline skips them cleanly. Gate: chapter renders; the three-model
capstone table in 10.4 produces sane numbers (Mamba ≤ LSTM ppl at comparable
params; if not, tune, and if it still isn't, report honestly rather than
cherry-pick). Commit per section.

**Phase 4 — sweep and reconcile.** Full-corpus crossref grep (§4); prose
references to deleted sections; orphaned-SVG cleanup; slides for every touched
file build (`make -j4 slides` is parallel-safe); `make figures` twice → second
run byte-identical (idempotency check); run the `figure-style-audit` skill on
both chapters. Gate: `make html` for the whole book, zero warnings attributable
to ch. 9–10. Commit: "ch9-10: crossref sweep, slides, figure audit".

**Phase 5 — full execution + capture + verify.**
```bash
make notebooks-pytorch notebooks-tensorflow notebooks-jax notebooks-mxnet
make run-all-notebooks          # scheduler owns concurrency; no outer -j
# after a green run (check logs/ for failures):
make capture-outputs FILES="chapter_recurrent-neural-networks/sequence.md … chapter_recurrent-modern/mamba.md"   # list every touched file
```
Capture caveats: (a) run capture only after ALL frameworks executed green for the
listed files, otherwise scope `--frameworks` (rule 2.2); (b) prune the outputs
store of deleted/renamed files (`outputs/<fw>/chapter_*/…` JSON + LFS image dirs
for rnn-scratch, rnn-concise, gru, deep-rnn, bi-rnn, machine-translation-and-dataset,
encoder-decoder, beam-search) — check `tools/audit_outputs.py --help` for a prune
mode before deleting by hand. Then:
```bash
.venv-build/bin/python tools/audit_outputs.py --verify-fresh   # or via make html
make html && make pdfs && make slides
```
Gate: 0 notebook failures, freshness gate clean, 4 PDFs build, book renders.
Commit: "ch9-10: execute all frameworks, bless outputs store".

**Phase 6 — review and report.** Self-review the diff at high effort (the
`code-review` skill at level high, or an equivalent adversarial pass): correctness
of every equation against the cited papers (ZOH formulas, gate equations, chrF
definition, HiPPO matrix), tab parity (does each framework's cell really compute
the same thing), exercise answerability. Fix, commit. Then write
`rnn-ssm-modernization-report.md`: per-section status, deviations from the
outline (with one-line justifications), measured runtimes, capstone-table
numbers, anything deferred, and open questions for Alex. Do not push; leave the
branch local and state its name and HEAD in the report.

## 9. Sub-agent dispatch guidance

Parallelize *within* phases, not across them. Safe parallel units: bib entries //
figure generators // pyproject (Phase 1); disjoint section files within a chapter
once the shared `#@save` APIs are frozen (Phases 2–3) — but sections that import
another section's `#@save` symbols need that section's code merged and `make lib`
run first, and only one agent may touch `_quarto.yml`/`CHAPTER_NUMBERING`/`d2l.bib`
at a time (do those edits yourself, serially). Never let two agents edit the same
file; never `git add -A` while any agent is mid-edit. Notebook *execution* is
never parallelized by you: the scheduler owns it.

## 10. Done criteria (all must hold)

- [ ] 13 section files match the outline's structure and cell inventory; 8 old
      files deleted; `_quarto.yml`, `CHAPTER_NUMBERING` consistent.
- [ ] All notebooks execute green: 4 frameworks for 9.1–10.2, 3 for 10.3–10.4.
- [ ] Outputs store captured for all touched files, pruned of deleted ones;
      `--verify-fresh` clean on the GPU box.
- [ ] `make html`, `make pdfs`, `make slides` all green; no dangling
      `:numref:`/`:cite:`; PDFs open and ch. 9–10 pages look right (spot-check).
- [ ] Figures: generated, byte-idempotent, house-style checklist passed,
      `figure-style-audit` run on both chapters; every figure visually inspected.
- [ ] No em-dashes in new prose (`grep -n "—\|---" <files>`, excluding YAML/code).
- [ ] Branch `rnn-ssm-modernization` with clean incremental commits; nothing
      pushed; report file written.
