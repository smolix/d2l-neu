# Implementation brief: ch. 12 "State Space Models" rebuild

*2026-07-19. The approved spec is
`reviews/ssm-chapter-review-and-proposal-2026-07-19.md` (read it in full
before writing anything; §5 holds the per-section specs, §6 the
cross-cutting ones, §8 Alex's decisions). This brief adds the build
contract: sequencing, notation, labels, #@save interfaces, validation
gates, and ops rules. It is binding for every author agent.*

## 0. Approved decisions (Alex, 2026-07-19)

Seven sections; seq2seq moves to the Language Models part ("seriously out
of date anyway and mostly for historical reasons" — reframe it as the
historical baseline, do not modernize it beyond the reframe); title stays
"State Space Models"; bi-RNN compressed significantly; scope approved.
Build ops: **max 2 concurrent agents; pause at 85% session usage** (agents
get a checkpoint message, wake after reset).

## 1. Hard sequencing rules

1. **No repo edits while the render/deploy pipeline is running.** The full
   notebook re-run + render + R2 + outputs-commit pipeline (Opus agent) must
   complete and PUSH before any chapter file, `CHAPTER_NUMBERING`,
   `_quarto.yml`, or `d2l/` edit lands. Pilots run in the session scratchpad
   only until then.
2. **Pilot before prose** for every experiment marked unpiloted in proposal
   §7. The 12.7 three-model sweep is a hard gate with a specified fallback.
   Existing pilots live in `scratchpad/ssm-proto/` and transfer.
3. **GPU discipline:** while the pipeline's *execution* phase runs, pilots
   are CPU-only (`CUDA_VISIBLE_DEVICES=""`); once it enters render phases,
   short GPU pilots are allowed.
4. Wave order (2 agents each, Fable for authoring):
   - **Wave 0 (main loop, not an agent):** seq2seq move (§4 below) +
     numbering/quarto edits, committed as one mechanical commit.
   - **Wave 1:** Agent A = 12.2+12.3 revisions (coupled: `#@save S4D`,
     dedup, `step()` APIs). Agent B = 12.4 `matrix-state.md`.
   - **Wave 2:** Agent C = 12.5 `deltanet.md`. Agent D = 12.6
     `test-time-regression.md`.
   - **Wave 3:** Agent E = 12.7 `hybrids.md`. Agent F = 12.1 lstm slim +
     `index.md` rewrite (front discussion; Fable).
   - **Review phase:** two Opus chapter-review passes (attention-style),
     Fable fix pass, full recapture, renders.
5. Each wave's agents announce `make lib` rebuilds to each other via their
   reports; a lib rebuild invalidates the other agent's venv imports
   mid-run (rerun affected cells).

## 2. Notation bible (binding, chapter-wide)

- State `S ∈ R^{d_k×d_v}`, keys index rows. Update
  `S_t = D_t S_{t-1} + k_t v_tᵀ`; read `o_t = S_tᵀ q_t`.
- DeltaNet: `S_t = (I − β_t k_t k_tᵀ) S_{t-1} + β_t k_t v_tᵀ`
  (equivalently `S_t = S_{t-1} − β_t k_t (k_tᵀ S_{t-1} − v_tᵀ)`).
  Derivation identity: `∇_S ½‖Sᵀk − v‖² = k(Sᵀk − v)ᵀ`.
- One margin note (12.4) on the transpose conventions in the literature;
  one paragraph (12.4) on ch. 10's `(S, z)` normalizer pair and where z
  went; unit-norm-keys assumption stated wherever the retrieval-error
  expansion is used.
- Keys L2-normalized in all delta-rule code (stability + matches shipped
  DeltaNet); `β = sigmoid(·)`, doubled to (0,2) only in the
  `allow_neg_eigval` discussion.

## 3. Files, labels, cell-ID prefixes

| File | Label | Cell-ID prefix |
|---|---|---|
| `lstm.md` (revise) | `sec_lstm` (keep) | existing `lstm-*` |
| `ssm.md` (revise) | `sec_ssm` (keep; new subsec `subsec_ssm-step`) | existing `ssm-*` |
| `mamba.md` (revise) | `sec_mamba` (keep) | existing `mamba-*` |
| `matrix-state.md` (new) | `sec_matrix-state` | `matrix-state-*` |
| `deltanet.md` (new) | `sec_deltanet` | `deltanet-*` |
| `test-time-regression.md` (new) | `sec_test-time-regression` | `test-time-regression-*` |
| `hybrids.md` (new) | `sec_hybrids` | `hybrids-*` |

ToC order in `index.md`/`_quarto.yml`: lstm, ssm, mamba, matrix-state,
deltanet, test-time-regression, hybrids. `chap_modern_rnn` stays the
chapter label (inbound refs depend on it). Existing labels `sec_ssm`,
`sec_mamba`, `subsec_parallel-scans` keep their meaning (ch. 10 cites
them). Never renumber by hand — `CHAPTER_NUMBERING` drives everything.
Cell IDs via `tools/add_cell_ids.py`; IDs never change once written.

## 4. seq2seq move (Wave 0, mechanical)

- `git mv chapter_recurrent-modern/seq2seq.md
  chapter_natural-language-processing-pretraining/seq2seq.md`; it becomes
  the FIRST section of ch. 17 (ahead of word2vec — ch. 17 already
  back-references it; landing it later would invert those references).
- `CHAPTER_NUMBERING`: remove from recurrent-modern block; insert as
  [17, 1] with the rest of ch. 17 shifting; `_quarto.yml` to match (dict
  order == quarto order, PDF invariant).
- `git mv outputs/<fw>/chapter_recurrent-modern/seq2seq
  outputs/<fw>/chapter_natural-language-processing-pretraining/seq2seq`
  for all four frameworks (keeps its 4 tabs for now; the Language Models
  part is not under the PT+JAX policy yet).
- Reframe: ONE new opening paragraph (historical baseline framing per
  Alex) + flip `:1102-1108` ("the subject of the next part" → attention as
  the escape already taken, backward refs to `chap_attention`); delete the
  dead `save_attention_weights` hook + its advertisements; drop the stale
  `bleu` justification sentence; keep all `#@save`s (incl.
  `show_list_len_pair_hist`). Do NOT otherwise modernize.
- Reference-check afterward (all six files listed in proposal §7) +
  `attention-scoring.md:524` (its forward ref now points to ch. 17 — text
  "we build such encoder-decoder models in full in `sec_seq2seq`" stays
  valid). Fix `attention-at-scale.md:1196`'s hardcoded "ch. 13" slide line.
- Slides: seq2seq's deck moves with the file (kicker rewrites are
  automatic from `CHAPTER_NUMBERING`).

## 5. #@save contracts (additions)

| Symbol | Defined in | Consumers |
|---|---|---|
| `S4D`, `S4DBlock` | ssm.md | mamba.md (dedup), hybrids.md |
| `S4D.step` (or `step()` on the block) | ssm.md | ssm.md inference subsec |
| `MambaBlock`, `Mamba`, and their `step()` | mamba.md | hybrids.md |
| (nothing from 12.4/12.5/12.6 — notebook-local) | | 12.7 rebuilds its tiny DeltaNet/GLA inline or imports `d2l.TransformerBlock` (ch. 11) for the attention stack |

12.7 uses `d2l.TransformerBlock` (pytorch no-arg factory / JAX rngs
factory contracts as established in ch. 11) for its pure-attention model
and the hybrid's attention layer. PT factories no-arg; JAX takes rngs;
JAX attention returns `(output, weights)` — respect the ch. 11 interface
conventions.

## 6. Per-section requirements (delta to proposal §5 — read that first)

Every section: PT+JAX tabs only; one imports cell; `<!-- slides -->` deck
re-authored (@cell-id references; new figures ALSO copied to `img/auto/`);
exercises per proposal §5's sketch; bib entries added for the section's
citations (IDs in proposal §6 — note MOHAWK = 2408.10189); every quoted
number regenerated at run time and rounded to run-to-run stability; run on
CPU where a GPU adds nothing (12.6 entirely; 12.5's mechanistic cells).
Figures via `tools/gen_mdl_ssm_*.py` generators importing
`tools/gen_mdl_figures.py`, byte-idempotent, house-style checklist,
render-and-inspect loop mandatory. Read `docs/writing-avoid.md` BEFORE
writing prose; the chapter must read like ch. 10/11 (study
`chapter_attention/attention-at-scale.md` and
`chapter_transformers/kv-cache.md` as the register models).

Stop-and-flag rule: if a pilot or validation contradicts a punchline in
the proposal (e.g. the 12.7 dissociation doesn't reproduce), STOP and
report — never weaken a cell to pass, never comment out core
functionality, never quietly change the claim.

## 7. Validation gates (per section, before capture)

1. `tools/add_cell_ids.py` run; `tools/lint_source.py` clean.
2. Cell-by-cell execution in `.venv-pytorch` AND `.venv-jax` (validate
   harness or `tools/run_one_notebook.py`), all green — "scheduler done: N
   ok, 0 failed" before ANY `make capture-outputs` (capture blesses failed
   notebooks' partial manifests).
3. Numerical checks in-notebook as asserts (chunked==loop, stepped==scan,
   dual==recurrence) — they are teaching content AND regression guards.
4. Figures regenerated twice → byte-identical; PNG-inspected.
5. `make -B slides-<fw> SLIDES_FILTER=chapter_recurrent-modern/<file>.md`
   renders clean.
6. After each wave: `make lib` + rerun of any section whose imports
   changed.

## 8. Ops: tokens and pausing

Max 2 concurrent agents. At 85% session usage: SendMessage both agents to
checkpoint (finish current cell, validate, report resumable state), launch
nothing new, ScheduleWakeup ~5 min past reset, resume from TaskList +
state reports. Prefer launching an agent only when there is budget for it
to plausibly finish its section.
