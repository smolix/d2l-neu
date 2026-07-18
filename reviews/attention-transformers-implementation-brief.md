# Implementation Brief: Attention + Transformers Rewrite

*Authoring conventions and contracts for the two-chapter rewrite. Read together
with `reviews/attention-transformers-review-and-proposal-2026-07-17.md` (the
content spec — per-notebook outlines live there, §2). Every authoring agent
reads BOTH documents, plus `CLAUDE.md` and `docs/writing-avoid.md`, before
writing a line.*

## 1. File map

```
chapter_attention/                       chapter_transformers/
  index.md            (A0)                index.md             (B0)
  queries-keys-values.md        (A1)      transformer-block.md (B1)
  attention-scoring.md          (A2)      gpt.md               (B2)
  multihead-attention.md        (A3)      kv-cache.md          (B3)
  positional-information.md     (A4)      encoders-decoders.md (B4)
  attention-at-scale.md         (A5)      vision-transformer.md(B5)
  what-attention-computes.md    (A6)      moe.md               (B6)
                                          scaling-laws.md      (B7)
```

Neither directory is in `_quarto.yml`/`CHAPTER_NUMBERING` yet — the atomic
switchover happens after authoring (task: integration). Until then the book
still renders the old chapter.

## 2. Hard rules (violations get rejected in review)

1. **Never edit `.qmd`** files; source `.md` only. **Never run `make lib`**
   until switchover — the old chapter still defines the same `#@save` names
   and `build_lib.py` resolves collisions by silent last-writer-wins.
2. **PyTorch + JAX tabs only** (`%%tab pytorch` / `%%tab jax`; untagged =
   all frameworks — do NOT leave training cells untagged). JAX = Flax **NNX**
   (`nnx.Module`, `nnx.Rngs`), optax for optimizers. Model files for idiom:
   `chapter_optimization/muon.md` (LM training, PT+JAX, slides),
   `chapter_attention-mechanisms-and-transformers/vision-transformer.md`
   (NNX blocks), `chapter_recurrent-modern/ssm.md` (timing cells).
3. **One imports cell** per framework near the top; no re-imports later.
   Prefer `d2l.` helpers (`d2l.plot`, `d2l.set_figsize`, `d2l.Vocab`,
   `d2l.TimeMachine`, `d2l.show_heatmaps`, …).
4. **3–5 top-level `##` sections** per file, with `###` subsections; end with
   `## Summary` (or `## Summary and Discussion`) then `## Exercises`.
5. **Code teaches; it does not draw.** Illustrative/schematic figures are
   pre-generated SVGs from a committed generator (see §5). A short
   `d2l.plot(...)` of a *computed* result is fine.
6. **Labels/refs**: every `##`-file gets `:label:` right under the H1;
   `:numref:`/`:ref:` only — never hardcoded chapter/section numbers.
   Required label strings (external refs depend on them) in §6.
7. **Results-precision policy** (CLAUDE.md): single seeded runs; quote only
   noise-stable precision; ranges and qualitative statements; never build a
   conclusion on a sub-noise difference. Fragile experiments (§7) get piloted
   BEFORE prose is written, with the claim templates below.
8. **PDF tripwires**: `$$` alone on its own lines with `:eqlabel:` on the
   next line; no `]` in image captions; no digit immediately after a closing
   `$`; `\left(\begin{smallmatrix}…` for parenthesized small matrices.
9. Writing style per `docs/writing-avoid.md`; proofs intuition-first; when a
   proof already lives in the math appendix, cross-reference it, don't
   duplicate (e.g. convexity/concentration material in `chapter_mdl-*`).
10. Compute guardrails: every notebook runs on **one GPU, ≤7.5 GB, target
    ≤~10 min per framework** (B7 may reach lr-scheduler scale, ~12 min).
    JAX runs under `XLA_PYTHON_CLIENT_MEM_FRACTION=.40` in the scheduler —
    size models accordingly.

## 3. Authoring workflow per notebook

1. Read the proposal's outline for your notebook (§2 of the proposal) — it is
   the spec: section list, experiments, and what NOT to include.
2. Pilot any flagged experiment first (§7); fix the numbers, then write prose.
3. Write the `.md` (both tabs). Validate every cell by executing the file's
   python blocks **in order** in each venv (`.venv-pytorch/bin/python`,
   `.venv-jax/bin/python`). For cross-file `d2l.X` symbols defined in the NEW
   chapters (not yet in the lib), inject them: run the defining file's blocks
   first in the same process, then `setattr(d2l, 'X', X)`. Never weaken a
   cell to make it pass.
4. Figures: edit/create the chapter generator (§5), regenerate, **render to
   PNG and visually inspect** (rsvg-convert loop per CLAUDE.md), iterate.
5. Run `python3 tools/add_cell_ids.py chapter_<x>/<file>.md` (idempotent),
   then write the `<!-- slides -->` section using the assigned `#<id>`s
   (format: copy the shape of `chapter_optimization/muon.md`'s slides block —
   `::: {.slide}` divs, `@<id>` / `@<id>@<fw>` placeholders, kicker line is
   auto-rewritten). Copy any figure referenced from slides into `img/auto/`
   (otherwise it is silently dropped).
6. Run `python3 tools/lint_source.py chapter_<x>/<file>.md` and fix findings.
7. Report: what you built, pilot numbers, validation output tails (both
   venvs), open issues. Do not git-commit.

## 4. Library contract (`#@save` registry)

Exactly these symbols are `#@save`d in the new chapters — no others without
flagging it:

| Symbol | Home | Notes |
|---|---|---|
| `masked_softmax` | A2 | dtype-safe (−inf via `finfo.min`/`masked_fill`) |
| `DotProductAttention` | A2 | scaled; takes `valid_lens`; dropout on weights |
| `MultiHeadAttention` | A3 | constructor `(num_hiddens, num_heads, dropout, bias=False)` — keep signature compatible with the frozen legacy encoder block's needs |
| `TinyCharLM` (working name) | A4 | attention-only char-LM: embed + stacked attention (+RoPE flag) + tied head; reused by A6 |
| `TransformerBlock` | B1 | configurable: `norm∈{'layer','rms'}`, `act∈{'gelu','swiglu'}`, `ffn_factory=None`, `attn_factory=None` (default builds `d2l.MultiHeadAttention`; B3 swaps in a GQA attention), pre/post flag |
| `GPT` (or `GPTLM`) | B2 | takes block config + `pos∈{'learned','rope'}`; pluggable FFN factory (B6) and attn factory (B3); RoPE implemented internally (self-contained class, mirroring `TinyCharLM._rope` — do not import A4's teaching `rope()`) |
| `PatchEmbedding`, `ViTBlock`, `ViT` | B5 | carried over from the old file, PT+JAX `#@save` (old file saved them only on the TF tab) |

NOT defined in the new chapters: `show_heatmaps` (use `d2l.show_heatmaps`;
its `#@save` moves to ch. 3 at switchover), `AdditiveAttention` (A2 shows the
scoring function inline in the history vignette; no library class),
`PositionalEncoding` (A4 implements sinusoidal/RoPE inline or as local
classes; the legacy class survives only in the quarantine file),
`TransformerEncoderBlock`/`TransformerEncoder`/`AttentionDecoder`/
`PositionWiseFFN`/`AddNorm` (legacy; quarantine or gone). Within the defining
file use bare names; across files use `d2l.X`.

## 5. Figures

- Generators: `tools/gen_mdl_attention_figures.py` and
  `tools/gen_mdl_transformers_figures.py`, importing the shared style from
  `tools/gen_mdl_figures.py`. Output `img/mdl-attention-<id>.svg` /
  `img/mdl-transformers-<id>.svg`. Byte-idempotent (no timestamps, no
  unseeded random). House checklist in CLAUDE.md (black axes/labels, 13–16 pt
  text, no text↔line intersections, no dead whitespace, balanced pairs).
- Legacy assets: the old chapter's hand-drawn SVGs (`qkv.svg`,
  `multi-head-attention.svg`, `transformer.svg`, `vit.svg`,
  `cnn-rnn-self-attention.svg`, attention-pattern trio from
  large-pretraining) may be **reused as-is** where they fit; anything redrawn
  or new goes through the generator. Paper-screenshot PNGs are banned.
  The A2 Bahdanau alignment vignette figure is a house-generated schematic.

## 6. Labels that MUST exist (external references)

`chapter_attention/index.md` → `:label:chap_attention` (new).
`chapter_transformers/index.md` → `:label:chap_transformers` (already on the
stub — keep the string). Section labels: `sec_queries-keys-values` +
`sec_attention-pooling` (both in A1 — pooling label goes on the kernel
subsection), `sec_attention-scoring-functions` + `subsec_batch_dot` (A2),
`sec_vision-transformer` (B5), `sec_transformer` + 
`sec_large-pretraining-transformers` (B4 — `sec_transformer` on the file,
the other on the taxonomy section). All other labels are free — use fresh
descriptive names (`sec_multihead-attention`, `sec_positional-information`,
`sec_attention-at-scale`, `sec_what-attention-computes`,
`sec_transformer-block`, `sec_gpt`, `sec_kv-cache`, `sec_encoders-decoders`,
`sec_moe`, `sec_scaling-laws` suggested). The old
`chap_attention-and-transformers` label is retired at switchover (12 external
refs get re-pointed one by one) — do not use it in new files. Cross-refs
between the two new chapters and to ch. 8 (`sec_text-sequence` BPE,
`sec_decoding`), ch. 9 (optimizers/schedules/muP), ch. 13 SSM
(`chap_...recurrent-modern` labels — check exact strings with grep) are
encouraged; verify every label you reference exists.

## 7. Pilots and claim templates (fragile experiments)

Run the pilot standalone in the venv, decide the claim from the numbers, THEN
write prose. Record pilot scripts + outputs in `scratchpad/` and quote the
results in your report.

- **A4 extrapolation** (train ctx 128 → eval 512, real text via
  `d2l.TimeMachine`): claim template "trained at context L, evaluated at 4L,
  the absolute-position model degrades badly; RoPE degrades more gracefully /
  ALiBi stays closest to flat" — keep only the parts the pilot supports;
  synthetic copy tasks are forbidden here (they invert the ordering).
- **A6 induction bump**: existence-not-location. "During training the loss on
  repeated sequences drops abruptly; a head emerges whose attention locks onto
  the token after the previous occurrence" — never quote the epoch.
- **B3 JAX timing**: fixed-size cache + index updates; warm up before timing;
  quote rough ratios ("several times faster"), never exact multipliers.
- **B7 scaling bend**: 4–5 sizes; claim "roughly a straight line over the
  small sizes, bending where the corpus saturates the larger models"; no
  fitted exponent decimals. Corpus: pilot decides between `d2l.TimeMachine`
  (may saturate too early) and a larger DATA_HUB text (if needed, flag the
  choice — do not silently add a new dataset dependency).
- **B2 GPT-2 load**: weights from the HF stable URL
  (`https://huggingface.co/openai-community/gpt2/resolve/main/model.safetensors`)
  registered in `d2l.DATA_HUB` with a pinned sha1 — verify the download and
  record the hash; mind the Conv1D transpose; `tiktoken` for the tokenizer;
  JAX side loads safetensors→numpy (no torch import in the JAX venv).

## 8. Chapter front discussions (A0/B0)

Written by the main session (not agents), after the section files stabilize —
substantial intros in the mold of `chapter_optimization/index.md` and other
recent chapters: what the chapter is about, why now, the arc through the
sections, honest history, `toc` block last.
