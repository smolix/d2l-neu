# Ch11 (Transformers) / Ch12 (State Space Models) — JAX-vs-PyTorch Coverage Parity Scan

Date: 2026-07-21. Scope: **coverage parity** between the PyTorch and JAX tabs only
(these Advanced-part chapters carry no tensorflow/mxnet). Read-only review — no
files edited, nothing committed.

Files reviewed:
- **Ch11 Transformers:** `chapter_transformers/{index,transformer-block,gpt,kv-cache,encoders-decoders,vision-transformer,moe,scaling-laws}.md`
- **Ch12 State Space Models:** `chapter_recurrent-modern/{index,lstm,ssm,mamba,matrix-state,deltanet,test-time-regression,hybrids}.md`

Method: a small parser (`/tmp/.../scratchpad/cov1112/extract_matrix2.py`) scans every
```` ```{.python .input #<id>} ```` fence, reads the `%%tab` line that follows, and
groups blocks by `#<id>`. It handles all three tab forms actually used in these
chapters (confirmed by exhaustive grep — no others exist): `%%tab pytorch`,
`%%tab jax`, and `%%tab pytorch, jax` (one shared block covering both). Per-file
cell counts were cross-checked against raw `grep -c '^```{\.python \.input'` counts
and match exactly (319 physical code blocks across the 14 content files) — the
matrix below is not an artifact of a parsing bug.

## Bottom line

**Zero genuine coverage gaps (type b) in either chapter.** Every one of the 180
distinct labeled code cells across both chapters has full PyTorch+JAX coverage —
either as a matched pair of framework-specific blocks or as one framework-neutral
shared block. There are no pytorch-only cells, no jax-only cells, and no stray
`%%tab tensorflow`/`%%tab mxnet` anywhere. All prose-level asymmetries
(`:begin_tab:` asides) are design-driven and well-justified. One soft, non-blocking
item is flagged below for human judgment (an exercise-depth asymmetry in `moe.md`,
not a hard gap). The bonus code-quality sweep (commented-out training loops,
stubs, TODOs) is clean.

| Category | Count |
|---|---|
| Genuine coverage gaps (type b) | **0** |
| Flagged for human judgment (soft, not a hard gap) | 1 (`moe.md` Ex. 5, see below) |
| Design-driven asymmetries reviewed & confirmed clean (type a) | 24 locations (listed below) |
| Tangential non-parity observation | 1 (Discussions-link inconsistency, affects both frameworks equally) |
| Bonus code-quality flags (commented-out core functionality, stubs, TODO) | **0** |
| Stray `%%tab tensorflow` / `%%tab mxnet` | **0** |

---

## 1. Per-file cell/tab matrix

"Paired" = two separate blocks under the same id, one `%%tab pytorch` + one
`%%tab jax`. "Shared" = one block tagged `%%tab pytorch, jax` (identical code for
both; some use an internal `tab.selected('pytorch'|'jax')` branch for a few
framework-specific lines inside an otherwise shared cell — see §1.3).

### Ch11 Transformers

| File | Distinct ids | Paired | Shared | Pytorch-only | Jax-only | Other/anomalous |
|---|---|---|---|---|---|---|
| index.md | 0 | 0 | 0 | 0 | 0 | 0 (TOC page, no code) |
| transformer-block.md | 10 | 10 | 0 | 0 | 0 | 0 |
| gpt.md | 11 | 11 | 0 | 0 | 0 | 0 |
| kv-cache.md | 16 | 16 | 0 | 0 | 0 | 0 |
| encoders-decoders.md | 14 | 14 | 0 | 0 | 0 | 0 |
| vision-transformer.md | 11 | 8 | 3 | 0 | 0 | 0 |
| moe.md | 9 | 8 | 1 | 0 | 0 | 0 |
| scaling-laws.md | 7 | 7 | 0 | 0 | 0 | 0 |
| **Ch11 total** | **78** | **74** | **4** | **0** | **0** | **0** |

### Ch12 State Space Models

| File | Distinct ids | Paired | Shared | Pytorch-only | Jax-only | Other/anomalous |
|---|---|---|---|---|---|---|
| index.md | 0 | 0 | 0 | 0 | 0 | 0 (TOC page, no code) |
| lstm.md | 19 | 8 | 11 | 0 | 0 | 0 |
| ssm.md | 20 | 15 | 5 | 0 | 0 | 0 |
| mamba.md | 23 | 14 | 9 | 0 | 0 | 0 |
| matrix-state.md | 8 | 6 | 2 | 0 | 0 | 0 |
| deltanet.md | 12 | 8 | 4 | 0 | 0 | 0 |
| test-time-regression.md | 11 | 9 | 2 | 0 | 0 | 0 |
| hybrids.md | 9 | 5 | 4 | 0 | 0 | 0 |
| **Ch12 total** | **102** | **65** | **37** | **0** | **0** | **0** |

**Grand total: 180 distinct cell ids, 139 paired + 41 shared, 0 pytorch-only, 0
jax-only, 0 anomalous.** (139×2 + 41 = 319 physical blocks = the raw grep count.)

### 1.1 What "shared" cells look like

The 41 shared cells are genuinely framework-neutral — a token scan of every shared
cell's body for framework-specific tells (`torch.`, `jnp.`, `nnx.`, `optax.`,
`.detach()`, `state_dict`, `import jax`, etc.) hit only 2 of 41, and both are the
legitimate `tab.selected(...)` idiom (see §1.3), not a bug. The rest are pure
`d2l.*`-helper code that runs unmodified under both frameworks (data loading,
trainer construction, scoreboard bookkeeping, simple metric prints).

### 1.2 Sanity checks performed

- **No dangling slide references.** Every `@<id>`/`@!<id>` placeholder in every
  file's `<!-- slides -->` section resolves to a real cell id (checked
  programmatically against the matrix above).
- **No `@<id>@<framework>` selector syntax used anywhere** in either chapter — the
  slide pipeline always pulls whichever framework's own executed notebook supplies
  the output at render time, which is safe precisely because every id has full
  two-framework (or shared) coverage.
- **No stray retired-framework residue.** `grep -in "tensorflow|mxnet"` across all
  16 files returns nothing — no stray `%%tab tensorflow`/`%%tab mxnet`, no leftover
  prose mentioning either.

### 1.3 The `tab.selected()` idiom (not an asymmetry)

`vision-transformer.md:356-361` and `:494-498` use one shared cell with an internal
branch to count parameters the idiomatic way per framework:

```python
if tab.selected('pytorch'):
    vit_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
if tab.selected('jax'):
    vit_params = sum(p.size for p in jax.tree.leaves(nnx.state(model, nnx.Param)))
```

Both branches run under both frameworks' notebooks (only one `if` fires per
execution); this is a normal, established idiom in this codebase, not a partial
implementation. Not a gap.

---

## 2. Prose parity (`:begin_tab:`/`:end_tab:`)

There are 8 lone `:begin_tab:`jax`` asides (no adjacent pytorch block) and **zero**
lone `:begin_tab:`pytorch`` asides — every pytorch tab that exists is immediately
paired with a jax tab. All 8 lone jax asides were read in full context and are
**design-driven**, explaining a genuine JAX/XLA-specific constraint that has no
PyTorch analogue (mirroring exactly the kind of acceptable asymmetry the task
brief anticipated — jit/compilation, static shapes, framework primitives):

| Location | Topic | Verdict |
|---|---|---|
| `gpt.md:464-472` | Fixed-size generation buffer (vs. PyTorch's naturally growing list) to avoid per-step XLA recompilation | design-driven |
| `gpt.md:634-639` | GPT-2 weight loading needs no transpose in JAX (layout happens to match `nnx.Linear`); explicitly notes what PyTorch users must do instead | design-driven |
| `kv-cache.md:316-322` | `jax.default_matmul_precision('highest')` to pin fp32 for a correctness check (TF32 is JAX's default here) | design-driven — same convention already established in `chapter_attention/attention-at-scale.md:366-369/385`, cross-referenced |
| `moe.md:219-226` | Dense-compute-every-expert formulation is specifically what keeps MoE JAX-friendly (XLA can't express data-dependent per-expert token counts) | design-driven |
| `encoders-decoders.md:713-719` | Fixed-buffer decode loop, reusing the `sec_gpt` idiom, for the same jit-recompilation reason | design-driven |
| `ssm.md:215-219` | `jax.lax.associative_scan` is a shipped primitive, so JAX's scan is a 3-line wrapper vs. PyTorch's hand-written Hillis–Steele doubling scan | design-driven — the clearest instance of the exact pattern anticipated by the task brief |
| `lstm.md:252-260` | `jax.lax.scan` vs. Python loop, because the loop unrolls under jit and compiles slowly for the LSTM's 8 matrix products/step; cross-references `sec_rnn-scratch` | design-driven |
| `mamba.md:166-174` | One-hot embedding lookup instead of a gather, because XLA's scatter-add gradient for ~33k colliding updates into a 10-row table is pathologically slow | design-driven |

The 5 places where `:begin_tab:`pytorch`` and `:begin_tab:`jax`` appear **paired**
back-to-back were all read and are substantively balanced — each side gets
comparable explanatory depth about its own framework's approach, not a filler
placeholder on one side:

- `scaling-laws.md:147-168` — PyTorch profiler (`aten::mm` attribution) vs. XLA
  `cost_analysis()`, each with its own quirk called out (attention-score FLOPs
  invisible to the PyTorch profiler; a SwiGLU width miscount specific to the JAX
  cost analyzer).
- `kv-cache.md:93-103` / `:105-118` — PyTorch's dict-based growing cache vs. JAX's
  preallocated fixed-shape cache, both explained at equal length.
- `kv-cache.md:439-444` / `:446-452` — timing-result commentary, one line each,
  same content module framework-specific caveat.
- `kv-cache.md:502-511` / `:513-519` — memory-bill verification: the PyTorch side
  reports a real bug the authors found by measuring (a `.contiguous()` fix for a
  view-into-fused-QKV leak) and the JAX side explains why its check is "almost
  circular" by construction. This is a good example of the two sides being
  *differently* interesting rather than symmetric filler — both are substantive.
- `lstm.md:901-903` / `:905-907` — per-framework Discourse "Discussions" link
  boilerplate (standard d2l footer, not content).

**No framework-specific section/subsection headers** (`grep` for a `##`/`###`/`####`
header containing PyTorch/JAX/TensorFlow/MXNet returns nothing in either chapter).
**No framework-specific figure captions or comparison tables** either.

One inline (non-`begin_tab`) aside is worth noting for completeness:
`kv-cache.md:628-629` mentions in passing that GQA's head-broadcast is "in PyTorch
via `enable_gqa`; the JAX kernel accepts grouped shapes natively" — a one-sentence,
evenhanded aside, not a gap.

---

## 3. Exercises parity

**Ch12 is fully framework-neutral** — no exercise in any of the 7 content files
mentions PyTorch, JAX, or a framework-specific API by name (except the boilerplate
Discussions links in `lstm.md`).

**Ch11** is neutral except two exercises in files that also carry framework-only
prose:

- `kv-cache.md` Exercise 4 (line 1299-1305) explicitly names *both* frameworks'
  version of the same defect (PyTorch's `torch.cat`-per-step growth; JAX's
  buffer-copy-unless-donated issue) and invites the reader to "fix either one" —
  balanced, not a gap.
- `moe.md` Exercise 5 (`moe.md:828-832`) asks the reader to hands-on
  "Implement the gathered alternative **in PyTorch**:
  for each expert, `torch.where` the indices... and scatter the results back,"
  with no equivalent hands-on JAX capacity-buffer implementation exercise.
  Exercise 2 does cover the JAX-relevant capacity-factor *concept* analytically for
  both readers, and the prose at `moe.md:219-226` already explains at length why a
  JAX capacity-buffer implementation is a substantially bigger lift (dynamic
  per-expert buffers, overflow masking) than the PyTorch gather — so the asymmetry
  is plausibly a deliberate scoping choice rather than an oversight. **Flagging
  this for human judgment rather than calling it a hard gap**: if the intent is a
  fully hands-on-symmetric exercise set, a JAX-side capacity-buffer exercise (or an
  explicit note on why it's out of scope) would close it.

---

## 4. Slides parity

No `@id@framework` overrides exist in either chapter, and no dangling references
(§1.2), so slide coverage inherits the 100% cell-level parity directly — whichever
framework's slide deck is built (`make slides-pytorch` / `make slides-jax`) pulls
that framework's own executed output for every referenced id.

Where the slides *do* call out a framework explicitly, they are deliberately
balanced. `kv-cache.md`'s slide deck pairs one JAX-specific `.d2l-note` (line
1349-1353, on preallocation/`dynamic_update_slice`) with one PyTorch-specific
`.d2l-note` (line 1386-1389, on the `.contiguous()` fix for the cache-view bug) at
analogous points in the narrative — each framework gets its own footgun
highlighted, not just one. No other file's slides mention a framework by name at
all. No rung or section is demonstrated in slides for one framework and skipped
for the other.

---

## 5. Length-disparity spot check (paired cells with lopsided code volume)

Beyond structural (tab-existence) parity, I compared body-line counts within every
paired cell to catch cases where one framework's implementation might be
substantively thinner under the same nominal pairing. 8 pairs exceeded a 1.8×
line-count ratio; all 8 were read in full and are design-driven, not gaps:

| Cell id | File | PyTorch / JAX lines | Why |
|---|---|---|---|
| `lstm-implementation-and-comparison-1` | lstm.md | 7 / 26 | `nn.GRU(...)` is a single fused call; Flax NNX has no equivalent turnkey multilayer+dropout GRU, so it's built from `nnx.GRUCell` primitives |
| `lstm-concise-implementation-1` | lstm.md | 10 / 26 | Same story for `nn.LSTM` vs. `nnx.LSTMCell` |
| `lstm-bidirectional-recurrent-networks` | lstm.md | 4 / 9 | `bidirectional=True` flag vs. `nnx.Bidirectional` wrapping two explicit sub-RNNs |
| `lstm-training-4` | lstm.md | 5 / 10 | See below — systemic `val_ppl` idiom |
| `ssm-a-mingru-language-model-5` | ssm.md | 4 / 12 | Same `val_ppl` idiom |
| `mamba-the-three-answers-measured-on-one-task-2` | mamba.md | 11 / 22 | Same `val_ppl` idiom, plus an `nnx.view(..., deterministic=True)` call to switch off dropout for eval |
| `kv-cache-the-memory-bill` | kv-cache.md | 12 / 4 | (Reverse direction.) PyTorch needs explicit `empty_cache()`/`memory_allocated()` before/after bookkeeping; JAX's fixed-size buffers make the check a direct `.nbytes` read — already explained in the adjacent `:begin_tab:` pair (§2) |
| `kv-cache-the-cached-forward-pass-1` | kv-cache.md | 35 / 18 | (Reverse direction.) PyTorch's growing-dict cache vs. JAX's preallocated-buffer cache — already explained in the adjacent `:begin_tab:` pair (§2) |

**The recurring `val_ppl` pattern is worth flagging as a systemic (not
chapter-local) framework-layer idiom**, not a ch11/ch12 content gap: it appears
**identically** in `lstm.md`, `ssm.md`, `mamba.md`, and `deltanet.md` — PyTorch
reads a value the `d2l.Trainer`/`Board` already logged
(`model.board.data['val_ppl'][-1].y`), while every JAX instance manually
recomputes perplexity by iterating `data.val_dataloader()`. Because it recurs
identically four times, it reflects a real (if slightly asymmetric) difference in
what the shared `d2l` library's PyTorch vs. JAX `Trainer`/`Board` track, applied
consistently — out of scope for a per-chapter content fix, and it does not leave
the JAX reader without a perplexity number.

---

## 6. Bonus: code-quality flags

- **Commented-out core functionality:** none found. Grepped for commented
  `trainer.fit(`, `model.fit(`, `.fit(`, `assert`, `.backward(`, `.step(`,
  `.zero_grad(`, `optimizer.`, `.train()`, and a commented-out `for` loop, across
  every code cell body in both chapters. Zero hits.
- **Stub markers:** no `TODO`/`FIXME`/`XXX`/`NotImplementedError`, no bare `pass`
  as a sole function body, no `...` ellipsis placeholders.
- **Stray tensorflow/mxnet:** none — confirmed by both a `%%tab tensorflow|mxnet`
  grep and a plain-text `tensorflow|mxnet` grep across all 16 files.

## 7. Tangential observation (not a PyTorch-vs-JAX issue)

`lstm.md` ends with real per-framework Discourse thread links
(`/t/1057` pytorch, `/t/18016` jax), while every other ch12 file
(`ssm`, `mamba`, `matrix-state`, `deltanet`, `test-time-regression`, `hybrids`)
ends with the generic un-slugged `https://d2l.discourse.group/` for both
frameworks equally, and ch11 files have no Discussions footer at all. This affects
both frameworks identically within each file, so it is not a coverage-parity
defect — noted only as a minor editorial loose end (likely placeholder links
pending real forum threads for the newer sections) in case it's useful for a
separate pass.

---

## Verdict

**Ch11 Transformers: clean.** All 78 cell ids have full coverage (74 paired + 4
shared, 0/0 pytorch-only/jax-only). Every framework-only prose aside is
design-driven and justified. One soft, non-blocking item for human judgment:
`moe.md` Exercise 5 is PyTorch-only with no JAX-side hands-on counterpart
(§3) — recommend a look, not a required fix.

**Ch12 State Space Models: clean, no exceptions.** All 102 cell ids have full
coverage (65 paired + 37 shared, 0/0 pytorch-only/jax-only). Every framework-only
prose aside is design-driven and justified. Exercises are fully framework-neutral.

**Total genuine (type b) coverage gaps found: 0.**
