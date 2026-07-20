# Ch. 13 "Computational Performance" rebuild — implementation brief

*2026-07-20. Alex approved the rebuild proposal
(`reviews/comp-perf-chapter-review-and-proposal-2026-07-20.md`) with the five
decisions recorded in Part A below. This brief is the handoff document: Alex is
picking this up later and a fresh agent (possibly with no memory of the
research phase) will implement it. Read this file first, then the proposal —
the proposal is the binding per-section spec; **where this brief amends it,
this brief wins.** The six research reports that ground every claim are
committed at `reviews/comp-perf-research/` (01 existing-chapter review, 02
curriculum survey, 03 framework-API audit with live measurements on this box,
03b JAX pedagogy survey, 04 mlss-efficiency deck inventory, 05 whole-book
coverage scan). The `smolix/mlss-efficiency` clone the research used lived in
a session scratchpad; re-clone from GitHub if needed (HTTPS; `gh` is
authenticated as smolix).*

Companion follow-up task in the session task list: "Implement comp-perf
rebuild per brief" (created alongside this file).

---

## Part A — Alex's decisions (2026-07-20), verbatim and binding

Alex's words in quotes; binding interpretation after each.

**A1. Hardware stays.** *"Hardware stays, at least key properties. Appendix is
more of a buyers guide, main body is meant to explain hardware properties."*
→ 13.2 stays in the chapter as specced (proposal §5/13.2). The division of
labor is now canon and should be stated in both places: **ch. 13.2 explains
hardware properties** (why the machine behaves the way it does — hierarchy,
shoreline, formats, interconnects, energy); **ch. 29 `sec_hardware_buyers` is
the buyers guide** (what to purchase). "At least key properties" licenses the
proposal's trims: the compact cross-vendor table stays, the edge-device menu
and market-pricing content from the MLSS deck are cut (fastest-decaying data).

**A2. Serving: soften upstream, defer the real treatment.** *"Yes, don't go
into detail on serving systems and soften upstream. The real serving section
will happen in the Language Models part in a lot more detail."*
→ Ch. 13 covers inference *economics* only (13.2's prefill-vs-decode roofline
reading). Serving **engines** (continuous batching, paged KV, speculative
decoding, serving stacks) are redirected to the **Language Models part** — not
to a vague "future chapter." Three upstream one-line prose edits are required;
exact current text in Part C2. Ch. 13's own `index.md` scope-fence paragraph
must also point serving at the Language Models part.

**A3. Pilot first.** *"yes, pilot first."*
→ The torchrun-from-a-notebook-cell idiom is primary for DDP (proposal §4.5),
gated on pilot P1 (Part D, Phase 0). The empirically verified fork harness
(`torch.multiprocessing.start_processes(..., start_method='fork')` with the
strict no-CUDA-in-parent-before-fork discipline) is the specified fallback. Do
not write any 13.6/13.7 prose before P1's verdict.

**A4. Ride-along approved, but INVISIBLE on the site.** *"mechanical
ride-along is ok but it should not show up in the rendered notebooks on the
website. it's just to have the code / library definitions ready. we will
likely remove mxnet & tf from the later notebooks, too."*
→ This **supersedes the proposal's §6.2/Open-Q4 plan** (which would have put
the mxnet/tf `split_batch`/`resnet18` blocks into ch. 19's
`image-augmentation.md` — a rendered page; vetoed). Use the repo's existing
**`LIB_ONLY_FILES` mechanism** instead — full recipe in Part C3. Note the
second sentence: mxnet/tf removal from ch. 17–19 is expected later, so the
legacy file must carry an explicit deletion condition in its header.

**A5. Runtime budget approved; 2-GPU demonstrability required.** *"runtime
budget for 45-60 minutes is fine. it can be more if really beneficial for
demonstration. we have a 4 GPU machine here but the notebooks should be
possible to demonstrate things on 2 GPUs, too."*
→ Budget per proposal §6.5 approved (~45–60 min/framework; exceed only where
a demo genuinely earns it). **New design rule for 13.5/13.6/13.7:** every
multi-GPU notebook must make its point on 2 GPUs as well as 4. Concretely:
- Device counts come from `d2l.num_gpus()` / `try_all_gpus()`, never a
  hardcoded 4. Sweeps run over k ∈ {1, 2, …, min(4, available)}.
- Every prose conclusion in those sections must hold at k=2 (e.g. "efficiency
  drops as k grows" is fine; "the fourth GPU is where X happens" is not).
  The capture box has 4 GPUs and captured outputs will show k up to 4; write
  the prose so a reader running on 2 sees a consistent story.
- Assertions/checks in cells must not require 4 devices (skip or degrade
  gracefully below the captured count; follow the pattern the existing
  `multiple-gpus.md` uses for 1-GPU readers, modernized).
- Scheduler marks stay as proposal §6.5 (13.5 = 2-GPU, 13.6/13.7 = 4-GPU
  whole-box for capture on this machine).

---

## Part B — state of the world (what is already done; do not redo)

- **Research complete** — the six reports in `reviews/comp-perf-research/`.
  Report 03's measurements were run live on this box at the current pins
  (torch 2.11.0 / torchvision 0.26.0 / jax 0.10.2 / flax 0.12.7): P2P
  disabled on all 4090 pairs; NCCL allreduce ≈ 2.2 GB/s busbw flat 2→4 GPUs;
  JAX psum on the same links 4.5–8.6 GB/s; `mp.spawn` fails under nbconvert
  (pickling), fork works iff parent untouched by CUDA; `torch.compile` 1.32×
  on a toy conv net; bf16 autocast 1.93× vs a fair TF32 baseline; TF32 is
  OFF by default at 2.11 (`'highest'`). Treat these as ground truth; do not
  re-derive, but DO re-measure inside the actual notebooks (that is the
  chapter's method).
- **Proposal written and approved (with Part A amendments)** — the per-section
  arcs (§5), figure table (§6.1), `#@save` table (§6.2, amended by A4),
  label-migration table (§6.3), kill list (§6.4), runtime budget (§6.5,
  amended by A5), precision policy (§6.6), citations/resources (§6.7), scope
  fences (§6.8, amended by A2), and pilot list (§7).
- **No chapter source files have been touched.** No pilots have been run. No
  figures generated. The existing chapter still renders and its outputs store
  is intact (all four frameworks).
- Chapter numbering at handoff: comp-perf is **ch. 13**
  (`chapter_computational-performance/` in `CHAPTER_NUMBERING`), after 9 Opt,
  10 Attention, 11 Transformers, 12 SSMs. CLAUDE.md's "Current status" bullet
  still describes an older order (comp-perf as 11) — fix it when this lands
  (Part D, Phase 3).

---

## Part C — decision-driven spec deltas (exact edits)

### C1. Hardware (A1) — no structural delta

13.2 as specced. Add one sentence each to 13.2's intro and ch. 29's
`sec_hardware_buyers` naming the properties-vs-buyers-guide division (the
ch. 29 edit is one line; do not otherwise touch ch. 29).

### C2. Serving softenings (A2) — three upstream one-line prose edits

These are **prose-only edits in non-code paragraphs** — they change no code
cells, so the outputs store for these files stays valid and **no re-execution
or recapture of these three files is needed** (the freshness gate keys on code
cells). Keep each sentence's rhythm; final wording is the implementer's, but
the semantic split is fixed: *kernels + memory hierarchies stay promised to
ch. 13; serving stacks/engines move to the Language Models part.*

1. `chapter_recurrent-modern/index.md` (~line 137–141). Current:
   > "The chapter teaches algorithms, not kernels: the chunked forms here are
   > twenty-line teaching implementations, and the Triton kernels, memory
   > hierarchies, and serving systems that make them fast belong to
   > :numref:`chap_performance`."
   Split the tail: kernels/memory hierarchies → `chap_performance`; serving
   systems → the Language Models part. (The very next sentence already sends
   pretraining to the Language Models part — attach serving there or mirror
   its construction.)

2. `chapter_recurrent-modern/hybrids.md` (~line 1024–1029). Current:
   > "…the systems story that makes the recurrent majority fast in practice —
   > the chunked forms of :numref:`subsec_ms-chunked` living as fused kernels,
   > and the serving stacks that exploit a mostly flat memory bill — which
   > belongs to :numref:`chap_performance`; and the pretraining and
   > post-training of full-scale language models, hybrid or not, which is the
   > subject of the Language Models part."
   Keep fused kernels → `chap_performance`; move "the serving stacks that
   exploit a mostly flat memory bill" into the Language-Models-part clause
   that already ends the sentence.

3. `chapter_transformers/kv-cache.md` (~line 1248–1249). Current:
   > "…the decode-side bandwidth arithmetic of this section is also what
   > *speculative decoding* exploits, drafting several tokens cheaply and
   > verifying them in one prefill-priced pass
   > :cite:`Leviathan.Kalman.Matias.2023`; the systems-level story belongs to
   > :numref:`chap_performance`."
   Redirect: the systems-level story of speculative decoding/serving belongs
   to the Language Models part. (Ch. 13.2 keeps only the decode-bandwidth
   *economics*, which this kv-cache section itself already teaches.)

Also: the new `chapter_computational-performance/index.md` scope-fence
paragraph (proposal §5/index) must say serving engines → Language Models part
(not "future efficiency chapter"). Other `chap_performance` refs
(`transformers/gpt.md:354`, `scaling-laws.md:672`, the ~20-ref list in
proposal §2a) are about kernels/parallelism/systems-of-training and stay
untouched.

### C3. The ride-along (A4) — LIB_ONLY_FILES recipe (supersedes proposal §6.2 row 4 and Open Q4)

Mechanism (already in the repo): `tools/build_lib.py` module-level
`LIB_ONLY_FILES` — source .md files that `make lib` scans for `#@save` blocks
but that are **not** in `_quarto.yml`/`CHAPTER_NUMBERING` (never rendered) and
that `gen_notebooks.py` **excludes** from notebook generation (it imports the
list). Precedent to mirror exactly:
`chapter_natural-language-processing-pretraining/legacy-attention-lib.md` —
copy its header-comment pattern (build-only rationale, what it carries, frozen
semantics, and an explicit deletion condition).

Steps, in order:
1. Create `chapter_computational-performance/legacy-multigpu-lib.md`:
   - Header comment modeled on legacy-attention-lib.md. Deletion condition:
     *"This file is deleted when ch. 17–19 (Language Models / Image Models
     parts) drop their mxnet/tensorflow tabs"* (per Alex, that removal is
     expected).
   - Move — **verbatim, byte-for-byte function bodies** — the mxnet and
     tensorflow `#@save` blocks for `split_batch` (from `multiple-gpus.md`)
     and `resnet18` (from `multiple-gpus-concise.md`). Mirror
     legacy-attention-lib.md's per-framework block formatting so
     `build_lib.py`'s tab extraction picks them up.
2. Append the new path to `LIB_ONLY_FILES` in `tools/build_lib.py`.
3. Verify **before** touching the chapter sources: run `make lib`, then
   `git diff d2l/mxnet.py d2l/tensorflow.py` must be **empty** (identical
   definitions extracted from the new location; LIB_ONLY_FILES entries are
   appended last, and with the chapter's own copies about to be deleted there
   is no shadowing concern — but do this verification while both copies still
   exist to prove the new file parses, expecting *no duplicate-definition
   surprise*: build_lib de-dupes by taking the rendered chapter's copy first.
   If the diff is non-empty or a duplicate appears, stop and inspect how
   legacy-attention-lib.md handles the same situation for `masked_softmax`).
4. Write the rebuilt 13.5/13.6 with PyTorch+JAX tabs only (their pytorch/jax
   `split_batch`/`resnet18` `#@save`s live on in the rewritten sections per
   proposal §6.2).
5. After the chapter rewrite deletes the old tf/mxnet tabs: `make lib` again;
   `git diff d2l/mxnet.py d2l/tensorflow.py` must again be empty. Then prove
   the consumers still work: at minimum
   `.venv-mxnet/bin/python -c "from d2l import mxnet as d2l; d2l.split_batch; d2l.resnet18"`
   and the tensorflow equivalent (`resnet18`); full re-execution of the
   consuming notebooks (ch. 18 `natural-language-inference-attention.md`
   mxnet; ch. 19 `image-augmentation.md`, `kaggle-cifar10.md` tf/mxnet) is
   NOT required if the lib files are byte-identical — their provenance
   fingerprints have not moved.

### C4. Pilot-first (A3) — see Phase 0. No spec delta beyond the gate.

### C5. Budget + 2-GPU rule (A5) — design rule in Part A5; scheduler marks per proposal §6.5.

---

## Part D — implementation plan

Model the workflow on the ch. 12 SSM rebuild (it shipped): pilots before
prose, figures before sections, measured outputs before quoted numbers, max
two concurrent build agents, Fable for content/judgment, Opus for review
passes, Sonnet for mechanical work.

### Phase 0 — pilots (all before any section prose; record results in `reviews/comp-perf-pilot-notes.md`)

Run each as a scratch notebook under the scratchpad, executed the same way
the build executes notebooks (jupyter nbconvert via the scheduler pathway is
the gold standard for P1's verdict — an idiom that works in a terminal but
not under nbconvert is a FAIL):

- **P1 (gates 13.6/13.7 structure): torchrun from a cell.** In an
  nbconvert-executed notebook: write a minimal DDP training script from a
  cell to a sidecar .py (the pattern: a `writefile`-style cell or
  `pathlib.Path.write_text`), then `subprocess.run(["torchrun",
  "--standalone", "--nproc-per-node=2", "train_ddp.py"])`, parse its stdout
  for per-rank confirmation. Success = clean exit, correct world_size, GPU
  utilization on both devices, AND the parent notebook can have touched CUDA
  earlier without consequence (test explicitly: allocate a tensor on cuda:0
  in an earlier cell). Repeat at nproc=4. On success → primary idiom. On
  failure → fork fallback: `start_processes(..., start_method='fork')` with
  ALL CUDA work (including the k=1 baseline) inside the harness; restructure
  13.6/13.7 accordingly (proposal §4.5).
- **P2: DDP ResNet-18 throughput at k ∈ {1,2,4}** (via P1's winning idiom),
  Fashion-MNIST-64. Deliverable: the honest scaling curve + efficiency
  numbers that 13.6's prose will be written around, plus confirmation the
  run fits the runtime budget.
- **P3: the 13.7 waterfall end-to-end** at two candidate widths of ch. 11's
  `d2l.GPT` (d=256 and d=512): baseline/compile/bf16/batch-up/checkpointing/
   2- and 4-GPU DP. Pick the width whose per-rung effects clear run-to-run
  noise comfortably (precision policy §6.6). Also confirms `d2l.GPT` +
  `train_lm` reuse works as the proposal assumes.
- **P4: memory-snapshot rendering path** — `torch.cuda.memory._record_memory_history`
  → `_dump_snapshot` → a viewable artifact that can appear in a rendered
  notebook (image or html— decide what the book embeds; a matplotlib
  rendering of the snapshot data is acceptable and more house-style than a
  screenshot). JAX side: `jax.profiler.save_device_memory_profile` usability
  headlessly.
- **P5: JAX 4-GPU under the scheduler** — a sharded-training toy notebook
  through `tools/notebook_scheduler.py` with a whole-box reservation; verify
  the memory-fraction handling (see ops trap list: the scheduler scales
  `XLA_PYTHON_CLIENT_MEM_FRACTION` with reserved slots; the single-notebook
  make path does NOT and OOMs — always use the scheduler for these).

STOP-and-reassess triggers: P1 and P5 failing both idioms/paths; P3 unable to
find a width where rungs clear noise (then re-scope 13.7's rung list).

### Phase 1 — figures

`tools/gen_mdl_perf_figures.py` importing the shared house style; the ~15
figures of proposal §6.1 (source-of-truth column there; MLSS deck numbers for
13.2's ladders/formats/shoreline — re-clone `smolix/mlss-efficiency`, use its
`style-contract.md` cheat sheet and `ref/` dossiers for the numbers; its
`verification/` folder flags the 5 known errors, already fixed in its .tex).
House rules are hard requirements: black axes/labels, 13–16 pt in-figure
text, no text↔line intersections, no dead whitespace, balanced pairs,
byte-idempotent output (`make figures`), and the **render-and-inspect loop**
(rsvg-convert → look at the PNG → fix → re-look; never trust label positions
from code). Use the `mdl-figure` skill for adding figures and
`figure-style-audit` for the chapter pass. Commit the SVGs.

### Phase 2 — sections

Write in dependency order: 13.1 → 13.2 → 13.3 → 13.4 → 13.5 → 13.6 → 13.7 →
index.md. The per-section spec is proposal §5 (arcs, demos, salvage,
exercises, slides beats) as amended by Part A. Binding conventions:

- `%%tab pytorch` / `%%tab jax` cells paired on shared stable `#<id>`s
  (assign via `tools/add_cell_ids.py`; IDs never change once written); one
  imports cell per framework near the top; prefer d2l helpers.
- PyTorch imports cell of every timing notebook sets
  `torch.set_float32_matmul_precision('high')` with its one-sentence
  justification (taught properly in 13.4).
- Every timed cell uses the upgraded `#@save d2l.Benchmark` (defined 13.1:
  warmup + device-sync built in; it has zero external consumers today, so
  the signature is free).
- Salvage sources per section: 13.1 ← async-computation.md's barrier
  taxonomy; 13.2 ← hardware.md's skeleton + exercises; 13.3 ← hybridize.md's
  benchmark arc; 13.5 ← multiple-gpus.md's from-scratch spine +
  parameterserver.md's ring derivation + history paragraph; 13.6 ←
  multiple-gpus-concise.md's resnet18 contract. Salvage means *rewrite in
  the new voice*, not paste — but keep the preserved code semantics exact
  where outputs-compatibility matters (split_batch/resnet18 signatures).
- Kill list (proposal §6.4) — none of these may appear as taught material:
  torch.jit.script/trace, allow_tf32 bool flags, FSDP1 wrapper,
  nn.DataParallel (one contrast sentence only), checkpoint() without
  use_reentrant, jax.pmap/nnx.pmap, jax.experimental.shard_map import path.
- Slides: `<!-- slides -->` section per file with `::: {.slide}` divs
  referencing cells by `@<id>`; beats per section listed in proposal §5.
- Exercises: numbered, per the book's format; proposal §5 lists them per
  section.
- Results precision (proposal §6.6): prose quotes "about 2×"-class
  magnitudes only; never per-run decimals; conclusions never rest on
  differences inside run-to-run noise; every quoted number must be one the
  regular recapture cycle will reproduce.
- New bib entries per proposal §6.7 (verify IDs against d2l.bib before
  adding; several may exist).
- Resources and Further Reading in index.md per ch. 9's format, from §6.7.

### Phase 3 — integration edits (small, surgical; most are one-liners)

1. `_quarto.yml`: replace the chapter's file list with the new seven
   (performance-model, hardware, compilation, memory-precision,
   multiple-gpus, multi-gpu-practice, fast-transformer).
2. `CHAPTER_NUMBERING` in `tools/d2l_preprocess.py`: same file set, [13, n]
   numbering. **Invariant: dict order must equal `_quarto.yml` chapter
   order** (PDF numbering pairs positionally — build-system.md §4.1).
3. `preliminaries/ndarray.md`: the one `sec_hybridize` inbound ref (TF tab)
   → retarget to `sec_compilation` and reword its clause (it currently
   speaks MXNet-era "hybridization" vocabulary).
4. The three serving softenings of Part C2.
5. One-line properties-vs-buyers sentence in ch. 29 `sec_hardware_buyers`
   (C1).
6. The legacy-lib recipe of Part C3 (file + LIB_ONLY_FILES entry).
7. Delete `async-computation.md`, `auto-parallelism.md`, `parameterserver.md`
   and `hybridize.md`, `multiple-gpus-concise.md` (contents reborn in the new
   files); delete the chapter's outputs trees for all deleted files (all four
   frameworks) and the tf/mxnet outputs trees for the whole chapter; delete
   the retired img/ assets (proposal §6.1 deletion list — the NVIDIA rasters
   etc.) after grepping for other referents.
8. CLAUDE.md "Current status": fix the stale chapter-order bullet (comp-perf
   is 13, transformers 11 is built, SSM is 12) — verified against
   `CHAPTER_NUMBERING`, per the update-build-docs-on-gaps rule.

### Phase 4 — execution, capture, render

1. `make lib` early and check the blast radius BEFORE captures: the d2l
   surface changes here (new `Benchmark` in d2l, rewritten pytorch/jax
   `split_batch`/`resnet18`) move the d2l lib fingerprint that capture
   provenance records. Run `tools/audit_outputs.py --verify-fresh` after
   `make lib` to see exactly which already-captured notebooks (this chapter
   and others importing changed symbols) go stale, and budget their
   re-execution. Keep the d2l surface diff minimal to keep this radius small.
2. Execute all new/changed notebooks **through
   `tools/notebook_scheduler.py`** (both frameworks). Force with `rm` of the
   relevant `_notebooks/<fw>/chapter_computational-performance/*.executed`
   stamps. NEVER finalize via bare `jupyter nbconvert` — capture verifies
   per-cell hashes + lib fingerprint recorded by the scheduler/make pathway
   and will REFUSE manual runs (this burned the ch. 12 endgame; see ops
   traps).
3. Gate: the scheduler run must end "done: 0 failed" **before** any
   `make capture-outputs` (capture blesses FAILED runs otherwise — known
   tooling gap).
4. Capture (`make capture-outputs FILES=...`), then
   `tools/audit_outputs.py --verify-fresh` fully green.
5. Slides: `make -B slides-pytorch SLIDES_FILTER="chapter_computational-performance/..."`
   and same for jax (CPU-only, `-j4`-safe per file set).
6. `make html` (single quarto render, never `-j`), then `make pdfs`. PDF
   tripwires (build-system.md §6.6): `$`-immediately-before-digit breaks
   math; `]` in image captions truncates alt-text; smallmatrix conventions.
   This chapter is table-heavy — check the PDF pages for ch. 13
   specifically.
7. `make check-all-artifacts`.

### Phase 5 — commit / push / deploy

- Branch: continue on **`rnn-ssm-modernization`** (the modernization
  campaign's branch — ch. 9/10/11/12 all landed there) unless Alex directs
  otherwise. Never main; no PR unless asked.
- Push via `gh`-authenticated HTTPS only (SSH needs the yubikey; `gh auth
  setup-git`, account smolix).
- Stage explicitly. NEVER stage: `ssm_feedback.md`, `scratchpad/`,
  `node_modules/`, `package*.json`, `.nbtest-logs/`, `logs/`, `_book/`,
  `_slides/`, `_pdf/`, `_notebooks/`. (Check `git status` for strays before
  every commit; there are long-lived local modifications in the tree —
  `bert-dataset.md`, `d2l/*.py`, `tools/run_one_notebook.py` — whose staging
  decisions belong to their own workstreams, EXCEPT the `d2l/*.py` changes
  that `make lib` legitimately regenerates from THIS chapter's `#@save`
  edits, which do ride along with the chapter commit.)
- Commit message trailer: `Claude-Session: <the implementing session's own
  claude.ai/code URL>`.
- Site deploy afterwards mirrors the ch. 12 pattern: `make all-quick` →
  `make check-all-artifacts` → rebuild notebook zips into `_book/notebooks`
  (quarto render wipes `_book/`; skipping this deleted the zips from the
  bucket once) → `make hosted-notebooks` → `make check-hosted-notebooks` →
  `make publish-notebooks-branch` → `bash tools/upload_r2.sh --delete` →
  live checks on d2l.smola.org. Only on Alex's go.

---

## Part E — operational traps (institutional knowledge; the implementing agent may not share this session's memory)

1. **Capture provenance requires the scheduler.** `capture_outputs.py`
   verifies per-cell source hashes + d2l lib fingerprint recorded at
   execution time by the make/scheduler pathway. Raw `nbconvert --execute`
   runs are refused at capture even when genuinely fresh. Iterate however
   you like; the FINAL pre-capture run of every changed notebook goes
   through `tools/notebook_scheduler.py` (rm `.executed` stamps to force).
2. **Capture blesses failed runs.** It writes partial manifests for FAILED
   notebooks. Only capture behind an explicit "scheduler done: 0 failed".
3. **BEST_OF_N masks real bugs** in `run_notebooks.py`: it keeps the best
   attempt even when all attempts fail the quality gate. When validating a
   training cell's numbers, measure true single-run (first-attempt,
   unseeded) scores.
4. **Background tasks die at T+1h** (harness SIGTERMs the process group).
   Anything possibly longer: `setsid nohup … &` with a done-marker file, and
   watch it with short re-armed waiters. Verify watcher liveness
   periodically — a silent dead monitor cost hours once.
5. **JAX GPU memory under the scheduler**: heavy JAX notebooks get a
   memory fraction scaled to reserved slots by the scheduler; the
   single-notebook `make -B …executed` path does NOT scale it and OOMs on
   heavies. Run JAX multi-GPU/heavy notebooks via the scheduler. 13.6/13.7's
   JAX variants claim all visible devices in one process — whole-box
   reservation (P5 verifies).
6. **Full-build ulimit**: concurrent full builds need `ulimit -u 8192` (the
   scheduler dies at 4096 with un-run notebooks tallied FAILED).
7. **`make html` is a single quarto render** — never wrap in `-j`; per-file
   subset renders flake under concurrency. Slides are `-j4`-safe.
8. **Never comment out core functionality** to make a stamp/gate pass (e.g.
   a failing `trainer.fit()`); fix or escalate.
9. **Never edit `.qmd`** (generated); source `.md` is the truth.
10. **Torch/jax pins are frozen** (2.11.0 / 0.10.2). Do not bump; the
    committed store keys on them.
11. **Prose precision**: quote only what re-execution reproduces (proposal
    §6.6). Performance numbers are thermally noisy — ranges and rounded
    ratios.
12. **Box facts** (report 03): 4×RTX 4090, driver 595.71, NO P2P (CNS on all
    pairs), NCCL allreduce ~2.2 GB/s flat 2→4 GPUs, JAX psum 4.5–8.6 GB/s.
    These are teaching data, not problems to fix. Don't burn time trying to
    enable P2P — it is a GeForce product-segmentation fact.
13. **d2l lib rebuilds have blast radius** (Phase 4.1): `make lib` regenerates
    d2l/*.py from ALL `#@save` blocks; notebooks in other chapters import
    d2l, so fingerprint-sensitive freshness can go stale outside this
    chapter. Audit before capturing; when unsure re-run affected notebooks.
14. **Agent staffing** (Alex's standing rule): Sonnet/Haiku for mechanical
    work, Opus for review, Fable for judgment-heavy composition — and Fable
    credits are budget-limited, so spend Fable turns on prose/judgment, not
    plumbing. Max 2 concurrent build agents; at 85% session usage,
    checkpoint and pause.

---

## Part F — acceptance checklist (the rebuild is done when all of these hold)

- [ ] P1–P5 pilot results recorded in `reviews/comp-perf-pilot-notes.md`;
      13.6/13.7 structured per P1's verdict.
- [ ] Seven new/rewritten sections + index per proposal §5 with Part A
      amendments; old five files deleted; `_quarto.yml` and
      `CHAPTER_NUMBERING` in matching order.
- [ ] Every §2a promise discharged (grep each inbound ref and read the
      receiving section); the three C2 softenings + ndarray.md retarget in.
- [ ] Kill-list APIs absent from all taught material (grep the chapter for
      each).
- [ ] 2-GPU demonstrability rule honored in 13.5/13.6/13.7 (no hardcoded
      device counts; prose valid at k=2).
- [ ] Figures: generated by `tools/gen_mdl_perf_figures.py`, byte-idempotent,
      house-style audited, visually inspected; no NVIDIA raster assets remain
      referenced.
- [ ] `legacy-multigpu-lib.md` in place; `make lib` twice-verified
      (C3 steps 3 and 5); mxnet/tf import checks pass.
- [ ] Both frameworks executed via the scheduler, "0 failed", captured;
      `verify-fresh` green across the store (including any lib-fingerprint
      blast radius).
- [ ] Slides re-rendered for the chapter; `make html`, `make pdfs`,
      `make check-all-artifacts` green; ch. 13 PDF pages eyeballed.
- [ ] CLAUDE.md status bullet corrected; committed on
      `rnn-ssm-modernization` with the session trailer; pushed via gh.
- [ ] Deploy only on Alex's explicit go.
