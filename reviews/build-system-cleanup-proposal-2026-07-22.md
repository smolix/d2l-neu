# Build-system cleanup & Makefile-simplification proposal (2026-07-22)

Audit of the build logic (`Makefile`, `tools/`, deploy scripts) with a plan to
clean up dead code, simplify the Makefile, and deliver five first-class
abilities: **full build · per-artifact partial builds · incremental builds ·
parallel/scheduled execution · deploy (Colab + R2)**.

## Status — IMPLEMENTED 2026-07-22 (phases 0–2 + two follow-ups)

Landed and verified with a green `make all-quick` (refactored Makefile + extracted
scripts, all 4 PDFs, `check-all-artifacts` clean):

- **Phase 0 — prune & docs.** Deleted `build.sh` + `tools/mkstamp.py`; archived the
  already-applied one-shots/probes to `tools/oneshot/` (with a README). **Kept**
  all university-adoption tooling as manual/occasional (Alex). Refreshed
  `docs/build-system.md` (§6.4, §6.6, new **§6.9**) and `docs/architecture.md`.
- **Phase 1 — modularize.** Split the 1032-line Makefile → a 263-line orchestrator
  + `make/*.mk` includes; extracted `tools/build_one_pdf.sh` +
  `tools/integrate_slides.sh` (killed the `$$$$` escaping). Verified
  behavior-preserving via a normalized `make -pn` database diff (zero targets/vars
  lost). **`make/notebooks.mk` = the scheduler core, relocated verbatim.**
- **Phase 1 — figures.** `make figures` now covers every chapter generator
  (incl. ch6/7/8/9) and is incremental per-generator + manual (Alex's steer).
- **Phase 2 — deploy.** `make deploy` / `publish-colab` (verified) / `upload-r2*`;
  gating independent, both on build success (Alex). `scratchpad/fullrun.sh` retired.

Follow-ups requested mid-pass, also landed:

- **Desktop breadcrumb fix** — `fix_crossref_numbers.py::fix_breadcrumbs()` now
  rewrites **both** the mobile and desktop breadcrumb blocks Quarto emits (was
  `re.search` = first/mobile only). Verified on real rendered HTML.
- **Logging sweep** — every substantive `make` recipe now tees to `logs/`.

**Deferred:** Phase 3 (make `outputs/` a tracked prerequisite so `all-quick` is a
no-op when nothing changed and only affected frameworks' PDFs rebuild). **Open
finding:** 8 `mdl-*` figures (la/cal/clf/opt) drift when regenerated with the
current venv — a matplotlib-version reproducibility gap that predates this work;
committed SVGs left as-is.

Changes are **not committed** — awaiting review.

## 0. Guardrail — what this proposal must NOT touch

The **notebook scheduler is correct and load-bearing** and stays exactly as-is.
It solves a genuinely hard problem — packing many notebooks onto GPUs that each
sit idle much of the time, mixing CPU-parallel work, and never tipping into
OOM or `ulimit -u` thread exhaustion. Cleanup **relocates** this code into a
Makefile include verbatim and changes **none** of its logic:

- `tools/notebook_scheduler.py`, `tools/run_one_notebook.py`,
  `tools/detect_resources.py`, `tools/notebook_run_plan.py`,
  `tools/notebook_run_summary.py`
- the per-notebook dependency machinery: `MANIFEST.mk` (via
  `scan_notebook_manifests.py`), `.d` files (via `scan_d2l_usage.py`),
  `EXEC_RULE`, and every `EXTRA_ENV_<fw>` / slot-derivation block
- the freshness/capture core: `capture_outputs.py`, `audit_outputs.py`,
  `refresh-stale`'s surgical stamp-removal trap fix

Everything below is the *scaffolding* around this core, which has accreted and
can be simplified without risk to scheduling.

## 1. What's wrong today (findings)

Four problems, evidence-backed:

### 1a. Dead / orphaned code in `tools/` (~130 scripts)
A transitive reachability audit from the real roots (Makefile, `bootstrap.sh`,
the three deploy scripts, `.vscode-extension/`) found a meaningful dead set.
There is **no CI and no pre-commit hook** in this checkout (`.git/hooks/` holds
only stock git-lfs hooks; no `.pre-commit-config.yaml`), so the Makefile + a few
loose scripts are the *only* things that keep a tool alive.

### 1b. The Makefile is a 1032-line monolith
One file mixes resource detection, venvs, library, figures, notebook
gen/exec, render (html/pdf/slides), hosted notebooks, universities, and store
management. Two structural pain points:
- **Giant inline shell inside recipes.** The `html` rule embeds ~40 lines of
  slide-integration shell (`Makefile:433-457`); `PDF_RULE` embeds ~40 lines of
  render/compile shell (`Makefile:834-872`). The latter forces the notorious
  quadruple-`$$$$` escaping because it lives inside a `define`/`$(eval)`.
- **No module boundaries.** No `include`s except the generated `MANIFEST.mk`/
  `.d` files, so the whole thing is read top-to-bottom to find anything.

### 1c. Deploy is not a build ability — it lives outside Make
- **No R2 upload target exists at all.** `tools/upload_r2.sh` (aws-cli →
  `staging-d2l` bucket, hash-manifest incremental, `--delete` reconciles against
  the live bucket) is a loose script the Makefile never references.
- **Colab publish is half-wired.** The Makefile's `publish-notebooks-branch`
  does the raw orphan force-push with **no verification**; the *verified* path
  (orphan-ness, per-fw counts, LFS-pointer check, raw-URL reachability) lives
  only in the loose `tools/publish_colab_notebooks.sh`, which is what actually
  gets used.
- **The top-level orchestrator is a throwaway.** `scratchpad/fullrun.sh`
  (`clean → all-quick → colab → R2`) is in gitignored `scratchpad/`. It also
  does **not gate R2 on Colab** — a failed Colab publish still uploads to R2.

### 1d. Top-level builds are never incremental
`rebuild-book-artifacts` unconditionally `rm`s `_book/index.html` and all four
PDFs every run (`Makefile:959-960`), so **`make all-quick` always pays a full
~12-min HTML render + a full 4× monolithic-PDF render even when nothing
changed.** Root cause: the committed `outputs/` store — where notebook *output*
changes actually live — is **not a Make prerequisite anywhere**, so Make can't
tell HTML/PDF are stale, and the blanket `rm` is the workaround. (Slides are the
exception: `gen_slides.py`'s per-deck content-hash makes the forced recipe a
near-no-op when nothing changed. `d2l_preprocess.py` is already content-aware
and does *not* churn `.qmd` mtimes — good.)

Note: HTML is whole-book monolithic **by design** — a subset render doesn't
amortize the cross-ref scan and flakes on cross-chapter `:numref:` (measured &
rejected, `build-system.md` §6.8). So "incremental HTML" means **skip entirely
when unchanged** and **rebuild only affected frameworks' PDFs** — not per-page
rendering.

### 1e. Docs drifted from yesterday's optimization commits
`build-system.md` §6.6 still says PDFs use `quarto render --to pdf` (now
`--to latex` + our own `xelatex ×3`); §6.4's `rebuild-book-artifacts` table
shows sequential html→pdf at `-j4` (now `html ∥ pdfs` at `$(RENDER_JOBS)`); no
section documents the slides isolated-clone + per-deck-hash incrementality; §11
hardcodes `-j4`. `build.sh` is stale/dangerous (still `--to pdf`, no store
awareness) and already contradicted by `CLAUDE.md`.

## 2. The five abilities — current vs. target

| Ability | Today | Target |
|---|---|---|
| **Full build** | `make all` (execute+build), `make all-quick` (from store) | keep; unchanged |
| **Executed notebooks** | `make run-all-notebooks` (scheduler) | keep; **untouched** |
| **HTML** | `make html` ✓ | keep; add real `outputs/` dep |
| **PDF** | `make pdfs` / `pdf-<fw>` ✓ | keep; per-fw incremental |
| **Colab notebooks** | split: unverified `publish-notebooks-branch` (make) + verified loose script | **one** verified `make publish-colab` |
| **Gzipped notebooks** | `make notebook-zips` ✓ | keep |
| **R2 upload** | loose `tools/upload_r2.sh`, no target | **`make upload-r2` / `deploy`** |
| **Partial builds** | mostly ✓ (`html`/`pdfs`/`slides`/`notebook-zips`) | + colab/R2 as targets |
| **Incremental** | notebooks ✓, slides ✓; **HTML/PDF always full** | HTML/PDF skip-if-unchanged |
| **Parallel/scheduled** | scheduler ✓, render fleet ✓ | keep; **untouched** |

The gaps are precisely: **R2/Colab as first-class targets** and **top-level
incremental skip for HTML/PDF**.

## 3. Proposed Makefile structure — split into `make/*.mk` includes

Behavior-preserving reorganization. The top `Makefile` keeps only variables,
`.PHONY` decls, `help`, the aggregate goals (`all`, `all-quick`, `deploy`), and
`include make/*.mk`. Each concern moves to its own include:

| Include | Owns | Source lines today |
|---|---|---|
| `make/resources.mk` | `detect_resources` plumbing, all slot vars, `RENDER_JOBS`, `SLIDE_WORKERS`, `EXTRA_ENV_<fw>` | ~73-127, 596-714, 786-791 |
| `make/venvs.mk` | `.venv-*/.synced`, TF cusolver fix, mxnet preflight, `kernels` | ~298-373 |
| `make/lib.mk` | `d2l/.built`, `figures` (+ **explicit generator list**, §5) | ~200-217, 286-296 |
| `make/notebooks.mk` | **[SCHEDULER CORE — verbatim]** gen (`IPYNB_RULE`, `.generated`, `MANIFEST.mk`, `.d`) + exec (`EXEC_RULE`, `run-*`) | ~463-807 |
| `make/store.mk` | `capture-outputs`, `audit-outputs`, `refresh-stale`, `render-fresh`, `test-trap` | ~219-284 |
| `make/render.mk` | `html`, `pdf`, `slides` (recipes thinned — §4) | ~401-461, 809-911 |
| `make/hosted.mk` | `hosted-notebooks`, `check-hosted-*`, `notebook-zips` | ~536-585, 913-931 |
| `make/deploy.mk` | **[NEW]** `publish-colab`, `upload-r2`, `deploy` (§6) | — |
| `make/universities.mk` | landing-page logo grid (independent of the book) | ~375-399 |

Same targets, same DAG — just navigable. This alone turns "where is X?" from a
scroll through 1032 lines into opening one 100-200-line file.

## 4. Extract the two inline-shell blobs into `tools/` scripts

- `tools/integrate_slides.sh` ← the html rule's slide-integration block
  (rsync `_slides/`→`_book/slides/`, strip symlinks, rewrite `../img` refs, copy
  slide-only images). The `html` recipe becomes one call.
- `tools/build_one_pdf.sh <fw>` ← `PDF_RULE`'s body (inject outputs, rsvg
  SVG→PDF, `quarto --to latex`, find+cp the `.tex`, `fix_latex.py`,
  `xelatex ×3`, publish to `_book/pdf/`). The `define` shrinks to a one-line
  call and **the quadruple-`$$$$` escaping disappears** (a plain script uses
  normal `$`). Both scripts become independently runnable and lintable.

## 5. `tools/` cleanup inventory

### DELETE — superseded, no live output, no references
| File | Why |
|---|---|
| `build.sh` | Legacy wrapper; `CLAUDE.md` says don't use it; still `--to pdf`, no store awareness, no stamps |
| `render_university_map.py` | Superseded by the google-map variant that `index.md` actually embeds |
| `run_hosted_full.py` | Release sweep never invoked by anything |
| `mkstamp.py` | Stamp helper; Makefile inlines all stamp logic, never calls it |
| `test_jax_threads.py`, `test_jax_threads2.py` | Ad-hoc thread-count probes; conclusions already baked into `EXTRA_ENV_jax` |

### ARCHIVE to `tools/oneshot/` — already-applied migrations / analyses
`migrate_slide_markers.py`, `strip_tab_all.py`, `split_tab_selected.py`,
`patch_slides_navlink_r2.sh`, `compare_convergence.py`, `extract_convergence.py`,
`enrich_university_metadata.py`, `promote_university_candidates.py`,
`normalize_logos.py`. (Keep in-tree for provenance, out of the way of the live
set. `render_google_university_map.py` also belongs here — its output is live
but it's a one-shot re-run manually.)

### KEEP but DOCUMENT as manual/diagnostic tools (not dead)
`audit_slides.py`, `check_slide_overflow.py`, `audit_notebook_results.py`,
`repro_mxnet_failures.py`, `repro_mxnet_runtime.py`, `set_r2_cache_control.sh`.
These are referenced only from docs as manual commands — add a "Manual &
diagnostic tools" section to `build-system.md` so they're discoverable and not
mistaken for dead in a later audit.

### FIX — live output, but disconnected from `make figures` (do **not** delete)
`make figures` globs only `tools/gen_mdl_*_figures.py`, so these generators —
whose SVGs **are committed and shown in the book** — are never regenerated:
`gen_opt_figures.py` (ch9), `gen_arch_convnets_figures.py` (ch7),
`gen_arch_convmodern_figures.py` (ch8), `gen_bg_{arch,io,memory,modules,numerics}_figures.py`
(ch6). Fix by replacing the glob with an **explicit `FIGURE_GENERATORS` list**
in `make/lib.mk` (lower-risk than renaming to the `gen_mdl_*` convention and its
skill-doc/import churn). After wiring them in, re-audit `bg_diagrams.py` — its
docstring says the `gen_bg_*` family should import it, but none do; if still
unused post-fix, it's genuinely dead. (This gap is already flagged in
`reviews/ch09-10-output-qc-findings-2026-07-21.md`.)

### DECIDE (needs Alex) — northstar parallel deploy path
`northstar_slides.py` is **live** (imported by `build_slides_index.py` and
`check_book_artifacts.py`). But `stage_northstar_slides.sh` +
`upload_northstar_r2.sh` are a *parallel, surgical* slide-deploy path used
during the ongoing north-star migration (81/164 decks). Once migration
completes they fold into the normal `make slides` + `upload-r2`. Keep for now;
retire when migration finishes.

## 6. Deploy layer (new `make/deploy.mk`)

Promote the throwaway `scratchpad/fullrun.sh` into first-class targets:

```
make publish-colab   # verified path: hosted-notebooks → check-hosted-notebooks
                     #   → publish_notebooks_branch.sh → orphan/count/LFS/raw verify
                     #   (subsumes the loose publish_colab_notebooks.sh; the
                     #    unverified publish-notebooks-branch becomes an alias or
                     #    is dropped so there's ONE safe path)
make upload-r2       # tools/upload_r2.sh (incremental); prereq: _book/ exists,
                     #   .env creds present (fail clearly if not)
make upload-r2-delete / upload-r2-full   # the --delete / --full variants
make deploy          # all-quick → publish-colab → upload-r2, gated:
                     #   build failure aborts; per-phase rc reported
```

Design decisions for Alex:
- **Gating:** gate both Colab and R2 on build success; keep Colab and R2
  independent of *each other* (they're separate artifacts) — or make R2 wait on
  Colab? (fullrun.sh currently doesn't; I lean "independent, both gated on build".)
- **Verified colab as the only path:** collapse `publish-notebooks-branch`
  (unverified) into `publish-colab` so nobody force-pushes without the
  orphan-ness check that prevents the PR-#30-into-main incident.
- `scratchpad/fullrun.sh` is then deleted (replaced by `make deploy`).

## 7. Incrementality fix (the real efficiency win)

Make the committed store a **tracked prerequisite** and drop the blanket
force-`rm`:

- Add `OUTPUT_MANIFESTS := $(wildcard outputs/**/*.json)` (or a single
  `outputs/.store-stamp` bumped by `capture-outputs`).
- `_book/index.html` depends on `.preprocess.stamp` + `$(OUTPUT_MANIFESTS)` +
  `_quarto.yml` + theme files. `_pdf/<fw>/…pdf` likewise per framework.
- Delete the `rm -f _book/index.html` and the PDF `rm` loop from
  `rebuild-book-artifacts`.

Result:
- `make all-quick` twice in a row → **second run is a no-op** (like slides today).
- One notebook's output changes → its manifest changes → HTML re-renders
  (whole-book, unavoidable) and **only the affected frameworks' PDFs** rebuild.
- A bare `make html` after an `outputs/`-only change now correctly detects
  staleness instead of silently serving stale HTML.

HTML stays whole-book monolithic (§1d caveat); per-page HTML is explicitly *not*
proposed (cross-ref correctness). The win is the no-op skip + per-framework PDF
scoping.

## 8. Phased plan (each phase independently shippable & verifiable; no CI, so
verification = a build)

| Phase | Scope | Risk | Verify |
|---|---|---|---|
| **0. Prune & document** | DELETE/ARCHIVE §5; add manual-tools doc section; refresh `build-system.md` §6.4/§6.6 + slides-incrementality; drop `build.sh` refs from `architecture.md` | none (no behavior change) | `make all-quick` green; `make figures` → clean `git diff img/` |
| **1. Modularize** | split into `make/*.mk`; extract `integrate_slides.sh` + `build_one_pdf.sh` | low (behavior-preserving) | `make -n <target>` recipe-diff ≈ empty for key targets; full `make all-quick` |
| **2. Deploy targets** | `make/deploy.mk`: `publish-colab`, `upload-r2*`, `deploy`; delete `scratchpad/fullrun.sh` | low | `make deploy` reproduces the from-scratch run (build+colab+R2 green) |
| **3. Incrementality** | `outputs/` as prereq; drop force-`rm` | medium | `all-quick` twice → 2nd no-op; touch one output → only that fw's PDF + HTML rebuild; full book still correct |
| **4. Figures glob fix** | explicit `FIGURE_GENERATORS`; wire ch6/7/8/9; re-audit `bg_diagrams.py` | low | `make figures` regenerates the previously-orphaned SVGs; `git diff img/` reviewed |

Phase 0 is pure win and reversible; phases 1-2 are the "simplify + fill the
deploy gap" the request centers on; phase 3 is the efficiency payoff; phase 4
closes a latent correctness gap.

## 9. Open questions for Alex

1. **Delete vs. archive** the one-shots (§5) — archive to `tools/oneshot/`, or
   just remove (git history keeps them)?
2. **Deploy gating** (§6) — R2 independent of Colab (both gated on build), or R2
   waits on Colab?
3. **Figures** (§5 FIX) — explicit generator list (proposed) vs. rename ch6/7/8/9
   generators to the `gen_mdl_*_figures.py` convention?
4. **Scope now** — do you want all four phases, or just 0-2 (cleanup + deploy)
   first and defer incrementality (phase 3)?
