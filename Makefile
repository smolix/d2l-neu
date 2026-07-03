# D2L Book Build System
#
# Usage:
#   make help                  Show this help
#   make html                  Build HTML book (multi-framework tabs)
#   make pdf-pytorch           Build PDF for PyTorch
#   make pdfs                  Build PDFs for all frameworks (safe with -j4)
#   make run-notebooks-pytorch Execute PyTorch notebooks (CPU/GPU queues)
#   make run-all-notebooks     Execute all frameworks (CPU/GPU queues)
#   make slides-pytorch        Build + render PyTorch slides (CPU)
#   make lib                   Build d2l Python package
#   make clean                 Remove build artifacts (keeps data/)
#
# Parallel-safe:
#   make -j4 pdfs              4 framework PDFs in parallel (separate dirs)
#   make -j4 notebooks         Generate (not execute) all notebooks in parallel
#
# Notebook execution is internally resource-scheduled. Use
# `run-all-notebooks` for all frameworks instead of wrapping multiple
# `run-notebooks-*` targets in an outer `make -j`.
# Parallel-safe (CPU-only, after slide refactor):
#   slides-*                   make -j4 slides works
#
# Host resources (GPUs, cores, RAM) are auto-detected by
# tools/detect_resources.py — run `make detect` to see the plan. It degrades
# cleanly with no GPU and on macOS (no /proc), so every CPU-only target below
# (html, all-quick, slides, pdfs, capture-outputs, and CPU notebooks) runs on
# an Apple-silicon laptop unchanged. Override detection on the command line:
# `make NUM_GPUS=2 GPU_SLOTS=4 CPU_SLOTS=2`.
#
# All build output is logged to logs/<target>-YYYYMMDD-HHMMSS.log

SHELL      := /bin/bash
.SHELLFLAGS := -o pipefail -c
.DEFAULT_GOAL := help

# ── Require GNU Make >= 4.3 ────────────────────────────────────────────
# This build relies on grouped targets (`a b &: prereqs`, GNU Make 4.3+).
# Older make (notably macOS's system /usr/bin/make = GNU Make 3.81) silently
# mis-parses `&:` as a target literally named `&`, emitting
# "overriding commands for target `&'" warnings and building incorrectly.
# Fail fast with a fix instead. On macOS install a modern make and use `gmake`:
#   sudo port install gmake     (MacPorts)   — then run `gmake` not `make`
_mk_maj := $(word 1,$(subst ., ,$(MAKE_VERSION)))
_mk_min := $(or $(word 2,$(subst ., ,$(MAKE_VERSION))),0)
_mk_ok  := $(shell [ "$(_mk_maj)" -gt 4 ] 2>/dev/null && echo y || \
             { [ "$(_mk_maj)" -eq 4 ] && [ "$(_mk_min)" -ge 3 ] && echo y; } 2>/dev/null)
ifneq ($(_mk_ok),y)
$(error GNU Make >= 4.3 required, but this is $(MAKE_VERSION) ($(MAKE)). \
  On macOS the system `make` is 3.81 — install gmake (`sudo port install gmake`) \
  and run `gmake` instead of `make`. See docs/build-system.md "Building on macOS")
endif

# Keep auto-generated stamps and manifests around. Without this, Make
# treats them as "intermediate" and deletes them after the chain that
# produced them completes (this caused the .generated files to be
# re-created on every run, breaking incremental builds).
.SECONDARY:

# Enable secondary expansion so we can refer to per-framework variables
# (EXECUTED_pytorch etc.) loaded from generated manifests when expanding
# pattern-rule prerequisites.
.SECONDEXPANSION:

SOURCE     ?= .
FRAMEWORKS := pytorch tensorflow jax mxnet
RUN_NOTEBOOK_TARGETS := $(addprefix run-notebooks-,$(FRAMEWORKS))
RUN_NOTEBOOK_CPU_TARGETS := $(addprefix run-notebooks-cpu-,$(FRAMEWORKS))
RUN_NOTEBOOK_GPU_TARGETS := $(addprefix run-notebooks-gpu-,$(FRAMEWORKS))
RUN_NOTEBOOK_MULTIGPU_TARGETS := $(addprefix run-notebooks-multigpu-,$(FRAMEWORKS))
SLIDE_STAMPS := $(addprefix _slides/,$(addsuffix /.built,$(FRAMEWORKS)))

# ── Host resource detection — single source of truth ──────────────────
# tools/detect_resources.py probes GPUs (nvidia-smi), CPU cores, RAM, and the
# open-file/process ulimits, then derives every slot count below. It degrades
# cleanly on a CPU-only host (no nvidia-smi → 0 GPUs) and on macOS (no /proc →
# sysctl + vm_stat fallback), so the same Makefile runs unchanged on a 4×GPU
# Linux server and an Apple-silicon laptop. Run `make detect` to see the plan.
#
# Per-job footprints are read from the environment by detect_resources.py; we
# export them so command-line overrides reach it, e.g.
#   make GPU_MIB_PER_jax=14000 run-all-notebooks
# PyTorch / TensorFlow / MXNet share the lighter 7.5 GiB / 8-core budget (their
# EXTRA_ENV_* tuning caps resident VRAM and threads). JAX is heavier (~11.5 GiB)
# because every JAX process keeps the CPU backend + XLA pools alive
# (d2l.Module.plot host-transfers metrics via d2l.cpu()).
GPU_MIB_PER_LIGHT ?= 7680
GPU_MIB_PER_jax   ?= 11776
CPU_PER_LIGHT     ?= 8
CPU_PER_jax       ?= 8
export GPU_MIB_PER_LIGHT GPU_MIB_PER_jax CPU_PER_LIGHT CPU_PER_jax

# detect_resources.py is consulted once (its probe is cached in /tmp), and each
# knob is resolved into a SIMPLY-EXPANDED variable via the _detected helper.
# Why := and not ?=: a `?=` $(shell …) is recursively expanded, so python would
# re-run on *every* reference to the variable — dozens of times during a deep
# `make -n` graph walk. The ifndef guard means a command-line override (e.g.
# `make CPU_SLOTS=2`) is already defined, so detection is skipped and the
# override wins. On the 4×24GB / 64-core box: GPU_SLOTS=12, CPU_SLOTS=6; on a
# CPU-only laptop: NUM_GPUS=1 (slot→device fallback), GPU_SLOTS=1, CPU_SLOTS=cores/8.
_DR := python3 tools/detect_resources.py --get
define _detected
ifndef $(1)
$(1) := $$(shell $$(_DR) $(1))
endif
endef
$(foreach v,NUM_GPUS GPU_SLOTS CPU_SLOTS,$(eval $(call _detected,$(v))))

# Multi-GPU notebooks use exactly 2 GPUs at <=GPU_MIB_PER_LIGHT each, so they
# are memory-packed onto disjoint GPU pairs (verified: 3 fit on a 24 GiB card
# across all 4 frameworks). tools/detect_resources.py is the single source for
# the packing plan (pairs × per_pair) and the jax preallocation fraction.
$(foreach v,NUM_GPU_PAIRS MULTIGPU_PER_PAIR MULTIGPU_SLOTS JAX_MGPU_MEM_FRACTION,$(eval $(call _detected,$(v))))
# Slide/PDF render fleet sizing — keeps (jobs × quarto Deno/V8 heap) under the
# RAM budget so the parallel `quarto render` never SIGKILLs (rc=137) or V8-OOMs
# (rc=133). See tools/detect_resources.py (RENDER_V8_MIN_MIB / RENDER_JOBS).
$(foreach v,RENDER_JOBS RENDER_V8_HEAP_MIB,$(eval $(call _detected,$(v))))
RENDER_V8_ENV := QUARTO_DENO_V8_OPTIONS=--max-old-space-size=$(RENDER_V8_HEAP_MIB),--max-heap-size=$(RENDER_V8_HEAP_MIB)
FILES         ?=
SLIDES_FILTER ?= $(FILES)

# Source files — the ultimate upstream for everything.
# Includes top-level index.md (landing page), which d2l_preprocess.py
# converts to index.qmd alongside the chapter files.
SRC_MDS := $(wildcard $(SOURCE)/chapter_*/*.md) $(SOURCE)/index.md
TOOLS   := $(wildcard tools/*.py)

# Logging: each recipe logs to logs/<target>-YYYYMMDD-HHMMSS.log
# PYTHONUNBUFFERED ensures output streams to the log file in real time.
LOGDIR := logs
TS     := $(shell date +%Y%m%d-%H%M%S)
export PYTHONUNBUFFERED := 1

# ── Phony targets ──────────────────────────────────────────

.PHONY: help all all-quick rebuild-book-artifacts check-all-artifacts html lib clean veryclean
.PHONY: pdf pdfs $(addprefix pdf-,$(FRAMEWORKS))
.PHONY: notebooks run-all-notebooks slides
.PHONY: capture-outputs audit-outputs verify-outputs-fresh refresh-stale render-fresh

# ── Help ───────────────────────────────────────────────────

help:
	@echo "d2l-neu Build System"
	@echo ""
	@echo "Targets ([any]=runs on macOS/CPU, [GPU]=needs NVIDIA GPUs):"
	@echo "  html                    [any] Build HTML book from committed outputs/ (CPU-only)"
	@echo "  all-quick               [any] html + pdfs + slides + notebooks + lib, NO execution"
	@echo "  pdf-<fw> / pdfs         [any] Build PDF(s) from committed outputs (pdfs safe with -j4)"
	@echo "  slides-<fw> / slides    [any] Build slides (CPU; slides safe with -j4)"
	@echo "  notebooks-<fw>         [any] Generate (not execute) notebooks for one framework"
	@echo "  notebooks               [any] Generate notebooks for all frameworks"
	@echo "  capture-outputs         [any] Bless executed _notebooks/ → committed outputs/ [FILES=...]"
	@echo "  audit-outputs           [any] Report stale notebooks (code drift) + store integrity"
	@echo "  verify-outputs-fresh    [any] Render gate: fail on stale inline outputs / orphaned ids"
	@echo "  lib                     [any] Build d2l Python package"
	@echo "  venv-<fw> / kernels     [any] Sync UV env / register ipykernels (for VS Code)"
	@echo "  detect                  [any] Print the auto-detected resource + parallelism plan"
	@echo "  run-notebooks-<fw>     [GPU*] Execute notebooks for one framework (CPU notebooks OK)"
	@echo "  run-all-notebooks      [GPU*] Execute all frameworks (CPU/GPU queues)"
	@echo "  refresh-stale          [GPU*] Re-execute only stale notebooks, then re-capture"
	@echo "  render-fresh           [GPU*] refresh-stale, then rebuild slides + html + pdfs"
	@echo "  all                     [GPU] Full pipeline: generate, execute, rebuild with outputs"
	@echo "  clean / veryclean       [any] Remove build artifacts (veryclean also drops data/)"
	@echo "  (*) CPU-only notebooks run anywhere; GPU notebooks are skipped/deferred without a GPU."
	@echo ""
	@echo "Variables:  SOURCE=$(SOURCE)  NUM_GPUS=$(NUM_GPUS)"
	@echo "           GPU_SLOTS=$(GPU_SLOTS)  CPU_SLOTS=$(CPU_SLOTS)"
	@echo "           FILES=$(FILES)  SLIDES_FILTER=$(SLIDES_FILTER)"
	@echo "Frameworks: $(FRAMEWORKS)"
	@echo "Logs:       $(LOGDIR)/<target>-YYYYMMDD-HHMMSS.log"

# ── Resource detection ─────────────────────────────────────
# Print the auto-detected host resources + the parallelism plan the build will
# use. Handy first command on a new machine (works on macOS / CPU-only).
.PHONY: detect
detect:
	@python3 tools/detect_resources.py --report

# ── d2l library ────────────────────────────────────────────

lib: d2l/.built

# ── Illustrative figures (committed SVGs; see the mdl-figure skill) ──
# Regenerate every committed img/mdl-*.svg from its generator. The shared house
# style lives in tools/gen_mdl_figures.py (Linear Algebra); each other chapter
# has tools/gen_mdl_<chapter>_figures.py that imports that style. Output is
# byte-idempotent (svg.hashsalt + metadata Date:None), so CI can gate on a clean
# `git diff img/` after running this. Notebooks never draw figures — they
# include these SVGs (see CLAUDE.md "Content authoring").
.PHONY: figures
figures: | .venv-pytorch/.synced
	@.venv-pytorch/bin/python tools/gen_mdl_figures.py
	@for g in $$(ls tools/gen_mdl_*_figures.py 2>/dev/null); do \
		echo "=== $$g ==="; .venv-pytorch/bin/python $$g || exit 1; \
	done
	@echo "Figures regenerated into img/ (verify: git diff --stat img/)"

# ── Committed outputs store (decoupled build; see docs/build-system.md) ──
# capture-outputs   distill executed _notebooks/ → committed outputs/ store
# audit-outputs     report stale notebooks (code-provenance drift) + integrity
# verify-outputs-fresh  render gate: fail on stale INLINE outputs / orphaned ids
# refresh-stale     re-execute ONLY stale notebooks, then re-capture
# render-fresh      refresh-stale, then rebuild slides + html + pdfs

capture-outputs:
	@mkdir -p $(LOGDIR)
	python3 tools/capture_outputs.py $(if $(FILES),FILES=$(FILES)) \
		2>&1 | tee $(LOGDIR)/capture-outputs-$(TS).log

audit-outputs:
	@python3 tools/audit_outputs.py

verify-outputs-fresh:
	@python3 tools/audit_outputs.py --verify-fresh

# Re-execute only what the audit reports stale, then re-capture those files.
# Execution goes through the unified scheduler (parallel GPU/CPU dispatch),
# NOT a serial per-notebook `make -B` loop — the old loop ran one notebook at a
# time with the GPU pool mostly idle. The scheduler is filtered to the stale set
# (`--files`) and, crucially, is NOT forced: a notebook that is stale-in-store
# but already executed (its .ipynb up to date, only never captured) is skipped
# by the scheduler and picked up by the single capture pass below — so this is a
# fast capture-only path when nothing actually needs a GPU re-run. `notebooks`
# is a prereq so .ipynb reflect current source before staleness is judged (the
# regen is mtime-conservative — unchanged notebooks are not re-executed).
refresh-stale: notebooks
	@mkdir -p $(LOGDIR)
	@stale="$$(python3 tools/audit_outputs.py --stale)"; \
	if [ -z "$$stale" ]; then echo "Nothing stale — store is fresh."; exit 0; fi; \
	echo "Stale (will re-execute if needed, then capture):"; echo "$$stale" | sed 's/^/  /'; \
	$(SCHED_ENV) python3 tools/notebook_scheduler.py --files "$$stale" \
		2>&1 | tee -a $(LOGDIR)/scheduler-$(TS).log; rc=$$?; \
	$(MAKE) capture-outputs FILES="$$(echo $$stale)"; \
	exit $$rc

# Fast recovery render: refresh stale outputs, then rebuild the artifacts with
# the SAME RAM-aware parallel fleet as `rebuild-book-artifacts` (RENDER_JOBS
# frameworks at once, each quarto capped to RENDER_V8_HEAP_MIB). The old form ran
# `$(MAKE) slides` with no -j, rendering the four frameworks one after another.
render-fresh: refresh-stale
	$(RENDER_V8_ENV) $(MAKE) -j$(RENDER_JOBS) slides
	$(MAKE) html
	$(RENDER_V8_ENV) $(MAKE) -j$(RENDER_JOBS) pdfs

# Grouped target (GNU Make 4.3+ `&:`): one invocation produces all four
# d2l/<fw>.py files plus the d2l/.built stamp. build_lib.py is content-
# aware: it only rewrites a d2l/<fw>.py whose content actually changed,
# so unchanged frameworks keep their mtime and don't invalidate downstream
# per-notebook .executed stamps.
d2l/.built d2l/pytorch.py d2l/tensorflow.py d2l/jax.py d2l/mxnet.py &: \
        $(SRC_MDS) tools/build_lib.py tools/d2l_preprocess.py
	@mkdir -p $(LOGDIR)
	@echo "=== Building d2l library ==="
	python3 tools/build_lib.py $(SOURCE) d2l/ 2>&1 | tee $(LOGDIR)/lib-$(TS).log
	@touch d2l/.built

# ── UV venvs ───────────────────────────────────────────────
#
# Preflight runs before every `uv sync`: if pyproject pins the mxnet wheel
# at a missing file path, `uv sync` for ANY framework fails during lock
# validation (uv walks all path-direct sources, even those in conflicting
# extras). The preflight is a fast no-op when the pin matches disk; when
# the wheel is missing it auto-bumps the pin to the newest `../mxnet/dist/`
# wheel and relocks. See tools/preflight_mxnet_pin.py for the trace.
.venv-%/.synced: pyproject.toml uv.lock | .preflight.mxnet-pin
	@mkdir -p $(LOGDIR)
	@echo "=== Syncing venv for $* ==="
	UV_PROJECT_ENVIRONMENT=.venv-$* uv sync --extra $* --extra run 2>&1 | tee $(LOGDIR)/venv-$*-$(TS).log
	@touch $@

# TensorFlow venv needs an extra post-sync step: the TF 2.21 wheel's RUNPATH
# omits nvidia/cusolver/lib, so TF silently falls back to CPU. Applying the
# cusolver symlinks at sync time means a fresh `uv sync` doesn't leave the venv
# in a CPU-only state. See tools/check_runtime_deps.py for the gory detail.
.venv-tensorflow/.synced: pyproject.toml uv.lock tools/check_runtime_deps.py | .preflight.mxnet-pin
	@mkdir -p $(LOGDIR)
	@echo "=== Syncing venv for tensorflow ==="
	UV_PROJECT_ENVIRONMENT=.venv-tensorflow uv sync --extra tensorflow --extra run 2>&1 | tee $(LOGDIR)/venv-tensorflow-$(TS).log
	@echo "=== Applying TF cusolver RUNPATH workaround ==="
	@python3 tools/check_runtime_deps.py tensorflow
	@touch $@

# Order-only preflight target. Always runs, but only mutates the project
# when the pinned mxnet wheel is missing on disk.
.PHONY: .preflight.mxnet-pin
.preflight.mxnet-pin:
	@python3 tools/preflight_mxnet_pin.py

# Dedicated venv that installs the Quarto CLI (via the quarto-cli PyPI package).
# Used by HTML and PDF recipes so `quarto` is a real declared dependency rather
# than an implicit system tool. Override-style rule for the `build` extra:
# `uv sync --extra build` is enough; no `--extra run` (no jupyter needed).
.venv-build/.synced: pyproject.toml uv.lock
	@mkdir -p $(LOGDIR)
	@echo "=== Syncing build venv (quarto-cli) ==="
	UV_PROJECT_ENVIRONMENT=.venv-build uv sync --extra build 2>&1 | tee $(LOGDIR)/venv-build-$(TS).log
	@touch $@

# Resolved path to quarto: prefer the project-local build venv when present.
QUARTO := .venv-build/bin/quarto

venv-%: .venv-%/.synced
	@echo "Venv .venv-$* is ready"

.venv-mxnet/.runtime-deps: .venv-mxnet/.synced tools/check_runtime_deps.py
	@echo "=== Checking MXNet native runtime deps ==="
	@python3 tools/check_runtime_deps.py mxnet
	@touch $@

RUNTIME_DEPS_mxnet := .venv-mxnet/.runtime-deps

# Pull the latest mxnet wheel URL from
# https://github.com/smolix/mxnet/releases/latest into pyproject.toml.
# Manual (not on every sync) so builds stay reproducible and we don't
# hit the unauthenticated GitHub API rate limit.
.PHONY: update-mxnet-wheel
update-mxnet-wheel:
	python3 tools/update_mxnet_wheel.py
	@echo "Run 'make venv-mxnet' to apply the bumped wheel."

# ── Jupyter kernels ────────────────────────────────────────
# Register one ipykernel per framework so VS Code can auto-select the
# right interpreter from the .ipynb's metadata.kernelspec.name.

.PHONY: kernels
kernels: $(addprefix .venv-,$(addsuffix /.synced,$(FRAMEWORKS)))
	@for fw in $(FRAMEWORKS); do \
	  echo "Registering kernel d2l-$$fw"; \
	  .venv-$$fw/bin/python -m ipykernel install --user \
	    --name d2l-$$fw --display-name "d2l ($$fw)"; \
	done
	@echo "All d2l kernels registered."

# ── Landing-page university grid ───────────────────────────
# tools/universities.json is the single source of truth for the logo grid on
# the landing page. The pipeline:
#   1. ../data/uni_evidence/UNIVERSITIES.tsv — consolidated course-evidence rows
#      (rebuilt by ../data/uni_evidence/_consolidate.py from per-region .md files)
#   2. tools/build_universities_json.py — merges TSV with existing logo files
#   3. tools/download_logos.py            — fetches missing logos (Wikipedia API,
#                                            incremental: rerun resumes)
#   4. tools/render_logo_grid.py          — emits <a href="..."><img></a> block
#                                            into index.md between markers
#
# Add new course evidence by appending to a region .md under
# /home/smola/d2l/data/uni_evidence/ then `make universities`.

.PHONY: universities universities-rebuild
universities:
	@cd /home/smola/d2l/data/uni_evidence && python3 _consolidate.py >/dev/null
	python3 tools/build_universities_json.py
	python3 tools/download_logos.py
	python3 tools/render_logo_grid.py

# Force re-fetch of all logos (slow; honors Wikipedia rate limits).
universities-rebuild:
	rm -f tools/universities.json
	$(MAKE) universities

# ── HTML book ──────────────────────────────────────────────

html: _book/index.html
	@echo "Output: _book/index.html"
	@echo "Log:    $(LOGDIR)/html-$(TS).log"

# Stage 1: preprocess d2l-en .md → .qmd + generate API docs
.preprocess.stamp: $(SRC_MDS) tools/d2l_preprocess.py tools/gen_api_doc.py d2l/.built
	@echo "=== Preprocessing .md → .qmd ==="
	python3 tools/d2l_preprocess.py $(SOURCE) . --primary pytorch
	python3 tools/gen_api_doc.py
	@touch $@

# Stage 2+3+4: inject (optional) + slides manifest + quarto render + fix numbering
_book/index.html: .preprocess.stamp _quarto.yml _d2l-theme.scss _d2l-style.css _d2l-tabs.html d2l.bib | .venv-build/.synced
	@mkdir -p $(LOGDIR)
	@echo "=== Verifying committed outputs are fresh ==="
	@python3 tools/audit_outputs.py --verify-fresh || \
		{ echo "Outputs stale — re-run + 'make capture-outputs', or 'make render-fresh'."; exit 1; }
	@echo "=== Building HTML book ==="
	@{ \
		if [ -d outputs ] || [ -d _notebooks ]; then \
			echo "Injecting notebook outputs (store preferred, _notebooks fallback)..."; \
			python3 tools/inject_outputs.py html; \
		fi; \
		echo "Building slides manifest (TOC button + landing page)..."; \
		python3 tools/build_slides_index.py; \
		"$(CURDIR)/$(QUARTO)" render --to html; \
		python3 tools/fix_crossref_numbers.py .; \
		python3 tools/add_cfasync.py _book; \
		if [ -d _slides ] && [ -f _slides/index.html ]; then \
			echo "Integrating _slides/ → _book/slides/ ..."; \
			rm -rf _book/slides; \
			mkdir -p _book/slides; \
			rsync -a --exclude='*.qmd' --exclude='_quarto.yml' \
				--exclude='.gitignore' --exclude='.quarto/' --exclude='errors/' \
				_slides/ _book/slides/; \
			echo "Stripping per-fw data/img symlinks (R2 storage bloat)..."; \
			find _book/slides -mindepth 2 -maxdepth 2 -type l \
				\( -name data -o -name img \) -delete; \
			echo "Rewriting deck '../img/' refs to '../../../img/' (single-source)..."; \
			find _book/slides -mindepth 3 -maxdepth 3 -name '*.html' \
				-exec perl -i -pe 's|src="\.\./img/|src="../../../img/|g' {} +; \
		fi; \
		if [ -d _pdf ]; then \
			echo "Staging PDFs into _book/pdf/ ..."; \
			mkdir -p _book/pdf; \
			for fw in $(FRAMEWORKS); do \
				src="_pdf/$$fw/_pdf/Dive-into-Deep-Learning-$$fw.pdf"; \
				[ -f "$$src" ] && cp "$$src" _book/pdf/ || true; \
			done; \
		fi; \
	} 2>&1 | tee $(LOGDIR)/html-$(TS).log

# ── Notebooks (generate) ──────────────────────────────────
# d2l/.built is order-only (|) here: gen_notebooks.py doesn't import d2l,
# so d2l/.built's mtime never needs to invalidate .generated. Making it a
# hard prereq caused a re-run: after sub-makes spawned, $(wildcard
# d2l/_blocks/*/MANIFEST.mk) resolved to a larger set than at the initial
# parse (MANIFEST.mk files didn't exist yet), making d2l/.built appear stale
# in sub-make evaluation. That advanced d2l/.built's mtime past .generated,
# causing gen_notebooks to re-fire mid-execution and wipe completed notebooks.

# Per-framework symlinks (img/data) — one-time setup, decoupled from
# notebook content gen. Was previously bundled into the .generated recipe,
# which meant a forced .generated rebuild would re-do this no-op too.
_notebooks/%/.symlinks:
	@mkdir -p _notebooks/$* data
	@[ -e "_notebooks/$*/img" ] || ln -s ../../img "_notebooks/$*/img"
	@if [ -L "_notebooks/$*/data" ]; then :; \
	elif [ -d "_notebooks/$*/data" ]; then \
		cp -rn "_notebooks/$*/data/." data/ 2>/dev/null || true; \
		rm -rf "_notebooks/$*/data"; \
		ln -s "$$(pwd)/data" "_notebooks/$*/data"; \
	else \
		ln -s "$$(pwd)/data" "_notebooks/$*/data"; \
	fi
	@touch $@

# Per-notebook .ipynb rule. A single source .md change rebuilds only the
# matching `_notebooks/<fw>/<chapter>/<file>.ipynb` for each framework that
# has that file in its MANIFEST — no per-framework batch wipe. gen_notebooks
# is called with --files for just this source; the in-script per-cell output
# merge restores outputs from the existing .ipynb when code is unchanged
# (e.g. a prose-only edit doesn't invalidate execution).
#
# d2l/.built and .venv-build/.synced are order-only: their mtimes shouldn't
# trigger a regen, but they must exist when the recipe runs.
# Per-notebook .ipynb rule. Used for **direct** invocation:
#   make _notebooks/pytorch/chapter_x/foo.ipynb
# i.e. an incremental edit where the user wants just one notebook
# regenerated. Each call spawns a ~0.5 s gen_notebooks startup; that's fine
# for one file but death-by-1000-cuts when N files are stale. For full
# rebuilds, the .generated batch rule below runs ONE gen_notebooks for the
# entire framework (~5 s for 139 notebooks vs ~70 s per-file × -j8).
#
# d2l/.built and .venv-build/.synced are order-only: their mtimes shouldn't
# trigger a regen, but they must exist when the recipe runs.
define IPYNB_RULE
_notebooks/$(1)/%.ipynb: %.md tools/gen_notebooks.py tools/d2l_preprocess.py tools/build_lib.py | d2l/.built _notebooks/$(1)/.symlinks .venv-build/.synced
	@mkdir -p $$(@D) $(LOGDIR)
	@PATH="$(CURDIR)/.venv-build/bin:$$$$PATH" \
	python3 tools/gen_notebooks.py $(SOURCE) _notebooks --convert --frameworks $(1) --files $$< 2>&1 | tee -a $(LOGDIR)/notebooks-$(1)-$(TS).log
endef
$(foreach fw,$(FRAMEWORKS),$(eval $(call IPYNB_RULE,$(fw))))

# Batch path: a single gen_notebooks invocation rebuilds every stale
# .ipynb for the framework. Touched as a stamp file `.generated`; when it
# runs, every .ipynb it produces gets a fresh mtime so the per-file rule
# (above) sees them as up-to-date and won't re-fire downstream.
#
# Used by `notebooks-<fw>` and as a transitive prereq of `run-notebooks-*`,
# so a full build stays fast. The per-file rule is still the canonical
# path for `make _notebooks/<fw>/foo.ipynb`-style direct invocations.
_notebooks/%/.generated: $(SRC_MDS) tools/gen_notebooks.py tools/d2l_preprocess.py tools/build_lib.py | d2l/.built _notebooks/%/.symlinks .venv-build/.synced
	@mkdir -p $(LOGDIR)
	@echo "=== Generating $* notebooks (batch) ==="
	@PATH="$(CURDIR)/.venv-build/bin:$$PATH" \
	python3 tools/gen_notebooks.py $(SOURCE) _notebooks --convert --frameworks $* 2>&1 | tee $(LOGDIR)/notebooks-$*-$(TS).log
	@touch $@

# `notebooks-<fw>` uses the batch path (fast) and also reaches each
# per-notebook .ipynb so direct dependents see the same set Make would
# build incrementally.
notebooks-%: _notebooks/%/.generated _notebooks/%/.symlinks
	@echo "Notebooks for $* in _notebooks/$*/ ($(words $(IPYNB_$*)) notebooks)"

notebooks: $(addprefix notebooks-,$(FRAMEWORKS))

# ── Notebooks (execute) — per-notebook granularity ────────
# Each .ipynb has its own .executed stamp. Editing one source .md (or one
# d2l/<fw>.py block) only invalidates the affected notebook(s). Aggregate
# targets split those stamps into CPU, single-GPU, and multi-GPU queues so
# Make admits work according to the resource it needs. The slot locks inside
# tools/run_one_notebook.py remain as a backstop for direct stamp builds.

define nvidia_ld_path
$(shell find .venv-$(1)/lib -path "*/nvidia/*/lib" -type d 2>/dev/null | paste -sd: -)
endef

# Per-framework manifest lists every _notebooks/<fw>/<rel>.executed target
# for that framework, split by execution resource. We use a CHEAP textual scan
# (tools/scan_notebook_manifests.py, ~1s for the whole tree) instead of the
# full gen_notebooks pipeline (~30s per framework) so `-include` doesn't
# trigger expensive work during Makefile parsing. The scan looks at #@tab /
# %%tab markers to know which (fw, src) pairs produce a notebook, then scans
# source text for GPU hints to classify the produced notebook.
$(addsuffix /MANIFEST.mk,$(addprefix _notebooks/,$(FRAMEWORKS))) &: \
        $(SRC_MDS) tools/scan_notebook_manifests.py
	@mkdir -p $(addprefix _notebooks/,$(FRAMEWORKS))
	@python3 tools/scan_notebook_manifests.py \
		--source $(SOURCE) --output-dir _notebooks

-include $(addsuffix /MANIFEST.mk,$(addprefix _notebooks/,$(FRAMEWORKS)))

# Per-notebook .d files capture each notebook's actual `d2l.<symbol>`
# usage and translate it into shard-file dependencies. With these, editing
# one `#@save` block invalidates only the notebooks that reference its
# symbol — not every notebook in the framework. The scan is fast (~1s)
# and runs whenever a source .md or the shard set changes.
DEP_FILES := $(foreach fw,$(FRAMEWORKS),$(patsubst %.executed,%.d,$(EXECUTED_$(fw))))

# A single rule emits every .d file (grouped output for one invocation).
# Depends on the source .md set so per-notebook .d files refresh when a
# notebook adds/removes a d2l.X reference, AND on the per-framework shard
# manifests so we pick up new/removed symbols.
$(DEP_FILES) &: $(SRC_MDS) tools/scan_d2l_usage.py \
        $(wildcard d2l/_blocks/*/MANIFEST.mk)
	@python3 tools/scan_d2l_usage.py \
		--source $(SOURCE) --output-dir _notebooks --shard-dir d2l/_blocks

-include $(DEP_FILES)

# Per-notebook execution pattern rule. The recipe runs ONE notebook via
# tools/run_one_notebook.py, which handles GPU slot locking (flock-based)
# and best-of-N retries for known-stochastic notebooks.
#
# Content deps come from the per-notebook `.d` file, which lists the exact
# `d2l/_blocks/<fw>/<symbol>.py` shards the notebook consumes (see
# tools/scan_d2l_usage.py). d2l/<fw>.py is order-only (must exist at run
# time, since notebooks `import d2l`, but its mtime doesn't trigger
# rebuilds — that's the .d's job).
# ── Per-framework execution env ────────────────────────────────────────
# Per-framework concurrency caps. Derived from each framework's footprint
# (GPU_MIB_PER_<fw> / CPU_PER_<fw>) so a heavier framework gets fewer
# slots than the global pool. Today only JAX is heavier than "light";
# caps for other frameworks would just equal the global pool, which is
# the same as "no cap" — `run_one_notebook.py` treats 0/unset as no cap.
$(foreach v,JAX_GPU_SLOTS JAX_CPU_SLOTS,$(eval $(call _detected,$(v))))

# Memory-and-thread-conservative defaults for JAX. PREALLOCATE=false: don't
# grab 75 % of GPU on init (1.6 GiB instead of ~19 GiB resident). OMP/BLAS=2:
# trims the unrelated "python" OpenMP pool by ~60 threads.
#
# We do NOT set JAX_PLATFORMS=cuda: nearly every JAX training notebook hits
# d2l.Module.plot(), which calls d2l.cpu() / jax.devices('cpu') to move
# metrics to host for matplotlib. Restricting to cuda makes those notebooks
# fail with "Unknown backend cpu". Leaving the CPU backend in costs ~64
# extra tf_XLAEigen threads (still well under the ulimit headroom).
EXTRA_ENV_jax := XLA_PYTHON_CLIENT_PREALLOCATE=false \
                 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 \
                 D2L_JAX_GPU_SLOTS=$(JAX_GPU_SLOTS) D2L_JAX_CPU_SLOTS=$(JAX_CPU_SLOTS)

# TensorFlow baseline is ~212 threads + ~22.5 GiB resident on the active GPU
# (TF preallocates ~all of VRAM on first device init). The mitigations:
#  * TF_FORCE_GPU_ALLOW_GROWTH=true   — allocate GPU memory on demand
#                                        (23.7 → 1.6 GiB total across 4 GPUs)
#  * TF_NUM_INTRAOP_THREADS / INTEROP_THREADS=2 — collapses the 64-thread
#                                        tf_Compute Eigen pool to 2
#  * OMP/BLAS/MKL=2                    — trims the ~62-thread "python" pool
#  * TF_CPP_MIN_LOG_LEVEL=3            — silence the boot-time TF info logs
# Combined: ~26 threads per TF process, 1.6 GiB GPU total. CPU device stays
# visible and usable (auto-parallelism's CPU benchmark still works), so no
# per-notebook split is needed.
EXTRA_ENV_tensorflow := TF_FORCE_GPU_ALLOW_GROWTH=true \
                        TF_NUM_INTRAOP_THREADS=2 TF_NUM_INTEROP_THREADS=2 \
                        OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 \
                        TF_CPP_MIN_LOG_LEVEL=3
# PyTorch is already lean: ~67 threads default (vs JAX 294, MX 304, TF 212),
# no GPU preallocation, lazy multi-device init, user-controlled allocator
# cache (torch.cuda.empty_cache()). The only oversized pool is `at::Threads`
# (32 OpenMP workers by default = nproc/2). Capping OMP/BLAS/MKL to 2 brings
# the total to ~5 threads with no measurable perf cost on d2l workloads.
EXTRA_ENV_pytorch := OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2

# MXNet 2.0 baseline is ~304 threads per process. Almost all of them come
# from libopenblas / libopencv OpenMP pools pulled in by the custom wheel;
# the `MXNET_*_NTHREADS` env vars empirically have NO effect on this build.
# OMP/BLAS/MKL=2 drops the count to ~56 threads (82 % reduction) with no
# measurable perf hit on tutorial workloads. GPU memory is fine by default
# (no init-time preallocation; pool retention is per-process so it doesn't
# bleed into the next notebook).
#
# Concurrency cap (D2L_MXNET_*_SLOTS, honored by run_one_notebook.py on top of
# the global pool): even at ~56 threads/proc, image-dataset notebooks spawn
# Gluon DataLoader worker subprocesses, and at the default GPU_SLOTS (~2/GPU →
# 8 here) the combined process/thread count exhausts `ulimit -u` (soft 4096 /
# hard 8192 on this host). A starved worker then fails to lazy-`dlopen` the
# bundled OpenCV and surfaces the misleading "Build with USE_OPENCV=1 for image
# resize operator" — even though OpenCV IS compiled in and works in isolation
# (see docs/mxnet-runtime-diagnostics.md, 2026-06-06). The same notebooks pass
# serially. We therefore derive the mxnet GPU cap from the open-file/process
# ulimits (tools/detect_resources.py: nofile//FD_PER_MXNET_JOB etc.) rather than
# a hardcoded 2; the board fix + the verified 3/GPU run make higher concurrency
# safe. Override with MXNET_GPU_SLOTS=N on the command line.
$(eval $(call _detected,MXNET_GPU_SLOTS))
MXNET_CPU_SLOTS ?= 2
# MXNET_CUDNN_LIB_CHECKING=0: the 20260607.1 wheel is compiled against cuDNN
# 9.23 but the only pip-available nvidia-cudnn-cu13 is 9.22 (ABI-compatible —
# conv/matmul verified correct). Without this, every GPU op prints a "cuDNN lib
# mismatch" line to stderr that gets captured into the notebook outputs.
EXTRA_ENV_mxnet := OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 \
                   MXNET_CUDNN_LIB_CHECKING=0 \
                   D2L_MXNET_GPU_SLOTS=$(MXNET_GPU_SLOTS) D2L_MXNET_CPU_SLOTS=$(MXNET_CPU_SLOTS)

define EXEC_RULE
_notebooks/$(1)/%.executed: _notebooks/$(1)/%.ipynb \
        | d2l/$(1).py .venv-$(1)/.synced $$(RUNTIME_DEPS_$(1))
	@mkdir -p $$(@D) $(LOGDIR)
	$$(eval NVIDIA_LIBS := $$(call nvidia_ld_path,$(1)))
	@UV_PROJECT_ENVIRONMENT=.venv-$(1) \
	PATH="$(CURDIR)/.venv-$(1)/bin:$$$$PATH" \
	LD_LIBRARY_PATH="$$(NVIDIA_LIBS)$$$${LD_LIBRARY_PATH:+:$$$$LD_LIBRARY_PATH}" \
	D2L_NUM_GPUS=$(NUM_GPUS) \
	D2L_GPU_SLOTS=$(GPU_SLOTS) \
	D2L_CPU_SLOTS=$(CPU_SLOTS) \
	D2L_NUM_GPU_PAIRS=$(NUM_GPU_PAIRS) \
	D2L_MULTIGPU_PER_PAIR=$(MULTIGPU_PER_PAIR) \
	D2L_JAX_MGPU_MEM_FRACTION=$(JAX_MGPU_MEM_FRACTION) \
	$$(EXTRA_ENV_$(1)) \
	.venv-$(1)/bin/python tools/run_one_notebook.py $(1) $$< \
		2>&1 | tee -a $(LOGDIR)/run-$(1)-$(TS).log
	@touch $$@
endef
$(foreach fw,$(FRAMEWORKS),$(eval $(call EXEC_RULE,$(fw))))

# Resource-specific aggregate targets. The public run-notebooks-* targets
# below invoke these through sub-makes with resource-specific -j limits.
run-notebooks-cpu-%: _notebooks/%/.generated $$(EXECUTED_CPU_$$*)
	@echo "CPU notebooks for $* up to date ($(words $(EXECUTED_CPU_$*)) total)"

run-notebooks-gpu-%: _notebooks/%/.generated $$(EXECUTED_GPU_$$*)
	@echo "GPU notebooks for $* up to date ($(words $(EXECUTED_GPU_$*)) total)"

run-notebooks-multigpu-%: _notebooks/%/.generated $$(EXECUTED_MULTI_GPU_$$*)
	@echo "Multi-GPU notebooks for $* up to date ($(words $(EXECUTED_MULTI_GPU_$*)) total)"

# Aggregate per-framework target. Generate notebooks before spawning resource
# queues so CPU and GPU sub-makes cannot race on _notebooks/<fw>/.generated.
#
# Each sub-make uses -k (--keep-going) so a single failing notebook does not
# stop Make from attempting the rest. The multi-GPU queue runs after the
# single-GPU queue drains regardless of single-GPU failures (`;` not `&&`).
# The aggregate stamp recipe still exits non-zero if any notebook failed, so
# `make all` propagates the failure to its caller. Use the post-run summary
# printed by tools/notebook_run_summary.py to see which notebooks failed.
run-notebooks-%: _notebooks/%/.generated
	@mkdir -p $(LOGDIR)
	@python3 tools/detect_resources.py --report || true
	@python3 tools/notebook_run_plan.py --framework $* || true
	@$(SCHED_ENV) python3 tools/notebook_scheduler.py --frameworks $* \
		2>&1 | tee -a $(LOGDIR)/scheduler-$*-$(TS).log; rc=$${PIPESTATUS[0]}; \
	python3 tools/notebook_run_summary.py --framework $* || true; \
	exit $$rc

# For a single notebook, invoke the stamp target directly:
#   make _notebooks/pytorch/chapter_x/foo.executed

# Execute all frameworks with one global CPU queue and one global GPU queue.
# `notebooks` runs first so CPU/GPU sub-makes cannot generate the same
# framework's notebooks concurrently. `d2l/.built` is order-only in the
# `.generated` rule so a mid-build mtime advance on d2l/.built (e.g. from
# the MANIFEST.mk wildcard settling after the first lib build) never
# invalidates an already-complete `.generated` stamp and re-runs gen_notebooks.
# -k on every sub-make keeps the queues draining even after individual
# notebook failures.
#
# Each queue is fed a single INTERLEAVED flat list of every framework's stamps
# (sorted by the path AFTER the framework, so a notebook's 4 framework variants
# are adjacent) rather than 4 grouped per-framework phony goals. GNU make drains
# one goal's prerequisites before the next, so the grouped form ran the
# frameworks ~sequentially: when a framework's own cap (e.g. JAX_GPU_SLOTS=8) is
# below GPU_SLOTS the leftover slots sat idle, and each framework left a slow-
# straggler tail. Interleaving keeps the GPU pool full with a framework mix
# throughout — one combined tail instead of four.
# Resource env handed to the unified scheduler (tools/notebook_scheduler.py):
# per-GPU slot capacity + per-GPU VRAM (heterogeneous-aware) + CPU slots.
$(foreach v,GPU_SLOTS_PER GPU_VRAM_PER GPU_MIB_PER_SLOT,$(eval $(call _detected,$(v))))
SCHED_ENV = D2L_GPU_SLOTS_PER='$(GPU_SLOTS_PER)' D2L_GPU_VRAM_PER='$(GPU_VRAM_PER)' \
	D2L_GPU_MIB_PER_SLOT=$(GPU_MIB_PER_SLOT) D2L_CPU_SLOTS=$(CPU_SLOTS)

# The unified scheduler (tools/notebook_scheduler.py) replaces the old "two
# background `make -jN` queues" orchestration: it owns the GPU/CPU/multi-GPU
# slot pools (resource allocation) and dispatches one item per free slot while
# never running two framework variants of the same notebook at once (scheduling
# — sequences a notebook's frameworks one at a time, killing the shared-dataset
# reorg race structurally). It shells `make <stamp>` per notebook with the
# device assigned via D2L_ASSIGNED_CUDA, reusing the per-framework EXEC_RULE env.
run-all-notebooks: notebooks
	@mkdir -p $(LOGDIR)
	@python3 tools/detect_resources.py --report || true
	@python3 tools/notebook_run_plan.py || true
	@$(SCHED_ENV) python3 tools/notebook_scheduler.py \
		2>&1 | tee -a $(LOGDIR)/scheduler-$(TS).log; rc=$${PIPESTATUS[0]}; \
	python3 tools/notebook_run_summary.py || true; \
	exit $$rc

# ── PDFs (per-framework, parallel-safe) ───────────────────

_pdf/%/.generated: $(SRC_MDS) tools/gen_pdf.py tools/d2l_preprocess.py tools/build_lib.py
	@mkdir -p $(LOGDIR)
	@echo "=== Generating PDF sources for $* ==="
	@python3 tools/gen_pdf.py $(SOURCE) _pdf/$* --framework $* 2>&1 | tee $(LOGDIR)/pdf-$*-gen-$(TS).log
	@touch $@

# PDF toolchain preflight: XeLaTeX (texlive-xetex) renders the .tex, and Quarto
# shells out to rsvg-convert (librsvg2-bin) for every SVG→PDF. Both are easy to
# be missing on a render box; without this the build failed deep inside quarto
# with a cryptic "_pdf/<fw>/Dive-into-Deep-Learning.tex: No such file" instead
# of naming the missing tool. Order-only prereq of every PDF target, so it runs
# once before the parallel renders.
.PHONY: pdf-preflight
pdf-preflight:
	@command -v xelatex >/dev/null 2>&1 || { echo "ERROR: xelatex not found on PATH — install TeX Live (e.g. apt install texlive-xetex texlive-latex-recommended texlive-fonts-recommended) for PDF builds."; exit 1; }
	@command -v rsvg-convert >/dev/null 2>&1 || { echo "ERROR: rsvg-convert not found on PATH — install librsvg2-bin (Quarto converts SVG→PDF with it)."; exit 1; }
	@echo "PDF toolchain OK: $$(xelatex --version 2>/dev/null | head -1), rsvg-convert $$(rsvg-convert --version 2>/dev/null)"

# Generate per-framework PDF rules (GNU Make only supports one % per pattern)
define PDF_RULE
_pdf/$(1)/_pdf/Dive-into-Deep-Learning-$(1).pdf: _pdf/$(1)/.generated | .venv-build/.synced pdf-preflight
	@mkdir -p $(LOGDIR)
	@echo "=== Building PDF ($(1)) ==="
	@{ \
		if [ -d _notebooks ]; then \
			echo "Injecting notebook outputs for $(1)..."; \
			python3 tools/inject_outputs.py pdf --framework $(1) --pdf-dir _pdf/$(1); \
		fi; \
		count=0; for svg in _pdf/$(1)/img/*.svg; do \
			[ -f "$$$$svg" ] || continue; \
			pdf="$$$${svg%.svg}.pdf"; \
			if [ ! -f "$$$$pdf" ] || [ "$$$$svg" -nt "$$$$pdf" ]; then \
				rsvg-convert -f pdf -o "$$$$pdf" "$$$$svg" 2>/dev/null && count=$$$$((count + 1)); \
			fi; \
		done; [ $$$$count -gt 0 ] && echo "Converted $$$$count SVGs to PDF" || true; \
		cd _pdf/$(1) && "$(CURDIR)/$(QUARTO)" render --to pdf; \
		cd "$(CURDIR)"; \
		python3 tools/fix_latex.py _pdf/$(1)/Dive-into-Deep-Learning.tex; \
		cd _pdf/$(1) && xelatex -interaction=nonstopmode Dive-into-Deep-Learning.tex > /dev/null 2>&1; \
		xelatex -interaction=nonstopmode Dive-into-Deep-Learning.tex > /dev/null 2>&1; \
		xelatex -interaction=nonstopmode Dive-into-Deep-Learning.tex > /dev/null 2>&1; \
		cd "$(CURDIR)"; \
		: 'Publish the fix_latex-patched PDF (hierarchical chapter/section'; \
		: 'numbering) — NOT quartos own pre-fix compile under _pdf/$(1)/_pdf/.'; \
		if [ -f _pdf/$(1)/Dive-into-Deep-Learning.pdf ]; then \
			mkdir -p _pdf/$(1)/_pdf; \
			mv -f _pdf/$(1)/Dive-into-Deep-Learning.pdf _pdf/$(1)/_pdf/Dive-into-Deep-Learning-$(1).pdf; \
		fi; \
		if [ -f _pdf/$(1)/_pdf/Dive-into-Deep-Learning-$(1).pdf ]; then \
			mkdir -p _book/pdf; \
			cp _pdf/$(1)/_pdf/Dive-into-Deep-Learning-$(1).pdf _book/pdf/; \
		fi; \
	} 2>&1 | tee $(LOGDIR)/pdf-$(1)-$(TS).log

pdf-$(1): _pdf/$(1)/_pdf/Dive-into-Deep-Learning-$(1).pdf
	@echo "Output: $$<"
	@echo "Log:    $(LOGDIR)/pdf-$(1)-$(TS).log"
endef

$(foreach fw,$(FRAMEWORKS),$(eval $(call PDF_RULE,$(fw))))

pdf: pdf-pytorch
pdfs: $(addprefix pdf-,$(FRAMEWORKS))

# ── Slides (per-framework) ────────────────────────────────
# CPU-only: gen_slides.py renders with --no-execute and injects code outputs
# from the committed store, so it needs only Quarto + nbformat (.venv-build) —
# never a framework/CUDA venv (it falls back to .venv-build for QUARTO_PYTHON).
# This is why slides build on a render-only host (e.g. macOS) and are
# parallel-safe across frameworks (see `slides:` below).

_slides/%/.built: $(SRC_MDS) tools/gen_slides.py tools/d2l_preprocess.py tools/build_lib.py | .venv-build/.synced
	@mkdir -p $(LOGDIR)
	@echo "=== Building $* slides ==="
	@# pipefail is global (.SHELLFLAGS) so gen_slides.py's non-zero exit on a
	@# failed deck render propagates through the `| tee` and aborts the recipe
	@# before `touch $@` — a failed slide build no longer stamps .built.
	PATH="$(CURDIR)/.venv-build/bin:$$PATH" \
	python3 tools/gen_slides.py $(SOURCE) _slides --frameworks $* \
		--render --workers 16 \
		$(if $(SLIDES_FILTER),--files $(SLIDES_FILTER)) \
		2>&1 | tee $(LOGDIR)/slides-$*-$(TS).log
	@touch $@

slides-%: _slides/%/.built
	@echo "Slides for $* in _slides/$*/"
	@echo "Log: $(LOGDIR)/slides-$*-$(TS).log"

# CPU-only after the slide refactor — parallel-safe across frameworks.
slides: $(addprefix slides-,$(FRAMEWORKS))

# ── Downloadable notebook zips (per framework) ────────────
# A first-class build output, linked from the navbar "Notebooks" menu: one
# d2l-<fw>.zip of that framework's executed notebooks. CPU-only / GPU-free, like
# PDFs and slides — the *code* comes from the generated _notebooks/ tree and the
# *outputs* are injected from the committed store (tools/build_notebook_zips.py),
# so it never needs a framework venv. Deterministic (fixed zip timestamps) so an
# unchanged build re-produces byte-identical zips and the R2 sync skips them.
NOTEBOOK_ZIP_DIR := _book/notebooks
.PHONY: notebook-zips
notebook-zips: notebooks
	@mkdir -p $(LOGDIR) $(NOTEBOOK_ZIP_DIR)
	@echo "=== Building per-framework notebook zips → $(NOTEBOOK_ZIP_DIR)/ ==="
	python3 tools/build_notebook_zips.py --out-dir $(NOTEBOOK_ZIP_DIR) \
		$(if $(FRAMEWORKS_FILTER),--frameworks $(FRAMEWORKS_FILTER)) \
		2>&1 | tee $(LOGDIR)/notebook-zips-$(TS).log

# ── Aggregate targets ─────────────────────────────────────

# Full pipeline: generate → execute → rebuild with outputs
all:
	$(MAKE) lib
	$(MAKE) notebooks
	@nb_rc=0; \
	$(MAKE) run-all-notebooks || nb_rc=$$?; \
	$(MAKE) rebuild-book-artifacts || exit $$?; \
	if [ $$nb_rc -ne 0 ]; then \
		echo "ERROR: notebook execution failed; rebuilt slides, HTML, and PDFs with available outputs."; \
		exit $$nb_rc; \
	fi

rebuild-book-artifacts:
	@echo "=== Rebuilding slides, HTML, and PDFs with current notebook outputs ==="
	@# Fail fast if the committed output images are unmaterialized Git-LFS
	@# pointers — slides render before the html verify-fresh gate, so without
	@# this they'd bake broken figures before anything noticed. (`git lfs pull`)
	@python3 tools/audit_outputs.py --check-lfs
	@rm -f $(SLIDE_STAMPS)
	@rm -f _book/index.html
	@for fw in $(FRAMEWORKS); do rm -f "_pdf/$$fw/_pdf/Dive-into-Deep-Learning-$$fw.pdf"; done
	@# Render slides/PDFs with a RAM-aware fleet: RENDER_JOBS frameworks in
	@# parallel, each quarto capped to RENDER_V8_HEAP_MIB. A hardcoded `-j4` at
	@# the old 24 GiB heap over-committed RAM on <128 GiB boxes (4×24=96 GiB →
	@# OOM-kill rc=137), and 24 GiB was too small for one framework's full
	@# single-project render anyway (V8 abort rc=133). See detect_resources.py.
	$(RENDER_V8_ENV) $(MAKE) -j$(RENDER_JOBS) slides
	$(MAKE) html
	@# Notebook download bundles — after html so quarto's render can't wipe
	@# _book/notebooks/; CPU-only, injects outputs from the committed store.
	$(MAKE) notebook-zips
	@# Clear stale render-scratch PDFs so the parallel PDF render can't
	@# skip-then-read a corrupt one (Quarto convert_svg, main.lua:7348). This
	@# makes `make all` self-sufficient without a preceding `make clean`.
	@find img/outputs -name '*.pdf' -delete 2>/dev/null || true
	$(RENDER_V8_ENV) $(MAKE) -j$(RENDER_JOBS) pdfs
	$(MAKE) check-all-artifacts

# Quick build without notebook execution
all-quick:
	$(MAKE) lib
	$(MAKE) notebooks
	$(MAKE) rebuild-book-artifacts

check-all-artifacts:
	@test -f _slides/index.html || { echo "ERROR: missing _slides/index.html"; exit 1; }
	@# Deep gate: core pages + zero broken (LFS-pointer) images + per-framework
	@# PDFs + full per-framework slide-deck coverage. Replaces the old
	@# "two index files exist" check that let OOM-dropped decks and broken
	@# images ship silently.
	@python3 tools/check_book_artifacts.py

# ── Clean ──────────────────────────────────────────────────

# `clean` preserves expensive state: ./data/ (downloaded datasets),
# logs/, and .upload-manifest-*.txt (sha256 manifest of the last R2
# upload — losing it forces a full re-upload). Use `veryclean` to wipe
# those too.
clean:
	rm -rf _book _pdf _notebooks _slides
	rm -rf d2l/_blocks
	rm -f img/*.pdf .preprocess.stamp d2l/.built _d2l-slides-data.html
	# img/outputs/ is render scratch; stale *.pdf there make Quarto's
	# convert_svg skip-then-read an old/corrupt PDF and abort the PDF build
	# (main.lua:7348 assertion). Clear them so PDFs always rebuild clean.
	find img/outputs -name '*.pdf' -delete 2>/dev/null || true
	rm -f $(wildcard _notebooks/*/.generated _notebooks/*/.executed)
	rm -f $(wildcard _notebooks/*/MANIFEST.mk)
	rm -f $(wildcard _pdf/*/.generated)
	rm -f $(wildcard _slides/*/.built)
	@echo "Cleaned build artifacts (kept ./data/, $(LOGDIR)/, and .upload-manifest-*.txt)"

veryclean: clean
	rm -rf data $(LOGDIR)
	rm -f .upload-manifest-*.txt
	@echo "Also deleted ./data/, $(LOGDIR)/, and .upload-manifest-*.txt"
