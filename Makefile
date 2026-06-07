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
# GPU slots are auto-detected from nvidia-smi (rule of thumb: 11GB per job,
# so a 24GB GPU runs 2 in parallel). Override on the command line:
# `make NUM_GPUS=2 GPU_SLOTS=4 CPU_SLOTS=2`.
#
# All build output is logged to logs/<target>-YYYYMMDD-HHMMSS.log

SHELL      := /bin/bash
.SHELLFLAGS := -o pipefail -c
.DEFAULT_GOAL := help

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

# Auto-detect GPU count and per-GPU memory via nvidia-smi.
# Returns "<num_gpus> <min_mib>". We use the *minimum* memory across
# detected GPUs (workers_per_gpu must be uniform — see run_one_notebook.py).
# Falls back to "0 0" when no GPUs are visible (CPU-only host).
_GPU_QUERY := $(shell nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | awk 'BEGIN{n=0;m=0} {n++; if(n==1||$$1<m) m=$$1} END{if(n==0) print "0 0"; else print n, m}')
DETECTED_NUM_GPUS := $(word 1,$(_GPU_QUERY))
DETECTED_GPU_MIB  := $(word 2,$(_GPU_QUERY))
# If no GPU was detected, fall back to NUM_GPUS=1 for slot-to-device mapping.
NUM_GPUS ?= $(if $(filter 0,$(DETECTED_NUM_GPUS)),1,$(DETECTED_NUM_GPUS))

# ── Per-framework resource footprint ──────────────────────────────────
# How much of a GPU / how many CPU cores a single notebook of each
# framework consumes at peak. Used to derive slot counts:
#   GPU_SLOTS    = NUM_GPUS × (min_GPU_MiB / GPU_MIB_PER_LIGHT)
#   CPU_SLOTS    = nproc / CPU_PER_LIGHT
#   <FW>_*_SLOTS = ... using the framework's own footprint
# All overridable on the command line: `make GPU_MIB_PER_jax=14000 ...`.
#
# PyTorch / TensorFlow / MXNet share the lighter 7.5 GiB / 10 cores
# budget (their EXTRA_ENV_* tuning caps resident VRAM and threads).
# JAX is heavier (~11.5 GiB / 15 cores) because every JAX process keeps
# the CPU backend + XLAEigen pool alive — d2l.Module.plot calls
# d2l.cpu() to host-transfer metrics, so we can't restrict to cuda only.
GPU_MIB_PER_LIGHT  ?= 7680
GPU_MIB_PER_jax    ?= 11776
CPU_PER_LIGHT      ?= 8
CPU_PER_jax        ?= 8

# Slot helpers. Each evaluates to "max(1, n*m / per)" using shell arith,
# falling back to 1 on CPU-only hosts where DETECTED_GPU_MIB=0.
_gpu_slots = $(shell n=$(NUM_GPUS); m=$(DETECTED_GPU_MIB); per=$(1); \
    if [ "$$m" -lt "$$per" ] || [ "$$per" -le 0 ]; then echo 1; \
    else echo $$(( n * m / per )); fi)
_cpu_slots = $(shell n=$$(nproc 2>/dev/null || echo 16); per=$(1); \
    if [ "$$per" -le 0 ]; then echo 1; \
    else x=$$(( n / per )); if [ "$$x" -lt 1 ]; then echo 1; else echo $$x; fi; fi)

# Global pool sized for the lightest framework's footprint. `make -j`
# should be at least GPU_SLOTS + CPU_SLOTS for full saturation.
# On this 4×24GB / 64-core box: GPU_SLOTS=12, CPU_SLOTS=6.
GPU_SLOTS ?= $(call _gpu_slots,$(GPU_MIB_PER_LIGHT))
CPU_SLOTS ?= $(call _cpu_slots,$(CPU_PER_LIGHT))

# Multi-GPU notebooks use exactly 2 GPUs at <=GPU_MIB_PER_LIGHT each, so they
# are memory-packed onto disjoint GPU pairs (verified: 3 fit on a 24 GiB card
# across all 4 frameworks). tools/detect_resources.py is the single source for
# the packing plan (pairs × per_pair) and the jax preallocation fraction.
NUM_GPU_PAIRS         ?= $(shell python3 tools/detect_resources.py --get NUM_GPU_PAIRS)
MULTIGPU_PER_PAIR     ?= $(shell python3 tools/detect_resources.py --get MULTIGPU_PER_PAIR)
MULTIGPU_SLOTS        ?= $(shell python3 tools/detect_resources.py --get MULTIGPU_SLOTS)
JAX_MGPU_MEM_FRACTION ?= $(shell python3 tools/detect_resources.py --get JAX_MGPU_MEM_FRACTION)
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
	@echo "Targets:"
	@echo "  html                    Build HTML book (multi-framework tabs)"
	@echo "  pdf-<fw>               Build PDF for one framework"
	@echo "  pdfs                    Build PDFs for all frameworks (safe with -j4)"
	@echo "  pdf                     Alias for pdf-pytorch"
	@echo "  notebooks-<fw>         Generate notebooks for one framework"
	@echo "  notebooks               Generate notebooks for all frameworks"
	@echo "  run-notebooks-<fw>     Execute notebooks for one framework (CPU/GPU queues)"
	@echo "  run-all-notebooks       Execute all frameworks (CPU/GPU queues)"
	@echo "  capture-outputs         Bless executed _notebooks/ → committed outputs/ store [FILES=...]"
	@echo "  audit-outputs           Report stale notebooks (code drift) + store integrity"
	@echo "  verify-outputs-fresh    Render gate: fail on stale inline outputs / orphaned ids"
	@echo "  refresh-stale           Re-execute only stale notebooks, then re-capture"
	@echo "  render-fresh            refresh-stale, then rebuild slides + html + pdfs"
	@echo "  slides-<fw>            Build slides for one framework (CPU)"
	@echo "  slides                  Build slides for all frameworks"
	@echo "  lib                     Build d2l Python package"
	@echo "  venv-<fw>              Sync UV environment for one framework"
	@echo "  kernels                 Register d2l-<fw> ipykernels (for VS Code)"
	@echo "  clean                   Remove build artifacts (keep data/)"
	@echo "  veryclean               Remove everything including data/"
	@echo "  all                     Full pipeline: generate, execute, rebuild with outputs"
	@echo "  all-quick               Build html + pdfs + notebooks + slides + lib (no execution)"
	@echo ""
	@echo "Variables:  SOURCE=$(SOURCE)  NUM_GPUS=$(NUM_GPUS)"
	@echo "           GPU_SLOTS=$(GPU_SLOTS)  CPU_SLOTS=$(CPU_SLOTS)"
	@echo "           FILES=$(FILES)  SLIDES_FILTER=$(SLIDES_FILTER)"
	@echo "Frameworks: $(FRAMEWORKS)"
	@echo "Logs:       $(LOGDIR)/<target>-YYYYMMDD-HHMMSS.log"

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
refresh-stale:
	@stale="$$(python3 tools/audit_outputs.py --stale)"; \
	if [ -z "$$stale" ]; then echo "Nothing stale — store is fresh."; exit 0; fi; \
	echo "Stale (will re-execute + capture):"; echo "$$stale" | sed 's/^/  /'; \
	for f in $$stale; do \
		ch=$$(dirname $$f); st=$$(basename $$f .md); \
		for fw in $(FRAMEWORKS); do \
			nb="_notebooks/$$fw/$$ch/$$st.ipynb"; \
			[ -f "$$nb" ] && $(MAKE) -B "_notebooks/$$fw/$$ch/$$st.executed" || true; \
		done; \
		$(MAKE) capture-outputs FILES="$$f"; \
	done

render-fresh: refresh-stale
	$(MAKE) slides
	$(MAKE) html
	$(MAKE) -j4 pdfs

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
		$(CURDIR)/$(QUARTO) render --to html; \
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
	@[ -e _notebooks/$*/img ] || ln -s ../../img _notebooks/$*/img
	@if [ -L _notebooks/$*/data ]; then :; \
	elif [ -d _notebooks/$*/data ]; then \
		cp -rn _notebooks/$*/data/. data/ 2>/dev/null || true; \
		rm -rf _notebooks/$*/data; \
		ln -s $$(pwd)/data _notebooks/$*/data; \
	else \
		ln -s $$(pwd)/data _notebooks/$*/data; \
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
JAX_GPU_SLOTS ?= $(call _gpu_slots,$(GPU_MIB_PER_jax))
JAX_CPU_SLOTS ?= $(call _cpu_slots,$(CPU_PER_jax))

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
MXNET_GPU_SLOTS ?= $(shell python3 tools/detect_resources.py --get MXNET_GPU_SLOTS)
MXNET_CPU_SLOTS ?= 2
EXTRA_ENV_mxnet := OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 \
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
	@python3 tools/detect_resources.py --report || true
	@python3 tools/notebook_run_plan.py --framework $* || true
	@cpu_rc=0; gpu_rc=0; mgpu_rc=0; \
	$(MAKE) --no-print-directory -k -j$(CPU_SLOTS) run-notebooks-cpu-$* & cpu=$$!; \
	( $(MAKE) --no-print-directory -k -j$(GPU_SLOTS) run-notebooks-gpu-$*; rc=$$?; \
	  $(MAKE) --no-print-directory -k -j$(MULTIGPU_SLOTS) run-notebooks-multigpu-$*; mrc=$$?; \
	  exit $$((rc || mrc)) ) & gpu=$$!; \
	wait $$gpu || gpu_rc=$$?; \
	wait $$cpu || cpu_rc=$$?; \
	python3 tools/notebook_run_summary.py --framework $* || true; \
	exit $$((gpu_rc || cpu_rc))

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
run-all-notebooks: notebooks
	@python3 tools/detect_resources.py --report || true
	@python3 tools/notebook_run_plan.py || true
	@cpu_rc=0; gpu_rc=0; \
	$(MAKE) --no-print-directory -k -j$(CPU_SLOTS) $(RUN_NOTEBOOK_CPU_TARGETS) & cpu=$$!; \
	( $(MAKE) --no-print-directory -k -j$(GPU_SLOTS) $(RUN_NOTEBOOK_GPU_TARGETS); rc=$$?; \
	  $(MAKE) --no-print-directory -k -j$(MULTIGPU_SLOTS) $(RUN_NOTEBOOK_MULTIGPU_TARGETS); mrc=$$?; \
	  exit $$((rc || mrc)) ) & gpu=$$!; \
	wait $$gpu || gpu_rc=$$?; \
	wait $$cpu || cpu_rc=$$?; \
	python3 tools/notebook_run_summary.py || true; \
	exit $$((gpu_rc || cpu_rc))

# ── PDFs (per-framework, parallel-safe) ───────────────────

_pdf/%/.generated: $(SRC_MDS) tools/gen_pdf.py tools/d2l_preprocess.py tools/build_lib.py
	@mkdir -p $(LOGDIR)
	@echo "=== Generating PDF sources for $* ==="
	@python3 tools/gen_pdf.py $(SOURCE) _pdf/$* --framework $* 2>&1 | tee $(LOGDIR)/pdf-$*-gen-$(TS).log
	@touch $@

# Generate per-framework PDF rules (GNU Make only supports one % per pattern)
define PDF_RULE
_pdf/$(1)/_pdf/Dive-into-Deep-Learning-$(1).pdf: _pdf/$(1)/.generated | .venv-build/.synced
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
		cd "$(CURDIR)"; \
		if [ -f _pdf/$(1)/_pdf/Dive-into-Deep-Learning.pdf ]; then \
			mv _pdf/$(1)/_pdf/Dive-into-Deep-Learning.pdf _pdf/$(1)/_pdf/Dive-into-Deep-Learning-$(1).pdf; \
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
	@rm -f $(SLIDE_STAMPS)
	@rm -f _book/index.html
	@for fw in $(FRAMEWORKS); do rm -f "_pdf/$$fw/_pdf/Dive-into-Deep-Learning-$$fw.pdf"; done
	$(MAKE) slides
	$(MAKE) html
	@# Clear stale render-scratch PDFs so the parallel PDF render can't
	@# skip-then-read a corrupt one (Quarto convert_svg, main.lua:7348). This
	@# makes `make all` self-sufficient without a preceding `make clean`.
	@find img/outputs -name '*.pdf' -delete 2>/dev/null || true
	$(MAKE) -j4 pdfs
	$(MAKE) check-all-artifacts

# Quick build without notebook execution
all-quick:
	$(MAKE) lib
	$(MAKE) notebooks
	$(MAKE) rebuild-book-artifacts

check-all-artifacts:
	@test -f _book/index.html || { echo "ERROR: missing _book/index.html"; exit 1; }
	@test -f _slides/index.html || { echo "ERROR: missing _slides/index.html"; exit 1; }
	@test -f _book/slides/index.html || { echo "ERROR: missing _book/slides/index.html"; exit 1; }
	@echo "Verified full build artifacts: _book/index.html and _book/slides/index.html"

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
