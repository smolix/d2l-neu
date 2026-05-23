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

# Auto-detect GPU count and memory via nvidia-smi.
# Rule of thumb: each notebook job needs ~11GB GPU memory, so a 24GB GPU
# runs 2 in parallel, a 48GB GPU runs 4, etc. We use the *minimum* memory
# across detected GPUs (workers_per_gpu must be uniform — see run_one_notebook.py).
# Falls back to "0 1" when no GPUs are visible (CPU-only host).
# Override on the command line: `make NUM_GPUS=2 GPU_SLOTS=4 ...`.
_GPU_QUERY := $(shell nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | awk 'BEGIN{n=0;m=0} {n++; if(n==1||$$1<m) m=$$1} END{per=int(m/11264); if(per<1) per=1; if(n==0) print "0 1"; else print n, n*per}')
DETECTED_NUM_GPUS := $(word 1,$(_GPU_QUERY))
DETECTED_GPU_SLOTS := $(word 2,$(_GPU_QUERY))
# If no GPU was detected, fall back to NUM_GPUS=1 for slot-to-device mapping.
NUM_GPUS   ?= $(if $(filter 0,$(DETECTED_NUM_GPUS)),1,$(DETECTED_NUM_GPUS))
_GPU_SLOTS_DEFAULT := $(if $(DETECTED_GPU_SLOTS),$(DETECTED_GPU_SLOTS),1)

# Per-notebook slot counts. Re-tuned for the post-Phase-1 build system:
#   GPU_SLOTS = NUM_GPUS × workers_per_GPU  (≥11GB VRAM per worker → 24GB
#                                            GPU runs 2 jobs at once)
#   CPU_SLOTS = max(1, cores / 16)          (≥16 cores per CPU notebook)
# `make -j` should be at least GPU_SLOTS + CPU_SLOTS for full saturation.
# Override either explicitly: `make GPU_SLOTS=8 CPU_SLOTS=4 ...`.
GPU_SLOTS ?= $(_GPU_SLOTS_DEFAULT)
CPU_SLOTS ?= $(shell n=$$(nproc 2>/dev/null || echo 16); echo $$(( n / 16 < 1 ? 1 : n / 16 )))
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

.venv-%/.synced: pyproject.toml uv.lock
	@mkdir -p $(LOGDIR)
	@echo "=== Syncing venv for $* ==="
	UV_PROJECT_ENVIRONMENT=.venv-$* uv sync --extra $* --extra run 2>&1 | tee $(LOGDIR)/venv-$*-$(TS).log
	@touch $@

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
	@echo "=== Building HTML book ==="
	@{ \
		if [ -d _notebooks ]; then \
			echo "Injecting notebook outputs..."; \
			python3 tools/inject_outputs.py html; \
		fi; \
		echo "Building slides manifest (TOC button + landing page)..."; \
		python3 tools/build_slides_index.py; \
		$(CURDIR)/$(QUARTO) render --to html; \
		python3 tools/fix_crossref_numbers.py .; \
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
				-exec sed -i 's|src="\.\./img/|src="../../../img/|g' {} +; \
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

_notebooks/%/.generated: $(SRC_MDS) tools/gen_notebooks.py tools/d2l_preprocess.py tools/build_lib.py d2l/.built | .venv-build/.synced
	@mkdir -p $(LOGDIR)
	@echo "=== Generating $* notebooks ==="
	@PATH="$(CURDIR)/.venv-build/bin:$$PATH" \
	python3 tools/gen_notebooks.py $(SOURCE) _notebooks --convert --frameworks $* 2>&1 | tee $(LOGDIR)/notebooks-$*-$(TS).log
	@mkdir -p data
	@[ -e _notebooks/$*/img ] || ln -s ../../img _notebooks/$*/img
	@if [ -L _notebooks/$*/data ]; then :; \
	elif [ -d _notebooks/$*/data ]; then \
		cp -rn _notebooks/$*/data/. data/ 2>/dev/null || true; \
		rm -rf _notebooks/$*/data; \
		ln -s $$(pwd)/data _notebooks/$*/data; \
	else \
		ln -s $$(pwd)/data _notebooks/$*/data; \
	fi
	@# Per-notebook .executed stamps depend on each .ipynb directly, and
	@# gen_notebooks.py preserves .ipynb mtime when content is unchanged
	@# (so unchanged notebooks don't re-execute). The .generated stamp's
	@# own mtime doesn't propagate to .executed targets via this chain.
	@touch $@

notebooks-%: _notebooks/%/.generated
	@echo "Notebooks for $* in _notebooks/$*/"

notebooks: $(addprefix notebooks-,$(FRAMEWORKS))

# .ipynb files are written by tools/gen_notebooks.py as a side effect of
# the .generated rule above. Per-framework pattern rule that connects any
# .ipynb to its framework's batch generator so per-notebook targets can
# list a real .ipynb prerequisite without "no rule to make target".
define IPYNB_FROM_GEN
_notebooks/$(1)/%.ipynb: _notebooks/$(1)/.generated
	@:
endef
$(foreach fw,$(FRAMEWORKS),$(eval $(call IPYNB_FROM_GEN,$(fw))))

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
	@echo "=== Running $* notebooks: GPU_SLOTS=$(GPU_SLOTS) CPU_SLOTS=$(CPU_SLOTS) ==="
	@cpu_rc=0; gpu_rc=0; mgpu_rc=0; \
	$(MAKE) --no-print-directory -k -j$(CPU_SLOTS) run-notebooks-cpu-$* & cpu=$$!; \
	( $(MAKE) --no-print-directory -k -j$(GPU_SLOTS) run-notebooks-gpu-$*; rc=$$?; \
	  $(MAKE) --no-print-directory -k -j1 run-notebooks-multigpu-$*; mrc=$$?; \
	  exit $$((rc || mrc)) ) & gpu=$$!; \
	wait $$gpu || gpu_rc=$$?; \
	wait $$cpu || cpu_rc=$$?; \
	python3 tools/notebook_run_summary.py --framework $* || true; \
	exit $$((gpu_rc || cpu_rc))

# For a single notebook, invoke the stamp target directly:
#   make _notebooks/pytorch/chapter_x/foo.executed

# Execute all frameworks with one global CPU queue and one global GPU queue.
# `notebooks` runs first so CPU/GPU sub-makes cannot generate the same
# framework's notebooks concurrently. -k on every sub-make keeps the queues
# draining even after individual notebook failures.
run-all-notebooks: notebooks
	@echo "=== Running all notebooks: GPU_SLOTS=$(GPU_SLOTS) CPU_SLOTS=$(CPU_SLOTS) ==="
	@cpu_rc=0; gpu_rc=0; \
	$(MAKE) --no-print-directory -k -j$(CPU_SLOTS) $(RUN_NOTEBOOK_CPU_TARGETS) & cpu=$$!; \
	( $(MAKE) --no-print-directory -k -j$(GPU_SLOTS) $(RUN_NOTEBOOK_GPU_TARGETS); rc=$$?; \
	  $(MAKE) --no-print-directory -k -j1 $(RUN_NOTEBOOK_MULTIGPU_TARGETS); mrc=$$?; \
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
# WARNING: do not run multiple slides-* targets with -j (GPU contention)

_slides/%/.built: $(SRC_MDS) tools/gen_slides.py tools/d2l_preprocess.py tools/build_lib.py | .venv-%/.synced .venv-build/.synced
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
	rm -f $(wildcard _notebooks/*/.generated _notebooks/*/.executed)
	rm -f $(wildcard _notebooks/*/MANIFEST.mk)
	rm -f $(wildcard _pdf/*/.generated)
	rm -f $(wildcard _slides/*/.built)
	@echo "Cleaned build artifacts (kept ./data/, $(LOGDIR)/, and .upload-manifest-*.txt)"

veryclean: clean
	rm -rf data $(LOGDIR)
	rm -f .upload-manifest-*.txt
	@echo "Also deleted ./data/, $(LOGDIR)/, and .upload-manifest-*.txt"
