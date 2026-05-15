# D2L Book Build System
#
# Usage:
#   make help                  Show this help
#   make html                  Build HTML book (multi-framework tabs)
#   make pdf-pytorch           Build PDF for PyTorch
#   make pdfs                  Build PDFs for all frameworks (safe with -j4)
#   make run-notebooks-pytorch Execute PyTorch notebooks (GPU)
#   make run-all-notebooks     Execute all frameworks sequentially (GPU)
#   make slides-pytorch        Build + render PyTorch slides (GPU)
#   make lib                   Build d2l Python package
#   make clean                 Remove build artifacts (keeps data/)
#
# Parallel-safe:
#   make -j4 pdfs              4 framework PDFs in parallel (separate dirs)
#   make -j4 notebooks         Generate (not execute) all notebooks in parallel
#
# NOT parallel-safe (GPU contention):
#   run-notebooks-*           Run one framework at a time
# Parallel-safe (CPU-only, after slide refactor):
#   slides-*                   make -j4 slides works
#
# GPU workers per framework: NUM_GPUS and PARALLEL_<fw> are auto-detected
# from nvidia-smi (rule of thumb: 11GB per job, so a 24GB GPU runs 2 in
# parallel). Override on the command line: `make NUM_GPUS=2 PARALLEL_pytorch=2`.
#
# All build output is logged to logs/<target>-YYYYMMDD-HHMMSS.log

SHELL      := /bin/bash
.SHELLFLAGS := -o pipefail -c
.DEFAULT_GOAL := help

SOURCE     ?= .
FRAMEWORKS := pytorch tensorflow jax mxnet

# Auto-detect GPU count and memory via nvidia-smi.
# Rule of thumb: each notebook job needs ~11GB GPU memory, so a 24GB GPU
# runs 2 in parallel, a 48GB GPU runs 4, etc. We use the *minimum* memory
# across detected GPUs (workers_per_gpu must be uniform — see run_notebooks.py).
# Falls back to "0 1" when no GPUs are visible (CPU-only host).
# Override on the command line: `make NUM_GPUS=2 PARALLEL_pytorch=2 ...`.
_GPU_QUERY := $(shell nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | awk 'BEGIN{n=0;m=0} {n++; if(n==1||$$1<m) m=$$1} END{per=int(m/11264); if(per<1) per=1; if(n==0) print "0 1"; else print n, n*per}')
DETECTED_NUM_GPUS := $(word 1,$(_GPU_QUERY))
DETECTED_PARALLEL := $(word 2,$(_GPU_QUERY))
# If no GPU was detected, fall back to NUM_GPUS=1 (CPU-only path in run_notebooks.py).
NUM_GPUS   ?= $(if $(filter 0,$(DETECTED_NUM_GPUS)),1,$(DETECTED_NUM_GPUS))
_PARALLEL_DEFAULT := $(if $(DETECTED_PARALLEL),$(DETECTED_PARALLEL),1)
PARALLEL_pytorch    ?= $(_PARALLEL_DEFAULT)
PARALLEL_tensorflow ?= $(_PARALLEL_DEFAULT)
PARALLEL_jax        ?= $(_PARALLEL_DEFAULT)
PARALLEL_mxnet      ?= $(_PARALLEL_DEFAULT)
FILES         ?=
SLIDES_FILTER ?= $(FILES)
NB_FILES      ?= $(FILES)

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

.PHONY: help all all-quick html lib clean veryclean
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
	@echo "  run-notebooks-<fw>     Execute notebooks for one framework (GPU)"
	@echo "  run-all-notebooks       Execute all frameworks sequentially (GPU)"
	@echo "  slides-<fw>            Build slides for one framework (GPU)"
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
	@echo "           PARALLEL: pytorch=$(PARALLEL_pytorch) tf=$(PARALLEL_tensorflow) jax=$(PARALLEL_jax) mxnet=$(PARALLEL_mxnet)"
	@echo "           FILES=$(FILES)  NB_FILES=$(NB_FILES)  SLIDES_FILTER=$(SLIDES_FILTER)"
	@echo "Frameworks: $(FRAMEWORKS)"
	@echo "Logs:       $(LOGDIR)/<target>-YYYYMMDD-HHMMSS.log"

# ── d2l library ────────────────────────────────────────────

lib: d2l/.built

d2l/.built: $(SRC_MDS) tools/build_lib.py tools/d2l_preprocess.py
	@mkdir -p $(LOGDIR)
	@echo "=== Building d2l library ==="
	python3 tools/build_lib.py $(SOURCE) d2l/ 2>&1 | tee $(LOGDIR)/lib-$(TS).log
	@touch $@

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

_notebooks/%/.generated: $(SRC_MDS) tools/gen_notebooks.py tools/d2l_preprocess.py tools/build_lib.py d2l/.built
	@mkdir -p $(LOGDIR)
	@echo "=== Generating $* notebooks ==="
	@python3 tools/gen_notebooks.py $(SOURCE) _notebooks --convert --frameworks $* 2>&1 | tee $(LOGDIR)/notebooks-$*-$(TS).log
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
	@touch $@

notebooks-%: _notebooks/%/.generated
	@echo "Notebooks for $* in _notebooks/$*/"

notebooks: $(addprefix notebooks-,$(FRAMEWORKS))

# ── Notebooks (execute) ───────────────────────────────────
# WARNING: do not run multiple run-notebooks-* targets with -j (GPU contention)

define nvidia_ld_path
$(shell find .venv-$(1)/lib -path "*/nvidia/*/lib" -type d 2>/dev/null | paste -sd: -)
endef

_notebooks/%/.executed: _notebooks/%/.generated d2l/.built | .venv-%/.synced
	@mkdir -p $(LOGDIR)
	@echo "=== Running $* notebooks ==="
	$(eval NVIDIA_LIBS := $(call nvidia_ld_path,$*))
	UV_PROJECT_ENVIRONMENT=.venv-$* \
	LD_LIBRARY_PATH="$(NVIDIA_LIBS)$${LD_LIBRARY_PATH:+:$$LD_LIBRARY_PATH}" \
	.venv-$*/bin/python tools/run_notebooks.py $* \
		--parallel $(PARALLEL_$*) --num-gpus $(NUM_GPUS) --continue-on-error \
		$(if $(NB_FILES),--files $(NB_FILES)) \
		2>&1 | tee $(LOGDIR)/run-$*-$(TS).log
	@touch $@

run-notebooks-%: _notebooks/%/.executed
	@echo "Notebooks for $* executed"
	@echo "Log: $(LOGDIR)/run-$*-$(TS).log"

# Sequential execution of all frameworks (GPU-safe)
run-all-notebooks:
	$(MAKE) run-notebooks-pytorch
	$(MAKE) run-notebooks-tensorflow
	$(MAKE) run-notebooks-jax
	$(MAKE) run-notebooks-mxnet

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

_slides/%/.built: $(SRC_MDS) tools/gen_slides.py tools/d2l_preprocess.py tools/build_lib.py | .venv-%/.synced
	@mkdir -p $(LOGDIR)
	@echo "=== Building $* slides ==="
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
	$(MAKE) run-all-notebooks
	$(MAKE) slides
	@echo "=== Rebuilding HTML and PDFs with notebook outputs ==="
	@rm -f _book/index.html
	@for fw in $(FRAMEWORKS); do rm -f "_pdf/$$fw/_pdf/Dive-into-Deep-Learning-$$fw.pdf"; done
	$(MAKE) html
	$(MAKE) -j4 pdfs

# Quick build without notebook execution
all-quick: html pdfs notebooks slides lib

# ── Clean ──────────────────────────────────────────────────

# `clean` preserves expensive state: ./data/ (downloaded datasets),
# logs/, and .upload-manifest-*.txt (sha256 manifest of the last R2
# upload — losing it forces a full re-upload). Use `veryclean` to wipe
# those too.
clean:
	rm -rf _book _pdf _notebooks _slides
	rm -f img/*.pdf .preprocess.stamp d2l/.built _d2l-slides-data.html
	rm -f $(wildcard _notebooks/*/.generated _notebooks/*/.executed)
	rm -f $(wildcard _pdf/*/.generated)
	rm -f $(wildcard _slides/*/.built)
	@echo "Cleaned build artifacts (kept ./data/, $(LOGDIR)/, and .upload-manifest-*.txt)"

veryclean: clean
	rm -rf data $(LOGDIR)
	rm -f .upload-manifest-*.txt
	@echo "Also deleted ./data/, $(LOGDIR)/, and .upload-manifest-*.txt"
