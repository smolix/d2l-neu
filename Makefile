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
#   run-notebooks-*, slides-*  Run one framework at a time
#
# All build output is logged to logs/<target>-YYYYMMDD-HHMMSS.log

SHELL      := /bin/bash
.DEFAULT_GOAL := help

SOURCE     ?= .
FRAMEWORKS := pytorch tensorflow jax mxnet
PARALLEL   ?= 4
NUM_GPUS   ?= 4
SLIDES_FILTER ?=
NB_FILES      ?=

# Source files — the ultimate upstream for everything
SRC_MDS := $(wildcard $(SOURCE)/chapter_*/*.md)
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
	@echo "  clean                   Remove build artifacts (keep data/)"
	@echo "  veryclean               Remove everything including data/"
	@echo "  all                     Full pipeline: generate, execute, rebuild with outputs"
	@echo "  all-quick               Build html + pdfs + notebooks + slides + lib (no execution)"
	@echo ""
	@echo "Variables:  SOURCE=$(SOURCE)  PARALLEL=$(PARALLEL)  NUM_GPUS=$(NUM_GPUS)"
	@echo "           SLIDES_FILTER=$(SLIDES_FILTER)  NB_FILES=$(NB_FILES)"
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

venv-%: .venv-%/.synced
	@echo "Venv .venv-$* is ready"

# ── HTML book ──────────────────────────────────────────────

html: _book/index.html
	@echo "Output: _book/index.html"
	@echo "Log:    $(LOGDIR)/html-$(TS).log"

# Stage 1: preprocess d2l-en .md → .qmd
.preprocess.stamp: $(SRC_MDS) tools/d2l_preprocess.py
	@echo "=== Preprocessing .md → .qmd ==="
	python3 tools/d2l_preprocess.py $(SOURCE) . --primary pytorch
	@touch $@

# Stage 2+3+4: inject (optional) + quarto render + fix numbering
_book/index.html: .preprocess.stamp _quarto.yml _d2l-theme.scss _d2l-style.css _d2l-tabs.html d2l.bib
	@mkdir -p $(LOGDIR)
	@echo "=== Building HTML book ==="
	@{ \
		if [ -d _notebooks ]; then \
			echo "Injecting notebook outputs..."; \
			python3 tools/inject_outputs.py html; \
		fi; \
		quarto render --to html; \
		python3 tools/fix_crossref_numbers.py .; \
	} 2>&1 | tee $(LOGDIR)/html-$(TS).log

# ── Notebooks (generate) ──────────────────────────────────

_notebooks/%/.generated: $(SRC_MDS) tools/gen_notebooks.py tools/d2l_preprocess.py tools/build_lib.py
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
		--parallel $(PARALLEL) --num-gpus $(NUM_GPUS) --continue-on-error \
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
_pdf/$(1)/_pdf/Dive-into-Deep-Learning-$(1).pdf: _pdf/$(1)/.generated
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
		cd _pdf/$(1) && quarto render --to pdf; \
		cd "$(CURDIR)"; \
		python3 tools/fix_latex.py _pdf/$(1)/Dive-into-Deep-Learning.tex; \
		cd _pdf/$(1) && xelatex -interaction=nonstopmode Dive-into-Deep-Learning.tex > /dev/null 2>&1; \
		xelatex -interaction=nonstopmode Dive-into-Deep-Learning.tex > /dev/null 2>&1; \
		cd "$(CURDIR)"; \
		if [ -f _pdf/$(1)/_pdf/Dive-into-Deep-Learning.pdf ]; then \
			mv _pdf/$(1)/_pdf/Dive-into-Deep-Learning.pdf _pdf/$(1)/_pdf/Dive-into-Deep-Learning-$(1).pdf; \
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
		--render --num-gpus $(NUM_GPUS) \
		$(if $(SLIDES_FILTER),--filter $(SLIDES_FILTER)) \
		2>&1 | tee $(LOGDIR)/slides-$*-$(TS).log
	@touch $@

slides-%: _slides/%/.built
	@echo "Slides for $* in _slides/$*/"
	@echo "Log: $(LOGDIR)/slides-$*-$(TS).log"

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
	$(MAKE) html pdfs

# Quick build without notebook execution
all-quick: html pdfs notebooks slides lib

# ── Clean ──────────────────────────────────────────────────

clean:
	rm -rf _book _pdf _notebooks _slides
	rm -f img/*.pdf .preprocess.stamp d2l/.built
	rm -f $(wildcard _notebooks/*/.generated _notebooks/*/.executed)
	rm -f $(wildcard _pdf/*/.generated)
	rm -f $(wildcard _slides/*/.built)
	@echo "Cleaned build artifacts (kept ./data/ and $(LOGDIR)/)"

veryclean: clean
	rm -rf data $(LOGDIR)
	@echo "Also deleted ./data/ and $(LOGDIR)/"
