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
.PHONY: notebooks run-all-notebooks slides notebook-env-locks notebook-zips
.PHONY: hosted-notebooks hosted-env-locks check-hosted-env-locks
.PHONY: check-hosted-runtime-contracts check-hosted-notebooks
.PHONY: check-hosted-docker check-hosted-docker-cpu check-hosted-docker-gpu
.PHONY: dry-run-notebooks-branch publish-notebooks-branch
.PHONY: capture-outputs audit-outputs verify-outputs-fresh refresh-stale render-fresh test-trap

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
	@echo "  notebook-env-locks      [any] Refresh downloadable CPU/GPU uv locks (network)"
	@echo "  notebook-zips           [any] Build runnable per-framework notebook downloads"
	@echo "  hosted-notebooks        [any] Stage PyTorch/TF/JAX/NumPy notebooks + site manifest"
	@echo "  hosted-env-locks        [any] Regenerate hosted profiles/constraints from uv.lock"
	@echo "  check-hosted-notebooks  [any] Verify hosted staging + runtime contracts"
	@echo "  check-hosted-docker     [GPU] Opt-in Colab CPU/GPU framework matrix (slow)"
	@echo "  dry-run-notebooks-branch      Build the generated branch commit without pushing"
	@echo "  publish-notebooks-branch      Replace and push the generated notebooks branch"
	@echo "  capture-outputs         [any] Bless executed _notebooks/ → committed outputs/ [FILES=...]"
	@echo "  audit-outputs           [any] Report stale notebooks (code drift) + store integrity"
	@echo "  verify-outputs-fresh    [any] Render gate: fail on stale inline outputs / orphaned ids"
	@echo "  test-trap               [any] Regression test for the refresh-stale freshness trap"
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

# ── Included rule modules ────────────────────────────────────────────
# The build DAG is split across make/*.mk (see
# reviews/build-system-cleanup-proposal-2026-07-22.md). resources.mk MUST
# come first (it defines the detection helpers the others use); notebooks.mk
# is the untouched resource-aware scheduler core.
include make/resources.mk
include make/venvs.mk
include make/lib.mk
include make/figures.mk
include make/store.mk
include make/notebooks.mk
include make/hosted.mk
include make/render.mk
include make/universities.mk
include make/deploy.mk

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
	@# Slides are force-re-checked (recipe re-runs, but gen_slides.py is internally
	@# incremental — 0 decks re-render when nothing changed). HTML and the PDFs are
	@# NOT force-removed: they list the committed outputs/ store as a prerequisite
	@# (make/render.mk OUTPUT_MANIFESTS), so Make re-renders them only when a source
	@# .md or an injected output manifest actually changed — and only the affected
	@# framework's PDF. An otherwise-unchanged rebuild is a no-op.
	@rm -f $(SLIDE_STAMPS)
	@# Clear stale render-scratch PDFs BEFORE the render fleet starts, so the
	@# parallel PDF render can't skip-then-read a corrupt one (Quarto
	@# convert_svg, main.lua:7348). Makes `make all` self-sufficient with no
	@# preceding `make clean`.
	@find img/outputs -name '*.pdf' -delete 2>/dev/null || true
	@# Render slides with a RAM-aware fleet: RENDER_JOBS frameworks in parallel,
	@# each quarto capped to RENDER_V8_HEAP_MIB. (Old hardcoded -j4 @ 24 GiB
	@# over-committed RAM on <128 GiB boxes → rc=137; 24 GiB starved a full
	@# framework render → rc=133. See detect_resources.py.)
	$(RENDER_V8_ENV) $(MAKE) -j$(RENDER_JOBS) slides
	@# HTML is LIGHT (~8 GiB V8, quarto's default heap); PDFs are HEAVY (~48 GiB
	@# each, RAM-capped to RENDER_JOBS). Render html CONCURRENTLY with the
	@# full-width heavy PDF pool — html as its own light make, pdfs at
	@# -j$(RENDER_JOBS) — so all RENDER_JOBS PDFs run at once instead of html
	@# stealing a heavy slot (which forced one PDF to wait a whole slot-time).
	@# Folding html into the same -j pool with -j(RENDER_JOBS+1) would be unsafe
	@# on smaller boxes (make could then run RENDER_JOBS+1 *heavy* PDFs → OOM),
	@# so keep them as two makes. They share no buildable target here: lib+venvs
	@# are already built, html owns .preprocess.stamp + chapter_*.qmd, pdfs own
	@# _pdf/<fw> — and the html recipe no longer stages _book/pdf (the pdf rule
	@# self-stages) — so the two concurrent makes never race. Peak RAM ≈
	@# RENDER_JOBS×heap + ~8 GiB.
	@$(MAKE) html & HTML_PID=$$!; \
	$(RENDER_V8_ENV) $(MAKE) -j$(RENDER_JOBS) pdfs; PDF_RC=$$?; \
	wait $$HTML_PID; HTML_RC=$$?; \
	if [ $$PDF_RC -ne 0 ] || [ $$HTML_RC -ne 0 ]; then \
		echo "ERROR: concurrent html/pdf render failed (html=$$HTML_RC pdf=$$PDF_RC)"; \
		exit 1; \
	fi
	@# Notebook download bundles — after html+pdfs so quarto's render can't wipe
	@# _book/notebooks/; CPU-only, injects outputs from the committed store.
	$(MAKE) notebook-zips
	$(MAKE) check-all-artifacts

# Quick build without notebook execution
all-quick:
	$(MAKE) lib
	$(MAKE) notebooks
	$(MAKE) rebuild-book-artifacts

check-all-artifacts:
	@mkdir -p $(LOGDIR)
	@test -f _slides/index.html || { echo "ERROR: missing _slides/index.html"; exit 1; }
	@# Deep gate: core pages + zero broken (LFS-pointer) images + per-framework
	@# PDFs + full per-framework slide-deck coverage. Replaces the old
	@# "two index files exist" check that let OOM-dropped decks and broken
	@# images ship silently.
	@python3 tools/check_book_artifacts.py 2>&1 | tee $(LOGDIR)/check-all-artifacts-$(TS).log

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
