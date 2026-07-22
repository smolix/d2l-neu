# make/render.mk — HTML book, per-framework PDFs, and slides (CPU render)
# Included by the top-level Makefile — see docs/build-system.md and
# reviews/build-system-cleanup-proposal-2026-07-22.md.

# Committed notebook-output store (outputs/<fw>/<chapter>/<stem>.json). A change to
# any manifest must re-render whatever injects it: HTML injects ALL frameworks
# (every tab) → depends on the whole store; each PDF injects only its own
# framework's outputs → depends on just that subtree (see PDF_RULE below). This is
# what lets an otherwise-unchanged `make all-quick` no-op and rebuild only the
# affected framework's PDF — it replaces the blanket force-rm that
# rebuild-book-artifacts used to do.
OUTPUT_MANIFESTS := $(wildcard outputs/*/*/*.json)

# ── HTML book ──────────────────────────────────────────────

html: _book/index.html
	@echo "Output: _book/index.html"
	@echo "Log:    $(LOGDIR)/html-$(TS).log"

# Stage 1: preprocess d2l-en .md → .qmd + generate API docs
.preprocess.stamp: $(SRC_MDS) tools/d2l_preprocess.py tools/gen_api_doc.py d2l/.built
	@mkdir -p $(LOGDIR)
	@echo "=== Preprocessing .md → .qmd ==="
	@{ python3 tools/d2l_preprocess.py $(SOURCE) . --primary pytorch \
		&& python3 tools/gen_api_doc.py; \
	} 2>&1 | tee $(LOGDIR)/preprocess-$(TS).log
	@touch $@

# Stage 2+3+4: inject (optional) + slides manifest + quarto render + fix numbering
_book/index.html: .preprocess.stamp _quarto.yml _d2l-theme.scss _d2l-style.css _d2l-tabs.html _d2l-notebooks.html tools/build_hosted_notebooks.py d2l.bib $(OUTPUT_MANIFESTS) | .venv-build/.synced
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
		echo "Building hosted-notebook manifest (buttons below Slides)..."; \
		python3 tools/build_hosted_notebooks.py manifest; \
		"$(CURDIR)/$(QUARTO)" render --to html; \
		python3 tools/fix_crossref_numbers.py .; \
		python3 tools/add_cfasync.py _book; \
		tools/integrate_slides.sh; \
		: 'PDFs are staged into _book/pdf by the pdf rule (PDF_RULE) itself,'; \
		: 'so html no longer copies them here — avoids a same-file cp race'; \
		: 'when html and pdfs render concurrently (rebuild-book-artifacts).'; \
	} 2>&1 | tee $(LOGDIR)/html-$(TS).log

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
_pdf/$(1)/_pdf/Dive-into-Deep-Learning-$(1).pdf: _pdf/$(1)/.generated $$(wildcard outputs/$(1)/*/*.json) | .venv-build/.synced pdf-preflight
	@mkdir -p $(LOGDIR)
	@echo "=== Building PDF ($(1)) ==="
	@QUARTO="$(CURDIR)/$(QUARTO)" tools/build_one_pdf.sh $(1) 2>&1 | tee $(LOGDIR)/pdf-$(1)-$(TS).log

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

# $$(wildcard outputs/$$*/*/*.json) (secondary expansion, $$* = framework): slides
# inject that framework's committed outputs (gen_slides.py), so an output change
# re-renders its decks and an unchanged store lets the recipe skip entirely — no
# force-rm needed in rebuild-book-artifacts.
_slides/%/.built: $(SRC_MDS) tools/gen_slides.py tools/d2l_preprocess.py tools/build_lib.py $$(wildcard outputs/$$*/*/*.json) | .venv-build/.synced
	@mkdir -p $(LOGDIR)
	@echo "=== Building $* slides ==="
	@# pipefail is global (.SHELLFLAGS) so gen_slides.py's non-zero exit on a
	@# failed deck render propagates through the `| tee` and aborts the recipe
	@# before `touch $@` — a failed slide build no longer stamps .built.
	PATH="$(CURDIR)/.venv-build/bin:$$PATH" \
	python3 tools/gen_slides.py $(SOURCE) _slides --frameworks $* \
		--render --workers $(SLIDE_WORKERS) \
		$(if $(SLIDES_FILTER),--files $(SLIDES_FILTER)) \
		2>&1 | tee $(LOGDIR)/slides-$*-$(TS).log
	@touch $@

slides-%: _slides/%/.built
	@echo "Slides for $* in _slides/$*/"
	@echo "Log: $(LOGDIR)/slides-$*-$(TS).log"

# CPU-only. gen_slides.py parallelizes WITHIN a framework (isolated clone
# projects, --workers), and frameworks run in parallel under
# `make -j$(RENDER_JOBS) slides`; SLIDE_WORKERS is sized so the product ≈ cores.
slides: $(addprefix slides-,$(FRAMEWORKS))
