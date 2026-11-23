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

# ALLOW_STALE=1 turns the --verify-fresh gate below into a warning, for a
# staging/preview render while stale outputs are known and a re-execution is
# planned (build-system.md §6.2a). Default (unset) stays strict. Stale cells
# render with their old outputs beside new code, so never ship such a build.
ALLOW_STALE ?=

html: _book/index.html
	@echo "Output: _book/index.html"
	@echo "Log:    $(LOGDIR)/html-$(TS).log"

# Stage 1: preprocess d2l-en .md → .qmd + generate API docs
.preprocess.stamp: $(SRC_MDS) tools/d2l_preprocess.py tools/preamble.py tools/gen_api_doc.py d2l/.built
	@mkdir -p $(LOGDIR)
	@echo "=== Preprocessing .md → .qmd ==="
	@{ python3 tools/d2l_preprocess.py $(SOURCE) . --primary pytorch \
		&& python3 tools/gen_api_doc.py; \
	} 2>&1 | tee $(LOGDIR)/preprocess-$(TS).log
	@touch $@

# index.qmd is generated from index.md by the preprocessing above and is
# gitignored, like chapter_*/*.qmd. A checkout that lacks it (it was tracked
# until 2026-11-22, so a pull deletes it) while the stamp is current re-runs
# the preprocessing instead of handing Quarto a missing landing page.
index.qmd: | .preprocess.stamp
	@test -f $@ || { rm -f .preprocess.stamp && $(MAKE) --no-print-directory .preprocess.stamp; }

# Stage 2+3+4: inject (optional) + slides manifest + quarto render + fix numbering
_book/index.html: .preprocess.stamp index.qmd _quarto.yml _d2l-theme.scss _d2l-style.css _d2l-tabs.html _d2l-notebooks.html tools/build_hosted_notebooks.py tools/inject_outputs.py d2l.bib $(OUTPUT_MANIFESTS) | .venv-build/.synced
	@mkdir -p $(LOGDIR)
	@echo "=== Verifying committed outputs are fresh ==="
	@python3 tools/audit_outputs.py --verify-fresh || \
		{ if [ "$(ALLOW_STALE)" = 1 ]; then \
			echo "WARNING: outputs stale; rendering anyway because ALLOW_STALE=1."; \
			echo "WARNING: staging/preview builds only, never a release (build-system.md §6.2a)."; \
		else \
			echo "Outputs stale — re-run + 'make capture-outputs', or 'make render-fresh'."; \
			echo "(Staging preview only: 'make html ALLOW_STALE=1' renders anyway, see build-system.md §6.2a.)"; \
			exit 1; \
		fi; }
	@# Slide-card section numbers (diagrams/sections.mjs): a stale map means the
	@# committed diagram SVGs show old numbers. Hardcoded "§N.M" in diagram
	@# code is reported here and is an error in lint_source.py --corpus.
	@python3 tools/section_refs.py --check-json || \
		{ echo "Chapter/section numbers changed: run 'make figures' and commit diagrams/section-numbers.json and img/."; exit 1; }
	@echo "=== Building HTML book ==="
	@# `set -e` makes the block stop at the first failing step (a failed quarto
	@# render used to be followed by the post-processing steps, whose exit status
	@# the block returned); pipefail (.SHELLFLAGS) carries that failure through
	@# the `| tee`. On failure, delete the target: quarto or integrate_slides.sh
	@# may already have written or touched _book/index.html, and make would then
	@# treat the failed render as current (2026-10-27, logs/fixround-allquick.log).
	@# Test $$? on its own line: a `|| …` after the pipeline would switch
	@# `set -e` off inside the block (bash ignores it left of `||`).
	@{ set -e; \
		if [ -d outputs ] || [ -d _notebooks ]; then \
			echo "Injecting notebook outputs (store preferred, _notebooks fallback)..."; \
			python3 tools/inject_outputs.py html; \
		fi; \
		echo "Building slides manifest (TOC button + landing page)..."; \
		python3 tools/build_slides_index.py; \
		echo "Building hosted-notebook manifest (buttons below Slides)..."; \
		python3 tools/build_hosted_notebooks.py manifest; \
		: 'The whole-book html render outgrew the ~8 GiB default V8 heap (rc=133,'; \
		: '2026-11-23); it runs alone, so it gets the per-job render heap.'; \
		$(RENDER_V8_ENV) "$(CURDIR)/$(QUARTO)" render --to html; \
		python3 tools/fix_crossref_numbers.py .; \
		python3 tools/build_date.py --stamp _book/index.html; \
		python3 tools/add_cfasync.py _book; \
		tools/integrate_slides.sh; \
		: 'PDFs are staged into _book/pdf by the pdf rule (PDF_RULE) itself.'; \
	} 2>&1 | tee $(LOGDIR)/html-$(TS).log; rc=$$?; \
	if [ $$rc -ne 0 ]; then \
		rm -f $@; echo "ERROR: HTML render failed; see $(LOGDIR)/html-$(TS).log"; \
		exit $$rc; \
	fi

# ── PDFs (per-framework, parallel-safe) ───────────────────

_pdf/%/.generated: $(SRC_MDS) tools/gen_pdf.py tools/d2l_preprocess.py tools/preamble.py tools/build_lib.py static/d2l-preamble.tex d2l.bib $(wildcard static/fonts/*)
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
	@# Quarto stages each render session under $$TMPDIR (default /tmp), which on
	@# this box is a small dedicated partition. Running out of it mid-render
	@# crashes quarto mid-.tex (2026-08-02: four parallel PDF sessions hit a
	@# /tmp filled with ~378 MB git-snapshot dirs the Codex CLI leaks per
	@# session, and truncated half-book PDFs were staged). Fail fast instead:
	@# 2 GiB comfortably covers RENDER_JOBS concurrent sessions. Clean stale
	@# /tmp/tmp.* snapshot dirs to recover space; D2L_SKIP_TMP_CHECK=1 overrides.
	@if [ -z "$$D2L_SKIP_TMP_CHECK" ]; then \
		avail_kb=$$(df -Pk "$${TMPDIR:-/tmp}" | awk 'NR==2 {print $$4}'); \
		if [ -n "$$avail_kb" ] && [ "$$avail_kb" -lt 2097152 ]; then \
			echo "ERROR: $${TMPDIR:-/tmp} has only $$((avail_kb / 1024)) MiB free (< 2 GiB);"; \
			echo "       quarto PDF sessions will die mid-render and stage truncated books."; \
			echo "       Clear stale /tmp/tmp.* snapshot dirs (Codex CLI leaks ~378 MB per"; \
			echo "       session) or set D2L_SKIP_TMP_CHECK=1 to proceed anyway."; \
			exit 1; \
		fi; \
	fi
	@echo "PDF toolchain OK: $$(xelatex --version 2>/dev/null | head -1), rsvg-convert $$(rsvg-convert --version 2>/dev/null)"

# Generate per-framework PDF rules (GNU Make only supports one % per pattern)
define PDF_RULE
_pdf/$(1)/_pdf/Dive-into-Deep-Learning-$(1).pdf: _pdf/$(1)/.generated tools/inject_outputs.py tools/build_one_pdf.sh tools/fix_latex.py $$(wildcard outputs/$(1)/*/*.json) | .venv-build/.synced pdf-preflight
	@mkdir -p $(LOGDIR)
	@echo "=== Building PDF ($(1)) ==="
	@QUARTO="$(CURDIR)/$(QUARTO)" tools/build_one_pdf.sh $(1) 2>&1 | tee $(LOGDIR)/pdf-$(1)-$(TS).log

pdf-$(1): _pdf/$(1)/_pdf/Dive-into-Deep-Learning-$(1).pdf
	@mkdir -p _book/pdf
	@cp -f "$$<" _book/pdf/
	@echo "Output: $$<"
	@echo "Log:    $(LOGDIR)/pdf-$(1)-$(TS).log"
endef

$(foreach fw,$(FRAMEWORKS),$(eval $(call PDF_RULE,$(fw))))

pdf: pdf-pytorch
pdfs: $(addprefix pdf-,$(FRAMEWORKS))

# ── Per-part PDFs (Basics, Advanced, Language Models, Mathematics) ───────
# One PDF per part, cut out of the full PDF_PART_FW book's fix_latex'd .tex
# and numbered from its .aux (tools/build_part_pdfs.py; build-system.md
# §6.6a): part cover, part-only TOC, the part's chapters with the full book's
# numbers, and the references the part cites. Needs the full book first, so
# it depends on that framework's PDF; XeLaTeX runs one part at a time.
PDF_PART_FW := pytorch
PDF_PARTS := basics advanced language-models mathematics-for-deep-learning
PDF_PART_FILES := $(foreach p,$(PDF_PARTS),_pdf/parts/Dive-into-Deep-Learning-$(p)-$(PDF_PART_FW).pdf)

$(PDF_PART_FILES) &: _pdf/$(PDF_PART_FW)/_pdf/Dive-into-Deep-Learning-$(PDF_PART_FW).pdf tools/build_part_pdfs.py _quarto.yml static/cover/d2l-part-covers-and-spines-pdf.zip | pdf-preflight
	@mkdir -p $(LOGDIR)
	@echo "=== Building per-part PDFs ($(PDF_PART_FW)) ==="
	@python3 tools/build_part_pdfs.py --framework $(PDF_PART_FW) --parts $(PDF_PARTS) \
		2>&1 | tee $(LOGDIR)/pdf-parts-$(TS).log

pdf-parts: $(PDF_PART_FILES)
	@mkdir -p _book/pdf/parts
	@cp -f $^ _book/pdf/parts/
	@echo "Output: _book/pdf/parts/"
	@echo "Log:    $(LOGDIR)/pdf-parts-$(TS).log"

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
_slides/%/.built: $(SRC_MDS) _d2l-slides.scss _d2l-slides-overlay.html tools/gen_slides.py tools/inject_outputs.py tools/d2l_preprocess.py tools/preamble.py tools/build_lib.py $$(wildcard outputs/$$*/*/*.json) | .venv-build/.synced
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

# ── Editable PowerPoint decks (per-framework) ─────────────
# Each rendered deck's injected .qmd also becomes <deck>.pptx beside its .html
# (tools/slides_to_pptx.py: pandoc's pptx writer + tools/pptx_fit.py, which
# builds the D2L template — logo, section footer, colours — and sizes every
# text box to fit). CPU-only, no Quarto render; incremental per deck (a deck is
# redone when its .qmd or the converter is newer than its .pptx). The book's
# right-sidebar "Slides" block links each deck's HTML, PDF, PowerPoint and
# LaTeX versions.
PPTX_TOOLS := tools/slides_to_pptx.py tools/pptx_fit.py static/logo-with-text.png
_slides/%/.pptx-built: _slides/%/.built $(PPTX_TOOLS) | .venv-build/.synced
	@mkdir -p $(LOGDIR)
	@echo "=== Building $* PowerPoint decks ==="
	PATH="$(CURDIR)/.venv-build/bin:$$PATH" \
	.venv-build/bin/python tools/slides_to_pptx.py --tree _slides/$* \
		--workers $(SLIDE_WORKERS) 2>&1 | tee $(LOGDIR)/slides-pptx-$*-$(TS).log
	@touch $@

slides-pptx-%: _slides/%/.pptx-built
	@echo "PowerPoint decks for $* in _slides/$*/"

slides-pptx: $(addprefix slides-pptx-,$(FRAMEWORKS))

# ── Beamer PDFs and their LaTeX source (per-framework) ───
# Each rendered deck also becomes <deck>.pdf and <deck>-latex.zip (the
# self-contained Beamer source: .tex, D2L theme, fitted sizes, figures) via
# tools/slides_to_beamer.py: pandoc's beamer writer, then up to 7 XeLaTeX
# passes that shrink any frame running into the footer. Needs XeLaTeX;
# incremental per deck like the PowerPoint step.
BEAMER_TOOLS := tools/slides_to_beamer.py tools/slides_to_pptx.py \
	tools/beamer/d2l-beamer.tex tools/beamer/outputs.lua static/logo-with-text.png
_slides/%/.beamer-built: _slides/%/.built $(BEAMER_TOOLS) | .venv-build/.synced
	@mkdir -p $(LOGDIR)
	@echo "=== Building $* Beamer PDFs and LaTeX bundles ==="
	PATH="$(CURDIR)/.venv-build/bin:$$PATH" \
	.venv-build/bin/python tools/slides_to_beamer.py --tree _slides/$* \
		--workers $(SLIDE_WORKERS) 2>&1 | tee $(LOGDIR)/slides-beamer-$*-$(TS).log
	@touch $@

slides-beamer-%: _slides/%/.beamer-built
	@echo "Beamer PDFs and LaTeX bundles for $* in _slides/$*/"

slides-beamer: $(addprefix slides-beamer-,$(FRAMEWORKS))

# CPU-only. gen_slides.py parallelizes WITHIN a framework (isolated clone
# projects, --workers), and frameworks run in parallel under
# `make -j$(RENDER_JOBS) slides`; SLIDE_WORKERS is sized so the product ≈ cores.
# `slides` builds the reveal.js decks, their PowerPoint versions, and their
# Beamer PDFs with LaTeX source.
slides: $(addprefix slides-,$(FRAMEWORKS)) $(addprefix slides-pptx-,$(FRAMEWORKS)) \
	$(addprefix slides-beamer-,$(FRAMEWORKS))
