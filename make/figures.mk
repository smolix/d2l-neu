# make/figures.mk — illustrative figure generators (committed SVGs).
# Included by the top-level Makefile — see docs/build-system.md and
# reviews/build-system-cleanup-proposal-2026-07-22.md.

# Illustrative figures are pre-generated static SVGs committed to img/ (see the
# mdl-figure skill + CLAUDE.md "Content authoring"). The shared house style lives
# in tools/gen_mdl_figures.py; each chapter's generator imports it.
#
# This target is INCREMENTAL and MANUAL (never part of all/all-quick, per Alex):
# a generator's figures regenerate only when its own script — or the shared house
# style it imports — changed. Generators are byte-idempotent (svg.hashsalt +
# Date:None), so re-running a changed generator yields a clean `git diff img/`
# unless the picture actually changed.
#
# Coverage: every standalone chapter generator, INCLUDING the ch6/7/8/9 ones
# (gen_bg_*, gen_arch_*, gen_opt_figures) that the old gen_mdl_*_figures.py glob
# silently skipped. Aggregator generators that import/runpy sub-modules get those
# sub-modules added as extra prerequisites below, so editing a sub-module also
# re-fires its group. (tools/bg_diagrams.py is currently imported by no generator
# — left in place pending a decision, see the proposal doc.)

FIGURE_GENERATORS := $(sort \
    tools/gen_mdl_figures.py \
    $(wildcard tools/gen_mdl_*_figures.py) \
    $(wildcard tools/gen_bg_*_figures.py) \
    $(wildcard tools/gen_arch_*_figures.py) \
    tools/gen_opt_figures.py)
FIGURE_STAMPS := $(patsubst tools/%.py,.figstamps/%,$(FIGURE_GENERATORS))

# Regenerate one generator's SVGs only when its script (or the shared house
# style it imports) changed. CPU-only; byte-idempotent.
.figstamps/%: tools/%.py tools/gen_mdl_figures.py | .venv-pytorch/.synced
	@mkdir -p .figstamps $(LOGDIR)
	@echo "=== figures: $* ==="
	@.venv-pytorch/bin/python tools/$*.py 2>&1 | tee $(LOGDIR)/figures-$*-$(TS).log
	@touch $@

# Aggregators pull their sub-modules in via import/runpy, so re-fire the whole
# group when a sub-module changes (the aggregator's own mtime wouldn't).
.figstamps/gen_mdl_attention_figures: $(wildcard tools/gen_mdl_attention_a*.py)
.figstamps/gen_mdl_transformers_figures: $(wildcard tools/gen_mdl_transformers_b*.py)
.figstamps/gen_mdl_tools_appendix_figures: tools/gen_tools_appendix_figures.py tools/arch_diagrams.py

.PHONY: figures
figures: $(FIGURE_STAMPS)
	@echo "Figures up to date (only changed generators ran). Verify: git diff --stat img/"
