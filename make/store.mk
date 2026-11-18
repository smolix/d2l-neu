# make/store.mk — committed notebook-output store (capture / audit / refresh)
# Included by the top-level Makefile — see docs/build-system.md and
# reviews/build-system-cleanup-proposal-2026-07-22.md.

# ── Committed outputs store (decoupled build; see docs/build-system.md) ──
# capture-outputs   distill executed _notebooks/ → committed outputs/ store
# audit-outputs     report stale notebooks (code-provenance drift) + integrity
# verify-outputs-fresh  render gate: fail on stale INLINE outputs / orphaned ids
# refresh-stale     re-execute ONLY stale notebooks, then re-capture
# render-fresh      refresh-stale, then rebuild slides + html + pdfs
# locks             list locked notebooks (docs/build-system.md "Locked notebooks")
# rerun-locked      execute named LOCKED notebooks (FILES=..., [FRAMEWORKS=...])

# UNLOCK=1 lets capture overwrite the manifest of a LOCKED notebook, and only
# one re-executed under the current source/lib (`make rerun-locked`).
capture-outputs:
	@mkdir -p $(LOGDIR)
	python3 tools/capture_outputs.py $(if $(FILES),FILES=$(FILES)) \
		$(if $(filter 1,$(UNLOCK)),--unlock) \
		2>&1 | tee $(LOGDIR)/capture-outputs-$(TS).log

audit-outputs:
	@mkdir -p $(LOGDIR)
	@python3 tools/audit_outputs.py 2>&1 | tee $(LOGDIR)/audit-outputs-$(TS).log

verify-outputs-fresh:
	@mkdir -p $(LOGDIR)
	@python3 tools/audit_outputs.py --verify-fresh 2>&1 | tee $(LOGDIR)/verify-outputs-fresh-$(TS).log

# Fast, self-contained regression test (no GPU, no venv) that reproduces the
# freshness-disagreement trap on a tmp fixture and asserts all three defenses:
# the audit flags the lib-stale stamp, `refresh-stale`'s stamp-removal forces a
# re-run, and the capture-side guard refuses to bless the stale outputs. Never
# touches the committed store. See docs/build-system.md §3.3. Also runs
# tools/test_freshness_deps.py: shard naming, the transitive .d closure, and
# the no-manifest rule (§3.4), tools/test_notebook_locks.py: notebook locks
# (§3.7), and tools/test_notebook_venvs.py: the separate .venv-lm (§6.4.2).
test-trap:
	@mkdir -p $(LOGDIR)
	@{ python3 tools/test_refresh_stale_trap.py && python3 tools/test_freshness_deps.py \
		&& python3 tools/test_notebook_locks.py \
		&& python3 tools/test_notebook_venvs.py; } 2>&1 | tee $(LOGDIR)/test-trap-$(TS).log

# CPU unit tests of the Language Models part's library code (rule T1 of the
# ch. 18 spec): every tools/test_lm_*.py plus tools/test_build_lib_async.py,
# run by unittest in .venv-lm (the part's own venv: transformers, openai, ...;
# docs/build-system.md §6.4.2). No GPU, no network unless a test opts in.
LM_TESTS := tools/test_build_lib_async.py $(sort $(wildcard tools/test_lm_*.py))
.PHONY: test-lm
test-lm: | .venv-lm/.synced
	@mkdir -p $(LOGDIR)
	@.venv-lm/bin/python -m unittest $(LM_TESTS) 2>&1 | tee $(LOGDIR)/test-lm-$(TS).log

# ── Locked notebooks (docs/build-system.md "Locked notebooks") ──
# A source .md carrying `<!-- execution: locked; requires: <class> -->` is never
# executed by a build: the scheduler (and run_one_notebook.py, for a direct
# `make <stamp>`) materializes its executed notebook from outputs/ and reports
# it as LOCKED (skipped). Only rerun-locked executes it; only
# `capture-outputs ... UNLOCK=1` replaces its manifest.
locks:
	@mkdir -p $(LOGDIR)
	@python3 tools/locks.py list 2>&1 | tee $(LOGDIR)/locks-$(TS).log

# Execute exactly the named locked notebooks through the scheduler, which
# removes their stamps and sets D2L_RERUN_LOCKED=1 so run_one_notebook.py
# executes instead of materializing. FRAMEWORKS=pytorch narrows the run.
# Capture right after: until then the run lives only in _notebooks/.
_comma := ,
_empty :=
_space := $(_empty) $(_empty)
ifneq ($(filter rerun-locked,$(MAKECMDGOALS)),)
ifeq ($(strip $(FILES)),)
$(error rerun-locked needs FILES=<chapter/file.md> [FRAMEWORKS=pytorch])
endif
endif
# Locked re-runs default to a 6 h per-notebook limit (a NOTEBOOK_TIMEOUTS entry
# in tools/runtime_env.py wins); LOCKED_TIMEOUT=<seconds> overrides it.
LOCKED_TIMEOUT ?= 21600
rerun-locked: notebooks
	@mkdir -p $(LOGDIR)
	@python3 tools/detect_resources.py --report || true
	@$(SCHED_ENV) D2L_LOCKED_TIMEOUT=$(LOCKED_TIMEOUT) \
		python3 tools/notebook_scheduler.py --rerun-locked \
		--files "$(FILES)" \
		--frameworks "$(subst $(_space),$(_comma),$(strip $(FRAMEWORKS)))" \
		2>&1 | tee -a $(LOGDIR)/rerun-locked-$(TS).log; rc=$${PIPESTATUS[0]}; \
	if [ $$rc -eq 0 ]; then \
		echo "Re-run done. Bless it: make capture-outputs FILES=\"$(FILES)\" UNLOCK=1"; \
	fi; \
	exit $$rc

# Re-execute exactly what the audit reports stale, then re-capture those files.
# Execution goes through the unified scheduler (parallel GPU/CPU dispatch),
# NOT a serial per-notebook `make -B` loop — the old loop ran one notebook at a
# time with the GPU pool mostly idle.
#
# THE TRAP this target must NOT fall into (see docs/build-system.md §3.3): the
# audit's staleness is code-provenance/**fingerprint** based, while the
# scheduler's own skip check (`tools/notebook_scheduler._stale`) is **mtime**
# based. A notebook whose *source* is unchanged but whose store is
# lib-fingerprint-stale (a `#@save` symbol it imports was edited elsewhere)
# regenerates a byte-identical .ipynb, keeps a stamp newer than that .ipynb, and
# so looks "fresh" to the scheduler — it is silently skipped, and the capture
# pass then blesses its *pre-edit* outputs under the new fingerprint (falsely
# green). `--files` alone does NOT fix this; the scheduler is not forced.
#
# Fix: `rm` exactly the audit's stale (framework, file) `.executed` stamps
# (emitted by `audit_outputs.py --stale-stamps`) BEFORE dispatch. A missing
# stamp makes both `_stale()` and Make itself re-execute the notebook, so only
# genuinely-fresh outputs are ever captured. `notebooks` is a prereq so .ipynb
# reflect current source before staleness is judged.
refresh-stale: notebooks
	@mkdir -p $(LOGDIR)
	@stale="$$(python3 tools/audit_outputs.py --stale)"; \
	if [ -z "$$stale" ]; then echo "Nothing stale — store is fresh."; exit 0; fi; \
	stamps="$$(python3 tools/audit_outputs.py --stale-stamps)"; \
	echo "Stale (will FORCE re-execute, then capture):"; echo "$$stale" | sed 's/^/  /'; \
	rm -f $$stamps; \
	$(SCHED_ENV) python3 tools/notebook_scheduler.py --files "$$stale" \
		2>&1 | tee -a $(LOGDIR)/scheduler-$(TS).log; rc=$$?; \
	crc=0; $(MAKE) capture-outputs FILES="$$(echo $$stale)" || crc=$$?; \
	if [ $$rc -eq 0 ]; then rc=$$crc; fi; \
	exit $$rc

# Fast recovery render: refresh stale outputs, then rebuild the artifacts with
# the SAME RAM-aware parallel fleet as `rebuild-book-artifacts` (RENDER_JOBS
# frameworks at once, each quarto capped to RENDER_V8_HEAP_MIB). The old form ran
# `$(MAKE) slides` with no -j, rendering the four frameworks one after another.
render-fresh: refresh-stale
	$(RENDER_V8_ENV) $(MAKE) -j$(RENDER_JOBS) slides
	$(MAKE) html
	$(RENDER_V8_ENV) $(MAKE) -j$(RENDER_JOBS) pdfs
	$(MAKE) pdf-parts
