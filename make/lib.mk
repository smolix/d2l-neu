# make/lib.mk — d2l Python library (rebuilt from #@save blocks)
# Included by the top-level Makefile — see docs/build-system.md and
# reviews/build-system-cleanup-proposal-2026-07-22.md.

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
