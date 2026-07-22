# make/notebooks.mk — SCHEDULER CORE: notebook generation + resource-scheduled execution (logic unchanged; relocated verbatim)
# Included by the top-level Makefile — see docs/build-system.md and
# reviews/build-system-cleanup-proposal-2026-07-22.md.

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
	@[ -e "_notebooks/$*/img" ] || ln -s ../../img "_notebooks/$*/img"
	@if [ -L "_notebooks/$*/data" ]; then :; \
	elif [ -d "_notebooks/$*/data" ]; then \
		cp -rn "_notebooks/$*/data/." data/ 2>/dev/null || true; \
		rm -rf "_notebooks/$*/data"; \
		ln -s "$$(pwd)/data" "_notebooks/$*/data"; \
	else \
		ln -s "$$(pwd)/data" "_notebooks/$*/data"; \
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
$(foreach v,JAX_GPU_SLOTS JAX_CPU_SLOTS JAX_TOTAL_SLOTS,$(eval $(call _detected,$(v))))

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
                 D2L_JAX_GPU_SLOTS=$(JAX_GPU_SLOTS) D2L_JAX_CPU_SLOTS=$(JAX_CPU_SLOTS) \
                 D2L_JAX_TOTAL_SLOTS=$(JAX_TOTAL_SLOTS)

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
$(eval $(call _detected,MXNET_GPU_SLOTS))
MXNET_CPU_SLOTS ?= 2
# MXNET_CUDNN_LIB_CHECKING=0: the 20260607.1 wheel is compiled against cuDNN
# 9.23 but the only pip-available nvidia-cudnn-cu13 is 9.22 (ABI-compatible —
# conv/matmul verified correct). Without this, every GPU op prints a "cuDNN lib
# mismatch" line to stderr that gets captured into the notebook outputs.
EXTRA_ENV_mxnet := OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 \
                   MXNET_CUDNN_LIB_CHECKING=0 \
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
	@mkdir -p $(LOGDIR)
	@python3 tools/detect_resources.py --report || true
	@python3 tools/notebook_run_plan.py --framework $* || true
	@$(SCHED_ENV) python3 tools/notebook_scheduler.py --frameworks $* \
		2>&1 | tee -a $(LOGDIR)/scheduler-$*-$(TS).log; rc=$${PIPESTATUS[0]}; \
	python3 tools/notebook_run_summary.py --framework $* || true; \
	exit $$rc

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
#
# Each queue is fed a single INTERLEAVED flat list of every framework's stamps
# (sorted by the path AFTER the framework, so a notebook's 4 framework variants
# are adjacent) rather than 4 grouped per-framework phony goals. GNU make drains
# one goal's prerequisites before the next, so the grouped form ran the
# frameworks ~sequentially: when a framework's own cap (e.g. JAX_GPU_SLOTS=8) is
# below GPU_SLOTS the leftover slots sat idle, and each framework left a slow-
# straggler tail. Interleaving keeps the GPU pool full with a framework mix
# throughout — one combined tail instead of four.
# Resource env handed to the unified scheduler (tools/notebook_scheduler.py):
# per-GPU slot capacity + per-GPU VRAM (heterogeneous-aware) + CPU slots.
$(foreach v,GPU_SLOTS_PER GPU_VRAM_PER GPU_MIB_PER_SLOT,$(eval $(call _detected,$(v))))
SCHED_ENV = D2L_GPU_SLOTS_PER='$(GPU_SLOTS_PER)' D2L_GPU_VRAM_PER='$(GPU_VRAM_PER)' \
	D2L_GPU_MIB_PER_SLOT=$(GPU_MIB_PER_SLOT) D2L_CPU_SLOTS=$(CPU_SLOTS) \
	D2L_JAX_TOTAL_SLOTS=$(JAX_TOTAL_SLOTS)

# The unified scheduler (tools/notebook_scheduler.py) replaces the old "two
# background `make -jN` queues" orchestration: it owns the GPU/CPU/multi-GPU
# slot pools (resource allocation) and dispatches one item per free slot while
# never running two framework variants of the same notebook at once (scheduling
# — sequences a notebook's frameworks one at a time, killing the shared-dataset
# reorg race structurally). It shells `make <stamp>` per notebook with the
# device assigned via D2L_ASSIGNED_CUDA, reusing the per-framework EXEC_RULE env.
run-all-notebooks: notebooks
	@mkdir -p $(LOGDIR)
	@python3 tools/detect_resources.py --report || true
	@python3 tools/notebook_run_plan.py || true
	@$(SCHED_ENV) python3 tools/notebook_scheduler.py \
		2>&1 | tee -a $(LOGDIR)/scheduler-$(TS).log; rc=$${PIPESTATUS[0]}; \
	python3 tools/notebook_run_summary.py || true; \
	exit $$rc
