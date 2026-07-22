# make/resources.mk — host resource detection & render/slide-fleet sizing
# Included by the top-level Makefile — see docs/build-system.md and
# reviews/build-system-cleanup-proposal-2026-07-22.md.

# ── Host resource detection — single source of truth ──────────────────
# tools/detect_resources.py probes GPUs (nvidia-smi), CPU cores, RAM, and the
# open-file/process ulimits, then derives every slot count below. It degrades
# cleanly on a CPU-only host (no nvidia-smi → 0 GPUs) and on macOS (no /proc →
# sysctl + vm_stat fallback), so the same Makefile runs unchanged on a 4×GPU
# Linux server and an Apple-silicon laptop. Run `make detect` to see the plan.
#
# Per-job footprints are read from the environment by detect_resources.py; we
# export them so command-line overrides reach it, e.g.
#   make GPU_MIB_PER_jax=14000 run-all-notebooks
# PyTorch / TensorFlow / MXNet share the lighter 7.5 GiB / 8-core budget (their
# EXTRA_ENV_* tuning caps resident VRAM and threads). JAX is heavier (~11.5 GiB)
# because every JAX process keeps the CPU backend + XLA pools alive
# (d2l.Module.plot host-transfers metrics via d2l.cpu()).
GPU_MIB_PER_LIGHT ?= 7680
GPU_MIB_PER_jax   ?= 11776
CPU_PER_LIGHT     ?= 8
CPU_PER_jax       ?= 8
export GPU_MIB_PER_LIGHT GPU_MIB_PER_jax CPU_PER_LIGHT CPU_PER_jax

# detect_resources.py is consulted once (its probe is cached in /tmp), and each
# knob is resolved into a SIMPLY-EXPANDED variable via the _detected helper.
# Why := and not ?=: a `?=` $(shell …) is recursively expanded, so python would
# re-run on *every* reference to the variable — dozens of times during a deep
# `make -n` graph walk. The ifndef guard means a command-line override (e.g.
# `make CPU_SLOTS=2`) is already defined, so detection is skipped and the
# override wins. On the 4×24GB / 64-core box: GPU_SLOTS=12, CPU_SLOTS=6; on a
# CPU-only laptop: NUM_GPUS=1 (slot→device fallback), GPU_SLOTS=1, CPU_SLOTS=cores/8.
_DR := python3 tools/detect_resources.py --get
define _detected
ifndef $(1)
$(1) := $$(shell $$(_DR) $(1))
endif
endef
$(foreach v,NUM_GPUS GPU_SLOTS CPU_SLOTS,$(eval $(call _detected,$(v))))

# Multi-GPU notebooks use exactly 2 GPUs at <=GPU_MIB_PER_LIGHT each, so they
# are memory-packed onto disjoint GPU pairs (verified: 3 fit on a 24 GiB card
# across all 4 frameworks). tools/detect_resources.py is the single source for
# the packing plan (pairs × per_pair) and the jax preallocation fraction.
$(foreach v,NUM_GPU_PAIRS MULTIGPU_PER_PAIR MULTIGPU_SLOTS JAX_MGPU_MEM_FRACTION,$(eval $(call _detected,$(v))))
# Slide/PDF render fleet sizing — keeps (jobs × quarto Deno/V8 heap) under the
# RAM budget so the parallel `quarto render` never SIGKILLs (rc=137) or V8-OOMs
# (rc=133). See tools/detect_resources.py (RENDER_V8_MIN_MIB / RENDER_JOBS).
$(foreach v,RENDER_JOBS RENDER_V8_HEAP_MIB,$(eval $(call _detected,$(v))))
RENDER_V8_ENV := QUARTO_DENO_V8_OPTIONS=--max-old-space-size=$(RENDER_V8_HEAP_MIB),--max-heap-size=$(RENDER_V8_HEAP_MIB)
# Per-framework slide-render workers. gen_slides.py now renders each deck in an
# ISOLATED clone project, so it parallelizes WITHIN a framework. With
# `make -j$(RENDER_JOBS) slides` running RENDER_JOBS frameworks at once, size
# workers so RENDER_JOBS × SLIDE_WORKERS ≈ cores — full CPU utilization without
# oversubscribing. Slide clones are light (--no-execute), so this is CPU- not
# RAM-bound. Override with SLIDE_WORKERS=N.
SLIDE_WORKERS ?= $(shell n=$$(nproc 2>/dev/null || echo 8); w=$$((n / $(RENDER_JOBS))); [ $$w -ge 1 ] && echo $$w || echo 1)
