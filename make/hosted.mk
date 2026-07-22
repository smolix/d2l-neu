# make/hosted.mk — public hosted/Colab notebook staging + downloadable zips
# Included by the top-level Makefile — see docs/build-system.md and
# reviews/build-system-cleanup-proposal-2026-07-22.md.

# Public, provider-neutral notebook tree. NumPy variants are derived from
# framework-independent sources in the PyTorch generation tree.
hosted-env-locks:
	@mkdir -p $(LOGDIR)
	@python3 tools/export_hosted_env.py generate 2>&1 | tee $(LOGDIR)/hosted-env-locks-$(TS).log

check-hosted-env-locks:
	@mkdir -p $(LOGDIR)
	@python3 tools/export_hosted_env.py check 2>&1 | tee $(LOGDIR)/check-hosted-env-locks-$(TS).log

hosted-notebooks: check-hosted-env-locks notebooks-pytorch notebooks-tensorflow notebooks-jax
	@mkdir -p $(LOGDIR)
	@python3 tools/build_hosted_notebooks.py build 2>&1 | tee $(LOGDIR)/hosted-notebooks-$(TS).log

check-hosted-runtime-contracts: check-hosted-env-locks d2l/.built \
		.venv-pytorch/.synced .venv-tensorflow/.synced .venv-jax/.synced
	@mkdir -p $(LOGDIR)
	@{ env MPLBACKEND=Agg OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 \
		.venv-pytorch/bin/python tools/check_hosted_runtime.py pytorch \
	&& env MPLBACKEND=Agg TF_FORCE_GPU_ALLOW_GROWTH=true \
		TF_NUM_INTRAOP_THREADS=2 TF_NUM_INTEROP_THREADS=2 \
		OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 \
		.venv-tensorflow/bin/python tools/check_hosted_runtime.py tensorflow \
	&& env MPLBACKEND=Agg XLA_PYTHON_CLIENT_PREALLOCATE=false \
		OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 \
		.venv-jax/bin/python tools/check_hosted_runtime.py jax; \
	} 2>&1 | tee $(LOGDIR)/check-hosted-runtime-contracts-$(TS).log

check-hosted-notebooks: hosted-notebooks check-hosted-runtime-contracts
	@mkdir -p $(LOGDIR)
	@{ PYTHONPATH=tools python3 -m unittest \
		tools/test_export_hosted_env.py tools/test_check_hosted_runtime.py \
		tools/test_build_hosted_notebooks.py tools/test_check_pip_delta.py \
		tools/test_run_hosted_docker.py \
	&& python3 tools/build_hosted_notebooks.py check; \
	} 2>&1 | tee $(LOGDIR)/check-hosted-notebooks-$(TS).log

# Slow, opt-in provider compatibility canary. It runs serial, executes no full
# notebook, and hard-limits every ephemeral container. Select cases or storage
# policy without changing the Makefile, for example:
#   make check-hosted-docker HOSTED_DOCKER_ARGS='--device cpu'
#   make check-hosted-docker HOSTED_DOCKER_ARGS='--prune-other-image'
HOSTED_DOCKER_ARGS ?=
check-hosted-docker: check-hosted-env-locks
	@mkdir -p $(LOGDIR)
	@python3 tools/run_hosted_docker.py $(HOSTED_DOCKER_ARGS) 2>&1 | tee $(LOGDIR)/check-hosted-docker-$(TS).log

check-hosted-docker-cpu: check-hosted-env-locks
	@mkdir -p $(LOGDIR)
	@python3 tools/run_hosted_docker.py --device cpu $(HOSTED_DOCKER_ARGS) 2>&1 | tee $(LOGDIR)/check-hosted-docker-cpu-$(TS).log

check-hosted-docker-gpu: check-hosted-env-locks
	@mkdir -p $(LOGDIR)
	@python3 tools/run_hosted_docker.py --device gpu $(HOSTED_DOCKER_ARGS) 2>&1 | tee $(LOGDIR)/check-hosted-docker-gpu-$(TS).log


# ── Downloadable notebook zips (per framework) ────────────
# A first-class build output, linked from the navbar "Notebooks" menu: one
# d2l-<fw>.zip of that framework's executed notebooks. Building the ZIP is
# CPU-only: code comes from _notebooks/, outputs from the committed store, and
# pinned reader environments from notebook_envs/. The ZIP also carries the d2l
# source, so readers can select its CPU or GPU lock and run it directly.
# Fixed ZIP timestamps keep unchanged builds byte-identical for the R2 sync.
NOTEBOOK_ZIP_DIR := _book/notebooks

notebook-env-locks:
	@mkdir -p $(LOGDIR)
	python3 tools/lock_notebook_envs.py \
		$(if $(FRAMEWORKS_FILTER),--frameworks $(FRAMEWORKS_FILTER)) \
		2>&1 | tee $(LOGDIR)/notebook-env-locks-$(TS).log

notebook-zips: notebooks
	@mkdir -p $(LOGDIR) $(NOTEBOOK_ZIP_DIR)
	@echo "=== Building per-framework notebook zips → $(NOTEBOOK_ZIP_DIR)/ ==="
	python3 tools/build_notebook_zips.py --out-dir $(NOTEBOOK_ZIP_DIR) \
		$(if $(FRAMEWORKS_FILTER),--frameworks $(FRAMEWORKS_FILTER)) \
		2>&1 | tee $(LOGDIR)/notebook-zips-$(TS).log
