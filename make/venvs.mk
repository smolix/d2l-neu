# make/venvs.mk — UV framework venvs, mxnet preflight, Quarto build venv, kernels
# Included by the top-level Makefile — see docs/build-system.md and
# reviews/build-system-cleanup-proposal-2026-07-22.md.

# ── UV venvs ───────────────────────────────────────────────
#
# Preflight runs before every `uv sync`: if pyproject pins the mxnet wheel
# at a missing file path, `uv sync` for ANY framework fails during lock
# validation (uv walks all path-direct sources, even those in conflicting
# extras). The preflight is a fast no-op when the pin matches disk; when
# the wheel is missing it auto-bumps the pin to the newest `../mxnet/dist/`
# wheel and relocks. See tools/preflight_mxnet_pin.py for the trace.
.venv-%/.synced: pyproject.toml uv.lock | .preflight.mxnet-pin
	@mkdir -p $(LOGDIR)
	@echo "=== Syncing venv for $* ==="
	UV_PROJECT_ENVIRONMENT=.venv-$* uv sync --extra $* --extra run 2>&1 | tee $(LOGDIR)/venv-$*-$(TS).log
	@touch $@

# TensorFlow venv needs an extra post-sync step: the TF 2.21 wheel's RUNPATH
# omits nvidia/cusolver/lib, so TF silently falls back to CPU. Applying the
# cusolver symlinks at sync time means a fresh `uv sync` doesn't leave the venv
# in a CPU-only state. See tools/check_runtime_deps.py for the gory detail.
.venv-tensorflow/.synced: pyproject.toml uv.lock tools/check_runtime_deps.py | .preflight.mxnet-pin
	@mkdir -p $(LOGDIR)
	@echo "=== Syncing venv for tensorflow ==="
	UV_PROJECT_ENVIRONMENT=.venv-tensorflow uv sync --extra tensorflow --extra run 2>&1 | tee $(LOGDIR)/venv-tensorflow-$(TS).log
	@echo "=== Applying TF cusolver RUNPATH workaround ==="
	@python3 tools/check_runtime_deps.py tensorflow
	@touch $@

# Order-only preflight target. Always runs, but only mutates the project
# when the pinned mxnet wheel is missing on disk.
.PHONY: .preflight.mxnet-pin
.preflight.mxnet-pin:
	@python3 tools/preflight_mxnet_pin.py

# Dedicated venv that installs the Quarto CLI (via the quarto-cli PyPI package).
# Used by HTML and PDF recipes so `quarto` is a real declared dependency rather
# than an implicit system tool. Override-style rule for the `build` extra:
# `uv sync --extra build` is enough; no `--extra run` (no jupyter needed).
.venv-build/.synced: pyproject.toml uv.lock
	@mkdir -p $(LOGDIR)
	@echo "=== Syncing build venv (quarto-cli) ==="
	UV_PROJECT_ENVIRONMENT=.venv-build uv sync --extra build 2>&1 | tee $(LOGDIR)/venv-build-$(TS).log
	@touch $@

# Resolved path to quarto: prefer the project-local build venv when present.
QUARTO := .venv-build/bin/quarto

venv-%: .venv-%/.synced
	@echo "Venv .venv-$* is ready"

.venv-mxnet/.runtime-deps: .venv-mxnet/.synced tools/check_runtime_deps.py
	@echo "=== Checking MXNet native runtime deps ==="
	@python3 tools/check_runtime_deps.py mxnet
	@touch $@

RUNTIME_DEPS_mxnet := .venv-mxnet/.runtime-deps

# Pull the latest mxnet wheel URL from
# https://github.com/smolix/mxnet/releases/latest into pyproject.toml.
# Manual (not on every sync) so builds stay reproducible and we don't
# hit the unauthenticated GitHub API rate limit.
.PHONY: update-mxnet-wheel
update-mxnet-wheel:
	@mkdir -p $(LOGDIR)
	python3 tools/update_mxnet_wheel.py 2>&1 | tee $(LOGDIR)/update-mxnet-wheel-$(TS).log
	@echo "Run 'make venv-mxnet' to apply the bumped wheel."

# ── Jupyter kernels ────────────────────────────────────────
# Register one ipykernel per framework so VS Code can auto-select the
# right interpreter from the .ipynb's metadata.kernelspec.name.

.PHONY: kernels
kernels: $(addprefix .venv-,$(addsuffix /.synced,$(FRAMEWORKS)))
	@for fw in $(FRAMEWORKS); do \
	  echo "Registering kernel d2l-$$fw"; \
	  .venv-$$fw/bin/python -m ipykernel install --user \
	    --name d2l-$$fw --display-name "d2l ($$fw)"; \
	done
	@echo "All d2l kernels registered."
