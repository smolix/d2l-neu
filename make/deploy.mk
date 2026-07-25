# make/deploy.mk — publishing & deployment (Colab orphan branch, R2 upload).
# Included by the top-level Makefile — see docs/build-system.md and
# reviews/build-system-cleanup-proposal-2026-07-22.md.
#
# These promote what used to live only in loose scripts (and the throwaway
# scratchpad/fullrun.sh) into first-class, dependency-aware targets. Every
# recipe tees to logs/ like the rest of the build.

# ── Colab notebooks (orphan `notebooks` branch) ──────────────────────────────
# publish-colab is the VERIFIED path: tools/publish_colab_notebooks.sh builds the
# hosted staging (make hosted-notebooks + make check-hosted-notebooks) then
# force-pushes the ORPHAN branch AND verifies orphan-ness (no shared history with
# main — structurally un-PR-able, the PR-#30 safeguard), per-framework counts,
# inline (non-LFS) img/, and raw-URL reachability. `make deploy` uses this.
.PHONY: publish-colab publish-colab-verify
publish-colab:
	@mkdir -p $(LOGDIR)
	@tools/publish_colab_notebooks.sh 2>&1 | tee $(LOGDIR)/publish-colab-$(TS).log

publish-colab-verify:
	@mkdir -p $(LOGDIR)
	@tools/publish_colab_notebooks.sh --verify-only 2>&1 | tee $(LOGDIR)/publish-colab-verify-$(TS).log

# Lower-level branch push, kept for debug/advanced use (check-hosted-notebooks
# runs first). Prefer `make publish-colab`, which adds the safety verification.
.PHONY: dry-run-notebooks-branch publish-notebooks-branch
dry-run-notebooks-branch: check-hosted-notebooks
	@mkdir -p $(LOGDIR)
	@tools/publish_notebooks_branch.sh _hosted_notebooks notebooks --dry-run 2>&1 | tee $(LOGDIR)/dry-run-notebooks-branch-$(TS).log

publish-notebooks-branch: check-hosted-notebooks
	@mkdir -p $(LOGDIR)
	@tools/publish_notebooks_branch.sh _hosted_notebooks notebooks 2>&1 | tee $(LOGDIR)/publish-notebooks-branch-$(TS).log

# ── R2 site upload (Cloudflare, aws-cli → bucket staging-d2l) ─────────────────
# tools/upload_r2.sh sources .env for R2 creds and does a hash-manifest
# incremental upload of _book/. Preflight guards the common footguns: _book/ must
# be built, .env must carry the creds, and the aws CLI must exist.
R2_UPLOAD := tools/upload_r2.sh

.PHONY: _r2-preflight
_r2-preflight:
	@test -f _book/index.html || { echo "ERROR: _book/ not built — run 'make html' or 'make all-quick' first."; exit 1; }
	@test -f .env || { echo "ERROR: .env with R2 creds (R2_ACCOUNT_ID, AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY) not found."; exit 1; }
	@command -v aws >/dev/null 2>&1 || { echo "ERROR: aws CLI not found (R2 uses the S3-compatible API)."; exit 1; }

.PHONY: upload-r2 upload-r2-delete upload-r2-full upload-r2-dry-run
upload-r2: | _r2-preflight
	@mkdir -p $(LOGDIR)
	@$(R2_UPLOAD) 2>&1 | tee $(LOGDIR)/upload-r2-$(TS).log
upload-r2-delete: | _r2-preflight
	@mkdir -p $(LOGDIR)
	@$(R2_UPLOAD) --delete 2>&1 | tee $(LOGDIR)/upload-r2-$(TS).log
upload-r2-full: | _r2-preflight
	@mkdir -p $(LOGDIR)
	@$(R2_UPLOAD) --full 2>&1 | tee $(LOGDIR)/upload-r2-$(TS).log
# Show what WOULD upload (manifest diff), no side effects — good for verification.
upload-r2-dry-run: | _r2-preflight
	@mkdir -p $(LOGDIR)
	@$(R2_UPLOAD) --dry-run 2>&1 | tee $(LOGDIR)/upload-r2-$(TS).log

# ── Full deploy: build from the committed store, then publish ─────────────────
# all-quick → then publish-colab AND upload-r2-delete. Both are gated on the
# build passing, but are INDEPENDENT of each other (per Alex): a Colab-publish
# failure does not block the R2 upload; both run and the combined status is
# reported (deploy exits non-zero if either failed). Each sub-step logs to its
# own logs/ file; deploy also records the publish phase to logs/deploy-*.log.
# For a pristine deploy, run `make clean && make deploy`.
.PHONY: deploy
deploy:
	@mkdir -p $(LOGDIR)
	$(MAKE) all-quick
	@{ echo "=== Build OK — publishing (colab + R2, independent, both gated on build) ==="; \
	colab_rc=0; r2_rc=0; \
	$(MAKE) publish-colab || colab_rc=$$?; \
	$(MAKE) upload-r2-delete || r2_rc=$$?; \
	echo "deploy summary: colab=$$colab_rc r2=$$r2_rc"; \
	if [ $$colab_rc -ne 0 ] || [ $$r2_rc -ne 0 ]; then \
		echo "ERROR: deploy had failures (colab=$$colab_rc r2=$$r2_rc)"; exit 1; \
	fi; } 2>&1 | tee $(LOGDIR)/deploy-$(TS).log
