#!/usr/bin/env bash
#
# Publish the Colab notebooks to the `notebooks` serving branch — end to end.
#
# WHY THIS EXISTS / THE ONE RULE
# ------------------------------
# The `notebooks` branch is a git ORPHAN: it has no shared history with `main`
# and holds ONLY the served `.ipynb` files plus their inline image assets. It is
# a *serving artifact* consumed by "Open in Colab" links
# (https://colab.research.google.com/github/<owner>/<repo>/blob/notebooks/<path>).
# It must NEVER be merged into or PR'd against `main` — doing so dumps hundreds of
# regenerable notebooks into the source tree. Keeping it an orphan makes that
# mistake impossible: GitHub refuses a PR between branches with no common history.
#
# This script (1) builds the hosted notebooks, (2) force-pushes them to the
# orphan branch via tools/publish_notebooks_branch.sh, and (3) VERIFIES the
# result — most importantly that the branch really is an orphan. If any check
# fails it exits non-zero so a bad publish can't pass silently.
#
# Usage:
#   tools/publish_colab_notebooks.sh [--no-build] [--dry-run] [--no-verify]
#                                     [--verify-only]
#     --no-build     skip `make hosted-notebooks`; publish the existing staging dir
#     --dry-run      build/verify but do not push (delegates --dry-run to the pusher)
#     --no-verify    skip post-push verification (not recommended)
#     --verify-only  do not build or publish; just verify the current remote branch
#                    (orphan-ness, notebook counts, inline assets, reachability)
#
# Env overrides: NOTEBOOKS_BRANCH (default notebooks), NOTEBOOKS_STAGING
#   (default _hosted_notebooks), NOTEBOOKS_REMOTE (default origin),
#   NOTEBOOKS_BASE (default main — the branch the orphan must NOT share history with).

set -euo pipefail

BRANCH="${NOTEBOOKS_BRANCH:-notebooks}"
STAGING="${NOTEBOOKS_STAGING:-_hosted_notebooks}"
REMOTE="${NOTEBOOKS_REMOTE:-origin}"
BASE="${NOTEBOOKS_BASE:-main}"
BUILD=1
DRY_RUN=
VERIFY=1
VERIFY_ONLY=0

for a in "$@"; do
  case "$a" in
    --no-build)    BUILD=0 ;;
    --dry-run)     DRY_RUN=--dry-run ;;
    --no-verify)   VERIFY=0 ;;
    --verify-only) VERIFY_ONLY=1; BUILD=0 ;;
    -h|--help)     grep '^#' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) echo "unknown arg: $a (see --help)" >&2; exit 2 ;;
  esac
done

root="$(git rev-parse --show-toplevel)"
cd "$root"

# ── 1. Build ────────────────────────────────────────────────────────────────
if [[ "$VERIFY_ONLY" == 1 ]]; then
  echo "==> Verify-only: not building or publishing; checking '$BRANCH' as published."
elif [[ "$BUILD" == 1 ]]; then
  echo "==> Building hosted notebooks (make hosted-notebooks && check-hosted-notebooks)…"
  make hosted-notebooks
  make check-hosted-notebooks
fi
# ── 2. Publish (orphan force-push) ────────────────────────────────────────────
# publish_notebooks_branch.sh creates the branch with `git checkout --orphan`,
# so the served branch never shares history with main. Do not "fix" that.
if [[ "$VERIFY_ONLY" == 0 ]]; then
  if [[ ! -d "$STAGING" ]]; then
    echo "ERROR: staging dir '$STAGING' not found. Run without --no-build, or set NOTEBOOKS_STAGING." >&2
    exit 1
  fi
  echo "==> Publishing '$STAGING' → orphan branch '$BRANCH'…"
  tools/publish_notebooks_branch.sh "$STAGING" "$BRANCH" $DRY_RUN
  if [[ -n "$DRY_RUN" ]]; then
    echo "==> Dry run: nothing pushed, verification skipped."
    exit 0
  fi
fi

# ── 3. Verify ─────────────────────────────────────────────────────────────────
if [[ "$VERIFY" == 0 ]]; then
  echo "==> Verification skipped (--no-verify)."
  exit 0
fi

echo "==> Verifying '$BRANCH'…"
# Capture stable SHAs — a later `git fetch` overwrites FETCH_HEAD, so never
# reuse FETCH_HEAD across fetches.
git fetch --quiet "$REMOTE" "$BRANCH"
nb_tip="$(git rev-parse FETCH_HEAD)"
git fetch --quiet "$REMOTE" "$BASE" 2>/dev/null || true
base_tip="$(git rev-parse FETCH_HEAD 2>/dev/null || echo '')"

fail=0

# 3a. THE load-bearing check: the branch must be an orphan (no shared history
#     with base), so it can never be PR'd/merged into main.
if [[ -n "$base_tip" ]] && git merge-base "$nb_tip" "$base_tip" >/dev/null 2>&1; then
  echo "  ✗ '$BRANCH' shares history with '$BASE' — it is NOT an orphan serving" >&2
  echo "    artifact and could be merged into $BASE. Aborting as unsafe." >&2
  fail=1
else
  echo "  ✓ orphan: no shared history with '$BASE' (cannot be PR'd/merged there)"
fi

# 3b. Notebooks are present.
total="$(git ls-tree -r --name-only "$nb_tip" | grep -c '\.ipynb$' || true)"
if [[ "${total:-0}" -gt 0 ]]; then
  echo "  ✓ $total notebooks published"
  for fw in pytorch jax tensorflow mxnet numpy; do
    n="$(git ls-tree -r --name-only "$nb_tip" | grep -c "^$fw/.*\.ipynb$" || true)"
    [[ "${n:-0}" -gt 0 ]] && echo "      $fw: $n"
  done
else
  echo "  ✗ no .ipynb files on '$BRANCH'" >&2
  fail=1
fi

# 3c. Image assets must be INLINE, not LFS pointers, or Colab shows broken
#     figures (it fetches raw blobs, which for an LFS file is the pointer text).
sample_svg="$(git ls-tree -r --name-only "$nb_tip" -- img 2>/dev/null | grep '\.svg$' | head -1 || true)"
if [[ -n "$sample_svg" ]]; then
  if git cat-file -p "$nb_tip:$sample_svg" 2>/dev/null | head -1 | grep -q 'version https://git-lfs'; then
    echo "  ✗ image assets are LFS pointers (e.g. $sample_svg) — Colab won't render them." >&2
    echo "    Ensure img/ is not LFS-tracked on this branch (.gitattributes should scope LFS to outputs/**)." >&2
    fail=1
  else
    echo "  ✓ image assets stored inline (Colab-renderable)"
  fi
fi

# 3d. Best-effort live reachability of one raw notebook URL.
if command -v curl >/dev/null 2>&1; then
  slug="$(git remote get-url "$REMOTE" 2>/dev/null | sed -E 's#\.git$##; s#^.*github\.com[:/]##')"
  sample_nb="$(git ls-tree -r --name-only "$nb_tip" | grep -E '^(pytorch|jax)/.*\.ipynb$' | head -1 || true)"
  if [[ -n "$slug" && -n "$sample_nb" ]]; then
    # Give GitHub a moment to serve the just-pushed ref.
    sleep 3
    code="$(curl -s -o /dev/null -w '%{http_code}' \
      "https://raw.githubusercontent.com/$slug/$BRANCH/$sample_nb" || echo 000)"
    if [[ "$code" == 200 ]]; then
      echo "  ✓ raw reachability OK ($sample_nb → HTTP 200)"
    else
      echo "  ! raw reachability check returned HTTP $code for $sample_nb (may be CDN lag)"
    fi
  fi
fi

if [[ "$fail" != 0 ]]; then
  echo "==> PUBLISH VERIFICATION FAILED." >&2
  exit 1
fi

slug="${slug:-<owner>/<repo>}"
echo "==> Done: '$BRANCH' @ ${nb_tip:0:9}"
echo "    Colab: https://colab.research.google.com/github/$slug/blob/$BRANCH/pytorch/<chapter>/<file>.ipynb"
