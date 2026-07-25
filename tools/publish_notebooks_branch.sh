#!/usr/bin/env bash
set -euo pipefail

staging="${1:-_hosted_notebooks}"
branch="${2:-notebooks}"
dry_run="${3:-}"
if [[ -n "$dry_run" && "$dry_run" != "--dry-run" ]]; then
  echo "usage: $0 [staging-dir] [branch] [--dry-run]" >&2
  exit 2
fi
root="$(git rev-parse --show-toplevel)"
staging="$(cd "$staging" && pwd)"
worktree="$(mktemp -d -t d2l-notebooks-XXXXXX)"
temp_branch="generated-notebooks-$$-${RANDOM}"

cleanup() {
  git -C "$root" worktree remove --force "$worktree" >/dev/null 2>&1 || true
  git -C "$root" branch -D "$temp_branch" >/dev/null 2>&1 || true
}
trap cleanup EXIT

# Publish the hosted notebooks as a git ORPHAN branch: it shares NO history with
# main, so GitHub structurally refuses any PR that would merge it back into main
# — the safeguard against dumping regenerable notebooks into the source tree (see
# the PR #30 incident). Colab needs only the raw .ipynb + inline img/ assets
# (img/ is NOT LFS-tracked — only outputs/** is — so Colab fetches raw blobs),
# not history or PR-reviewability, so we push the orphan directly (no PR needed).
git -C "$root" worktree add --detach "$worktree" HEAD
git -C "$worktree" checkout --orphan "$temp_branch"
git -C "$worktree" rm -rf . >/dev/null 2>&1 || true   # drop the inherited tree
cp -a "$staging"/. "$worktree"/
git -C "$worktree" add --all
git -C "$worktree" commit -q -m "Publish hosted Colab notebooks (orphan serving artifact)"

if [[ "$dry_run" == "--dry-run" ]]; then
  echo "Dry run: generated ORPHAN commit $(git -C "$worktree" rev-parse --short HEAD)"
  echo "Dry run: would force-push HEAD to refs/heads/$branch (orphan)"
  exit 0
fi

remote_tip="$(git -C "$root" ls-remote origin "refs/heads/$branch" | awk '{print $1}')"
if [[ -n "$remote_tip" ]]; then
  git -C "$worktree" push --force-with-lease="refs/heads/$branch:$remote_tip" \
    origin "HEAD:refs/heads/$branch"
else
  git -C "$worktree" push origin "HEAD:refs/heads/$branch"
fi
