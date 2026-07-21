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

# Base the published branch on main (NOT an orphan) so it shares history with
# main and stays reviewable via a PR. The hosted notebooks are overlaid on
# main's tree; their image assets live under img/, which is not LFS-tracked
# (only outputs/** is), so Colab renders them from raw blobs. Earlier this
# branch was a git orphan, which GitHub refuses to open a PR for
# ("no history in common with main").
git -C "$root" fetch --quiet origin main
git -C "$root" worktree add --detach "$worktree" origin/main
git -C "$worktree" checkout -B "$temp_branch"
cp -a "$staging"/. "$worktree"/
git -C "$worktree" add --all
git -C "$worktree" commit -q -m "Publish hosted Colab notebooks (re-rooted on main)"

if [[ "$dry_run" == "--dry-run" ]]; then
  echo "Dry run: generated commit $(git -C "$worktree" rev-parse --short HEAD)"
  echo "Dry run: would push HEAD to refs/heads/$branch"
  exit 0
fi

remote_tip="$(git -C "$root" ls-remote origin "refs/heads/$branch" | awk '{print $1}')"
if [[ -n "$remote_tip" ]]; then
  git -C "$worktree" push --force-with-lease="refs/heads/$branch:$remote_tip" \
    origin "HEAD:refs/heads/$branch"
else
  git -C "$worktree" push origin "HEAD:refs/heads/$branch"
fi
