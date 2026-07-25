#!/usr/bin/env bash
# One-time remediation: fix the in-deck navbar "Slides" link on every deck
# already live in the staging-d2l R2 bucket.
#
# The deck overlay used to point "Slides" at ../../../index.html, which
# resolves to the book-root index.html instead of the slides landing
# (slides/index.html = ../../index.html from a deck). The source is fixed
# in _d2l-slides-overlay.html, so any *re-rendered* deck is already
# correct — but the ~520 frozen legacy decks on the bucket still carry the
# old link. Re-rendering them risks restyling/clipping (the reason they're
# frozen), so we patch the one link in place instead.
#
# Safe + byte-precise: downloads each deck, changes ONLY the matched
# substring, re-uploads with text/html. Idempotent — decks already fixed
# (e.g. the re-rendered north-star ones) match nothing and are skipped.
#
# Reads credentials from .env (R2_ACCOUNT_ID + AWS creds).
#
# Usage:
#   ./tools/patch_slides_navlink_r2.sh --dry-run   # list what would change
#   ./tools/patch_slides_navlink_r2.sh             # apply
#   UPLOAD_JOBS=16 ./tools/patch_slides_navlink_r2.sh

set -euo pipefail
cd "$(cd "$(dirname "$0")/.." && pwd)"

set -a; [[ -f .env ]] && source .env; set +a
: "${R2_ACCOUNT_ID:?Error: R2_ACCOUNT_ID not set (need .env)}"

BUCKET="staging-d2l"
export EP="https://${R2_ACCOUNT_ID}.r2.cloudflarestorage.com"
export REGION="${R2_REGION:-auto}"
JOBS="${UPLOAD_JOBS:-16}"
export DRY="${1:-}"

OLD='href="../../../index.html">Slides'
NEW='href="../../index.html">Slides'
export OLD NEW BUCKET

echo "Listing deck HTMLs under s3://${BUCKET}/slides/ ..."
KEYS=$(mktemp); trap 'rm -f "$KEYS"' EXIT
aws s3 ls "s3://${BUCKET}/slides/" --recursive --endpoint-url "$EP" --region "$REGION" \
  | awk '{print $4}' \
  | grep -E '^slides/(pytorch|tensorflow|jax|mxnet)/chapter_[^/]+/[^/]+\.html$' \
  | grep -v '/index\.html$' > "$KEYS"
echo "  $(wc -l < "$KEYS") deck(s)."

patch_one() {
  local key="$1" tmp; tmp=$(mktemp)
  if ! aws s3 cp "s3://${BUCKET}/$key" "$tmp" --endpoint-url "$EP" --region "$REGION" >/dev/null 2>&1; then
    echo "DLFAIL $key"; rm -f "$tmp"; return
  fi
  if grep -qF "$OLD" "$tmp"; then
    if [[ "$DRY" == "--dry-run" ]]; then
      echo "WOULDFIX $key"; rm -f "$tmp"; return
    fi
    # macOS/BSD sed; the slashes in the paths make '#' a safer delimiter.
    sed -i '' "s#$(printf '%s' "$OLD" | sed 's/[.[*\\/]/\\&/g')#${NEW}#g" "$tmp" 2>/dev/null \
      || python3 - "$tmp" <<'PY'
import sys
p=sys.argv[1]; s=open(p).read()
open(p,'w').write(s.replace('href="../../../index.html">Slides','href="../../index.html">Slides'))
PY
    if aws s3 cp "$tmp" "s3://${BUCKET}/$key" --endpoint-url "$EP" --region "$REGION" \
         --content-type "text/html; charset=utf-8" >/dev/null 2>&1; then
      echo "FIXED $key"
    else echo "ULFAIL $key"; fi
  else
    echo "SKIP $key"
  fi
  rm -f "$tmp"
}
export -f patch_one

xargs -P "$JOBS" -I{} bash -c 'patch_one "$@"' _ {} < "$KEYS" | tee /tmp/navlink_patch.log
echo "=== summary ==="
awk '{print $1}' /tmp/navlink_patch.log | sort | uniq -c
