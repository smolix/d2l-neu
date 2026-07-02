#!/usr/bin/env bash
# Surgically upload ONLY the north-star slide files to the staging-d2l R2
# bucket — the exact set emitted by stage_northstar_slides.sh in
# _book/slides/.northstar-upload.txt.
#
# Unlike upload_r2.sh (a whole-_book hash sync), this touches nothing but
# the listed files, so the ~137 live legacy decks are guaranteed untouched
# — the right tool for a gradual, low-risk migration. Each file is copied
# with an explicit content-type; everything else on the bucket is left
# exactly as it is.
#
# Reads credentials from .env (R2_ACCOUNT_ID + AWS creds), same as
# upload_r2.sh.
#
# Usage:
#   ./tools/upload_northstar_r2.sh            # upload the staged set
#   ./tools/upload_northstar_r2.sh --dry-run  # preview, upload nothing

set -euo pipefail
cd "$(cd "$(dirname "$0")/.." && pwd)"

# `set -a` so bare `KEY=value` lines in .env are exported to the aws subprocess
# (else "Unable to locate credentials"); .env uses no `export` prefixes.
[[ -f .env ]] && { set -a; source .env; set +a; }
: "${R2_ACCOUNT_ID:?Error: R2_ACCOUNT_ID not set (need .env)}"

BUCKET="staging-d2l"
BOOK_DIR="_book"
LIST="$BOOK_DIR/slides/.northstar-upload.txt"
ENDPOINT="https://${R2_ACCOUNT_ID}.r2.cloudflarestorage.com"

DRY=""
[[ "${1:-}" == "--dry-run" ]] && DRY="--dryrun"

[[ -f "$LIST" ]] || { echo "Error: $LIST not found. Run ./tools/stage_northstar_slides.sh first." >&2; exit 1; }

n=$(grep -c . "$LIST")
echo "Uploading $n north-star file(s) to s3://${BUCKET}/  ${DRY:+(dry run)}"

while IFS= read -r rel; do
  [[ -n "$rel" ]] || continue
  local_path="$BOOK_DIR/$rel"
  if [[ ! -f "$local_path" ]]; then
    echo "  !! missing locally, skipping: $local_path" >&2
    continue
  fi
  case "$rel" in
    *.html) ct="text/html; charset=utf-8" ;;
    *.css)  ct="text/css; charset=utf-8" ;;
    *.svg)  ct="image/svg+xml" ;;
    *.json) ct="application/json; charset=utf-8" ;;
    *.js)   ct="application/javascript; charset=utf-8" ;;
    *)      ct="" ;;
  esac
  # R2's S3 API requires region "auto" (the local aws config may default
  # to a real AWS region, which R2 rejects).
  args=(--endpoint-url "$ENDPOINT" --region "${R2_REGION:-auto}")
  [[ -n "$ct" ]] && args+=(--content-type "$ct")
  # staging is served from pub-*.r2.dev (no purgeable CF cache); no-cache
  # makes browsers revalidate via ETag so re-uploads show without a hard
  # refresh. Override CACHE_CONTROL=... for a cacheable prod domain.
  args+=(--cache-control "${CACHE_CONTROL:-no-cache}")
  [[ -n "$DRY" ]] && args+=("$DRY")
  echo "  → $rel"
  aws s3 cp "$local_path" "s3://${BUCKET}/${rel}" "${args[@]}" >/dev/null
done < "$LIST"

echo "Done.${DRY:+ (dry run — nothing uploaded)}"
