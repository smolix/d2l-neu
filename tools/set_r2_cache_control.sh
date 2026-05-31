#!/usr/bin/env bash
# Stamp Cache-Control on staging-d2l R2 objects so browsers REVALIDATE
# instead of serving a stale, heuristically-cached copy.
#
# Why this instead of a Cloudflare "cache purge"?
#   The staging site is served from a pub-*.r2.dev managed domain. That
#   domain belongs to Cloudflare, not to you, so the Cloudflare
#   cache-purge API cannot target it. In practice r2.dev serves objects
#   straight from R2 — responses carry no CF-Cache-Status / Age header, so
#   there is no edge cache to purge. The staleness you see after
#   re-uploading to the SAME url is your *browser*: an SVG/HTML with no
#   Cache-Control header is cached heuristically off Last-Modified. Setting
#   `Cache-Control: no-cache` makes the browser revalidate via ETag (cheap
#   304s) so re-uploads show up without a hard refresh.
#
# NOTE: this does NOT retroactively evict a copy already sitting in your
# browser cache. Do ONE hard refresh (Cmd/Ctrl+Shift+R) after running it;
# from then on no-cache keeps things fresh. For a real production domain
# you would instead use a custom domain + long max-age + cache-busting
# filenames; no-cache is the right choice for an actively-reviewed staging
# bucket.
#
# Reads credentials from .env (R2_ACCOUNT_ID + AWS creds).
#
# Usage:
#   ./tools/set_r2_cache_control.sh PATH [PATH ...]   # stamp these bucket paths
#   ./tools/set_r2_cache_control.sh --prefix img/     # stamp everything under a prefix
#   CACHE_CONTROL='public, max-age=300' ./tools/set_r2_cache_control.sh ...
#
# Paths are bucket-relative, e.g. chapter_preliminaries/probability.html

set -euo pipefail
cd "$(cd "$(dirname "$0")/.." && pwd)"
[[ -f .env ]] && source .env
: "${R2_ACCOUNT_ID:?Error: R2_ACCOUNT_ID not set (need .env)}"

BUCKET="${BUCKET:-staging-d2l}"
ENDPOINT="https://${R2_ACCOUNT_ID}.r2.cloudflarestorage.com"
REGION="${R2_REGION:-auto}"        # R2's S3 API requires region "auto"
CC="${CACHE_CONTROL:-no-cache}"

ct_for() {
  case "$1" in
    *.html) echo "text/html; charset=utf-8" ;;
    *.css)  echo "text/css; charset=utf-8" ;;
    *.js)   echo "application/javascript; charset=utf-8" ;;
    *.json) echo "application/json; charset=utf-8" ;;
    *.svg)  echo "image/svg+xml" ;;
    *.png)  echo "image/png" ;;
    *.pdf)  echo "application/pdf" ;;
    *)      echo "application/octet-stream" ;;
  esac
}

# In-place server-side copy with new metadata (REPLACE keeps the object but
# rewrites Content-Type + Cache-Control).
stamp() {
  local k="$1" ct
  ct="$(ct_for "$k")"
  aws s3 cp "s3://$BUCKET/$k" "s3://$BUCKET/$k" \
    --endpoint-url "$ENDPOINT" --region "$REGION" \
    --metadata-directive REPLACE --content-type "$ct" --cache-control "$CC" >/dev/null \
    && echo "  stamped  $k   ($ct; Cache-Control: $CC)"
}

if [[ "${1:-}" == "--prefix" ]]; then
  pfx="${2:?--prefix needs a value}"
  aws s3api list-objects-v2 --bucket "$BUCKET" --prefix "$pfx" \
    --endpoint-url "$ENDPOINT" --region "$REGION" \
    --query 'Contents[].Key' --output text \
    | tr '\t' '\n' | while read -r k; do [[ -n "$k" ]] && stamp "$k"; done
else
  [[ $# -gt 0 ]] || { echo "usage: $0 PATH [PATH ...]  |  --prefix PREFIX" >&2; exit 1; }
  for k in "$@"; do stamp "$k"; done
fi

echo "Done. Do one hard refresh (Cmd/Ctrl+Shift+R) to drop copies already in your browser cache."
