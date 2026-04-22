#!/usr/bin/env bash
# Upload _book/ to Cloudflare R2 bucket "staging-d2l".
#
# Reads credentials from .env in the project root.
#
# Usage:
#   ./tools/upload_r2.sh              # sync _book/ to staging-d2l
#   ./tools/upload_r2.sh --dry-run    # preview without uploading
#   ./tools/upload_r2.sh --delete     # also remove files from bucket not in _book/

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

if [[ -f "$PROJECT_DIR/.env" ]]; then
    # shellcheck disable=SC1091
    source "$PROJECT_DIR/.env"
fi

BUCKET="staging-d2l"
BOOK_DIR="${BOOK_DIR:-_book}"
DRY_RUN=""
DELETE=""

for arg in "$@"; do
    case "$arg" in
        --dry-run)  DRY_RUN="--dryrun" ;;
        --delete)   DELETE="--delete" ;;
        *)          echo "Unknown arg: $arg"; exit 1 ;;
    esac
done

if [[ -z "${R2_ACCOUNT_ID:-}" ]]; then
    echo "Error: R2_ACCOUNT_ID is not set." >&2
    exit 1
fi

if [[ ! -d "$BOOK_DIR" ]]; then
    echo "Error: $BOOK_DIR not found. Run 'quarto render' first." >&2
    exit 1
fi

ENDPOINT="https://${R2_ACCOUNT_ID}.r2.cloudflarestorage.com"
S3_ARGS=(--endpoint-url "$ENDPOINT")

echo "Uploading ${BOOK_DIR}/ → s3://${BUCKET}/"
echo "Endpoint: ${ENDPOINT}"
[[ -n "$DRY_RUN" ]] && echo "(dry run — no files will be uploaded)"
echo

# Pass 1: HTML files with explicit content-type and charset
aws s3 sync "$BOOK_DIR" "s3://${BUCKET}/" \
    "${S3_ARGS[@]}" \
    --content-type "text/html; charset=utf-8" \
    --exclude "*" --include "*.html" \
    $DRY_RUN $DELETE

# Pass 2: CSS/JS with correct content-types (R2 auto-detect is unreliable)
aws s3 sync "$BOOK_DIR" "s3://${BUCKET}/" \
    "${S3_ARGS[@]}" \
    --content-type "text/css; charset=utf-8" \
    --exclude "*" --include "*.css" \
    $DRY_RUN

aws s3 sync "$BOOK_DIR" "s3://${BUCKET}/" \
    "${S3_ARGS[@]}" \
    --content-type "application/javascript; charset=utf-8" \
    --exclude "*" --include "*.js" \
    $DRY_RUN

aws s3 sync "$BOOK_DIR" "s3://${BUCKET}/" \
    "${S3_ARGS[@]}" \
    --content-type "application/json; charset=utf-8" \
    --exclude "*" --include "*.json" \
    $DRY_RUN

# Pass 3: everything else (images, fonts, PDFs — auto-detected)
aws s3 sync "$BOOK_DIR" "s3://${BUCKET}/" \
    "${S3_ARGS[@]}" \
    --exclude "*.html" --exclude "*.css" --exclude "*.js" --exclude "*.json" \
    $DRY_RUN $DELETE

echo
echo "Done. Uploaded to s3://${BUCKET}/"
