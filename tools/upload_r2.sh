#!/usr/bin/env bash
# Upload _book/ to Cloudflare R2 bucket "staging-d2l".
#
# Hash-based incremental sync: maintains `.upload-manifest-<bucket>.txt`
# with `sha256  path` for every file uploaded last time. On each run we
# hash _book/ again, diff against the manifest, and upload only the
# files whose content changed. quarto rewrites file mtimes on every
# render so `aws s3 sync` (which compares size+mtime) re-uploads
# everything; hashing avoids that.
#
# Reads credentials from .env in the project root.
#
# Usage:
#   ./tools/upload_r2.sh              # incremental sync of _book/
#   ./tools/upload_r2.sh --dry-run    # preview without uploading or
#                                     # touching the manifest
#   ./tools/upload_r2.sh --delete     # also remove bucket files that
#                                     # no longer exist locally
#   ./tools/upload_r2.sh --full       # ignore manifest, re-upload all

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

if [[ -f "$PROJECT_DIR/.env" ]]; then
    # shellcheck disable=SC1091
    source "$PROJECT_DIR/.env"
fi

BUCKET="staging-d2l"
BOOK_DIR="${BOOK_DIR:-_book}"
MANIFEST_FILE="${PROJECT_DIR}/.upload-manifest-${BUCKET}.txt"

DRY_RUN=""
DELETE=false
FORCE_FULL=false

for arg in "$@"; do
    case "$arg" in
        --dry-run)  DRY_RUN="--dryrun" ;;
        --delete)   DELETE=true ;;
        --full)     FORCE_FULL=true ;;
        *)          echo "Unknown arg: $arg"; exit 1 ;;
    esac
done

if [[ -z "${R2_ACCOUNT_ID:-}" ]]; then
    echo "Error: R2_ACCOUNT_ID is not set." >&2
    exit 1
fi

if [[ ! -d "$BOOK_DIR" ]]; then
    echo "Error: $BOOK_DIR not found. Run 'make html' first." >&2
    exit 1
fi

ENDPOINT="https://${R2_ACCOUNT_ID}.r2.cloudflarestorage.com"
S3_ARGS=(--endpoint-url "$ENDPOINT")

# ── Hash the local tree ─────────────────────────────────────
# sha256sum format: `<64-hex>  <path>` — 64 + 2 + path.
echo "Hashing $BOOK_DIR/ ..."
NEW_MANIFEST=$(mktemp)
WORK_DIR=$(mktemp -d)
trap 'rm -f "$NEW_MANIFEST"; rm -rf "$WORK_DIR"' EXIT

# `-L` follows symlinks so e.g. the per-fw `slides/<fw>/img` symlink
# resolves to its content (matches what `aws s3 sync` would upload).
# Full-line lex sort (not `-k2`) keeps both manifests in the byte-wise
# order that `comm` expects.
(cd "$BOOK_DIR" && find -L . -type f -print0 \
  | xargs -0 -P 8 sha256sum) | LC_ALL=C sort > "$NEW_MANIFEST"

TOTAL=$(wc -l < "$NEW_MANIFEST")
echo "  $TOTAL files."

# ── Decide: full or incremental ─────────────────────────────
MODE="incremental"
if [[ ! -f "$MANIFEST_FILE" ]]; then
    MODE="full"
    echo "No prior manifest — performing full sync."
elif $FORCE_FULL; then
    MODE="full"
    echo "--full requested — performing full sync."
fi

# ── Helper: pick a Content-Type for a path ──────────────────
content_type_for() {
    local path=$1
    case "$path" in
        *.html) echo "text/html; charset=utf-8" ;;
        *.css)  echo "text/css; charset=utf-8" ;;
        *.js)   echo "application/javascript; charset=utf-8" ;;
        *.json) echo "application/json; charset=utf-8" ;;
        *)      echo "" ;;  # let aws auto-detect
    esac
}
export -f content_type_for

# ── Upload one file with proper content-type ────────────────
upload_one() {
    local rel=$1
    local ct
    ct=$(content_type_for "$rel")
    local args=("${S3_ARGS[@]}")
    if [[ -n "$ct" ]]; then
        args+=(--content-type "$ct")
    fi
    aws s3 cp "$BOOK_DIR/$rel" "s3://${BUCKET}/$rel" \
        "${args[@]}" $DRY_RUN
}

# We export S3_ARGS / BOOK_DIR / BUCKET / DRY_RUN so xargs -I{} bash -c
# can call upload_one in subshells.
export S3_ARGS_STR="${S3_ARGS[*]}"
export BOOK_DIR BUCKET DRY_RUN

upload_changed() {
    local list=$1
    local count
    count=$(wc -l < "$list")
    if [[ $count -eq 0 ]]; then
        echo "  Nothing to upload."
        return
    fi
    echo "  Uploading $count file(s)..."
    # Parallel uploads. Each line in $list is a relative path under
    # $BOOK_DIR. We pipe-feed xargs to handle paths with spaces via -d.
    # shellcheck disable=SC2016
    xargs -a "$list" -d '\n' -P 8 -I{} bash -c '
        rel="$1"
        case "$rel" in
            *.html) ct="text/html; charset=utf-8" ;;
            *.css)  ct="text/css; charset=utf-8" ;;
            *.js)   ct="application/javascript; charset=utf-8" ;;
            *.json) ct="application/json; charset=utf-8" ;;
            *)      ct="" ;;
        esac
        if [[ -n "$ct" ]]; then
            aws s3 cp "$BOOK_DIR/$rel" "s3://${BUCKET}/$rel" \
                $S3_ARGS_STR --content-type "$ct" $DRY_RUN
        else
            aws s3 cp "$BOOK_DIR/$rel" "s3://${BUCKET}/$rel" \
                $S3_ARGS_STR $DRY_RUN
        fi
    ' _ {}
}

# ── Full sync (first run or --full) ─────────────────────────
if [[ "$MODE" == "full" ]]; then
    echo "Uploading $BOOK_DIR/ → s3://${BUCKET}/"
    [[ -n "$DRY_RUN" ]] && echo "(dry run — no files will be uploaded)"

    # Pass 1: HTML
    aws s3 sync "$BOOK_DIR" "s3://${BUCKET}/" \
        "${S3_ARGS[@]}" \
        --content-type "text/html; charset=utf-8" \
        --exclude "*" --include "*.html" \
        $DRY_RUN
    # Pass 2: CSS
    aws s3 sync "$BOOK_DIR" "s3://${BUCKET}/" \
        "${S3_ARGS[@]}" \
        --content-type "text/css; charset=utf-8" \
        --exclude "*" --include "*.css" \
        $DRY_RUN
    # Pass 3: JS
    aws s3 sync "$BOOK_DIR" "s3://${BUCKET}/" \
        "${S3_ARGS[@]}" \
        --content-type "application/javascript; charset=utf-8" \
        --exclude "*" --include "*.js" \
        $DRY_RUN
    # Pass 4: JSON
    aws s3 sync "$BOOK_DIR" "s3://${BUCKET}/" \
        "${S3_ARGS[@]}" \
        --content-type "application/json; charset=utf-8" \
        --exclude "*" --include "*.json" \
        $DRY_RUN
    # Pass 5: rest
    DEL_FLAG=""
    $DELETE && DEL_FLAG="--delete"
    aws s3 sync "$BOOK_DIR" "s3://${BUCKET}/" \
        "${S3_ARGS[@]}" \
        --exclude "*.html" --exclude "*.css" \
        --exclude "*.js" --exclude "*.json" \
        $DRY_RUN $DEL_FLAG
else
    # ── Incremental sync ────────────────────────────────────
    echo "Diffing against $MANIFEST_FILE"

    # Lines unique to NEW = added or content-changed.
    # We diff full lines (hash + path) so a path with a changed hash
    # shows up here.
    LC_ALL=C comm -13 "$MANIFEST_FILE" "$NEW_MANIFEST" \
        | awk '{print substr($0, 67)}' > "$WORK_DIR/changed.txt"

    # Lines unique to OLD = deleted or content-changed. To get strictly
    # removed paths, diff the path columns.
    awk '{print substr($0, 67)}' "$MANIFEST_FILE" \
        | LC_ALL=C sort > "$WORK_DIR/old-paths.txt"
    awk '{print substr($0, 67)}' "$NEW_MANIFEST" \
        | LC_ALL=C sort > "$WORK_DIR/new-paths.txt"
    LC_ALL=C comm -23 "$WORK_DIR/old-paths.txt" "$WORK_DIR/new-paths.txt" \
        > "$WORK_DIR/removed.txt"

    REMOVED=$(wc -l < "$WORK_DIR/removed.txt")
    CHANGED=$(wc -l < "$WORK_DIR/changed.txt")
    echo "  $CHANGED to upload, $REMOVED removed locally"
    [[ -n "$DRY_RUN" ]] && echo "(dry run — no changes will be applied)"

    upload_changed "$WORK_DIR/changed.txt"

    if $DELETE && [[ $REMOVED -gt 0 ]]; then
        echo "  Removing $REMOVED file(s) from bucket..."
        while IFS= read -r rel; do
            aws s3 rm "s3://${BUCKET}/$rel" "${S3_ARGS[@]}" $DRY_RUN
        done < "$WORK_DIR/removed.txt"
    elif [[ $REMOVED -gt 0 ]]; then
        echo "  ($REMOVED file(s) gone locally — pass --delete to remove)"
    fi
fi

# ── Persist new manifest ────────────────────────────────────
# Skip on dry-run so the next real run still sees the prior state.
if [[ -z "$DRY_RUN" ]]; then
    mv "$NEW_MANIFEST" "$MANIFEST_FILE"
    # Don't rm-trap a file we just moved.
    trap 'rm -rf "$WORK_DIR"' EXIT
    echo "Manifest written: $MANIFEST_FILE ($TOTAL files)"
fi

echo "Done."
