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
#   ./tools/upload_r2.sh --no-stage-slides
#                                     # do not refresh _book/slides from _slides
#
# Set UPLOAD_JOBS to tune local hashing and parallel incremental uploads
# (default: 32).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

if [[ -f "$PROJECT_DIR/.env" ]]; then
    # `set -a` so the bare `KEY=value` lines in .env are *exported*, not just
    # set as shell vars — otherwise the `aws` subprocess can't see
    # AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY and dies with "Unable to locate
    # credentials". (.env intentionally uses bare assignments, no `export`.)
    set -a
    # shellcheck disable=SC1091
    source "$PROJECT_DIR/.env"
    set +a
fi

BUCKET="staging-d2l"
BOOK_DIR="${BOOK_DIR:-_book}"
MANIFEST_FILE="${PROJECT_DIR}/.upload-manifest-${BUCKET}.txt"
UPLOAD_JOBS="${UPLOAD_JOBS:-32}"

DRY_RUN=""
DELETE=false
FORCE_FULL=false
STAGE_SLIDES=true

for arg in "$@"; do
    case "$arg" in
        --dry-run)         DRY_RUN="--dryrun" ;;
        --delete)          DELETE=true ;;
        --full)            FORCE_FULL=true ;;
        --no-stage-slides) STAGE_SLIDES=false ;;
        *)                 echo "Unknown arg: $arg"; exit 1 ;;
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

if ! [[ "$UPLOAD_JOBS" =~ ^[1-9][0-9]*$ ]]; then
    echo "Error: UPLOAD_JOBS must be a positive integer." >&2
    exit 1
fi

# Slide rendering writes fresh decks to _slides/. The public site serves
# them from _book/slides/, which is normally refreshed by `make html`.
# Refresh that subtree here too so a slide-only rebuild followed by upload
# actually publishes the new decks instead of hashing a stale _book/.
if $STAGE_SLIDES && [[ -d _slides ]]; then
    echo "Staging current _slides/ into $BOOK_DIR/slides/ ..."
    python3 tools/build_slides_index.py
    rm -rf "$BOOK_DIR/slides"
    mkdir -p "$BOOK_DIR/slides"
    rsync -a --exclude='*.qmd' --exclude='_quarto.yml' \
        --exclude='.gitignore' --exclude='.quarto/' --exclude='errors/' \
        _slides/ "$BOOK_DIR/slides/"
    find "$BOOK_DIR/slides" -mindepth 2 -maxdepth 2 -type l \
        \( -name data -o -name img \) -delete
    find "$BOOK_DIR/slides" -mindepth 3 -maxdepth 3 -name '*.html' \
        -exec perl -i -pe 's|src="\.\./img/|src="../../../img/|g' {} +
fi

ENDPOINT="https://${R2_ACCOUNT_ID}.r2.cloudflarestorage.com"
# R2 only accepts its own region names (wnam/enam/.../auto); pin `auto` so the
# upload is independent of the ambient ~/.aws/config default region.
S3_ARGS=(--endpoint-url "$ENDPOINT" --region auto)

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
(cd "$BOOK_DIR" && find -L . -path '*/.quarto/*' -prune -o -type f -print0 \
  | xargs -0 -P "$UPLOAD_JOBS" sha256sum) | LC_ALL=C sort > "$NEW_MANIFEST"

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

# Export scalar values so the xargs worker can rebuild its own argv array.
# `S3_ARGS` (an array) cannot survive an env hop, so the worker functions
# below rebuild the endpoint / dry-run flags from `R2_ENDPOINT` and
# `DRY_RUN` directly.
export BOOK_DIR BUCKET
export R2_ENDPOINT="$ENDPOINT"
export DRY_RUN="${DRY_RUN:-}"

# ── Upload one file with proper content-type ────────────────
upload_one() {
    local rel=$1
    local ct
    case "$rel" in
        *.html) ct="text/html; charset=utf-8" ;;
        *.css)  ct="text/css; charset=utf-8" ;;
        *.js)   ct="application/javascript; charset=utf-8" ;;
        *.json) ct="application/json; charset=utf-8" ;;
        *)      ct="" ;;
    esac
    local args=(--endpoint-url "$R2_ENDPOINT" --region auto)
    if [[ -n "$ct" ]]; then
        args+=(--content-type "$ct")
    fi
    if [[ -n "${DRY_RUN:-}" ]]; then
        args+=("$DRY_RUN")
    fi
    aws s3 cp "$BOOK_DIR/$rel" "s3://${BUCKET}/$rel" "${args[@]}"
}

# ── Delete one bucket object ────────────────────────────────
delete_one() {
    local rel=$1
    local args=(--endpoint-url "$R2_ENDPOINT" --region auto)
    if [[ -n "${DRY_RUN:-}" ]]; then
        args+=("$DRY_RUN")
    fi
    aws s3 rm "s3://${BUCKET}/$rel" "${args[@]}"
}

export -f upload_one delete_one

upload_changed() {
    local list=$1
    local count
    count=$(wc -l < "$list")
    if [[ $count -eq 0 ]]; then
        echo "  Nothing to upload."
        return
    fi
    echo "  Uploading $count file(s) with $UPLOAD_JOBS worker(s)..."
    # Each line in $list is a relative path under $BOOK_DIR. NUL-delimit via tr
    # so paths with spaces stay intact and xargs -0 works on both GNU and BSD
    # (macOS) xargs (which lacks GNU's -a/-d).
    # shellcheck disable=SC2016
    tr '\n' '\0' < "$list" | xargs -0 -P "$UPLOAD_JOBS" -I{} \
        bash -c 'upload_one "$1"' _ {}
}

# ── Full sync (first run or --full) ─────────────────────────
if [[ "$MODE" == "full" ]]; then
    echo "Uploading $BOOK_DIR/ → s3://${BUCKET}/"
    [[ -n "$DRY_RUN" ]] && echo "(dry run — no files will be uploaded)"

    # Pass 1: HTML
    aws s3 sync "$BOOK_DIR" "s3://${BUCKET}/" \
        "${S3_ARGS[@]}" \
        --content-type "text/html; charset=utf-8" \
        --exclude "*" --include "*.html" --exclude "*/.quarto/*" \
        $DRY_RUN
    # Pass 2: CSS
    aws s3 sync "$BOOK_DIR" "s3://${BUCKET}/" \
        "${S3_ARGS[@]}" \
        --content-type "text/css; charset=utf-8" \
        --exclude "*" --include "*.css" --exclude "*/.quarto/*" \
        $DRY_RUN
    # Pass 3: JS
    aws s3 sync "$BOOK_DIR" "s3://${BUCKET}/" \
        "${S3_ARGS[@]}" \
        --content-type "application/javascript; charset=utf-8" \
        --exclude "*" --include "*.js" --exclude "*/.quarto/*" \
        $DRY_RUN
    # Pass 4: JSON
    aws s3 sync "$BOOK_DIR" "s3://${BUCKET}/" \
        "${S3_ARGS[@]}" \
        --content-type "application/json; charset=utf-8" \
        --exclude "*" --include "*.json" --exclude "*/.quarto/*" \
        $DRY_RUN
    # Pass 5: rest
    DEL_FLAG=""
    $DELETE && DEL_FLAG="--delete"
    aws s3 sync "$BOOK_DIR" "s3://${BUCKET}/" \
        "${S3_ARGS[@]}" \
        --exclude "*.html" --exclude "*.css" \
        --exclude "*.js" --exclude "*.json" --exclude "*/.quarto/*" \
        $DRY_RUN $DEL_FLAG

    if $DELETE; then
        echo "Removing bucket .quarto cache objects..."
        aws s3 rm "s3://${BUCKET}/" "${S3_ARGS[@]}" --recursive \
            --exclude "*" --include "*/.quarto/*" $DRY_RUN
    fi
else
    # ── Incremental sync ────────────────────────────────────
    echo "Diffing against $MANIFEST_FILE"

    # Lines unique to NEW = added or content-changed.
    # We diff full lines (hash + path) so a path with a changed hash
    # shows up here.
    LC_ALL=C comm -13 "$MANIFEST_FILE" "$NEW_MANIFEST" \
        | awk '{p=substr($0, 67); sub("^\\./", "", p); print p}' \
        > "$WORK_DIR/changed.txt"

    # Lines unique to OLD = deleted or content-changed. To get strictly
    # removed paths, diff the path columns.
    awk '{p=substr($0, 67); sub("^\\./", "", p); print p}' "$MANIFEST_FILE" \
        | LC_ALL=C sort > "$WORK_DIR/old-paths.txt"
    awk '{p=substr($0, 67); sub("^\\./", "", p); print p}' "$NEW_MANIFEST" \
        | LC_ALL=C sort > "$WORK_DIR/new-paths.txt"
    LC_ALL=C comm -23 "$WORK_DIR/old-paths.txt" "$WORK_DIR/new-paths.txt" \
        > "$WORK_DIR/removed.txt"

    REMOVED=$(wc -l < "$WORK_DIR/removed.txt")
    CHANGED=$(wc -l < "$WORK_DIR/changed.txt")
    echo "  $CHANGED to upload, $REMOVED removed locally"
    [[ -n "$DRY_RUN" ]] && echo "(dry run — no changes will be applied)"

    upload_changed "$WORK_DIR/changed.txt"

    if $DELETE && [[ $REMOVED -gt 0 ]]; then
        echo "  Removing $REMOVED file(s) from bucket with $UPLOAD_JOBS worker(s)..."
        # shellcheck disable=SC2016
        tr '\n' '\0' < "$WORK_DIR/removed.txt" | xargs -0 -P "$UPLOAD_JOBS" -I{} \
            bash -c 'delete_one "$1"' _ {}
        echo "  Removing bucket .quarto cache objects..."
        aws s3 rm "s3://${BUCKET}/" "${S3_ARGS[@]}" --recursive \
            --exclude "*" --include "*/.quarto/*" $DRY_RUN
    elif [[ $REMOVED -gt 0 ]]; then
        echo "  ($REMOVED file(s) gone locally — pass --delete to remove)"
    elif $DELETE; then
        echo "  Removing bucket .quarto cache objects..."
        aws s3 rm "s3://${BUCKET}/" "${S3_ARGS[@]}" --recursive \
            --exclude "*" --include "*/.quarto/*" $DRY_RUN
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
