#!/usr/bin/env bash
# Gradual slide migration: overlay only the *north-star* decks onto
# _book/slides/, leaving every legacy deck exactly as it is.
#
# Why not the wholesale staging in upload_r2.sh? That does
# `rm -rf _book/slides && rsync all of _slides`, which replaces all ~137
# legacy decks with freshly-regenerated ones (new no-scroll pipeline +
# new theme), changing their hashes and clobbering the live versions.
# We want "substitute the north-star deck whenever one is available; keep
# the old deck otherwise" — so we copy only the north-star decks (+ the
# refreshed index, the new content-hashed theme CSS, and the plot assets
# they reference) on top of the existing _book/slides/ tree.
#
# A deck is north-star iff tools/northstar_slides.py says so (its source
# slides block uses the north-star vocabulary). The legacy decks keep
# referencing the old-hash theme CSS, which stays in libs/ untouched, so
# the two render styles coexist.
#
# Output: writes _book/slides/.northstar-upload.txt — the exact set of
# slides-relative paths that changed, so a surgical R2 upload can push
# only those (see the companion command printed at the end).
#
# Usage:
#   ./tools/stage_northstar_slides.sh

set -euo pipefail
cd "$(cd "$(dirname "$0")/.." && pwd)"

SLIDES=_slides
BOOK=_book/slides
FRAMEWORKS=(pytorch tensorflow jax mxnet)

[[ -d "$SLIDES" ]] || { echo "Error: $SLIDES not found (build slides first)." >&2; exit 1; }
[[ -d "$BOOK"   ]] || { echo "Error: $BOOK not found (need an existing book build to overlay onto)." >&2; exit 1; }

# North-star decks, as 'chapter_dir/stem' lines.
mapfile -t NS_DECKS < <(python3 tools/northstar_slides.py . --list)
[[ ${#NS_DECKS[@]} -gt 0 ]] || { echo "No north-star decks detected — nothing to stage."; exit 0; }

echo "North-star decks (${#NS_DECKS[@]}):"
printf '  ✦ %s\n' "${NS_DECKS[@]}"

# Refresh the landing index + manifest (badges the north-star decks).
echo "Rebuilding slides index..."
python3 tools/build_slides_index.py --slides-dir "$SLIDES" --source . >/dev/null

# Additive asset sync: brings the new content-hashed theme CSS and any
# plot assets the north-star decks reference. No --delete, so legacy
# assets (incl. the old-hash theme CSS) are preserved.
echo "Syncing shared assets (libs/, img/) additively..."
rsync -a "$SLIDES/libs/" "$BOOK/libs/"
[[ -d "$SLIDES/img" ]] && rsync -a "$SLIDES/img/" "$BOOK/img/"

# Copy index + manifest.
cp "$SLIDES/index.html" "$BOOK/index.html"
[[ -f "$SLIDES/manifest.json" ]] && cp "$SLIDES/manifest.json" "$BOOK/manifest.json"

UPLOAD_LIST="$BOOK/.northstar-upload.txt"
: > "$UPLOAD_LIST"
echo "slides/index.html" >> "$UPLOAD_LIST"

# Copy each north-star deck for every framework that has it, and collect
# the libs/ + img/ assets each deck references (slides-relative paths).
declare -A REF_ASSETS
for deck in "${NS_DECKS[@]}"; do
  chap="${deck%/*}"; stem="${deck##*/}"
  for fw in "${FRAMEWORKS[@]}"; do
    src="$SLIDES/$fw/$chap/$stem.html"
    [[ -f "$src" ]] || continue
    mkdir -p "$BOOK/$fw/$chap"
    cp "$src" "$BOOK/$fw/$chap/$stem.html"
    echo "slides/$fw/$chap/$stem.html" >> "$UPLOAD_LIST"
    # Assets referenced as ../../libs/... or ../../img/... resolve to
    # slides/libs/... and slides/img/... respectively.
    while IFS= read -r ref; do
      REF_ASSETS["slides/$ref"]=1
    done < <(grep -oE '\.\./\.\./(libs|img)/[^"]+' "$src" | sed -E 's#^\.\./\.\./##' | sort -u)
  done
done

# A libs asset only needs uploading if no *legacy* deck already references
# it (the shared reveal core/plugins are already live; only the new theme
# CSS is genuinely new). Build the set of legacy-referenced libs assets.
LEGACY_LIBS=$(mktemp)
trap 'rm -f "$LEGACY_LIBS"' EXIT
ns_re=$(printf '%s\n' "${NS_DECKS[@]}" | sed 's#/#/#' | paste -sd'|' -)
for fw in "${FRAMEWORKS[@]}"; do
  while IFS= read -r f; do
    rel="${f#"$SLIDES/$fw/"}"          # chapter/stem.html
    key="${rel%.html}"                  # chapter/stem
    [[ "$key" =~ ^(${ns_re})$ ]] && continue   # skip north-star decks
    grep -ohE '\.\./\.\./libs/[^"]+' "$f" 2>/dev/null | sed -E 's#^\.\./\.\./#slides/#'
  done < <(find "$SLIDES/$fw" -name '*.html')
done | sort -u > "$LEGACY_LIBS"

# Emit referenced assets that are NEW (img always; libs only if not in the
# legacy set).
for asset in "${!REF_ASSETS[@]}"; do
  case "$asset" in
    slides/img/*) echo "$asset" >> "$UPLOAD_LIST" ;;
    slides/libs/*) grep -qxF "$asset" "$LEGACY_LIBS" || echo "$asset" >> "$UPLOAD_LIST" ;;
  esac
done

LC_ALL=C sort -u "$UPLOAD_LIST" -o "$UPLOAD_LIST"
echo
echo "Staged $(($(wc -l < "$UPLOAD_LIST"))) file(s) → $BOOK/  (list: $UPLOAD_LIST)"
echo "Surgical upload:  ./tools/upload_northstar_r2.sh        (--dry-run to preview)"
