#!/usr/bin/env bash
# Parallel Quarto HTML render.
#
# `quarto render` is single-threaded AND does not amortize: every page costs the
# full per-invocation pandoc + crossref scan (~40-50s each), whether you render
# one file, a file list, or a project:render subset — batching N files into one
# invocation just renders them sequentially at ~40s/file (measured: 1 file 40s,
# 10 files 392s, 10-file project:render 529s). So the ONLY lever is concurrency:
# run each page as its own `quarto render <file>` and fan them across cores.
# Measured: 10 concurrent single-file renders finish in 41s (== one file), 0
# errors. Wall ≈ ceil(num_pages / C) * ~45s.
#
# Concurrent same-project single-file renders are safe: disjoint output pages,
# shared .quarto/ is read-mostly, site_libs writes are idempotent. Cross-page
# crossrefs still resolve because each invocation scans the whole project for the
# crossref DB; logical numbering is fixed afterward by fix_crossref_numbers.py. A
# subset render does NOT regenerate the project search.json, so the Makefile
# rebuilds it via build_search_index.py.
#
# Each render is wrapped in `timeout` because quarto occasionally HANGS at exit
# under concurrency (the page HTML is already written by then); timeout reaps the
# straggler without losing output.
#
# Env: RENDER_JOBS=C concurrency (default cores-8, clamped [4,60]), QUARTO=path,
#      RENDER_TIMEOUT=per-file seconds (default 360).
set -uo pipefail
cd "$(dirname "$0")/.."
QUARTO="${QUARTO:-.venv-build/bin/quarto}"
cores=$(nproc 2>/dev/null || echo 8)
C="${RENDER_JOBS:-$(( cores - 8 ))}"
[ "$C" -lt 4 ] && C=4
[ "$C" -gt 60 ] && C=60
RENDER_TIMEOUT="${RENDER_TIMEOUT:-360}"

# Order inputs heaviest-first (qmd byte size ≈ render cost). xargs dispatches in
# input order, so heaviest-first is LPT scheduling: the slow pages start first
# and the pool drains evenly, minimizing the tail.
mapfile -t FILES < <(grep -oE '[A-Za-z0-9_./-]+\.qmd' _quarto.yml | sort -u \
  | while read -r f; do printf '%s\t%s\n' "$(stat -c %s "$f" 2>/dev/null || echo 0)" "$f"; done \
  | sort -rn | cut -f2)
total=${#FILES[@]}
[ "$total" -eq 0 ] && { echo "no qmd inputs found in _quarto.yml"; exit 1; }
[ "$C" -gt "$total" ] && C=$total
echo "=== parallel HTML render: $total pages, $C concurrent single-file renders ==="

tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT
ERR_RE='^ERROR:|Rendering of .* failed|Error running'
t0=$(date +%s)

# Concurrency pool: one `quarto render <file>` per page, C at a time. Each log's
# first line is its source path (log names can't be reversed unambiguously).
render_one() {  # $1=file $2=logdir
  local f="$1" log="$2/$(printf '%s' "$1" | tr / _).log"
  printf 'SRCFILE %s\n' "$f" >"$log"
  timeout -k 5 "$RENDER_TIMEOUT" "$QUARTO" render "$f" --to html >>"$log" 2>&1
}
export -f render_one
export QUARTO RENDER_TIMEOUT
printf '%s\n' "${FILES[@]}" | xargs -P "$C" -I{} bash -c 'render_one "$1" "'"$tmp"'"' _ {}

t1=$(date +%s)

# High concurrency occasionally makes a render (a) trip a transient Lua-filter
# error, or (b) fail to read the project config and fall back to STANDALONE mode,
# writing the page next to its source instead of _book/ (so _book/ is missing it
# despite a clean log). Both are flakes that a single sequential render fixes.
# Collect pages with a log error OR a missing _book output and re-render each
# SEQUENTIALLY (proven reliable), so a flake never fails the build.
declare -A want_retry=()
while IFS= read -r f; do want_retry["$f"]=1; done < <(
  grep -lE "$ERR_RE" "$tmp"/*.log 2>/dev/null | xargs -r -I{} sed -n 's/^SRCFILE //p' {})
for f in "${FILES[@]}"; do
  [ -f "_book/${f%.qmd}.html" ] || want_retry["$f"]=1
done
RETRY=("${!want_retry[@]}")
if [ "${#RETRY[@]}" -gt 0 ]; then
  echo ">>> retrying ${#RETRY[@]} page(s) sequentially after transient concurrency flakes: ${RETRY[*]}"
  for f in "${RETRY[@]}"; do render_one "$f" "$tmp"; done
fi

# Remove any page HTML a standalone-fallback render left in the SOURCE tree
# (output belongs only in _book/; no .html is ever tracked under a source dir).
strays=0
for f in "${FILES[@]}"; do
  s="${f%.qmd}.html"
  [ -f "$s" ] && { rm -f "$s"; strays=$((strays+1)); }
done
[ "$strays" -gt 0 ] && echo ">>> removed $strays stray source-tree .html from standalone-fallback renders"

missing=0; errored=0
for f in "${FILES[@]}"; do
  out="_book/${f%.qmd}.html"
  [ -f "$out" ] || { echo "  MISSING OUTPUT: $out"; missing=$((missing+1)); }
done
# any error remaining AFTER the retry pass is a genuine failure
for log in "$tmp"/*.log; do
  grep -qiE "$ERR_RE" "$log" 2>/dev/null && {
    errored=$((errored+1))
    echo "--- errors in $(sed -n 's/^SRCFILE //p' "$log") ---"
    grep -iE "$ERR_RE" "$log" | head -5; }
done
t2=$(date +%s)
echo ">>> parallel render: $((total-missing))/$total pages present in $((t2-t0))s (pool $((t1-t0))s, C=$C), missing=$missing, errored=$errored"
{ [ "$missing" -gt 0 ] || [ "$errored" -gt 0 ]; } && exit 1 || exit 0
