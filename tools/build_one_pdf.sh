#!/usr/bin/env bash
# Build one framework's PDF from generated _pdf/<fw>/ sources.
#
# Extracted verbatim (behavior-preserving) from the Makefile PDF_RULE so the
# recipe no longer needs the quadruple-$ escaping that a define/$(eval) template
# forces. Run from the project root:
#
#   QUARTO=/abs/path/to/quarto tools/build_one_pdf.sh <framework>
#
# Produces _pdf/<fw>/_pdf/Dive-into-Deep-Learning-<fw>.pdf and copies it to
# _book/pdf/.
#
# Control flow (;-separated, no `set -e`) intentionally mirrors the original
# recipe: only a missing .tex hard-fails here. `check_book_artifacts.py`
# (make check-all-artifacts) is the backstop that verifies each framework's PDF
# actually got produced.

fw="$1"
[ -n "$fw" ] || { echo "usage: build_one_pdf.sh <framework>" >&2; exit 2; }
root="$(pwd)"
quarto="${QUARTO:-$root/.venv-build/bin/quarto}"

if [ -d _notebooks ]; then
	echo "Injecting notebook outputs for $fw..."
	python3 tools/inject_outputs.py pdf --framework "$fw" --pdf-dir "_pdf/$fw"
fi

# Pre-convert every SVG → PDF (Quarto shells rsvg-convert per figure); only
# reconvert when the SVG is newer than its PDF.
count=0
for svg in "_pdf/$fw"/img/*.svg; do
	[ -f "$svg" ] || continue
	pdf="${svg%.svg}.pdf"
	if [ ! -f "$pdf" ] || [ "$svg" -nt "$pdf" ]; then
		rsvg-convert -f pdf -o "$pdf" "$svg" 2>/dev/null && count=$((count + 1))
	fi
done
[ "$count" -gt 0 ] && echo "Converted $count SVGs to PDF" || true

# Render LaTeX ONLY — skip quarto's ~2 discarded xelatex passes (we compile the
# fix_latex-patched .tex ourselves below). `quarto --to latex` writes the book to
# _pdf/<fw>/_pdf/book-latex/Dive-into-Deep-Learning.tex with image paths relative
# to the PROJECT ROOT (chapter_x/../img/...), so copy it up to _pdf/<fw>/ and
# compile there (where chapter_*/ + img/ live).
cd "_pdf/$fw" && "$quarto" render --to latex
cd "$root"

src_tex="$(find "_pdf/$fw" -name Dive-into-Deep-Learning.tex -path '*book-latex*' -print -quit 2>/dev/null)"
[ -n "$src_tex" ] || src_tex="$(find "_pdf/$fw" -name Dive-into-Deep-Learning.tex -printf '%T@ %p\n' 2>/dev/null | sort -rn | head -1 | cut -d' ' -f2-)"
[ -n "$src_tex" ] || { echo "ERROR: quarto --to latex produced no Dive-into-Deep-Learning.tex under _pdf/$fw"; exit 1; }
cp -f "$src_tex" "_pdf/$fw/Dive-into-Deep-Learning.tex"
python3 tools/fix_latex.py "_pdf/$fw/Dive-into-Deep-Learning.tex"

cd "_pdf/$fw"
xelatex -interaction=nonstopmode Dive-into-Deep-Learning.tex > /dev/null 2>&1
xelatex -interaction=nonstopmode Dive-into-Deep-Learning.tex > /dev/null 2>&1
xelatex -interaction=nonstopmode Dive-into-Deep-Learning.tex > /dev/null 2>&1
cd "$root"

# Publish the fix_latex-patched PDF (hierarchical chapter/section numbering) —
# NOT quarto's own pre-fix compile under _pdf/<fw>/_pdf/.
if [ -f "_pdf/$fw/Dive-into-Deep-Learning.pdf" ]; then
	mkdir -p "_pdf/$fw/_pdf"
	mv -f "_pdf/$fw/Dive-into-Deep-Learning.pdf" "_pdf/$fw/_pdf/Dive-into-Deep-Learning-$fw.pdf"
fi
if [ -f "_pdf/$fw/_pdf/Dive-into-Deep-Learning-$fw.pdf" ]; then
	mkdir -p _book/pdf
	cp "_pdf/$fw/_pdf/Dive-into-Deep-Learning-$fw.pdf" _book/pdf/
fi
