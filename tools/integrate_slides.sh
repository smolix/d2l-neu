#!/usr/bin/env bash
# Integrate rendered _slides/ into _book/slides/ (called from the html recipe).
#
# Extracted verbatim (behavior-preserving) from the Makefile html rule. No-op if
# _slides/ hasn't been rendered. Run from the project root.

[ -d _slides ] && [ -f _slides/index.html ] || exit 0

echo "Integrating _slides/ → _book/slides/ ..."
rm -rf _book/slides
mkdir -p _book/slides
rsync -a --exclude='*.qmd' --exclude='_quarto.yml' \
	--exclude='.gitignore' --exclude='.quarto/' --exclude='errors/' \
	--exclude='.deckhashes.json' --exclude='.render-*' \
	_slides/ _book/slides/

echo "Rebuilding the slides landing page in the frame of this render..."
python3 tools/build_slides_index.py --landing-only --out _book/slides/index.html

echo "Stripping per-fw data/img symlinks (R2 storage bloat)..."
find _book/slides -mindepth 2 -maxdepth 2 -type l \
	\( -name data -o -name img \) -delete

echo "Rewriting deck '../img/' refs to '../../../img/' (single-source)..."
find _book/slides -mindepth 3 -maxdepth 3 -name '*.html' \
	-exec perl -i -pe 's|src="\.\./img/|src="../../../img/|g' {} +

echo "Copying slide-only img assets into _book/img/ ..."
{ find _book/slides -mindepth 3 -maxdepth 3 -name '*.html' \
	-exec grep -ho 'src="\.\./\.\./\.\./img/[^"]*"' {} + || true; } | \
	sed 's|src="\.\./\.\./\.\./img/||; s|"$||' | sort -u | \
	while read -r f; do
		if [ ! -f "_book/img/$f" ] && [ -f "img/$f" ]; then
			cp "img/$f" "_book/img/$f"
			echo "  + img/$f (slide-only)"
		fi
	done
