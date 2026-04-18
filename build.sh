#!/bin/bash
# D2L Book Build Pipeline
#
# Usage:
#   ./build.sh html           Build HTML book (multi-framework with tabs)
#   ./build.sh pdf [fw]       Build PDF book (single framework, default: pytorch)
#   ./build.sh notebooks [fw...] Build Jupyter notebooks (default: all frameworks)
#   ./build.sh slides [fw]    Build Reveal.js slides (default: all frameworks)
#   ./build.sh lib            Build d2l Python package
#   ./build.sh all            Build everything
#   ./build.sh clean          Remove build artifacts (keeps downloaded data)
#   ./build.sh veryclean      Remove build artifacts AND downloaded data
#
# Prerequisites:
#   - Python 3.9+
#   - Quarto 1.7+
#   - XeLaTeX (for PDF)
#   - rsvg-convert (for SVG→PDF conversion)

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SOURCE="${D2L_SOURCE:-../d2l-en}"
cd "$SCRIPT_DIR"

# ── Helpers ──

# Make _notebooks/<fw>/data a symlink to ./data so all frameworks share the
# same download cache. Migrates any existing real data dir in-place.
link_shared_data() {
    local fw_dir="$1"
    mkdir -p "$SCRIPT_DIR/data"
    if [ -L "$fw_dir/data" ]; then
        return
    fi
    if [ -d "$fw_dir/data" ]; then
        # Move contents into the shared dir without overwriting existing files.
        cp -rn "$fw_dir/data/." "$SCRIPT_DIR/data/" 2>/dev/null || true
        rm -rf "$fw_dir/data"
    fi
    ln -s "$SCRIPT_DIR/data" "$fw_dir/data"
}

convert_svgs() {
    local dir="$1"
    local count=0
    for svg in "$dir"/*.svg; do
        [ -f "$svg" ] || continue
        local pdf="${svg%.svg}.pdf"
        if [ ! -f "$pdf" ] || [ "$svg" -nt "$pdf" ]; then
            rsvg-convert -f pdf -o "$pdf" "$svg" 2>/dev/null && count=$((count + 1))
        fi
    done
    echo "  Converted $count SVGs to PDF in $dir/"
}

# ── Commands ──

build_html() {
    echo "=== Building HTML book ==="
    python3 tools/d2l_preprocess.py "$SOURCE" . --primary pytorch
    quarto render --to html
    python3 tools/fix_crossref_numbers.py .
    echo "  Output: _book/index.html"
}

build_pdf() {
    local fw="${1:-pytorch}"
    echo "=== Building PDF ($fw) ==="

    # Generate single-framework sources
    python3 tools/gen_pdf.py "$SOURCE" _pdf --framework "$fw"

    # Convert SVGs to PDF for XeLaTeX
    convert_svgs _pdf/img

    # Render via Quarto (produces .tex + initial .pdf)
    cd _pdf
    quarto render --to pdf
    cd ..

    # Post-process LaTeX for correct chapter/section hierarchy
    python3 tools/fix_latex.py _pdf/Dive-into-Deep-Learning.tex

    # Recompile the fixed LaTeX (two passes for TOC and references)
    cd _pdf
    xelatex -interaction=nonstopmode Dive-into-Deep-Learning.tex > /dev/null 2>&1
    xelatex -interaction=nonstopmode Dive-into-Deep-Learning.tex > /dev/null 2>&1
    cd ..

    # Clean up intermediate PDF images (keep only in _pdf/img during build)
    echo "  Output: _pdf/_pdf/Dive-into-Deep-Learning.pdf"
}

build_notebooks() {
    local fws=("$@")
    if [ ${#fws[@]} -gt 0 ]; then
        echo "=== Building Jupyter notebooks (${fws[*]}) ==="
        python3 tools/gen_notebooks.py "$SOURCE" _notebooks --convert --frameworks "${fws[@]}"
    else
        echo "=== Building Jupyter notebooks ==="
        python3 tools/gen_notebooks.py "$SOURCE" _notebooks --convert
    fi
    # Symlink img/ and share the data/ download cache across all frameworks
    for fw_dir in _notebooks/*/; do
        [ -d "$fw_dir" ] || continue
        [ ! -e "${fw_dir}img" ] && ln -s ../../img "${fw_dir}img"
        link_shared_data "${fw_dir%/}"
    done
    echo "  Output: _notebooks/{pytorch,tensorflow,jax,mxnet}/"
    echo "  Shared data cache: ./data/"
}

run_notebooks() {
    local fw="${1:?Usage: $0 run-notebooks <framework> [extra-args...]}"
    shift
    build_lib
    build_notebooks "$fw"
    echo "=== Running $fw notebooks ==="
    export UV_PROJECT_ENVIRONMENT=".venv-$fw"
    uv sync --extra "$fw" --extra run
    # MXNet CUDA libs are pip-installed under nvidia/*/lib; add them to LD_LIBRARY_PATH.
    local nvidia_libs
    nvidia_libs=$(find "$SCRIPT_DIR/.venv-$fw/lib" -path "*/nvidia/*/lib" -type d 2>/dev/null | paste -sd: -)
    if [ -n "$nvidia_libs" ]; then
        export LD_LIBRARY_PATH="$nvidia_libs${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
    fi
    uv run --no-sync python tools/run_notebooks.py "$fw" --continue-on-error "$@"
}

build_slides() {
    local fws="${1:-pytorch tensorflow jax mxnet}"
    echo "=== Building slides ==="
    python3 tools/gen_slides.py "$SOURCE" _slides --frameworks $fws --render
    echo "  Output: _slides/"
}

build_lib() {
    echo "=== Building d2l package ==="
    python3 tools/build_lib.py "$SOURCE" d2l/
    echo "  Output: d2l/"
}

clean() {
    echo "=== Cleaning build artifacts (keeping ./data/) ==="
    rm -rf _book _pdf _notebooks _slides
    rm -f img/*.pdf
    echo "  Done. Use 'veryclean' to also delete ./data/."
}

veryclean() {
    echo "=== Cleaning build artifacts AND downloaded data ==="
    rm -rf _book _pdf _notebooks _slides data
    rm -f img/*.pdf
    echo "  Done."
}

# ── Main ──

case "${1:-}" in
    html)           build_html ;;
    pdf)            build_pdf "${2:-pytorch}" ;;
    notebooks)      shift; build_notebooks "$@" ;;
    run-notebooks)  shift; run_notebooks "$@" ;;
    slides)         build_slides "${2:-}" ;;
    lib)            build_lib ;;
    all)
        build_html
        build_pdf pytorch
        build_notebooks
        build_slides
        build_lib
        ;;
    clean)          clean ;;
    veryclean)      veryclean ;;
    *)
        echo "Usage: $0 {html|pdf|notebooks|run-notebooks|slides|lib|all|clean|veryclean}"
        echo "  pdf [framework]            - default: pytorch"
        echo "  notebooks [framework...]   - default: all"
        echo "  run-notebooks <framework>  - execute notebooks (uses uv)"
        echo "  slides [framework]         - default: all"
        exit 1
        ;;
esac
