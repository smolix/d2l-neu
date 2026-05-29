#!/usr/bin/env bash
# Bootstrap a fresh host for d2l-neu.
#
# Two modes (see docs/build-system.md for the decoupled build model):
#
#   ./bootstrap.sh            DOCUMENT mode (default) — everything needed to
#                             edit + render the book/slides CPU-only from the
#                             committed outputs/ store. No CUDA, no GPU, no
#                             framework venvs. This is what you want on a
#                             laptop / fresh server to "work on the document".
#
#   ./bootstrap.sh --pdf      Also install TeX Live + rsvg so `make pdfs` works.
#
#   ./bootstrap.sh --full     DOCUMENT + --pdf, and a reminder that notebook
#                             *execution* (`make run-all-notebooks`) additionally
#                             needs the per-framework GPU venvs (~20 GB CUDA
#                             wheels), synced on first use.
#
# Idempotent: skips steps already done. Needs sudo for apt installs.
set -euo pipefail
cd "$(dirname "$(readlink -f "$0")")"

MODE=doc
case "${1:-}" in
    ''|--doc)   MODE=doc ;;
    --pdf)      MODE=pdf ;;
    --full)     MODE=full ;;
    -h|--help)  sed -n '2,20p' "$0"; exit 0 ;;
    *) echo "Unknown arg: $1 (use --doc | --pdf | --full)"; exit 1 ;;
esac

step() { printf '\n\033[1;34m==> %s\033[0m\n' "$*"; }
have() { command -v "$1" >/dev/null 2>&1; }
SUDO=""; [ "$EUID" -ne 0 ] && SUDO="sudo"

# ── 1. uv ─────────────────────────────────────────────────────────────
if have uv; then
    step "uv already installed: $(uv --version)"
else
    step "Installing uv (standalone installer)"
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
    have uv || { echo "uv install failed; $HOME/.local/bin not on PATH"; exit 1; }
    echo "uv installed: $(uv --version)"
fi

# ── 2. System packages ────────────────────────────────────────────────
# git-lfs is REQUIRED in every mode: the committed outputs/ store keeps its
# image assets in Git LFS (see docs/build-system.md §2.2). Without it, a fresh
# checkout has pointer files instead of images and the rendered book/slides
# show broken figures.
# nodejs is for the slide diagram engine (diagrams/*.mjs → img/auto/*.svg).
# TeX Live + librsvg are PDF-only (modes --pdf / --full).
APT_PKGS=(git-lfs nodejs)
if [ "$MODE" != doc ]; then
    APT_PKGS+=(texlive-xetex texlive-latex-extra texlive-latex-recommended
               texlive-fonts-extra librsvg2-bin)
fi
missing=()
for p in "${APT_PKGS[@]}"; do
    dpkg -s "$p" >/dev/null 2>&1 || missing+=("$p")
done
if [ "${#missing[@]}" -eq 0 ]; then
    step "All required apt packages already installed (mode: $MODE)"
else
    step "Installing apt packages: ${missing[*]}"
    if [ -z "$SUDO" ] && [ "$EUID" -ne 0 ]; then
        echo "Need sudo or root to install: ${missing[*]}" >&2; exit 1
    fi
    $SUDO apt-get update
    $SUDO apt-get install -y "${missing[@]}"
fi

# ── 3. Git LFS: hooks + materialize the outputs/ store ────────────────
step "Configuring Git LFS and pulling store assets"
git lfs install --local
# If the repo was cloned before git-lfs existed, working-tree assets are
# pointer stubs; smudge them into real files. Best-effort (no-op without a
# remote / when assets aren't pushed yet).
git lfs pull 2>/dev/null || echo "  (git lfs pull skipped — no remote objects yet)"

# ── 4. logs/ + PDF preamble ───────────────────────────────────────────
step "Ensuring logs/ exists"
mkdir -p logs

if [ "$MODE" != doc ] && [ ! -f static/d2l-preamble.tex ]; then
    step "Creating static/d2l-preamble.tex (referenced by tools/gen_pdf.py)"
    cat > static/d2l-preamble.tex <<'PREAMBLE'
% LaTeX preamble for "Dive into Deep Learning" PDF builds.
% Referenced by tools/gen_pdf.py via `include-in-header`.
% Quarto already loads fontspec/hyperref/amsmath/amssymb/graphicx/longtable/booktabs,
% so this file only adds project-specific tweaks.
\usepackage{fvextra}
\DefineVerbatimEnvironment{Highlighting}{Verbatim}{commandchars=\\\{\},breaklines,breakanywhere,fontsize=\small}
\usepackage{float}
\usepackage[strings]{underscore}
\setcounter{tocdepth}{1}
\sloppy
PREAMBLE
fi

# ── 5. Build venv (Quarto + nbformat) ─────────────────────────────────
# `quarto-cli` pulls in the jupyter/nbformat stack Quarto needs to parse
# {python} cells even with execute disabled. No framework, no CUDA.
step "Syncing .venv-build (provides quarto + nbformat)"
UV_PROJECT_ENVIRONMENT=.venv-build uv sync --extra build

# ── 6. d2l library (CPU-only; enables .md → .qmd preprocessing) ────────
# build_lib.py just parses #@save blocks from the source .md — no framework
# import — so this works on a render-only host and lets `make html` run.
step "Building d2l library (make lib)"
make lib

# ── 7. Sanity check ───────────────────────────────────────────────────
step "Sanity check (mode: $MODE)"
echo "  uv          : $(uv --version)"
echo "  git-lfs     : $(git lfs version 2>/dev/null | head -1)"
echo "  node        : $(node --version 2>/dev/null || echo MISSING)"
echo "  quarto      : $(.venv-build/bin/quarto --version 2>/dev/null)"
echo "  store       : $(find outputs -name '*.json' 2>/dev/null | wc -l) manifests, $(find outputs -type f \( -name '*.png' -o -name '*.svg' \) 2>/dev/null | wc -l) assets"
echo "  LFS pointers left unsmudged: $(git lfs ls-files 2>/dev/null | grep -c ' - ' || echo 0)  (should be 0)"
if [ "$MODE" != doc ]; then
    echo "  xelatex     : $(xelatex --version 2>/dev/null | head -1)"
    echo "  rsvg-convert: $(rsvg-convert --version 2>/dev/null)"
fi
echo "  GPUs        : $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | wc -l) detected"

cat <<NEXT

==> Bootstrap complete (mode: $MODE).

DOCUMENT WORK — CPU-only, reads the committed outputs/ store, no GPU:
    make html                                  # render the book
    make -j4 slides                            # render all slide decks
    make -B slides-pytorch SLIDES_FILTER=chapter_preliminaries/ndarray.md   # one deck
    # edit prose / <!-- slides --> blocks / _d2l-*.scss / _quarto.yml, then re-render
    node diagrams/render.mjs --out img/auto    # regenerate slide diagrams
NEXT
if [ "$MODE" != doc ]; then
cat <<NEXT
    make -j4 pdfs                              # render PDFs (needs the TeX you just installed)
NEXT
fi
cat <<NEXT

CHANGING A NOTEBOOK'S OUTPUTS — needs the per-framework GPU venv (synced on
first use, ~20 GB CUDA wheels) and a GPU:
    make -B _notebooks/<fw>/<chapter>/<file>.executed   # re-execute one notebook
    make capture-outputs FILES=<chapter>/<file>.md      # bless into the store
    make audit-outputs                                  # what's stale + integrity
See docs/build-system.md for the full model and the four canonical flows.
NEXT
