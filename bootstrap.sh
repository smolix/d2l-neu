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
# Idempotent: skips steps already done. System-package installs need sudo and
# use the host's package manager (apt-get on Debian/Ubuntu, MacPorts `port` on
# macOS); anything already on PATH is left alone.
set -euo pipefail
# cd to repo root (this script's directory). BSD readlink lacks -f, so prefer
# GNU greadlink when present and fall back to a pwd-based resolve otherwise.
if command -v greadlink >/dev/null 2>&1; then
    cd "$(dirname "$(greadlink -f "$0")")"
elif readlink -f "$0" >/dev/null 2>&1; then
    cd "$(dirname "$(readlink -f "$0")")"
else
    cd "$(dirname "$0")"
fi

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

# The Makefile uses grouped-target rules (`targets &: prereqs`), a GNU Make
# >= 4.3 feature. Apple ships GNU Make 3.81 as `make`, which mis-parses `&` as
# a literal target ("overriding commands for target `&`") and breaks grouped
# output. MacPorts/Homebrew provide `gmake` (4.x); prefer it when `make` is too
# old. Exported so the printed next-step commands name the right binary.
make_ge_43() {
    local v maj min
    v="$("$1" --version 2>/dev/null | sed -n '1s/.*GNU Make //p')"
    [ -n "$v" ] || return 1
    maj=${v%%.*}; min=${v#*.}; min=${min%%.*}
    [ "${maj:-0}" -gt 4 ] 2>/dev/null && return 0
    [ "${maj:-0}" -eq 4 ] 2>/dev/null && [ "${min:-0}" -ge 3 ] 2>/dev/null
}
MAKE=make
if ! make_ge_43 make; then
    if have gmake && make_ge_43 gmake; then
        MAKE=gmake
    else
        echo "WARNING: GNU Make >= 4.3 not found ('$(make --version 2>/dev/null | sed -n 1p)')." >&2
        echo "  The Makefile's grouped-target rules ('&:') need >= 4.3." >&2
        echo "  macOS: 'sudo port install gmake' (MacPorts) and use 'gmake' for build targets." >&2
    fi
fi
export MAKE

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
# These are *capabilities* we need, probed by the binary they put on PATH —
# not distro-specific package names. We never install something already on
# PATH, and we never invoke a package manager the host doesn't have.
#
#   git-lfs       REQUIRED, every mode — the committed outputs/ store keeps its
#                 image assets in Git LFS (docs/build-system.md §2.2). Without
#                 it a fresh checkout has pointer stubs, not images, and the
#                 rendered book/slides show broken figures.
#   node          slide diagram engine (diagrams/*.mjs → img/auto/*.svg).
#   xelatex       PDF builds only (modes --pdf / --full).
#   rsvg-convert  PDF builds only.
NEED_BINS=(git-lfs node)
if [ "$MODE" != doc ]; then
    NEED_BINS+=(xelatex rsvg-convert)
fi

missing_bins=()
for b in "${NEED_BINS[@]}"; do
    have "$b" || missing_bins+=("$b")
done

if [ "${#missing_bins[@]}" -eq 0 ]; then
    step "All required system tools already present (mode: $MODE): ${NEED_BINS[*]}"
else
    # Map each missing capability to its package name on the host's manager.
    # (TeX needs several sub-packages beyond xelatex itself for the LaTeX
    # classes the book uses; they share the same names on apt and MacPorts.)
    TEX_PKGS="texlive-xetex texlive-latex-extra texlive-latex-recommended texlive-fonts-extra"
    declare -A APT_PKG=(
        [git-lfs]=git-lfs [node]=nodejs
        [xelatex]="$TEX_PKGS" [rsvg-convert]=librsvg2-bin
    )
    declare -A PORT_PKG=(
        [git-lfs]=git-lfs [node]=nodejs22
        [xelatex]="$TEX_PKGS" [rsvg-convert]=librsvg
    )

    pkgs=()
    if [ "$(uname -s)" = Darwin ] && have port; then
        for b in "${missing_bins[@]}"; do pkgs+=(${PORT_PKG[$b]}); done
        step "Installing MacPorts packages for missing tools (${missing_bins[*]}): ${pkgs[*]}"
        $SUDO port install "${pkgs[@]}"
    elif have apt-get; then
        for b in "${missing_bins[@]}"; do pkgs+=(${APT_PKG[$b]}); done
        # dpkg check trims TeX sub-packages already present (no probe binary).
        need=()
        for p in "${pkgs[@]}"; do dpkg -s "$p" >/dev/null 2>&1 || need+=("$p"); done
        if [ "${#need[@]}" -eq 0 ]; then
            step "Required apt packages already installed (mode: $MODE)"
        else
            step "Installing apt packages: ${need[*]}"
            $SUDO apt-get update
            $SUDO apt-get install -y "${need[@]}"
        fi
    else
        step "Missing required tools: ${missing_bins[*]}"
        echo "No supported package manager found (need apt-get on Debian/Ubuntu" >&2
        echo "or MacPorts 'port' on macOS). Install these tools manually, then" >&2
        echo "re-run ./bootstrap.sh:" >&2
        echo "    ${missing_bins[*]}" >&2
        exit 1
    fi
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
step "Building d2l library ($MAKE lib)"
"$MAKE" lib

# ── 7. Sanity check ───────────────────────────────────────────────────
step "Sanity check (mode: $MODE)"
echo "  uv          : $(uv --version)"
echo "  git-lfs     : $(git lfs version 2>/dev/null | head -1)"
echo "  node        : $(node --version 2>/dev/null || echo MISSING)"
echo "  make        : $("$MAKE" --version 2>/dev/null | sed -n 1p) (using '$MAKE')"
echo "  quarto      : $(.venv-build/bin/quarto --version 2>/dev/null)"
echo "  store       : $(find outputs -name '*.json' 2>/dev/null | wc -l) manifests, $(find outputs -type f \( -name '*.png' -o -name '*.svg' \) 2>/dev/null | wc -l) assets"
echo "  LFS pointers left unsmudged: $(git lfs ls-files 2>/dev/null | grep -c ' - ' || true)  (should be 0)"
if [ "$MODE" != doc ]; then
    echo "  xelatex     : $(xelatex --version 2>/dev/null | head -1)"
    echo "  rsvg-convert: $(rsvg-convert --version 2>/dev/null)"
fi
echo "  GPUs        : $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | wc -l) detected"

cat <<NEXT

==> Bootstrap complete (mode: $MODE).

DOCUMENT WORK — CPU-only, reads the committed outputs/ store, no GPU:
    $MAKE html                                  # render the book
    $MAKE -j4 slides                            # render all slide decks
    $MAKE -B slides-pytorch SLIDES_FILTER=chapter_preliminaries/ndarray.md   # one deck
    # edit prose / <!-- slides --> blocks / _d2l-*.scss / _quarto.yml, then re-render
    node diagrams/render.mjs --out img/auto     # regenerate slide diagrams
NEXT
if [ "$MODE" != doc ]; then
cat <<NEXT
    $MAKE -j4 pdfs                              # render PDFs (needs the TeX you just installed)
NEXT
fi
cat <<NEXT

CHANGING A NOTEBOOK'S OUTPUTS — needs the per-framework GPU venv (synced on
first use, ~20 GB CUDA wheels) and a GPU:
    $MAKE -B _notebooks/<fw>/<chapter>/<file>.executed   # re-execute one notebook
    $MAKE capture-outputs FILES=<chapter>/<file>.md      # bless into the store
    $MAKE audit-outputs                                  # what's stale + integrity
See docs/build-system.md for the full model and the four canonical flows.
NEXT
