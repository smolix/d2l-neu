#!/usr/bin/env bash
# Bootstrap a fresh host for the d2l-neu build.
#
# Idempotent: skip steps that are already done. Run on a fresh Ubuntu/Debian
# host with sudo available; after this completes, `make all` should run.
#
#   ./bootstrap.sh
#
set -euo pipefail
cd "$(dirname "$(readlink -f "$0")")"

step() { printf '\n\033[1;34m==> %s\033[0m\n' "$*"; }
have() { command -v "$1" >/dev/null 2>&1; }

# ── 1. uv ─────────────────────────────────────────────────────────────
if have uv; then
    step "uv already installed: $(uv --version)"
else
    step "Installing uv (standalone installer)"
    curl -LsSf https://astral.sh/uv/install.sh | sh
    # Installer puts uv into ~/.local/bin; make it usable in this shell.
    export PATH="$HOME/.local/bin:$PATH"
    have uv || { echo "uv install failed; $HOME/.local/bin not on PATH"; exit 1; }
    echo "uv installed: $(uv --version)"
fi

# ── 2. System packages (TeX Live + librsvg) ───────────────────────────
# texlive-xetex:               xelatex binary itself
# texlive-latex-extra:         fvextra.sty and friends (referenced by the
#                              project's static/d2l-preamble.tex)
# texlive-latex-recommended:   underscore.sty, float.sty
# texlive-fonts-extra:         fonts pulled in by the d2l preamble
# librsvg2-bin:                `rsvg-convert` CLI used by gen_pdf.py and
#                              Quarto's _svg-to-pdf.lua filter
APT_PKGS=(
    texlive-xetex
    texlive-latex-extra
    texlive-latex-recommended
    texlive-fonts-extra
    librsvg2-bin
)
missing=()
for p in "${APT_PKGS[@]}"; do
    dpkg -s "$p" >/dev/null 2>&1 || missing+=("$p")
done
if [ "${#missing[@]}" -eq 0 ]; then
    step "All required apt packages already installed"
else
    step "Installing apt packages: ${missing[*]}"
    if [ "$EUID" -ne 0 ] && ! have sudo; then
        echo "Need sudo or root to install: ${missing[*]}" >&2
        exit 1
    fi
    SUDO=${EUID:+sudo}; [ "$EUID" -eq 0 ] && SUDO=""
    $SUDO apt-get update
    $SUDO apt-get install -y "${missing[@]}"
fi

# ── 3. logs/ directory ────────────────────────────────────────────────
# Make targets create logs/ themselves via `mkdir -p`, but `tee` calls in
# scripts that pipe output to logs/foo.log fail if logs/ doesn't exist
# yet. Pre-create.
step "Ensuring logs/ exists"
mkdir -p logs

# ── 4. static/d2l-preamble.tex ────────────────────────────────────────
# Gitignored (TeX intermediates) but the PDF build references it via
# `include-in-header`. Re-create it if missing on a fresh checkout.
if [ ! -f static/d2l-preamble.tex ]; then
    step "Creating static/d2l-preamble.tex (referenced by tools/gen_pdf.py)"
    cat > static/d2l-preamble.tex <<'PREAMBLE'
% LaTeX preamble for "Dive into Deep Learning" PDF builds.
% Referenced by tools/gen_pdf.py via `include-in-header`.
% Quarto already loads fontspec/hyperref/amsmath/amssymb/graphicx/longtable/booktabs,
% so this file only adds project-specific tweaks.

% Allow line breaks inside long verbatim spans (URLs, code lines).
\usepackage{fvextra}
\DefineVerbatimEnvironment{Highlighting}{Verbatim}{commandchars=\\\{\},breaklines,breakanywhere,fontsize=\small}

% Better float placement for figures inside narrow sections.
\usepackage{float}

% Allow underscores in identifiers without escaping.
\usepackage[strings]{underscore}

% Show subsections in the TOC.
\setcounter{tocdepth}{1}

% Looser hyphenation to reduce overfull \hbox warnings in code-heavy paragraphs.
\sloppy
PREAMBLE
else
    step "static/d2l-preamble.tex already present"
fi

# ── 5. Quarto + build venv ────────────────────────────────────────────
# `quarto-cli` is declared in the `build` extra; `uv sync --extra build`
# installs it into .venv-build. The make rules also do this on demand,
# but doing it now warms the cache.
step "Syncing .venv-build (provides quarto)"
UV_PROJECT_ENVIRONMENT=.venv-build uv sync --extra build

# ── 6. Sanity check ───────────────────────────────────────────────────
step "Sanity check"
echo "  uv         : $(uv --version)"
echo "  xelatex    : $(xelatex --version 2>/dev/null | head -1)"
echo "  rsvg-convert: $(rsvg-convert --version 2>/dev/null)"
echo "  quarto     : $(.venv-build/bin/quarto --version 2>/dev/null)"
echo "  GPUs       : $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | wc -l) detected"

cat <<'NEXT'

==> Bootstrap complete. Next:

    make help          # see targets
    make all           # full build: notebooks + slides + html + pdfs
    make all-quick     # everything except notebook execution (fast smoke test)

The first `make all` triggers per-framework venv syncs (.venv-pytorch,
.venv-tensorflow, .venv-jax, .venv-mxnet) which download ~20 GB of CUDA
wheels. Subsequent invocations are incremental — the new Phase 1/2
build system rebuilds only the notebooks whose source actually changed.
NEXT
