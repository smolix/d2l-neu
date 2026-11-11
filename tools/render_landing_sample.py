#!/usr/bin/env python3
"""Render the landing page's code sample from the book source and the output store.

The landing page (index.md) shows one real section of the book: the model
class and the training cell of the concise softmax regression section, in
every framework that section covers, next to the training plot that cell
produced when the notebook was last executed. Both come from the repository:

  * the code from the source cells, found by their stable cell ids;
  * the plot from the committed store, outputs/<fw>/<chapter>/<file>.json
    and the asset it names, copied to static/landing/sample/.

The rendered HTML replaces the block between the markers
    <!-- @sample-begin ... -->
    <!-- @sample-end -->
in index.md. The tab markup is the book's own (.panel-tabset with
data-group="framework"), so _d2l-tabs.html switches it and the reader's choice
of framework carries over into the book.

Code is highlighted by pandoc (through Quarto, as for the book pages), so it
takes the site's code-block style. Run it again after the section or its
stored outputs change:

    python3 tools/render_landing_sample.py
"""
from __future__ import annotations

import hashlib
import html
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
INDEX = REPO / "index.md"

SOURCE = "chapter_linear-classification/softmax-regression-concise.md"
CELLS = [
    "softmax-regression-concise-defining-the-model",
    "softmax-regression-concise-training",
]
# The cell whose stored output (the training plot) is shown.
PLOT_CELL = "softmax-regression-concise-training"

# Tab order and labels follow the book (tools/d2l_preprocess.py FRAMEWORKS).
FRAMEWORKS = ["pytorch", "tensorflow", "jax", "mxnet"]
LABELS = {"pytorch": "PyTorch", "tensorflow": "TensorFlow", "jax": "JAX", "mxnet": "MXNet"}

ASSET_DIR = "static/landing/sample"
BEGIN = "<!-- @sample-begin"
END = "<!-- @sample-end -->"

# Hosted notebooks (tools/build_hosted_notebooks.py: REPOSITORY, BRANCH).
try:
    sys.path.insert(0, str(REPO / "tools"))
    from build_hosted_notebooks import BRANCH, REPOSITORY  # type: ignore
except Exception:  # pragma: no cover - fall back to the values used there
    REPOSITORY, BRANCH = "smolix/d2l-neu", "notebooks"

CELL_RE = re.compile(r"^```\{\.python \.input #(?P<id>[\w-]+)\}\n(?P<body>.*?)^```",
                     re.M | re.S)


def source_cells(text: str) -> dict[str, dict[str, str]]:
    """{cell_id: {framework: code}} for the cells in CELLS."""
    cells: dict[str, dict[str, str]] = {}
    for m in CELL_RE.finditer(text):
        cid = m.group("id")
        if cid not in CELLS:
            continue
        lines = m.group("body").rstrip("\n").split("\n")
        fws = FRAMEWORKS
        if lines and lines[0].startswith("%%tab"):
            fws = [f.strip() for f in lines[0][len("%%tab"):].split(",") if f.strip()]
            if fws == ["all"]:
                fws = FRAMEWORKS
            lines = lines[1:]
        code = "\n".join(re.sub(r"\s*#@save\s*$", "", ln) for ln in lines).strip("\n")
        for fw in fws:
            cells.setdefault(cid, {})[fw] = code
    missing = [c for c in CELLS if c not in cells]
    if missing:
        raise SystemExit(f"{SOURCE}: cell(s) not found: {', '.join(missing)}")
    return cells


def quarto_bin() -> str | None:
    for cand in (os.environ.get("QUARTO"), str(REPO / ".venv-build" / "bin" / "quarto"),
                 shutil.which("quarto")):
        if cand and Path(cand).exists():
            return cand
    return None


def highlight(code: str, quarto: str | None) -> str:
    """Pandoc-highlighted HTML (the book's sourceCode markup), anchors stripped."""
    if quarto is None:
        print("warning: quarto not found; code is not highlighted", file=sys.stderr)
        return ('<div class="sourceCode"><pre class="sourceCode python"><code class="sourceCode python">'
                + html.escape(code) + "</code></pre></div>")
    fence = "`" * 4
    out = subprocess.run(
        [quarto, "pandoc", "-f", "markdown", "-t", "html", "--wrap=none"],
        input=f"{fence}python\n{code}\n{fence}\n", capture_output=True, text=True, check=True,
    ).stdout
    out = re.sub(r'<a href="#cb\d+-\d+" aria-hidden="true" tabindex="-1"></a>', "", out)
    out = re.sub(r'\s+id="cb\d+(-\d+)?"', "", out)
    out = re.sub(r"<span>(.*?)</span>(\n|(?=</code>))", r"\1\2", out)  # per-line wrappers
    return out.strip()


def svg_size(path: Path) -> tuple[int, int]:
    head = path.read_text(encoding="utf-8")[:2000]
    m = re.search(r'<svg[^>]*\swidth="([\d.]+)(pt|px)?"[^>]*\sheight="([\d.]+)(pt|px)?"', head, re.S)
    if not m:
        return 400, 290
    scale = 4 / 3 if m.group(2) == "pt" else 1
    return round(float(m.group(1)) * scale), round(float(m.group(3)) * scale)


def stored_plot(fw: str) -> tuple[str, int, int] | None:
    """Copy the stored training plot of fw into ASSET_DIR; (src, w, h) or None."""
    stem = Path(SOURCE).with_suffix("")
    manifest = REPO / "outputs" / fw / f"{stem}.json"
    if not manifest.exists():
        return None
    cell = json.loads(manifest.read_text(encoding="utf-8"))["cells"].get(PLOT_CELL, {})
    assets = [o["asset"] for o in cell.get("outputs", []) if o.get("asset")]
    if not assets:
        return None
    src = REPO / "outputs" / fw / stem.parent / assets[0]
    rel = f"{ASSET_DIR}/{fw}-{Path(assets[0]).name}"
    dst = REPO / rel
    dst.parent.mkdir(parents=True, exist_ok=True)
    if not dst.exists() or dst.read_bytes() != src.read_bytes():
        shutil.copyfile(src, dst)
    digest = hashlib.sha256(dst.read_bytes()).hexdigest()[:10]
    w, h = svg_size(dst)
    return f"{rel}?v={digest}", w, h


def notebook_counts() -> dict[str, int]:
    return {fw: sum(1 for _ in (REPO / "outputs" / fw).rglob("*.json"))
            for fw in FRAMEWORKS if (REPO / "outputs" / fw).is_dir()}


def render(cells: dict[str, dict[str, str]]) -> str:
    quarto = quarto_bin()
    page = Path(SOURCE).with_suffix(".html").as_posix()
    nb = Path(SOURCE).with_suffix(".ipynb").as_posix()
    fws = [fw for fw in FRAMEWORKS if all(fw in cells[c] for c in CELLS)]
    out = [f"{BEGIN} (auto-generated by tools/render_landing_sample.py from {SOURCE} and outputs/) -->",
           '<div class="panel-tabset d2l-sample" data-group="framework">',
           '<ul class="nav nav-tabs" role="tablist">']
    for i, fw in enumerate(fws):
        active = " active" if i == 0 else ""
        out.append(
            f'<li class="nav-item" role="presentation"><a class="nav-link{active}" '
            f'id="d2l-sample-tab-{fw}" data-bs-toggle="tab" data-bs-target="#d2l-sample-{fw}" '
            f'role="tab" aria-controls="d2l-sample-{fw}" aria-selected="{str(i == 0).lower()}" '
            f'href="#d2l-sample-{fw}">{LABELS[fw]}</a></li>')
    out.append("</ul>")
    out.append('<div class="tab-content">')
    for i, fw in enumerate(fws):
        active = " active show" if i == 0 else ""
        out.append(f'<div id="d2l-sample-{fw}" class="tab-pane fade{active}" role="tabpanel" '
                   f'aria-labelledby="d2l-sample-tab-{fw}">')
        out.append('<div class="d2l-sample-body">')
        out.append('<div class="d2l-sample-code">')
        for c in CELLS:
            out.append(highlight(cells[c][fw], quarto))
        out.append("</div>")
        plot = stored_plot(fw)
        if plot:
            src, w, h = plot
            colab = f"https://colab.research.google.com/github/{REPOSITORY}/blob/{BRANCH}/{fw}/{nb}"
            out.append('<figure class="d2l-sample-out">')
            out.append(f'<img src="{src}" width="{w}" height="{h}" loading="lazy" '
                       f'alt="Training loss, validation loss and validation accuracy per epoch '
                       f'({LABELS[fw]})">')
            out.append(f'<figcaption>The stored output of the last cell, from the book&#8217;s '
                       f'{LABELS[fw]} run. <a href="{page}">Read this section</a> &middot; '
                       f'<a href="{colab}">Open in Colab</a></figcaption>')
            out.append("</figure>")
        out.append("</div>")
        out.append("</div>")
    out.append("</div>")
    out.append("</div>")
    counts = notebook_counts()
    if counts:
        parts = [f"{counts[fw]} in {LABELS[fw]}" for fw in FRAMEWORKS if fw in counts]
        out.append(f'<p class="d2l-sample-note">Every output in the book comes from running its '
                   f'notebooks: {", ".join(parts[:-1])} and {parts[-1]}.</p>')
    out.append(END)
    return "\n".join(out)


def main() -> None:
    cells = source_cells((REPO / SOURCE).read_text(encoding="utf-8"))
    block = render(cells)
    text = INDEX.read_text(encoding="utf-8")
    pattern = re.compile(rf"^[ \t]*{re.escape(BEGIN)}.*?{re.escape(END)}[ \t]*$", re.M | re.S)
    if not pattern.search(text):
        raise SystemExit("index.md: @sample-begin/@sample-end markers not found")
    new = pattern.sub(lambda _: block, text, count=1)
    if new != text:
        INDEX.write_text(new, encoding="utf-8")
        print(f"updated {INDEX}")
    else:
        print(f"unchanged {INDEX}")


if __name__ == "__main__":
    main()
