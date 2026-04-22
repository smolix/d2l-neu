# Plan: Notebook Launch Buttons (Colab, SageMaker, Kaggle, Binder)

## Context

d2l-en's old d2lbook-based build had Colab and SageMaker integration: it
generated platform-specific notebooks, pushed them to GitHub repos, and
injected "Open in Colab" buttons into the HTML. The new Quarto build
(d2l-neu) has none of this — it generates per-framework notebooks in
`_notebooks/<fw>/` but has no cloud-platform launch support.

This plan adds launch buttons for four platforms: **Colab**, **Kaggle**,
**SageMaker**, and **Binder** (Binder only for CPU-only notebooks). The
buttons appear in the rendered HTML, linked to notebooks hosted in
per-platform GitHub repos.

---

## Files to create

### 1. `tools/gen_launch_notebooks.py` (~200 lines)

Generates platform-specific notebooks from the existing `_notebooks/<fw>/*.ipynb`.

**Input**: `_notebooks/<fw>/<chapter>/<name>.ipynb` (already produced by `gen_notebooks.py`)
**Output**: `_launch/<platform>/<fw>/<chapter>/<name>.ipynb`

**Per-platform transforms:**

| Transform | Colab | Kaggle | SageMaker | Binder |
|-----------|-------|--------|-----------|--------|
| Kernel | `python3` | `python3` | framework Conda kernel | `python3` |
| Pip install cell | Yes | Yes | Yes (d2l only) | No (use requirements.txt) |
| GPU metadata | `"accelerator": "GPU"` in Colab metadata | No (manual in UI) | No | N/A |
| SVG URL rewrite | `img/` → absolute `https://d2l.ai/_images/` | No | No | No |
| Filter | All notebooks | All notebooks | All notebooks | **CPU-only** (skip GPU) |

**Dependency resolution** — reads `pyproject.toml` at build time:

```python
import tomllib

def get_cloud_deps(framework):
    """Return pip install list for a framework on cloud platforms."""
    with open('pyproject.toml', 'rb') as f:
        cfg = tomllib.load(f)
    version = cfg['project']['version']
    base = [f'd2l=={version}']
    extras = cfg['project']['optional-dependencies'].get(framework, [])
    # Filter out nvidia-* (pre-installed on GPU platforms) and run extras
    filtered = [p for p in extras if not p.startswith('nvidia-')]
    return base + filtered
```

Platform-specific exclusions on top of this:
- **Colab**: additionally exclude `torch`, `torchvision`, `tensorflow`
  (pre-installed). Keep `gymnasium`, `gpytorch`, `syne-tune`, `flax`, etc.
- **Kaggle**: same exclusions as Colab (torch/tf pre-installed)
- **SageMaker**: only install `d2l=={version}` (Conda kernels have everything)
- **Binder**: full deps including framework package (no GPU libs since CPU-only).
  Written to `_launch/binder/<fw>/requirements.txt` at repo root level.

Reuses `GPU_KEYWORDS` and `notebook_uses_gpu()` pattern from
`tools/runtime_env.py` for the Binder CPU-only filter.

### 2. `tools/inject_launch_buttons.py` (~150 lines)

Post-processes `_book/chapter_*/*.html` to inject launch buttons.

For each HTML page, checks whether a corresponding notebook exists in
`_launch/colab/<fw>/`. If so, injects a button bar. Buttons are grouped by
framework using `data-framework="<fw>"` attributes — the JS in
`_d2l-tabs.html` shows/hides the correct set on tab switch.

**URL patterns per platform:**

```
Colab:     https://colab.research.google.com/github/d2l-ai/d2l-{fw}-colab/blob/master/{path}.ipynb
Kaggle:    https://kaggle.com/kernels/welcome?src=https://github.com/d2l-ai/d2l-{fw}-colab/blob/master/{path}.ipynb
SageMaker: https://studiolab.sagemaker.aws/import/github/d2l-ai/d2l-{fw}-sagemaker/blob/master/{path}.ipynb
Binder:    https://mybinder.org/v2/gh/d2l-ai/d2l-{fw}-binder/HEAD?filepath={path}.ipynb
```

Note: Kaggle's `src=` points to the same GitHub URL as Colab (Kaggle
imports from any GitHub .ipynb URL). No separate Kaggle repo needed.

**Injection point**: after the `<h1>` element in each chapter page. HTML
structure:

```html
<div class="d2l-launch-buttons" data-framework="pytorch">
  <a href="..." target="_blank" title="Open in Google Colab">
    <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Colab">
  </a>
  <a href="..." target="_blank" title="Open in Kaggle">
    <img src="https://kaggle.com/static/images/open-in-kaggle.svg" alt="Kaggle">
  </a>
  <a href="..." target="_blank" title="Open in SageMaker Studio Lab">
    <img src="https://studiolab.sagemaker.aws/studiolab.svg" alt="SageMaker">
  </a>
  <!-- Binder link only if CPU-only notebook -->
  <a href="..." target="_blank" title="Open in Binder">
    <img src="https://mybinder.org/badge_logo.svg" alt="Binder">
  </a>
</div>
<!-- repeat for other frameworks with style="display:none" -->
```

Uses official SVG badges from each platform (loaded from their CDNs, not
local copies).

### 3. Changes to `_d2l-tabs.html`

Add ~15 lines of JS to the existing framework tab sync function. When tabs
switch, show/hide `.d2l-launch-buttons[data-framework="..."]` divs to match
the selected framework. Pattern identical to how tabset visibility already
works (lines 58-63 of the existing code).

### 4. Changes to `_d2l-style.css`

Add ~10 lines for `.d2l-launch-buttons` styling: float right after h1,
inline badges with small gaps, subtle opacity hover effect.

### 5. Changes to `Makefile`

```makefile
# New target: generate platform-specific notebooks
_launch/.stamp: $(foreach fw,$(FRAMEWORKS),_notebooks/$(fw)/.generated)
	@mkdir -p $(LOGDIR)
	@echo "=== Generating launch notebooks ==="
	python3 tools/gen_launch_notebooks.py _notebooks _launch 2>&1 | tee $(LOGDIR)/launch-$(TS).log
	@touch $@

launch: _launch/.stamp
```

Modify the `_book/index.html` recipe to inject buttons after quarto render:

```makefile
_book/index.html: .preprocess.stamp _quarto.yml ... _launch/.stamp
	...
	quarto render --to html
	python3 tools/inject_launch_buttons.py _book _launch
	python3 tools/fix_crossref_numbers.py .
```

Add `_launch` to the `clean` target.

### 6. `tools/upload_launch_repos.sh` (~40 lines)

Deployment script (modeled on d2lbook's `upload_github.sh`). For each
platform+framework combo, clones the target GitHub repo, replaces notebook
contents, commits, and pushes. Called manually or from CI — not part of the
regular build.

```bash
# Usage: ./tools/upload_launch_repos.sh <platform> <framework>
# Example: ./tools/upload_launch_repos.sh colab pytorch
```

---

## GitHub repo mapping

| Platform | Repo pattern | Contents |
|----------|-------------|----------|
| Colab | `d2l-ai/d2l-{fw}-colab` (existing) | All notebooks |
| SageMaker | `d2l-ai/d2l-{fw}-sagemaker` (existing) | All notebooks |
| Kaggle | No new repos needed | Uses Colab repo URLs via `?src=` |
| Binder | `d2l-ai/d2l-{fw}-binder` (new, 3 repos) | CPU-only notebooks + `requirements.txt` |

MXNet Binder repo may not be worth creating (legacy framework). Could skip.

---

## Dependency flow

```
pyproject.toml (source of truth)
       │
       ▼
gen_launch_notebooks.py reads [project] version + [optional-dependencies]
       │
       ├─► Colab/Kaggle pip cell: d2l=={ver} + framework extras (minus nvidia-*, minus pre-installed)
       ├─► SageMaker pip cell:    d2l=={ver} only
       └─► Binder requirements.txt: d2l=={ver} + base deps + framework extras (minus nvidia-*, CPU variant)
```

For Binder, JAX needs `jax[cpu]` instead of `jax[cuda12]`. The script maps
`jax[cuda12]` → `jax[cpu]` for the Binder platform.

---

## Build order

```
gen_notebooks.py          (existing — produces _notebooks/<fw>/*.ipynb)
       │
       ▼
gen_launch_notebooks.py   (new — produces _launch/<platform>/<fw>/*.ipynb)
       │
       ▼
quarto render --to html   (existing — produces _book/)
       │
       ▼
inject_launch_buttons.py  (new — patches _book/ HTML with buttons)
       │
       ▼
fix_crossref_numbers.py   (existing)
```

---

## Verification

1. `make notebooks-pytorch` then `python3 tools/gen_launch_notebooks.py _notebooks _launch`
   - Check `_launch/colab/pytorch/chapter_linear-regression/linear-regression.ipynb` has pip install cell and GPU metadata
   - Check `_launch/binder/pytorch/` excludes GPU notebooks (e.g. no `use-gpu.ipynb`)
   - Check `_launch/binder/pytorch/requirements.txt` has correct deps with `torch` (CPU)
2. `make html` (which now includes button injection)
   - Open `_book/chapter_linear-regression/linear-regression.html` in browser
   - Verify Colab/Kaggle/SageMaker badges appear after the h1
   - Switch framework tabs — badges should update to the correct framework's links
   - On a CPU-only page (e.g. `chapter_appendix-mathematics-for-deep-learning/`), verify Binder badge also appears
   - On a GPU page, verify Binder badge is absent
3. Click a Colab badge — should open the correct notebook in Colab (once repos are populated)
4. Verify PDF build is unaffected (buttons are HTML-only)
