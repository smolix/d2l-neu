# Implementation guide — applying a section's report (PyTorch pass)

You are an **Opus agent implementing the review recommendations for ONE section**,
to the same standard as the already-finished sections (§3.1, §3.2, §3.4, §3.6,
§3.7). Match that bar: intuition-first prose, tight and concrete, every code cell
teaches, figures that carry an idea. Full effort.

## Read first
1. This guide.
2. Your section's **report** (path in your task). Its **Section 4 implementation
   spec** is your task list: apply every **P0** and **P1**; do **P2** if cheap.
   The report's drafted prose/code is a strong start, but adapt it to the live
   source and the rules below.
3. Your **source `.md`** (the source of truth — edit it directly; never `.qmd`).
   Cross-check the served page and the executed-output manifests where useful.

## Scope of THIS pass (PyTorch first)
- **Apply now:** prose, structure, exercises, figures, and **PyTorch** code edits.
- **Defer (do NOT apply — list in your return):** JAX / TensorFlow / MXNet-specific
  *code* fixes. A later batch handles the other three frameworks.
- **MXNet is KEPT** as a co-equal tab. **Discard any "tombstone / de-emphasize /
  archived MXNet" recommendation** from your report.

## Hard rules
- **No `---` (em-dashes), anywhere** (prose, captions, exercises). Use a comma,
  parentheses, a colon, or reword. Strip `---` out of any drafted text you reuse.
- **Edit ONLY your section `.md`** — plus, for a figure, your *own* new generator
  file and its SVG (see Figures). Do **not** touch other sections, the shared
  chapter generators, `d2l.bib`, or `_quarto.yml`.
- **Do NOT run notebooks / `make` / re-execute anything.** Make the PyTorch source
  edits, but do not execute. **List every notebook whose PyTorch outputs your code
  edits invalidate** (relpath) in your return, with what the corrected output
  should look like. The main thread batches all re-execution + the render.
- **Citations:** only cite keys already in `d2l.bib`. Available (verify by grep):
  `Loshchilov.Hutter.2019`, `Belkin.Hsu.Ma.ea.2019`, `nakkiran2021deep`,
  `Recht.Roelofs.Schmidt.ea.2019`, `Koh.Sagawa.Marklund.ea.2021`,
  `Lipton.Wang.Smola.2018`, `Saerens.Latinne.Decaestecker.2002`,
  `Gal.Ghahramani.2016`, `He.Zhang.Ren.ea.2015`, `zhang2021understanding`.
  If you need a key that is NOT present, **do not edit the bib** — use the citation
  and FLAG the missing key in your return. Confirm each `:numref:`/`:cite:` target
  exists (grep the repo) before relying on it; flag any that don't.

## Figures (do them, conflict-free)
Only for genuine **schematic/conceptual** figures the report calls for. A
photographic/anatomical reference image (e.g. a real neuron) is **not** redrawn.
A *computed* data plot that teaches a result stays an inline `d2l.plot` cell in the
notebook (not pre-generated).

1. Create your **own** generator `tools/gen_mdl_<your-slug>_figures.py` (unique to
   your section, so agents never collide), with this header:
   ```python
   import os, sys
   sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
   import gen_mdl_figures as fl
   np, plt = fl.np, fl.plt
   BLUE, ORANGE, GREEN, GRAY, LIGHT = fl.BLUE, fl.ORANGE, fl.GREEN, fl.GRAY, fl.LIGHT
   ```
   Mirror `tools/gen_mdl_linreg_figures.py` (study `fig_ridge_geometry` and
   `fig_oo_classes`): end each fig with `fl.save(fig, "mdl-<cslug>-<id>")`, reuse
   `fl.arrow`/`fl.clean_axes`/`fl.axis_cross` and the palette by name, use
   `set_aspect("equal")` for schematics, and **compute real numbers** where the
   picture must be exact. Add a `FIGURES=[...]` list and copy the `main()` from
   `gen_mdl_linreg_figures.py`. Chapter slug `<cslug>`: `clf` (ch.4), `mlp` (ch.5).
2. Generate, check idempotence, and **look at it**:
   ```bash
   .venv-pytorch/bin/python tools/gen_mdl_<your-slug>_figures.py    # writes img/mdl-<cslug>-<id>.svg
   .venv-pytorch/bin/python tools/gen_mdl_<your-slug>_figures.py    # run twice; `git diff img/` must be empty
   rsvg-convert -w 900 -o /tmp/<your-slug>.png img/mdl-<cslug>-<id>.svg
   ```
   Then **Read `/tmp/<your-slug>.png` and inspect it.** Iterate until it is clean,
   uncluttered, correctly proportioned, with no colliding/overflowing labels. Never
   ship a figure you have not viewed.
3. Include it in your `.md` (caption: no `]` inside, no `---`) and reference it with
   `:numref:`:
   ```
   ![Caption.](../img/mdl-<cslug>-<id>.svg)
   :label:`fig_mdl-<cslug>-<id>`
   ```

## Quality bar
The standard to match: the §3.6 restored polynomial demo (computed + a clean
U-curve), the §3.7 ridge/lasso figure, the §3.2 class diagram, the §3.1
intuition-first opening. Be that good; do not pad.

## Return (compact — this is what the main thread acts on)
1. The change IDs applied (one line each).
2. Figures created: id + confirmation you viewed/verified them.
3. **Notebooks needing PyTorch re-execution** (exact relpaths) + expected output.
4. Any citation key missing from `d2l.bib`.
5. JAX/TF/MXNet code fixes you DEFERRED (for the frameworks batch).
6. Cross-file issues or anything you could not resolve.
