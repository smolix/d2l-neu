# d2l Tools — VS Code Extension

Workspace tooling for the d2l-neu source repository.

## Features

- **Sync daemon.** Watches `_notebooks/<fw>/**/*.ipynb`. On save,
  invokes `tools/sync_back.py` to round-trip cell edits back to the
  source `.md`. Status bar shows current state (`✓ idle` / `syncing…` /
  `⚠ conflict`).
- **Edit Framework View** (`Cmd+E Cmd+J`). Opens the per-framework
  notebook for the active `.md` source file. Regenerates the `.ipynb`
  if stale.
- **Switch Framework** (`Cmd+E Cmd+S`). Closes the current notebook,
  generates the requested framework's notebook, opens it.
- **Watch Slides** (`Cmd+E Cmd+W`). Spawns
  `tools/watch_slides.py` on the active `.md` source. Browser tab
  opens at `http://localhost:4444/`. Source edits trigger live reload.
- **Reveal Source for Cell.** From a notebook cell, jumps to the
  matching `#<id>` in source `.md`.
- **Lint Source.** Runs `tools/lint_source.py` on save, surfacing
  issues in the Problems pane. Manual command available too.
- **Open Slide Preview.** One-shot render + browser open.

## Running notebooks locally (per-framework kernels)

Since the build was decoupled (see `docs/build-system.md`), the CPU-capable
notebooks can be **executed in-editor**, not just on the GPU box — this is how
you refresh a notebook's committed outputs surgically.

One-time, register the four ipykernels so an opened `.ipynb` auto-selects the
right interpreter from its `metadata.kernelspec.name` (`d2l-<fw>`):

```bash
make kernels        # registers d2l-pytorch / d2l-tensorflow / d2l-jax / d2l-mxnet
```

Then **Edit Framework View** (`Cmd+E Cmd+J`) opens the notebook with its kernel
pre-selected; run cells normally. After editing/running, refresh the committed
store for that file with one command — **Run & Capture Notebook** (`Cmd+E
Cmd+R`). It saves the active notebook, runs `sync_back` (in-editor edits → source
`.md`), then `make -B …executed` + `make capture-outputs` in a `d2l run+capture`
terminal. Equivalent to running by hand:

```bash
make -B _notebooks/<fw>/<chapter>/<file>.executed   # re-execute (or just run in-editor)
make capture-outputs FILES=<chapter>/<file>.md      # bless into outputs/
```

The freshness gate is host-capability-aware (`docs/build-system.md` §3.3a): a
CPU/Apple-Silicon machine renders the whole book and only needs to re-run the
CPU notebooks it touches; GPU/multi-GPU notebooks are deferred to the GPU box.

## Building

```bash
cd .vscode-extension
npm install
npm run compile
npm run package    # produces .vsix
```

Install the `.vsix` via:

```bash
code --install-extension d2l-tools-0.1.0.vsix
```

## Configuration

- `d2l.python` — Python interpreter (default `python3`).
- `d2l.syncDaemon.enabled` — start sync daemon on activation.
- `d2l.syncDaemon.debounceMs` — debounce window after a notebook save.
- `d2l.slidePreview.port` — port for the slide preview server.
