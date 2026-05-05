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
