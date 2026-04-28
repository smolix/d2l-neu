#!/usr/bin/env python3
"""Long-running slide preview watcher.

Workflow:
  1. Generate a slide deck for one source file + one framework.
  2. Inject any cached notebook outputs.
  3. Spawn `quarto preview` on the deck (browser tab opens automatically).
  4. Watch the source `.md`. On modify (debounced ~250ms): re-generate,
     re-inject. Quarto preview detects the .qmd change and reloads.
  5. On Ctrl-C / SIGTERM: kill the quarto preview subprocess and exit.

Usage:
    python tools/watch_slides.py --fw <framework> --file <chapter/file.md>
        [--port 4444] [--host localhost]

The deck `.qmd` is written into `_slides/<fw>/<chapter>/<file>.qmd`.
Inputs and outputs use the same conventions as `make slides`.
"""

import argparse
import os
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))


def _which_python(fw):
    """Pick a venv python that has nbformat (required by quarto)."""
    for cand in (Path(f'.venv-{fw}'), Path('.venv-pytorch'), Path('.venv-jax'),
                 Path('.venv-tensorflow'), Path('.venv-mxnet')):
        py = cand.absolute() / 'bin' / 'python'
        if py.exists():
            return str(py)
    return sys.executable


def _gen_and_inject(source: Path, fw: str, slides_root: Path, log_lock):
    from gen_slides import generate_slides_qmd
    from inject_outputs import inject_slides

    fw_dir = slides_root / fw
    fw_dir.mkdir(parents=True, exist_ok=True)
    rel = source if not source.is_absolute() else source.relative_to(Path.cwd())
    dst = fw_dir / Path(str(rel).replace('.md', '.qmd'))
    dst.parent.mkdir(parents=True, exist_ok=True)

    warnings = []
    content = generate_slides_qmd(source, fw, warnings)
    if content is None:
        with log_lock:
            print(f'[watch] {source}: no slides for {fw}', flush=True)
        return None

    # Atomic write
    tmp = dst.with_suffix(dst.suffix + '.tmp')
    tmp.write_text(content, encoding='utf-8')
    os.replace(tmp, dst)

    # Ensure _slides/_quarto.yml override + img/data symlinks
    yml = slides_root / '_quarto.yml'
    if not yml.exists():
        yml.write_text('project:\n  type: default\n', encoding='utf-8')
    img_link = fw_dir / 'img'
    if not img_link.exists():
        img_link.symlink_to(Path('../../img'))
    data_link = fw_dir / 'data'
    if not data_link.exists():
        data_dir = Path.cwd() / 'data'
        data_dir.mkdir(exist_ok=True)
        data_link.symlink_to(data_dir)

    # Inject if notebook exists for this fw
    notebooks_dir = Path('_notebooks')
    if (notebooks_dir / fw).exists():
        img_outputs = slides_root / 'img' / 'outputs'
        try:
            inject_slides(str(slides_root), fw, str(notebooks_dir),
                           str(img_outputs))
        except Exception as e:
            with log_lock:
                print(f'[watch] inject failed: {e}', flush=True)

    if warnings:
        with log_lock:
            for w in warnings[:5]:
                print(f'[watch] {w}', flush=True)
    return dst


def main():
    parser = argparse.ArgumentParser(
        description='Watch a slide source and live-preview in browser')
    parser.add_argument('--fw', required=True,
                        choices=['pytorch', 'tensorflow', 'jax', 'mxnet'])
    parser.add_argument('--file', required=True, type=Path,
                        help='Source .md path (e.g. chapter_foo/bar.md)')
    parser.add_argument('--slides-dir', type=Path, default=Path('_slides'))
    parser.add_argument('--port', type=int, default=4444)
    parser.add_argument('--host', default='localhost')
    parser.add_argument('--debounce-ms', type=int, default=250)
    args = parser.parse_args()

    source = args.file.resolve()
    if not source.exists():
        print(f'error: source not found: {source}', file=sys.stderr)
        sys.exit(1)

    slides_root = args.slides_dir.resolve()
    log_lock = threading.Lock()

    # Initial gen + inject
    print(f'[watch] generating slides for {args.fw} from {source.name}',
          flush=True)
    dst = _gen_and_inject(source, args.fw, slides_root, log_lock)
    if dst is None:
        sys.exit(2)

    # Spawn quarto preview
    env = {**os.environ, 'QUARTO_PYTHON': _which_python(args.fw)}
    print(f'[watch] starting quarto preview on http://{args.host}:{args.port}/',
          flush=True)
    preview = subprocess.Popen(
        ['quarto', 'preview', str(dst), '--to', 'revealjs',
         '--no-watch-inputs', '--port', str(args.port), '--host', args.host],
        env=env)

    # Watch source .md
    try:
        from watchdog.observers import Observer
        from watchdog.events import FileSystemEventHandler
    except ImportError:
        print('error: watchdog not installed — install with `pip install watchdog`',
              file=sys.stderr)
        preview.terminate()
        sys.exit(3)

    last_event = [0.0]
    pending = [False]
    debounce_lock = threading.Lock()

    def maybe_rebuild():
        # Sleep to coalesce events; if a fresher event arrived, abandon.
        time.sleep(args.debounce_ms / 1000.0)
        with debounce_lock:
            if not pending[0]:
                return
            pending[0] = False
        with log_lock:
            t0 = time.time()
            print(f'[watch] rebuilding…', flush=True)
        _gen_and_inject(source, args.fw, slides_root, log_lock)
        with log_lock:
            print(f'[watch] reloaded ({time.time() - t0:.2f}s)', flush=True)

    class Handler(FileSystemEventHandler):
        def on_modified(self, event):
            if Path(event.src_path).resolve() != source:
                return
            with debounce_lock:
                last_event[0] = time.time()
                if pending[0]:
                    return
                pending[0] = True
            threading.Thread(target=maybe_rebuild, daemon=True).start()

    observer = Observer()
    observer.schedule(Handler(), str(source.parent), recursive=False)
    observer.start()

    def shutdown(*_):
        with log_lock:
            print('[watch] shutting down', flush=True)
        observer.stop()
        try:
            preview.terminate()
            preview.wait(timeout=2)
        except subprocess.TimeoutExpired:
            preview.kill()
        sys.exit(0)

    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    try:
        preview.wait()
    finally:
        observer.stop()
        observer.join()


if __name__ == '__main__':
    main()
