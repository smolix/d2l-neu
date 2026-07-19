#!/usr/bin/env python3
"""Generate pinned CPU/GPU pylock files for downloadable notebook bundles.

The inputs live in notebook_envs/<framework>/{cpu,gpu}.in. Locks keep the local
D2L project editable as ../.. while stored in the repository; the ZIP builder
rewrites that one path to . because locks live at the archive root.
"""

import argparse
import subprocess
from pathlib import Path

FRAMEWORKS = ('pytorch', 'jax', 'tensorflow', 'mxnet')
PYTHON_RANGE = '>=3.12,<3.13'


def jobs_for(framework):
    if framework == 'mxnet':
        return [
            ('cpu', ['--python-platform', 'x86_64-unknown-linux-gnu']),
            ('cpu-macos', ['--python-platform', 'aarch64-apple-darwin']),
            ('gpu', ['--python-platform', 'x86_64-unknown-linux-gnu']),
        ]
    cpu = ['--universal']
    gpu = ['--python-platform', 'x86_64-unknown-linux-gnu']
    if framework == 'pytorch':
        cpu += ['--torch-backend', 'cpu']
        gpu += ['--torch-backend', 'cu128']
    return [('cpu', cpu), ('gpu', gpu)]


def generate(root, framework, name, options):
    source_name = 'cpu.in' if name == 'cpu-macos' else f'{name}.in'
    source = Path('notebook_envs') / framework / source_name
    output = Path('notebook_envs') / framework / f'pylock.{name}.toml'
    cmd = ['uv', 'pip', 'compile', str(source), '--format', 'pylock.toml',
           '--output-file', str(output), '--python-version', '3.12',
           '--no-sources', '--quiet', '--custom-compile-command',
           'make notebook-env-locks', *options]
    subprocess.run(cmd, cwd=root, check=True)

    path = root / output
    text = path.read_text()
    old_range = 'requires-python = ">=3.12"'
    if text.count(old_range) != 1:
        raise RuntimeError(f'{output}: expected one generated Python range')
    text = text.replace(old_range, f'requires-python = "{PYTHON_RANGE}"')
    editable = 'directory = { path = "../../", editable = true }'
    if text.count(editable) != 1:
        raise RuntimeError(f'{output}: expected one editable ../.. d2l path')
    path.write_text(text)
    print(f'  wrote {output}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--frameworks', default=','.join(FRAMEWORKS))
    args = parser.parse_args()
    selected = [x.strip() for x in args.frameworks.split(',') if x.strip()]
    unknown = set(selected) - set(FRAMEWORKS)
    if unknown:
        parser.error(f'unknown framework(s): {", ".join(sorted(unknown))}')
    root = Path(__file__).resolve().parents[1]
    for framework in selected:
        for name, options in jobs_for(framework):
            generate(root, framework, name, options)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
