#!/usr/bin/env python3
"""Split multi-framework code blocks that use tab.selected() into per-framework blocks.

Transforms:
    ```{.python .input}
    %%tab pytorch, mxnet, tensorflow
    class Foo:
        if tab.selected('pytorch'):
            self.x = nn.Linear(10, 20)
        if tab.selected('mxnet'):
            self.x = nn.Dense(20)
    ```

Into:
    ```{.python .input}
    %%tab pytorch
    class Foo:
        self.x = nn.Linear(10, 20)
    ```

    ```{.python .input}
    %%tab mxnet
    class Foo:
        self.x = nn.Dense(20)
    ```

    ```{.python .input}
    %%tab tensorflow
    ...
    ```

Usage:
    python tools/split_tab_selected.py <source_dir> [--dry-run]
"""

import re
import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from build_lib import flatten_tab_branches

FRAMEWORKS = ['pytorch', 'tensorflow', 'jax', 'mxnet']


def split_file(path, dry_run=False):
    """Split multi-framework tab.selected() blocks in a single .md file.

    Returns number of blocks split.
    """
    text = path.read_text(encoding='utf-8')
    lines = text.split('\n')
    result = []
    i = 0
    splits = 0

    while i < len(lines):
        line = lines[i]

        # Detect code fence opening: ```{.python .input ...}
        if re.match(r'^```\{\.python', line):
            fence_open = line
            code_lines = []
            i += 1
            while i < len(lines) and not re.match(r'^```\s*$', lines[i]):
                code_lines.append(lines[i])
                i += 1
            fence_close = lines[i] if i < len(lines) else '```'
            i += 1

            # Check: multi-framework %%tab with tab.selected()
            # Skip leading blank lines
            first = 0
            while first < len(code_lines) and code_lines[first].strip() == '':
                first += 1
            code_text = '\n'.join(code_lines)
            tab_m = re.match(r'^%%tab\s+([\w,\s]+)$',
                             code_lines[first]) if first < len(code_lines) else None

            if (tab_m
                    and 'tab.selected(' in code_text):
                raw = tab_m.group(1).strip()
                if raw == 'all':
                    fws = list(FRAMEWORKS)
                else:
                    fws = [f.strip() for f in raw.split(',')]
                    fws = [f for f in fws if f in FRAMEWORKS]

                if len(fws) > 1:
                    # Body is everything after the %%tab line
                    body = '\n'.join(code_lines[first + 1:])

                    for j, fw in enumerate(fws):
                        flat = flatten_tab_branches(body, fw).rstrip('\n')
                        result.append(fence_open)
                        result.append(f'%%tab {fw}')
                        result.append(flat)
                        result.append(fence_close)
                        if j < len(fws) - 1:
                            result.append('')

                    splits += 1
                    continue

            # No split needed — emit as-is
            result.append(fence_open)
            result.extend(code_lines)
            result.append(fence_close)
            continue

        result.append(line)
        i += 1

    if splits > 0 and not dry_run:
        path.write_text('\n'.join(result), encoding='utf-8')

    return splits


def main():
    parser = argparse.ArgumentParser(
        description='Split tab.selected() blocks into per-framework blocks')
    parser.add_argument('source', type=Path, help='d2l-en source directory')
    parser.add_argument('--dry-run', action='store_true',
                        help='Report changes without writing')
    args = parser.parse_args()

    total = 0
    for md in sorted(args.source.glob('chapter_*/*.md')):
        n = split_file(md, dry_run=args.dry_run)
        if n:
            print(f'  {md.relative_to(args.source)}: {n} blocks split')
            total += n

    action = 'would split' if args.dry_run else 'split'
    print(f'\n  Total: {action} {total} blocks')


if __name__ == '__main__':
    main()
