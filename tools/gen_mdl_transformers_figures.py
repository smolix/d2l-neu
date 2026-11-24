#!/usr/bin/env python3
"""Aggregate generator for chapter_transformers figures.

The chapter's figures were authored in per-section generator scripts
(gen_mdl_transformers_b*.py). This wrapper exists so `make figures`'s
gen_mdl_*_figures.py glob regenerates them all; each part script is
byte-idempotent on its own.
"""
import pathlib
import re
import runpy
import sys

HERE = pathlib.Path(__file__).parent
sys.path.insert(0, str(HERE))
import gen_mdl_figures as fl  # noqa: E402  (the same module the parts import)

# The parts share fl.WRITTEN in this one process. Each part post-processes and
# copies to img/auto only its own figures; written[] collects every path.
written = []
for part in sorted(HERE.glob('gen_mdl_transformers_b*.py')):
    print(f'-- {part.name}')
    start = len(fl.WRITTEN)
    runpy.run_path(str(part), run_name='__main__')
    written += fl.WRITTEN[start:] if len(fl.WRITTEN) >= start else fl.WRITTEN

# Unitless root sizes for every figure (figure guide), including the parts
# that do not rewrite their own (b2, b5). Idempotent: only "...pt" matches.
for path in written:
    with open(path) as fh:
        svg = fh.read()
    head, rest = svg.split('<svg', 1)
    tag, body = rest.split('>', 1)
    new = re.sub(r'(width|height)="([0-9.]+)pt"',
                 lambda m: f'{m.group(1)}="{float(m.group(2)) * 4 / 3:.2f}"', tag)
    if new != tag:
        with open(path, 'w') as fh:
            fh.write(head + '<svg' + new + '>' + body)
