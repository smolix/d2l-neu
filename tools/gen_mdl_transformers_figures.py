#!/usr/bin/env python3
"""Aggregate generator for chapter_transformers figures.

The chapter's figures were authored in per-section generator scripts
(gen_mdl_transformers_b*.py). This wrapper exists so `make figures`'s
gen_mdl_*_figures.py glob regenerates them all; each part script is
byte-idempotent on its own.
"""
import pathlib
import runpy

HERE = pathlib.Path(__file__).parent
for part in sorted(HERE.glob('gen_mdl_transformers_b*.py')):
    print(f'-- {part.name}')
    runpy.run_path(str(part), run_name='__main__')
