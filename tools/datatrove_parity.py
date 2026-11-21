#!/usr/bin/env python3
"""Datatrove's decisions on documents, for the parity table of 20.2
(`tools/build_lm_data_sources.py parity` calls this script).

Runs in a scratch environment with datatrove 0.10.1 and spacy (the
`lm` extra has neither): it imports nothing from d2l.

    <python with datatrove> tools/datatrove_parity.py IN.jsonl OUT.jsonl

IN.jsonl holds {"id", "text"} rows. OUT.jsonl gets, per row, the decision
of each of FineWeb's four quality filters (datatrove's settings in FineWeb's
pipeline) on the ORIGINAL text, as the reason string datatrove returns for a
removal or null for a keep, and `pipeline`: the first filter that removes
the document when the four run in FineWeb's order on one document (C4's
filter removes lines before FineWeb's filter sees the text), or null.
"""
import json
import sys

from datatrove.data import Document
from datatrove.pipeline.filters import (C4QualityFilter, FineWebQualityFilter,
                                        GopherQualityFilter,
                                        GopherRepetitionFilter)

FILTERS = {  # family: the filter as FineWeb's pipeline configures it
    'gopher_repetition': GopherRepetitionFilter(),
    'gopher_quality': GopherQualityFilter(),
    'c4': C4QualityFilter(filter_no_terminal_punct=False),
    'fineweb': FineWebQualityFilter()}


def decide(f, doc):
    """None for a keep, else datatrove's reason (or 'removed')."""
    out = f.filter(doc)
    if out is True:
        return None
    if isinstance(out, tuple):
        return None if out[0] else str(out[1])
    return None if out else 'removed'


def main(src, dst):
    from importlib.metadata import version
    ver = version('datatrove')
    with open(src, encoding='utf-8') as fin, \
            open(dst, 'w', encoding='utf-8') as fout:
        for line in fin:
            r = json.loads(line)
            row = {'id': r['id']}
            for name, f in FILTERS.items():  # each on the original text
                row[name] = decide(f, Document(text=r['text'], id=r['id']))
            doc, row['pipeline'] = Document(text=r['text'], id=r['id']), None
            for name, f in FILTERS.items():  # FineWeb's order, one document
                if decide(f, doc) is not None:
                    row['pipeline'] = name
                    break
            row['datatrove_version'] = ver
            fout.write(json.dumps(row, sort_keys=True) + '\n')


if __name__ == '__main__':
    main(*sys.argv[1:3])
