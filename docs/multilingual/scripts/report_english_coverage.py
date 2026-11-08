#!/usr/bin/env python3
"""Refresh reference counts against the saved live navigation snapshot, offline."""
import collections
import json
from pathlib import Path
from urllib.parse import urlsplit

BASE = Path(__file__).resolve().parents[1]
ROOT = BASE.parents[1]


def main():
    concepts = [json.loads(s) for s in (BASE / 'concepts.jsonl').read_text().splitlines() if s.strip()]
    manifest = json.loads((BASE / 'english-coverage.json').read_text())
    by_stem = collections.defaultdict(set)
    by_chapter = collections.defaultdict(set)
    for concept in concepts:
        for ref in concept['book_refs']:
            source = ref.get('path') or urlsplit(ref['url']).path.lstrip('/')
            by_stem[str(Path(source).with_suffix(''))].add(concept['id'])
            if ref.get('path'):
                by_chapter[Path(source).parts[0]].add(concept['id'])
    manifest['concept_count'] = len(concepts)
    manifest['reference_count_note'] = 'Concept IDs with a reference to the page stem; .md/.qmd/.html share a stem. These are chosen source references, not a count of every term occurrence or evidence of exhaustive coverage.'
    for page in manifest['live_pages']:
        stem = str(Path(urlsplit(page['url']).path.lstrip('/')).with_suffix(''))
        page['referenced_concepts'] = len(by_stem[stem])
    (BASE / 'english-coverage.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n')
    missing = [p for p in manifest['live_pages'] if not p.get('local_source')]
    lines = ['# English source coverage', '',
             f"Live navigation inspected {manifest['checked_at']} at {manifest['live_url']}. Local HEAD `{manifest['local_git_head']}`. The JSON manifest preserves the snapshot's URLs, source mappings, and hashes. Local files may be modified relative to HEAD; a path match does not establish matching live and local contents.", '',
             f"The snapshot contains {len(manifest['live_pages'])} pages; {len(missing)} have no same-stem local Markdown or Quarto file. The catalog contains **{len(concepts):,} concepts**. It is an extensive starting catalog, not an assertion that every technical expression in the book has been captured.", '',
             'Counts below identify the chosen source references. A concept normally cites one defining passage, so zero references do not mean that a chapter contains no covered terms. These counts do not measure all occurrences or semantic review. See [notes/english-audit.md](notes/english-audit.md) for the audit boundary.', '',
             '| Local chapter | Referenced concepts |', '| --- | ---: |']
    for path in sorted(ROOT.glob('chapter_*')):
        if path.is_dir():
            lines.append(f'| {path.name} | {len(by_chapter[path.name])} |')
    lines += ['', '## Catalog topics', '', '| Topic | Concepts |', '| --- | ---: |']
    for topic, count in sorted(collections.Counter(c['topic'] for c in concepts).items()):
        lines.append(f'| {topic} | {count} |')
    lines += ['', '## Live pages without same-stem local source', '']
    lines += [f"- [{page['title']}]({page['url']})" for page in missing]
    (BASE / 'ENGLISH_COVERAGE.md').write_text('\n'.join(lines) + '\n')
    print(f'Refreshed English reference counts: {len(concepts)} concepts, {len(manifest["live_pages"])} snapshot pages.')


if __name__ == '__main__':
    main()
