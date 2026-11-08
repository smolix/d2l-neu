#!/usr/bin/env python3
"""Check quoted D2L expressions at original-file line locators in local caches.

This verifies provenance, not semantic correctness. Missing caches and non-GitHub
sources are explicitly reported as untested, not counted as passing.
"""
import html
import json
import re
import subprocess
import unicodedata
from pathlib import Path
from urllib.parse import unquote

BASE = Path(__file__).resolve().parents[1]
CACHE = Path('/tmp/d2l-glossary-corpora')


def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def norm(text):
    return re.sub(r'\s+', '', unicodedata.normalize('NFC', html.unescape(text)).casefold().replace('*', '').replace('_', ''))


def main():
    sources = {s['id']: s for p in (BASE / 'sources').glob('*.jsonl') for s in rows(p)}
    tested = []
    failures = []
    untested = []
    revisions = {}
    for p in sorted((BASE / 'terms').glob('*.jsonl')):
        for term in rows(p):
            if term['status'] not in {'verified_d2l', 'verified_external', 'invariant'}:
                continue
            for ev in term['evidence']:
                record = {'locale': p.stem, 'concept_id': term['concept_id'], 'source_id': ev['source_id'], 'locator': ev['locator']}
                src = sources.get(ev['source_id'], {})
                url = ev['locator']
                if re.fullmatch(r'line \d+', url):
                    url = src.get('url', '').split('#')[0] + '#L' + url.split()[-1]
                match = re.match(r'https://github.com/([^/]+)/([^/]+)/blob/([^/]+)/(.+?)#L(\d+)', url)
                if not match:
                    untested.append(record | {'reason': 'Not a GitHub original-file line locator'})
                    continue
                owner, repo, commit, path, line = match.groups()
                local = CACHE / repo / unquote(path)
                if not local.exists():
                    untested.append(record | {'reason': 'Original source not in local corpus cache'})
                    continue
                if repo not in revisions:
                    try:
                        revisions[repo] = subprocess.check_output(['git', '-C', str(CACHE / repo), 'rev-parse', 'HEAD'], text=True).strip()
                    except subprocess.CalledProcessError:
                        revisions[repo] = None
                if not revisions[repo] or not revisions[repo].startswith(commit):
                    untested.append(record | {'reason': 'Cached source revision differs from cited revision'})
                    continue
                n = int(line)
                lines = local.read_text().splitlines()
                excerpt = ' '.join(lines[max(n - 1, 0):n + 2])
                record['resolved_url'] = url
                tested.append(record)
                if norm(ev['source_term']) not in norm(excerpt):
                    failures.append(record | {'source_term': ev['source_term'], 'reason': 'Source expression absent at cited original-file line (including next two lines)'})
    report = {'method': 'NFC/case/whitespace/Markdown-emphasis-normalized source expression occurrence at cited original-file line and next two lines. Does not establish meaning or good usage.', 'checked': len(tested), 'failed': len(failures), 'untested': len(untested), 'failures': failures, 'checks': tested, 'untested_evidence': untested}
    (BASE / 'evidence-audit.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    print(f"Original-file evidence: {len(tested)} checked, {len(failures)} failed, {len(untested)} other/missing-cache evidence items untested.")
    for row in failures[:25]:
        print(row['locale'], row['concept_id'], row['resolved_url'])
    raise SystemExit(bool(failures))


if __name__ == '__main__':
    main()
