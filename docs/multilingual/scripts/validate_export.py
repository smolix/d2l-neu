#!/usr/bin/env python3
"""Validate canonical terminology data and regenerate review/automation exports.

Run from any directory: python docs/multilingual/scripts/validate_export.py
Uses only the Python standard library. Never modifies canonical glossary shards.
"""
from __future__ import annotations

import argparse
import collections
import csv
import json
import re
import unicodedata
from pathlib import Path
from urllib.parse import quote

BASE = Path(__file__).resolve().parents[1]
ROOT = BASE.parents[1]
LOCALES = ['zh-Hans', 'ja', 'ko', 'es', 'ar', 'pt-BR', 'vi', 'fr', 'hi', 'tr', 'de']
STATUSES = {'verified_d2l', 'verified_external', 'proposed', 'unresolved', 'invariant'}
ELIGIBLE = {'verified_d2l', 'verified_external', 'invariant'}


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(''.join(json.dumps(row, ensure_ascii=False, sort_keys=True) + '\n' for row in rows), encoding='utf-8')


def cell(value):
    return str(value or '').replace('|', '\\|').replace('\n', ' ')


def evidence_link(ev, sources):
    locator = str(ev['locator'])
    if locator.startswith('#'):
        locator = sources[ev['source_id']]['url'].split('#')[0] + locator
    elif re.fullmatch(r'line \d+', locator) and 'github.com/' in sources[ev['source_id']]['url']:
        locator = sources[ev['source_id']]['url'].split('#')[0] + '#L' + locator.split()[-1]
    if locator.startswith('http'):
        target = quote(locator, safe=":/?#[]@!$&'()*+,;=%")
        return f"[{cell(ev['source_id'])}]({target})"
    return f"{cell(ev['source_id'])}: {cell(locator)}"


def main():
    errors = []
    warnings = []
    concepts = read_jsonl(BASE / 'concepts.jsonl')
    by_id = {c['id']: c for c in concepts}
    excerpts = {r['id']: r for r in read_jsonl(BASE / 'english-source-excerpts.jsonl')}
    if len(by_id) != len(concepts):
        errors.append('Duplicate English concept IDs')
    redirects_path = BASE / 'concept-redirects.json'
    if redirects_path.exists():
        for old, new in json.loads(redirects_path.read_text()).items():
            if old in by_id or new not in by_id:
                errors.append(f'Invalid concept redirect: {old} -> {new}')
    for c in concepts:
        for field in ['id', 'en', 'aliases_en', 'topic', 'sense', 'book_refs']:
            if field not in c:
                errors.append(f"{c.get('id')}: missing {field}")
        if not c.get('book_refs'):
            errors.append(f"{c['id']}: no English source reference")
        if c.get('source_review') not in {'sense_spot_checked', 'prose_occurrence_checked'}:
            errors.append(f"{c['id']}: missing or unknown English review level")
        for ref in c.get('book_refs', []):
            if ref.get('path'):
                path = ROOT / ref['path']
                snapshot = excerpts.get(ref.get('snapshot_id'))
                if ref.get('snapshot_id') and (not snapshot or snapshot.get('original_path') != ref['path'] or snapshot.get('original_locator') != ref['locator'] or not snapshot.get('original_excerpt')):
                    errors.append(f"{c['id']}: invalid English excerpt reference")
                if not path.is_file() and not snapshot:
                    errors.append(f"{c['id']}: nonexistent English source {ref['path']}")
            elif not (str(ref.get('url', '')).startswith('https://d2l.smola.org/') or re.fullmatch(r'https://github\.com/(?:d2l-ai/d2l-en|smolix/d2l-neu)/blob/[0-9a-f]{40}/[^?#]+(?:#L[0-9]+)?', str(ref.get('url', '')))):
                errors.append(f"{c['id']}: source requires local path or published book URL")
            if not ref.get('locator'):
                errors.append(f"{c['id']}: source lacks locator")
    sources = []
    for path in sorted((BASE / 'sources').glob('*.jsonl')):
        sources.extend(read_jsonl(path))
    source_map = {}
    for src in sources:
        if src['id'] in source_map and src != source_map[src['id']]:
            errors.append(f"Conflicting source metadata: {src['id']}")
        source_map[src['id']] = src
    terms = {}
    coverage = {}
    for locale in LOCALES:
        path = BASE / 'terms' / f'{locale}.jsonl'
        if not path.exists():
            errors.append(f'Missing locale: {locale}')
            continue
        rows = read_jsonl(path)
        terms[locale] = {r['concept_id']: r for r in rows}
        if len(rows) != len(terms[locale]):
            errors.append(f'{locale}: duplicate concept IDs')
        missing = set(by_id) - set(terms[locale])
        extra = set(terms[locale]) - set(by_id)
        if missing or extra:
            errors.append(f'{locale}: {len(missing)} missing / {len(extra)} unknown concepts')
        counts = collections.Counter()
        topics = collections.defaultdict(collections.Counter)
        for row in rows:
            cid = row['concept_id']
            tag = f'{locale}/{cid}'
            for field in ['preferred', 'alternatives', 'status', 'source_ids', 'evidence', 'notes']:
                if field not in row:
                    errors.append(f'{tag}: missing {field}')
            status = row.get('status')
            counts[status] += 1
            topics[by_id.get(cid, {}).get('topic', 'unknown')][status] += 1
            if status not in STATUSES:
                errors.append(f'{tag}: unknown status {status}')
            preferred = row.get('preferred')
            if status == 'unresolved' and preferred not in (None, ''):
                errors.append(f'{tag}: unresolved entry has preferred term')
            if status != 'unresolved' and not preferred:
                errors.append(f'{tag}: non-unresolved entry lacks preferred term')
            if preferred:
                if unicodedata.normalize('NFC', preferred) != preferred:
                    errors.append(f'{tag}: preferred is not NFC')
                if '\ufffd' in preferred or any(unicodedata.category(ch) == 'Cc' for ch in preferred):
                    errors.append(f'{tag}: replacement/control character in preferred')
                if re.search('[\u202a-\u202e\u2066-\u2069]', preferred):
                    errors.append(f'{tag}: embedded bidi control in preferred')
                if preferred.casefold() == by_id.get(cid, {}).get('en', '').casefold():
                    warnings.append(f'{tag}: English retained; inspect loanword/name rationale')
            for alternative in row.get('alternatives', []):
                if not isinstance(alternative, str) or not alternative.strip():
                    errors.append(f'{tag}: blank/non-string alternative')
                elif unicodedata.normalize('NFC', alternative) != alternative or any(unicodedata.category(ch) == 'Cc' for ch in alternative):
                    errors.append(f'{tag}: alternative has non-NFC/control text')
            for sid in row.get('source_ids', []):
                if sid not in source_map:
                    errors.append(f'{tag}: unknown source {sid}')
            if status in ELIGIBLE and not row.get('evidence'):
                errors.append(f'{tag}: eligible term has no evidence')
            for ev in row.get('evidence', []):
                for field in ['source_id', 'locator', 'source_term', 'context']:
                    if not ev.get(field):
                        errors.append(f'{tag}: evidence lacks {field}')
                sid = ev.get('source_id')
                if sid not in source_map or sid not in row.get('source_ids', []):
                    errors.append(f'{tag}: evidence source is unregistered/unlisted: {sid}')
                elif source_map[sid].get('locale') not in (locale, 'multilingual'):
                    errors.append(f'{tag}: evidence source has locale {source_map[sid].get("locale")}')
            if status == 'invariant' and not row.get('notes'):
                errors.append(f'{tag}: invariant requires rationale')
        coverage[locale] = {'total': len(rows), 'statuses': dict(counts), 'by_topic': {k: dict(v) for k, v in sorted(topics.items())}}
    if errors:
        print('\n'.join(errors))
        raise SystemExit(f'Validation failed: {len(errors)} errors; no exports written.')

    write_jsonl(BASE / 'sources.jsonl', [source_map[k] for k in sorted(source_map)])
    export = BASE / 'exports'
    export.mkdir(exist_ok=True)
    with (export / 'glossary-wide.csv').open('w', newline='', encoding='utf-8-sig') as f:
        fields = ['concept_id', 'en', 'topic', 'sense', 'english_source_review'] + [col for locale in LOCALES for col in (locale, f'{locale}_status')]
        writer = csv.DictWriter(f, fieldnames=fields, lineterminator='\n')
        writer.writeheader()
        for c in concepts:
            row = {'concept_id': c['id'], 'en': c['en'], 'topic': c['topic'], 'sense': c['sense'], 'english_source_review': c.get('source_review', 'prose_occurrence_checked')}
            for locale in LOCALES:
                t = terms[locale][c['id']]
                row[locale] = t['preferred'] or ''
                row[f'{locale}_status'] = t['status']
            writer.writerow(row)
    for locale in LOCALES:
        selected = []
        table = [f'# English → {locale}', '', 'Generated by `scripts/validate_export.py`; edit canonical JSONL, not this file.', '', 'Only `verified_d2l`, `verified_external`, and reviewed `invariant` entries are in the automation export. These statuses indicate source attestation, not native-speaker certification.', '', '| English | Preferred | Status | Concept / sense | Evidence |', '| --- | --- | --- | --- | --- |']
        for c in concepts:
            t = terms[locale][c['id']]
            links = '; '.join(evidence_link(ev, source_map) for ev in t['evidence'])
            table.append(f"| {cell(c['en'])} | {cell(t['preferred'])} | {t['status']} | {cell(c['id'])}: {cell(c['sense'])} | {links} |")
            if t['status'] in ELIGIBLE:
                selected.append({'concept_id': c['id'], 'en': c['en'], 'aliases_en': c['aliases_en'], 'sense': c['sense'], 'english_source_review': c.get('source_review', 'prose_occurrence_checked'), 'locale': locale, 'preferred': t['preferred'], 'status': t['status'], 'source_ids': t['source_ids']})
        write_jsonl(export / 'automation' / f'{locale}.jsonl', selected)
        (export / f'{locale}.md').write_text('\n'.join(table) + '\n', encoding='utf-8')
    (BASE / 'coverage.json').write_text(json.dumps({'concepts': len(concepts), 'locales': coverage, 'retained_english_review_flags': warnings}, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    report = ['# Coverage and review gaps', '', f'{len(concepts)} English concepts; {len(LOCALES)} target languages. Counts are generated from canonical records.', '', '| Locale | D2L attested | External attested | Invariant | Proposed | Unresolved | Automation eligible |', '| --- | ---: | ---: | ---: | ---: | ---: | ---: |']
    for locale, info in coverage.items():
        n = info['statuses']
        report.append('| ' + ' | '.join([locale] + [str(n.get(s, 0)) for s in ['verified_d2l', 'verified_external', 'invariant', 'proposed', 'unresolved']] + [str(sum(n.get(s, 0) for s in ELIGIBLE))]) + ' |')
    report += ['', '## Remaining review queue', '', 'Proposals and unresolved entries are excluded from automation exports. Consult each canonical record for notes, alternatives, and evidence; `coverage.json` provides topic-level counts.', '']
    for locale in LOCALES:
        gaps = [f"`{c['id']}` ({terms[locale][c['id']]['status']})" for c in concepts if terms[locale][c['id']]['status'] not in ELIGIBLE]
        report += [f'### {locale}', '', ', '.join(gaps) or 'No unverified entries.', '']
    (BASE / 'COVERAGE.md').write_text('\n'.join(report).rstrip() + '\n', encoding='utf-8')
    print(f'Validated {len(concepts)} concepts, {sum(len(t) for t in terms.values())} term records, {len(source_map)} sources across {len(LOCALES)} locales.')
    print(f'Exports regenerated. {len(warnings)} retained-English review flags recorded in coverage.json.')


if __name__ == '__main__':
    argparse.ArgumentParser(description=__doc__).parse_args()
    main()
