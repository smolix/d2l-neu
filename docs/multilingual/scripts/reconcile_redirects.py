#!/usr/bin/env python3
"""Reconcile existing locale shards after a reviewed English concept merge.

Only active retired IDs are merged. Preserve every pre-merge variant in an
archive; never replay archived decisions over a later reviewer correction.
"""
import json
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]
ELIGIBLE = {'verified_d2l', 'verified_external', 'invariant'}


def read(path):
    return [json.loads(s) for s in path.read_text().splitlines() if s.strip()]


def dump(path, rows):
    path.write_text(''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in rows))


def main():
    redirects = json.loads((BASE / 'concept-redirects.json').read_text())
    concepts = read(BASE / 'concepts.jsonl')
    ids = {c['id'] for c in concepts}
    assert all(old not in ids and new in ids for old, new in redirects.items())
    archive_path = BASE / 'candidates' / 'merged-variant-terms.jsonl'
    archive = read(archive_path) if archive_path.exists() else []
    outputs = []
    merged_count = 0
    for path in sorted((BASE / 'terms').glob('*.jsonl')):
        rows = read(path)
        groups = {}
        for row in rows:
            cid = redirects.get(row['concept_id'], row['concept_id'])
            if cid not in ids:
                raise ValueError(f'{path.name}: unknown concept {cid}')
            groups.setdefault(cid, []).append(row)
        if set(groups) != ids:
            raise ValueError(f'{path.name}: incomplete concept coverage')
        output = []
        for concept in concepts:
            cid = concept['id']
            group = groups[cid]
            if len(group) == 1 and group[0]['concept_id'] == cid:
                output.append(group[0])
                continue
            # An attested equivalent can replace a proposal. Among equally
            # attested choices, preserve the retained concept's reviewed term.
            chosen = max(group, key=lambda r: (r['status'] in ELIGIBLE,
                                               r['concept_id'] == cid,
                                               r['status'] != 'unresolved'))
            result = dict(chosen)
            result['concept_id'] = cid
            result['merged_from'] = sorted(r['concept_id'] for r in group if r['concept_id'] != cid)
            result['notes'] += ' Equivalent concept IDs were merged; other pre-merge translations and evidence are preserved in candidates/merged-variant-terms.jsonl and do not inherit this preferred term\'s status.'
            output.append(result)
            for row in group:
                archive.append({'locale': path.stem, 'canonical_concept_id': cid, 'original': row})
            merged_count += 1
        outputs.append((path, output))
    # Validate every shard before writing any output.
    if merged_count:
        dump(archive_path, archive)
        for path, rows in outputs:
            dump(path, rows)
    print(f'Reconciled {merged_count} locale/concept groups; {len(concepts)} canonical concepts.')


if __name__ == '__main__':
    main()
