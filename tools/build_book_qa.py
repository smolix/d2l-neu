"""Build the book's question set lm-book-qa-v1 (chapter 19, section 19.5).

Stages (chapter 19 spec, part 1.5, with the author's override of
2026-11-14: a frontier model verifies every dev and test item):

  S0 corpus r3.1 and the targets      S5 neighbour check (dev, test)
  S1 anchors per split and type       S6 closed-book flag (Qwen3-4B + judge)
  S2 drafts (DeepSeek-V4.1-Flash)     S7 type quotas (dev, test)
  S3 automatic filters                S8 frontier verification (dev, test)
  S4 checklist (Qwen3.8-27B)          S9 write the zip; S10 review sheet

Every API call goes through chapter 18's d2l.ChatAPI (one d2l.APIRecording
per model, fixed cost shares per stage cache), and every GPU stage writes its
outputs, keyed by candidate id, to <cache>/local/. With --mode replay the
script reads all of it from the cache folder (or the registered artifact
lm-book-qa-api-cache-v1) and rebuilds the set bit for bit, with neither a GPU
nor an API key.

Usage (from the repository root, in .venv-lm, HF_HOME set):
  python tools/build_book_qa.py --mode record --smoke
  python tools/build_book_qa.py --mode record --out data/lm/authoring/qa
  python tools/build_book_qa.py --mode replay --cache-dir <folder> --out <dir>
  python tools/build_book_qa.py --mode record --calibrate <generations.jsonl> \
      --out <dir>   (labels lm-rag-judge-labels-v1; --smoke first)
The key is read from OPENROUTER_API_KEY by the launching shell; it is never
printed, logged or written.
"""
import argparse
import asyncio
import collections
import datetime
import hashlib
import json
import math
import os
import random
import re
import sys
import zipfile

W = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, W)
sys.path.insert(0, os.path.join(W, 'tools', 'lm'))

import book_qa_prompts as P  # noqa: E402

CORPUS_VERSION = 'r3.1'
KEY, CACHE_KEY = 'lm-book-qa-v1', 'lm-book-qa-api-cache-v1'
DRAFTER, DRAFTER_AT = 'deepseek/deepseek-v4.1-flash-1017', 'novita'
CHECKER, CHECKER_AT = 'qwen/qwen3.8-27b', 'deepinfra/bf16'
VERIFIER, VERIFIER_AT = 'anthropic/claude-opus-5.5', 'anthropic'
CAP_USD = {DRAFTER: 20.0, CHECKER: 15.0, VERIFIER: 15.0}
SHARES = {  # fraction of a model's cap that each stage cache may spend
    'smoke-draft': 0.02, 's2-draft': 0.85,
    'smoke-check': 0.02, 's4-check': 0.70, 's5-neighbours': 0.18,
    'calib-check': 0.10,
    'smoke-verify': 0.04, 's8-verify': 0.73, 'calib-verify': 0.23}
ARM_MODEL = {'smoke-draft': DRAFTER, 's2-draft': DRAFTER,
             'smoke-check': CHECKER, 's4-check': CHECKER,
             's5-neighbours': CHECKER, 'calib-check': CHECKER,
             'smoke-verify': VERIFIER, 's8-verify': VERIFIER,
             'calib-verify': VERIFIER}
TYPES = ('single', 'explain', 'numeric', 'code', 'multihop', 'unanswerable')
QUOTA = {'single': 120, 'explain': 40, 'numeric': 20, 'code': 20,
         'multihop': 20, 'unanswerable': 20}
# Anchors per type and split: ceil(factor x quota); the factor of a type is
# raised (up to all eligible anchors) when its dev/test pass rate is low
# (raised after the dev/test pilot of S2-S4, 2026-11-14: anchors with a
# surviving candidate were 30/36 numeric, 33/36 code, 19-24/36 multihop and
# 27-28/36 unanswerable per split, before S5 and S8)
FACTOR = {'single': 1.8, 'explain': 1.8, 'numeric': 2.5, 'code': 2.5,
          'multihop': 3.0, 'unanswerable': 3.0}
TRAIN_MULTIHOP, TRAIN_UNANSWERABLE = 200, 200
MIN_TOKENS, SECTION_CHARS = 80, 6000
NEAR_DUP, NEAR_DUP_LOG = 0.90, 0.85
SEED = 0
DRAFT_PARAMS = dict(max_tokens=3000, temperature=0.6,
                    response_format={'type': 'json_object'})
# two passages: the pilot's multihop drafts ran out of tokens while reasoning
MULTIHOP_DRAFT_PARAMS = dict(DRAFT_PARAMS, max_tokens=8000)
CHECK_PARAMS = dict(max_tokens=600, temperature=0, seed=0,
                    response_format={'type': 'json_object'})
NEIGHBOR_PARAMS = dict(max_tokens=600, temperature=0, seed=0,
                       response_format={'type': 'json_object'})
VERIFY_PARAMS = dict(max_tokens=1500,  # the endpoint takes no temperature
                     response_format={'type': 'json_object'})
GEN, JUDGE = 'Qwen/Qwen3-4B', 'Qwen/Qwen3.5-4B'
# Set version 2 (2026-11-14): the closed-book flag re-judged by the judge of
# 19.5 chosen against frontier-model labels (lm-rag-judge-labels-v1)
JUDGE_V2 = {'id': 'Qwen/Qwen3.8-27B-FP8',
            'revision': 'c3a462cc3d38cf351f3b26cbd763469eeb74ef81',
            'served': 'Qwen/Qwen3.8-27B-FP8', 'thinking': True,
            'max_new_tokens': 3000, 'prompt': 'RAG_JUDGE_V1',
            'engine': 'vLLM 0.33.0, pipeline parallel over two RTX 4090',
            'default_params': {'extra_body': {'chat_template_kwargs': {
                'enable_thinking': True}}}}
KAPPA_SAMPLE = 40  # S4-rejected dev/test candidates the verifier also sees
STOP = set(('''a an the of to in on for and or but is are was were be been
being by with as at from that this these those it its into than then which
what when where who whom whose why how does do did can could should would
will shall may might must not no yes if so such there their they them we our
you your he she his her i me my one two both each any all some more most
other another also only very same between about over under after before
while during through per via up down out own just''').split())
CONTEXT_PHRASES = re.compile(
    r'\b(?:this|that|these|the above|the following|the preceding|the given|'
    r'the provided|the shown|the previous|the current)\s+(?:section|passage|'
    r'text|chapter|paragraph|excerpt|snippet|code|cell|listing|figure|table|'
    r'example|function|class|implementation|equation|experiment|discussion)'
    r'\b|\b(?:the passage|the excerpt|the snippet|the text|above|below)\b|'
    r'\bas shown\b|\baccording to the (?:text|passage|section|author)',
    re.I)
IDENT = re.compile(
    r'`[^`]+`|\b[A-Za-z_]\w*_\w+\b|\b[a-z]+[A-Z]\w*\b|\b[A-Z][a-z0-9]+'
    r'[A-Z]\w*\b|\b(?!e\.g|i\.e)[A-Za-z_]\w*\.[A-Za-z_]\w*\b|\b\w+\(')


def log(*a):
    print(*a, flush=True)


def sha256(b):
    return hashlib.sha256(b if isinstance(b, bytes) else b.encode()).hexdigest()


def item_seed(s):
    return int(hashlib.sha1(s.encode()).hexdigest()[:8], 16) % 2**31


def dumps(obj):
    return json.dumps(obj, ensure_ascii=False, sort_keys=False)


# --------------------------------------------------------------- S0 / S1


class Book:
    """The corpus under its three chunkings, with per-document blocks."""

    def __init__(self, d2l, root):
        self.c = d2l.book_corpus(root=root)
        if self.c.version != CORPUS_VERSION:
            raise SystemExit(f'corpus {self.c.version} is not {CORPUS_VERSION}')
        self.fixed = {o: d2l.book_corpus('fixed', size=256, overlap=o,
                                         root=root) for o in (0, 64)}
        self.d2l = d2l
        self._blocks = {}

    def doc_text(self, doc_id):
        return self.c.docs[doc_id]['text']

    def blocks(self, doc_id):
        if doc_id not in self._blocks:
            self._blocks[doc_id] = self.d2l._r3_parse_blocks(
                self.doc_text(doc_id))
        return self._blocks[doc_id]

    def chapter(self, doc_id):
        return self.c.docs[doc_id]['chapter']


def is_eligible(chunk):
    return chunk['kind'] in ('prose', 'mixed') and chunk['n_tokens'] >= MIN_TOKENS


def prose_only(text):
    text = re.sub(r'^```.*?^```', ' ', text, flags=re.M | re.S)
    text = re.sub(r'\$\$.*?\$\$', ' ', text, flags=re.S)
    text = re.sub(r'\$[^$\n]+\$', ' ', text)
    text = re.sub(r'`[^`\n]+`', ' ', text)
    return re.sub(r'(?:Section|Figure|Table|Chapter|Equation) "[^"]*"', ' ',
                  text)


def has_number(chunk):
    return re.search(r'(?<![\w.])\d+(?:[.,]\d+)?%?(?![\w])',
                     prose_only(chunk['text'])) is not None


def has_code(chunk):
    return bool(chunk['cell_ids']) and '```python' in chunk['text']


def round_robin(items, key, rng_tag):
    """Items grouped by key, each group shuffled with a seeded generator,
    then interleaved one per group in sorted group order."""
    groups = collections.defaultdict(list)
    for x in items:
        groups[key(x)].append(x)
    for g in groups:
        random.Random(f'{rng_tag}|{g}|{SEED}').shuffle(groups[g])
    out, keys = [], sorted(groups)
    while any(groups[g] for g in keys):
        for g in keys:
            if groups[g]:
                out.append(groups[g].pop(0))
    return out


def find_crossref_pairs(book, split):
    """(A, B) chunk pairs in two documents of split: A's text cites the
    title of a section of B (the cleaned form of a :numref:); B is the first
    eligible chunk at or after that section's heading."""
    c = book.c
    docs = [d for d in c.docs if c.split_of(d) == split]
    titles = collections.defaultdict(list)
    for d in docs:
        for h in c.docs[d]['headings']:
            if h['level'] >= 1 and h.get('label'):
                titles[h['title']].append((d, h['char_start']))
    by_doc = collections.defaultdict(list)
    for ch in c.chunks:
        if c.split_of(ch['doc_id']) == split and is_eligible(ch):
            by_doc[ch['doc_id']].append(ch)
    pairs, seen = [], set()
    for d in docs:
        for a in by_doc[d]:
            for m in re.finditer(r'(?:Section|Chapter) "([^"]+)"', a['text']):
                hits = [t for t in titles.get(m.group(1), []) if t[0] != d]
                if len(hits) != 1:
                    continue
                db, start = hits[0]
                b = next((x for x in by_doc[db] if x['char_end'] > start),
                         None)
                if b is not None and (a['id'], b['id']) not in seen:
                    seen.add((a['id'], b['id']))
                    pairs.append((a, b, 'crossref'))
    return pairs


def h2_sections(book, split):
    """H2 sections of the documents of split: (doc_id, start, end, title)."""
    out = []
    for d, doc in book.c.docs.items():
        if book.c.split_of(d) != split:
            continue
        hs = doc['headings']
        for i, h in enumerate(hs):
            if h['level'] != 2:
                continue
            end = next((x['char_start'] for x in hs[i + 1:]
                        if x['level'] <= 2), len(doc['text']))
            if end - h['char_start'] >= 800:
                out.append((d, h['char_start'], end, h['title']))
    return out


def assign_types(chunks, targets):
    """Target type per chunk in order: the eligible type (code needs a code
    cell, numeric a number in the prose) with the largest remaining share
    of its target; chunks are skipped once every target is met."""
    left, out = dict(targets), []
    order = ('code', 'numeric', 'explain', 'single')
    for ch in chunks:
        ok = [t for t in order if left.get(t, 0) > 0 and (
            t in ('single', 'explain') or (t == 'code' and has_code(ch))
            or (t == 'numeric' and has_number(ch)))]
        if not ok:
            if all(v <= 0 for v in left.values()):
                break
            continue
        t = max(ok, key=lambda t: left[t] / targets[t])
        left[t] -= 1
        out.append((ch, t))
    return out, left


def anchors_for_split(book, split, private_exclude=None):
    """The anchors of one split (S1), in drafting order."""
    c, rng_tag = book.c, f'r3-anchors-v1|{split}'
    pool = [ch for ch in c.chunks if c.split_of(ch['doc_id']) in (
        ('dev', 'test') if split == 'private' else (split,))
            and is_eligible(ch)]
    if private_exclude:
        pool = [ch for ch in pool if ch['id'] not in private_exclude]
    order = round_robin(pool, lambda ch: book.chapter(ch['doc_id']), rng_tag)
    anchors, notes = [], {}
    if split == 'train':
        n_mh, n_un = TRAIN_MULTIHOP, TRAIN_UNANSWERABLE
    else:
        n_mh = math.ceil(FACTOR['multihop'] * QUOTA['multihop'])
        n_un = math.ceil(FACTOR['unanswerable'] * QUOTA['unanswerable'])
    # multihop: cross-reference pairs, then same-chapter pairs
    pairs = [] if split == 'private' else find_crossref_pairs(book, split)
    pairs = round_robin(pairs, lambda p: book.chapter(p[0]['doc_id']),
                        rng_tag + '|mh')
    used, mh = set(), []
    for a, b, how in pairs:
        if len(mh) == n_mh:
            break
        if split != 'train' and (a['id'] in used or b['id'] in used):
            continue
        mh.append((a, b, how))
        used |= {a['id'], b['id']}
    notes['multihop_crossref_pairs_available'] = len(pairs)
    if len(mh) < n_mh:  # fill from two documents of the same chapter
        by_ch = collections.defaultdict(list)
        for ch in order:
            if split == 'train' or ch['id'] not in used:
                by_ch[book.chapter(ch['doc_id'])].append(ch)
        fill = []
        for chap in sorted(by_ch):
            xs = by_ch[chap]
            for i, a in enumerate(xs):
                b = next((x for x in xs[i + 1:] if x['doc_id'] != a['doc_id']
                          and (split == 'train' or x['id'] not in used)), None)
                if b is not None:
                    fill.append((a, b, 'same_chapter'))
        fill = round_robin(fill, lambda p: book.chapter(p[0]['doc_id']),
                           rng_tag + '|mhfill')
        for a, b, how in fill:
            if len(mh) == n_mh:
                break
            if split != 'train' and (a['id'] in used or b['id'] in used):
                continue
            mh.append((a, b, how))
            used |= {a['id'], b['id']}
    notes['multihop_same_chapter_fill'] = sum(h == 'same_chapter'
                                              for _, _, h in mh)
    # single-anchor types
    single_pool = [ch for ch in order if split == 'train' or
                   ch['id'] not in used]
    if split == 'train':
        base = sum(QUOTA[t] for t in ('single', 'explain', 'numeric', 'code'))
        targets = {t: math.ceil(QUOTA[t] / base * len(single_pool)) + 1
                   for t in ('single', 'explain', 'numeric', 'code')}
    else:
        targets = {t: math.ceil(FACTOR[t] * QUOTA[t])
                   for t in ('single', 'explain', 'numeric', 'code')}
    typed, left = assign_types(single_pool, targets)
    notes['type_shortfall_anchors'] = {t: v for t, v in left.items()
                                       if v > 0 and split != 'train'}
    for ch, t in typed:
        anchors.append({'aid': f'{split}|{t}|{ch["id"]}', 'split': split,
                        'type': t, 'chunks': [ch['id']], 'how': None})
    for a, b, how in mh:
        anchors.append({'aid': f'{split}|multihop|{a["id"]}+{b["id"]}',
                        'split': split, 'type': 'multihop',
                        'chunks': [a['id'], b['id']], 'how': how})
    # unanswerable: H2 sections
    if split != 'private':
        secs = round_robin(h2_sections(book, split),
                           lambda s: book.chapter(s[0]), rng_tag + '|un')
        for d, s, e, title in secs[:n_un]:
            ids = [ch['id'] for ch in book.c._by_doc[d]
                   if ch['char_end'] > s and ch['char_start'] < e]
            anchors.append({'aid': f'{split}|unanswerable|{d}@{s}',
                            'split': split, 'type': 'unanswerable',
                            'chunks': ids, 'section': [d, s, e],
                            'how': None})
    return anchors, notes


# --------------------------------------------------------------------- S2


def fewshot_block():
    parts = []
    for ex in P.FEWSHOT_V1:
        for e in ex['answer']['evidence']:
            assert e in ex['passage'], 'few-shot evidence is not verbatim'
        parts.append(f'Example ({ex["type"]}). Passage:\n<<<\n'
                     f'{ex["passage"]}\n>>>\nA good item:\n'
                     f'{json.dumps(ex["answer"], ensure_ascii=False)}')
    return ('Examples of good items, from other passages of the book:\n\n'
            + '\n\n'.join(parts))


def section_text(book, anchor):
    d, s, e = anchor['section']
    text = book.doc_text(d)[s:e]
    if len(text) > SECTION_CHARS:
        cut = text.rfind('\n\n', 0, SECTION_CHARS)
        text = text[:cut if cut > SECTION_CHARS // 2 else SECTION_CHARS]
    return text


def draft_messages(book, anchor):
    D, c = P.DRAFT_V1, book.c
    t = anchor['type']
    common = dict(type_rule=P.TYPE_RULES[t], rules=D['rules'],
                  format=D['format'].replace('<type>', t))
    if t == 'unanswerable':
        d = anchor['section'][0]
        first = c[anchor['chunks'][0]]
        user = D['user_unanswerable'].format(
            heading=c.header(first), text=section_text(book, anchor),
            type_rule=P.TYPE_RULES[t])
    elif t == 'multihop':
        a, b = (c[i] for i in anchor['chunks'])
        user = fewshot_block() + '\n\n---\n\n' + D['user_multihop'].format(
            heading_a=c.header(a), text_a=a['text'], heading_b=c.header(b),
            text_b=b['text'], **common)
    else:
        ch = c[anchor['chunks'][0]]
        user = fewshot_block() + '\n\n---\n\n' + D['user_single'].format(
            heading=c.header(ch), text=ch['text'], qtype=t, **common)
    return [{'role': 'system', 'content': D['system']},
            {'role': 'user', 'content': user}]


def parse_json_obj(text):
    """The first JSON object in text (code fences allowed), or None."""
    if not text:
        return None
    t = text.strip()
    t = re.sub(r'^```(?:json)?\s*|\s*```$', '', t)
    try:
        return json.loads(t)
    except ValueError:
        pass
    i, j = t.find('{'), t.rfind('}')
    if 0 <= i < j:
        try:
            return json.loads(t[i:j + 1])
        except ValueError:
            return None
    return None


def yes(v):
    return isinstance(v, str) and v.strip().lower().startswith('y') or v is True


# --------------------------------------------------------------------- S3


def find_evidence(text, ev):
    """(start, end) of a verbatim quote in text; whitespace runs may differ
    (the quote may join lines). None if absent."""
    ev = (ev or '').strip()
    if len(ev) < 8:
        return None
    i = text.find(ev)
    if i >= 0:
        return i, i + len(ev)
    toks = ev.split()
    if not toks:
        return None
    m = re.search(r'\s+'.join(map(re.escape, toks)), text)
    return (m.start(), m.end()) if m else None


def expand_span(book, doc_id, s, e):
    """A gold span: the evidence [s, e) of a document widened to its
    enclosing sentences, or to the whole block for code, math and tables."""
    text = book.doc_text(doc_id)
    for kind, bs, be in book.blocks(doc_id):
        if bs <= s < be:
            if kind in ('code', 'math', 'table'):
                return bs, max(be, min(e, len(text)))
            sent = book.d2l._r3_sentences(text[bs:be])
            ss = [(bs + a, bs + b) for a, b in sent
                  if bs + b > s and bs + a < e]
            if ss:
                return ss[0][0], max(ss[-1][1], e)
            return bs, be
    return s, e


def words(text):
    return re.findall(r'[a-z0-9]+', text.lower())


def strip_identifiers(text):
    return IDENT.sub(' | ', text)


def lexical_check(question, anchor_text):
    q = [w for w in words(strip_identifiers(question))]
    a = words(strip_identifiers(anchor_text))
    qs, as_ = set(q) - set(STOP), set(a) - set(STOP)
    jac = len(qs & as_) / max(1, len(qs | as_))
    # 5-grams of words within identifier-free stretches
    def ngrams(text):
        out = set()
        for part in strip_identifiers(text).split('|'):
            ws = words(part)
            out |= {tuple(ws[i:i + 5]) for i in range(len(ws) - 4)}
        return out
    shared = ngrams(question) & ngrams(anchor_text)
    return jac, sorted(' '.join(g) for g in shared)


def identifier_flag(item):
    return bool(IDENT.search(item['question']) or IDENT.search(
        item['reference']))


# ------------------------------------------------------------- API plumbing


class APIs:
    """One d2l.APIRecording per model; clients per stage cache (arm)."""

    def __init__(self, d2l, mode, folder):
        self.d2l, self.mode, self.folder = d2l, mode, folder
        os.makedirs(os.path.join(folder, 'local'), exist_ok=True)
        lp = os.path.join(folder, 'listing.json')
        if not os.path.exists(lp):
            if mode == 'replay':
                raise SystemExit(f'{lp} missing')
            rec = d2l.APIRecording(mode, os.path.dirname(folder), 1.0,
                                   {DRAFTER: DRAFTER_AT})
            rec.snapshot_listing([DRAFTER, CHECKER, VERIFIER], {},
                                 key=os.path.basename(folder))
        with open(lp) as f:
            self.listing = json.load(f)
        self.recs = {}
        for m, at in ((DRAFTER, DRAFTER_AT), (CHECKER, CHECKER_AT),
                      (VERIFIER, VERIFIER_AT)):
            r = d2l.APIRecording(mode, os.path.dirname(folder), CAP_USD[m],
                                 {m: at})
            r.listing = self.listing
            self.recs[m] = r
        self.clients = {}

    def client(self, arm):
        if arm not in self.clients:
            m = ARM_MODEL[arm]
            # the verifier's endpoint cannot switch reasoning off
            reasoning = ({'enabled': arm == 'calib-check'} if m == CHECKER
                         else {'effort': 'low'})
            self.clients[arm] = self.recs[m].hosted(
                m, os.path.join(self.folder, arm + '.jsonl'),
                share=SHARES[arm], reasoning=reasoning, timeout=300.0,
                max_retries=10)
        return self.clients[arm]

    def run(self, arm, requests, concurrency=24):
        """Results of requests (dicts with messages and parameters) in
        order; failures are exceptions. Raises on a replay miss or the cap."""
        if not requests:
            return []
        api = self.client(arm)
        log(f'{arm}: {len(requests)} requests ({self.mode})')
        out = asyncio.run(self.d2l.chat_batch(api, requests,
                                              concurrency=concurrency))
        for attempt in range(6 if self.mode == 'record' else 0):
            todo = [i for i, r in enumerate(out)
                    if isinstance(r, BaseException)]
            if not todo:
                break
            import time
            log(f'{arm}: {len(todo)} failed ({out[todo[0]]!r:.120}); '
                f'retrying in 60 s')
            time.sleep(60)
            again = asyncio.run(self.d2l.chat_batch(
                api, [requests[i] for i in todo], concurrency=concurrency))
            for i, r in zip(todo, again):
                out[i] = r
        bad = [r for r in out if isinstance(r, BaseException)]
        live = sum(1 for r in out if not isinstance(r, BaseException)
                   and not r.from_cache)
        log(f'{arm}: {len(out) - len(bad)} ok ({live} live), {len(bad)} '
            f'failed; recorded spend {self.spent(arm):.4f} USD')
        if bad:
            log(f'  first failure: {bad[0]!r:.300}')
        return out

    def spent(self, arm):
        path = os.path.join(self.folder, arm + '.calls.jsonl')
        return sum(c.get('cost') or 0.0 for c in self.d2l.api_calls(path)
                   if c.get('status') == 200)

    def recorded(self, arm):
        path = os.path.join(self.folder, arm + '.calls.jsonl')
        calls = self.d2l.api_calls(path)
        ok = [c for c in calls if c.get('status') == 200]
        return {'calls': len(ok), 'failed_attempts': len(calls) - len(ok),
                'usd': round(sum(c.get('cost') or 0.0 for c in ok), 6),
                'served': sorted({str(c.get('served_model')) for c in ok}),
                'providers': sorted({str(c.get('provider')) for c in ok})}


def served(r):
    return {'model': r.model, 'provider': r.provider}


def local_path(folder, name):
    return os.path.join(folder, 'local', name)


def read_jsonl(path):
    if not os.path.exists(path):
        return []
    with open(path, encoding='utf-8') as f:
        return [json.loads(l) for l in f if l.strip()]


def append_jsonl(path, rows):
    with open(path, 'a', encoding='utf-8') as f:
        for r in rows:
            f.write(dumps(r) + '\n')


# -------------------------------------------------------------- the build


class Build:
    def __init__(self, args):
        os.environ.setdefault('HF_HOME', '/home/smola/.cache/hf-lm')
        os.chdir(os.path.join(W, 'chapter_lm-encoders'))
        from d2l import torch as d2l
        self.d2l, self.args, self.mode = d2l, args, args.mode
        if args.cache_dir is None:  # replay: the registered artifact
            args.cache_dir = (os.path.join(W, 'data', 'lm-api', 'record',
                                           CACHE_KEY) if self.mode == 'record'
                              else d2l.api_unpack(CACHE_KEY))
        self.folder = os.path.abspath(args.cache_dir)
        self.api = APIs(d2l, self.mode, self.folder)
        log('loading the corpus')
        self.book = Book(d2l, os.path.join(W, 'data', 'lm'))
        self.stats = collections.defaultdict(collections.Counter)
        self.notes = {}

    # S1 ---------------------------------------------------------------
    def anchors(self, splits):
        out = []
        for s in splits:
            a, notes = anchors_for_split(self.book, s)
            self.notes[f'anchors_{s}'] = notes
            for x in a:
                self.stats['anchors'][f'{s}/{x["type"]}'] += 1
            out += a
        return out

    # S2 ---------------------------------------------------------------
    def draft(self, anchors, arm='s2-draft'):
        reqs = [dict(messages=draft_messages(self.book, a),
                     seed=item_seed(a['aid']), **(
                         MULTIHOP_DRAFT_PARAMS if a['type'] == 'multihop'
                         else DRAFT_PARAMS)) for a in anchors]
        res = self.api.run(arm, reqs, concurrency=self.args.concurrency)
        cands, failed = [], 0
        for a, r in zip(anchors, res):
            if isinstance(r, BaseException):
                failed += 1
                continue
            obj = parse_json_obj(r.text)
            got = (obj or {}).get('candidates') if isinstance(obj, dict) else None
            if not isinstance(got, list) or not got:
                self.stats['draft_parse_fail'][a['type']] += 1
                continue
            for k, cand in enumerate(got[:2]):
                if not isinstance(cand, dict):
                    continue
                ev = cand.get('evidence') or []
                ev = [ev] if isinstance(ev, str) else list(ev)
                kf = cand.get('key_facts') or []
                kf = [kf] if isinstance(kf, str) else list(kf)
                cands.append({
                    'cid': f'{a["aid"]}|{k}', 'anchor': a, 'k': k,
                    'split': a['split'], 'type': a['type'],
                    'question': str(cand.get('question') or '').strip(),
                    'reference': str(cand.get('reference') or '').strip(),
                    'key_facts': [str(x).strip() for x in kf if str(x).strip()][:3],
                    'evidence': [str(x) for x in ev],
                    'drafter': {'requested': f'{DRAFTER}@{DRAFTER_AT}',
                                'served': r.model, 'provider': r.provider}})
        self.stats['draft_failed_calls']['all'] += failed
        return cands, failed

    # S3 ---------------------------------------------------------------
    def filters_abce(self, cands):
        c, out = self.book.c, []
        for x in cands:
            t, a = x['type'], x['anchor']
            reason = None
            if not x['question'] or (t != 'unanswerable' and (
                    not x['reference'] or not x['key_facts'])):
                reason = 'empty_field'
            spans = []
            if reason is None and t != 'unanswerable':
                chunks = [c[i] for i in a['chunks']]
                evs = x['evidence']
                if t == 'multihop':
                    if len(evs) < 2:
                        reason = 'evidence'
                    else:
                        found = [find_evidence(chunks[0]['text'], evs[0]),
                                 find_evidence(chunks[1]['text'], evs[1])]
                        if None in found:  # swapped quotes
                            found = [find_evidence(chunks[0]['text'], evs[1]),
                                     find_evidence(chunks[1]['text'], evs[0])]
                        if None in found:
                            reason = 'evidence'
                        else:
                            spans = [(ch, f) for ch, f in zip(chunks, found)]
                else:
                    f = next((find_evidence(chunks[0]['text'], e)
                              for e in evs if find_evidence(
                                  chunks[0]['text'], e)), None)
                    if f is None:
                        reason = 'evidence'
                    else:
                        spans = [(chunks[0], f)]
                gold = []
                for ch, (s, e) in spans:
                    gs, ge = expand_span(self.book, ch['doc_id'],
                                         ch['char_start'] + s,
                                         ch['char_start'] + e)
                    gold.append({'doc_id': ch['doc_id'], 'char_start': gs,
                                 'char_end': ge})
                x['gold_spans'] = gold
            else:
                x['gold_spans'] = []
            if reason is None and CONTEXT_PHRASES.search(x['question']):
                reason = 'context_phrase'
            if reason is None:
                atext = ('\n'.join(c[i]['text'] for i in a['chunks'])
                         if t != 'unanswerable' else section_text(self.book, a))
                jac, shared = lexical_check(x['question'], atext)
                x['lexical'] = {'jaccard': round(jac, 4), 'shared_5grams':
                                shared[:3]}
                if jac > 0.5 or shared:
                    reason = 'lexical'
            if reason is None and t != 'unanswerable':
                for corp in (c, self.book.fixed[0], self.book.fixed[64]):
                    ids = {i for g in x['gold_spans'] for i in
                           corp.span_to_chunk_ids(g['doc_id'], g['char_start'],
                                                  g['char_end'], 0.5)}
                    if not ids:
                        reason = 'no_relevant_chunk'
                        break
            if reason is not None:
                self.stats[f's3_reject_{reason}'][t] += 1
            else:
                out.append(x)
        return out

    def near_duplicates(self, cands):
        """S3(d): drop a candidate whose question has cosine > 0.9 with an
        earlier kept one (order: dev, test, train, then drafting order)."""
        path = local_path(self.folder, 's3d.jsonl')
        ids = [x['cid'] for x in cands]
        digest = sha256('\n'.join(ids))
        rows = read_jsonl(path)
        if not rows or rows[0].get('candidates_sha256') != digest:
            if self.mode == 'replay':
                raise SystemExit('s3d.jsonl does not match the candidates')
            import numpy as np
            d2l = self.d2l
            model, tok = d2l.load_encoder('Qwen/Qwen3-Embedding-0.6B',
                                          padding_side='left')
            Z = d2l.embed_texts(model, tok, [x['question'] for x in cands],
                                pooling='last', batch_size=128,
                                max_len=128).numpy()
            del model
            d2l.lm_free()
            S = Z @ Z.T
            kept, rows = [], [{'candidates_sha256': digest,
                               'n': len(cands), 'model':
                               'Qwen/Qwen3-Embedding-0.6B'}]
            dropped = set()
            for j in range(len(cands)):
                if kept:
                    sims = S[j, kept]
                    near = np.nonzero(sims > NEAR_DUP_LOG)[0]
                    drop = bool((sims > NEAR_DUP).any())
                    for i in near:
                        rows.append({'a': ids[kept[i]], 'b': ids[j],
                                     'cos': round(float(sims[i]), 4),
                                     'decision': 'drop' if drop else 'keep'})
                    if drop:
                        dropped.add(ids[j])
                        continue
                kept.append(j)
            with open(path, 'w', encoding='utf-8') as f:
                for r in rows:
                    f.write(dumps(r) + '\n')
        dropped = {r['b'] for r in rows[1:] if r['decision'] == 'drop'}
        out = []
        for x in cands:
            if x['cid'] in dropped:
                self.stats['s3_reject_near_duplicate'][x['type']] += 1
            else:
                out.append(x)
        return out

    # S4 ---------------------------------------------------------------
    def anchor_window(self, x):
        """Text of the anchor(s) as the checker sees it."""
        c, a = self.book.c, x['anchor']
        if x['type'] == 'unanswerable':
            return section_text(self.book, a)
        parts = []
        for n, i in enumerate(a['chunks']):
            ch = c[i]
            name = (f'Passage {"AB"[n]} ' if len(a['chunks']) > 1 else
                    'Passage ')
            parts.append(f'{name}(book location: {c.header(ch)}):\n<<<\n'
                         f'{ch["text"]}\n>>>')
        return '\n\n'.join(parts)

    def check(self, cands, arm='s4-check'):
        C, reqs = P.CHECK_V1, []
        for x in cands:
            if x['type'] == 'unanswerable':
                user = C['user_unanswerable'].format(
                    text=self.anchor_window(x), question=x['question'])
            else:
                user = C['user'].format(
                    passages=self.anchor_window(x), question=x['question'],
                    reference=x['reference'], key_facts='\n'.join(
                        f'- {k}' for k in x['key_facts']))
            reqs.append(dict(messages=[
                {'role': 'system', 'content': C['system']},
                {'role': 'user', 'content': user}], **CHECK_PARAMS))
        res = self.api.run(arm, reqs, concurrency=self.args.concurrency)
        out = []
        for x, r in zip(cands, res):
            t = x['type']
            if isinstance(r, BaseException):
                self.stats['s4_failed_calls'][t] += 1
                continue
            v = parse_json_obj(r.text)
            if not isinstance(v, dict):
                self.stats['s4_reject_parse'][t] += 1
                continue
            x['checker'] = {'requested': f'{CHECKER}@{CHECKER_AT}',
                            'served': r.model, 'provider': r.provider}
            if t == 'unanswerable':
                checks = {'self_contained': yes(v.get('self_contained')),
                          'reference_supported': not yes(v.get('answered')),
                          'key_facts_complete': True, 'unique_answer': True,
                          'general_knowledge': yes(v.get('general_knowledge'))}
            else:
                checks = {k: yes(v.get(k)) for k in (
                    'self_contained', 'reference_supported',
                    'key_facts_complete', 'unique_answer',
                    'general_knowledge')}
                x['checker_answer'] = str(v.get('answer') or '')[:500]
            x['checks'] = checks
            fails = [k for k in ('self_contained', 'reference_supported',
                                 'key_facts_complete', 'unique_answer')
                     if not checks[k]]
            for k in fails:
                self.stats[f's4_fail_{k}'][t] += 1
            if fails:
                self.stats['s4_reject'][t] += 1
                x['s4_pass'] = False
            else:
                x['s4_pass'] = True
                out.append(x)
        return out

    # S5 ---------------------------------------------------------------
    def neighbour_lists(self, cands):
        """Hybrid top-10 over the corpus for each candidate (gold chunks
        removed for answerable items); computed once, in local/."""
        path = local_path(self.folder, 's5_neighbours.jsonl')
        have = {r['cid']: r['ids'] for r in read_jsonl(path)}
        todo = [x for x in cands if x['cid'] not in have]
        if todo:
            if self.mode == 'replay':
                raise SystemExit(f'{len(todo)} neighbour lists missing')
            d2l = self.d2l
            if not hasattr(self, 'retriever'):
                self.retriever = d2l.BookRetriever(self.book.c)
            rows = []
            for i in range(0, len(todo), 64):
                batch = todo[i:i + 64]
                hits = self.retriever.search_batch(
                    [x['question'] for x in batch], k=30, mode='hybrid')
                for x, hs in zip(batch, hits):
                    gold = self.gold_ids(x, include_alt=False)
                    ids = [h.chunk_id for h in hs if h.chunk_id not in gold]
                    rows.append({'cid': x['cid'], 'ids': ids[:10]})
            append_jsonl(path, rows)
            have.update({r['cid']: r['ids'] for r in rows})
        return have

    def gold_ids(self, x, include_alt=True, corpus=None):
        corp = corpus or self.book.c
        spans = x.get('gold_spans', []) + (x.get('alt_spans', [])
                                            if include_alt else [])
        return {i for s in spans for i in corp.span_to_chunk_ids(
            s['doc_id'], s['char_start'], s['char_end'], 0.5)}

    def passages_block(self, ids):
        c = self.book.c
        return '\n\n'.join(f'[{n}] {c.header(c[i])}\n{c[i]["text"]}'
                           for n, i in enumerate(ids, start=1))

    def neighbours(self, cands, lists, arm='s5-neighbours'):
        N, reqs = P.NEIGHBOR_V1, []
        for x in cands:
            ids = lists[x['cid']]
            if x['type'] == 'unanswerable':
                user = N['user_unanswerable'].format(
                    question=x['question'], passages=self.passages_block(ids))
            else:
                user = N['user'].format(
                    question=x['question'], reference=x['reference'],
                    passages=self.passages_block(ids))
            reqs.append(dict(messages=[
                {'role': 'system', 'content': N['system']},
                {'role': 'user', 'content': user}], **NEIGHBOR_PARAMS))
        res = self.api.run(arm, reqs, concurrency=self.args.concurrency)
        c = self.book.c
        for x, r in zip(cands, res):
            t, ids = x['type'], lists[x['cid']]
            x['s5_pass'] = False
            if isinstance(r, BaseException):
                self.stats['s5_failed_calls'][t] += 1
                continue
            v = parse_json_obj(r.text)
            if not isinstance(v, dict):
                self.stats['s5_reject_parse'][t] += 1
                continue
            nums = v.get('chunk')
            def listed(key):
                src = nums.get(key) if isinstance(nums, dict) else nums
                out = []
                for n in src if isinstance(src, list) else []:
                    try:
                        n = int(n)
                    except (TypeError, ValueError):
                        continue
                    if 1 <= n <= len(ids) and ids[n - 1] not in out:
                        out.append(ids[n - 1])
                return out
            if t == 'unanswerable':
                if yes(v.get('answers')):
                    self.stats['s5_reject_answered'][t] += 1
                    continue
                x['s5_pass'], x['alt_spans'] = True, []
                continue
            if yes(v.get('contradicts')):
                self.stats['s5_reject_contradicts'][t] += 1
                continue
            if yes(v.get('answers_better')):
                self.stats['s5_reject_answers_better'][t] += 1
                continue
            alts = listed('answers_equally') if yes(
                v.get('answers_equally')) else []
            x['alt_spans'] = [{'doc_id': c[i]['doc_id'],
                               'char_start': c[i]['char_start'],
                               'char_end': c[i]['char_end']} for i in alts]
            if alts:
                self.stats['s5_alt_items'][t] += 1
                self.stats['s5_alt_spans'][t] += len(alts)
            if any(c.split_of(c[i]['doc_id']) != x['split'] for i in alts):
                x['cross_split_alt'] = True
                self.stats['s5_cross_split_alt'][t] += 1
                continue
            x['s5_pass'] = True

    # S6 ---------------------------------------------------------------
    def closed_book(self, items):
        path = local_path(self.folder, 's6.jsonl')
        have = {r['cid']: r for r in read_jsonl(path)}
        todo = [x for x in items if x['cid'] not in have]
        if todo:
            if self.mode == 'replay':
                raise SystemExit(f'{len(todo)} closed-book results missing')
            d2l = self.d2l
            import time
            t0 = time.time()
            model, tok = d2l.load_hf(GEN)
            order = sorted(range(len(todo)), key=lambda i: len(
                todo[i]['question']))
            msgs = [d2l.rag_prompt(todo[i]['question'], None) for i in order]
            answers = [None] * len(todo)
            B = self.args.gen_batch
            for s in range(0, len(msgs), B * 8):
                gens = d2l.generate(model, tok, msgs[s:s + B * 8],
                                    max_new_tokens=256, chat=True,
                                    enable_thinking=False, batch_size=B)
                for j, g in zip(order[s:s + B * 8], gens):
                    answers[j] = g.text.strip()
                log(f'  closed-book {min(s + B * 8, len(msgs))}/{len(msgs)} '
                    f'({time.time() - t0:.0f} s)')
            del model
            d2l.lm_free()
            judge, jtok = d2l.load_hf(JUDGE)
            items_j = [self.item_view(x) for x in todo]
            res = [{'answer': a, 'cited_ids': []} for a in answers]
            verdicts = d2l.rag_judge(judge, jtok, items_j, res,
                                     [[] for _ in todo],
                                     batch_size=self.args.judge_batch)
            del judge
            d2l.lm_free()
            rows = [{'cid': x['cid'], 'answer': a, 'correct': v['correct']}
                    for x, a, v in zip(todo, answers, verdicts)]
            append_jsonl(path, rows)
            have.update({r['cid']: r for r in rows})
            log(f'S6: {len(todo)} items in {time.time() - t0:.0f} s')
        for x in items:
            x['closed_book_correct'] = bool(have[x['cid']]['correct'])
            x['closed_book_answer'] = have[x['cid']]['answer']
        if self.args.set_version == 2:
            self.closed_book_v2(items)

    def closed_book_v2(self, items):
        """Set version 2: closed_book_correct re-judged by JUDGE_V2
        (Qwen3.8-27B, thinking on, RAG_JUDGE_V1 prompt) behind an
        OpenAI-compatible endpoint (--judge-url: a local vLLM server);
        verdicts keyed by candidate id in --judge-v2-file (default
        <cache>/local/s6_judge_v2.jsonl), which replay reads."""
        path = self.args.judge_v2_file or local_path(self.folder,
                                                     's6_judge_v2.jsonl')
        have = {r['cid']: r for r in read_jsonl(path)}
        todo = [x for x in items if x['cid'] not in have]
        if todo:
            if self.mode == 'replay' or not self.args.judge_url:
                raise SystemExit(f'{len(todo)} closed-book v2 verdicts '
                                 f'missing ({path}); record with --judge-url')
            import queue
            import threading
            import time
            t0, lock = time.time(), threading.Lock()
            # chunks of 256 items with fixed boundaries over all items, each
            # with its own ChatAPI cache (a restart reuses every response);
            # two workers per server keep it busy while a chunk's slowest
            # judgments finish
            pending = {x['cid'] for x in todo}
            jobs = queue.Queue()
            for c in range(0, len(items), 256):
                part = [x for x in items[c:c + 256] if x['cid'] in pending]
                if part:
                    jobs.put((c // 256, part))
            done = [0]

            def work(url):
                while True:
                    try:
                        c, part = jobs.get_nowait()
                    except queue.Empty:
                        return
                    api = self.d2l.ChatAPI(
                        url, JUDGE_V2['served'], mode='record',
                        cache=os.path.join(os.path.dirname(path),
                                           f's6_judge_v2.c{c:03d}.jsonl'),
                        timeout=1800.0, max_retries=2,
                        default_params=JUDGE_V2['default_params'])
                    verdicts = self.d2l.rag_judge(
                        api, None, [self.item_view(x) for x in part],
                        [{'answer': x['closed_book_answer'],
                          'cited_ids': []} for x in part],
                        [[] for _ in part],
                        batch_size=self.args.judge_concurrency)
                    rows = [{'cid': x['cid'], 'correct': v['correct']}
                            for x, v in zip(part, verdicts)]
                    with lock:
                        append_jsonl(path, rows)
                        have.update({r['cid']: r for r in rows})
                        done[0] += len(part)
                        log(f'S6 v2: {done[0]}/{len(todo)} '
                            f'({time.time() - t0:.0f} s)')
            urls = self.args.judge_url.split(',')
            threads = [threading.Thread(target=work, args=(u,))
                       for u in urls for _ in range(2)]
            for t in threads:
                t.start()
            for t in threads:
                t.join()
            missing = [x for x in todo if x['cid'] not in have]
            if missing:
                raise SystemExit(f'{len(missing)} closed-book v2 verdicts '
                                 'missing after the run: run again')
        n_none = 0
        for x in items:
            v = have[x['cid']]['correct']
            n_none += v is None
            x['closed_book_correct_v1'] = x['closed_book_correct']
            x['closed_book_correct'] = bool(v)
        self.stats['s6_v2']['judged'] = len(items)
        self.stats['s6_v2']['no_verdict'] = n_none
        self.stats['s6_v2']['changed'] = sum(
            x['closed_book_correct'] != x['closed_book_correct_v1']
            for x in items)

    def item_view(self, x):
        return {'question': x['question'], 'reference': x['reference'] or
                'The book does not answer this.', 'key_facts': x['key_facts'],
                'answerable': x['type'] != 'unanswerable'}

    # S8 ---------------------------------------------------------------
    def gold_marked(self, x):
        """The anchor chunk(s) with each gold span between markers."""
        c, out = self.book.c, []
        for n, g in enumerate(x['gold_spans']):
            ch = c[x['anchor']['chunks'][n]] if n < len(
                x['anchor']['chunks']) else None
            text = self.book.doc_text(g['doc_id'])
            s = min(g['char_start'], ch['char_start']) if ch else g['char_start']
            e = max(g['char_end'], ch['char_end']) if ch else g['char_end']
            win = (text[s:g['char_start']] + '<<gold>>' +
                   text[g['char_start']:g['char_end']] + '<</gold>>' +
                   text[g['char_end']:e])
            head = c.header(ch) if ch else g['doc_id']
            out.append(f'Passage {n + 1} (book location: {head}):\n<<<\n'
                       f'{win}\n>>>')
        return '\n\n'.join(out)

    def verify(self, cands, lists, arm='s8-verify'):
        V, reqs = P.VERIFY_V1, []
        for x in cands:
            if x['type'] == 'unanswerable':
                user = V['user_unanswerable'].format(
                    question=x['question'], text=section_text(
                        self.book, x['anchor']),
                    passages=self.passages_block(lists[x['cid']]))
            else:
                alts = x.get('alt_spans') or []
                alt = ''
                if alts:
                    alt = ('\nOther passages judged to answer the question '
                           'as well as the gold evidence:\n' + '\n\n'.join(
                               f'<<<\n{self.book.doc_text(s["doc_id"])[s["char_start"]:s["char_end"]]}\n>>>'
                               for s in alts) + '\n')
                user = V['user'].format(
                    qtype=x['type'], question=x['question'],
                    reference=x['reference'], key_facts='\n'.join(
                        f'- {k}' for k in x['key_facts']),
                    passages=self.gold_marked(x), alternatives=alt)
            reqs.append(dict(messages=[
                {'role': 'system', 'content': V['system']},
                {'role': 'user', 'content': user}], **VERIFY_PARAMS))
        # the verifier's route allows 20 requests per minute on this account
        res = self.api.run(arm, reqs, concurrency=5)
        for x, r in zip(cands, res):
            if isinstance(r, BaseException):
                x['frontier'] = None
                self.stats['s8_failed_calls'][x['type']] += 1
                continue
            v = parse_json_obj(r.text)
            if not isinstance(v, dict):
                x['frontier'] = {'verdict': 'reject', 'rationale':
                                 'unparsable verifier reply',
                                 'served': r.model}
                self.stats['s8_parse_fail'][x['type']] += 1
                continue
            keys = (('self_contained', 'answered_by_book', 'general_knowledge')
                    if x['type'] == 'unanswerable' else (
                        'self_contained', 'reference_supported',
                        'key_facts_complete', 'unique_answer',
                        'general_knowledge'))
            f = {k: 'yes' if yes(v.get(k)) else 'no' for k in keys}
            ok = (f['self_contained'] == 'yes' and (
                f['answered_by_book'] == 'no' if x['type'] == 'unanswerable'
                else all(f[k] == 'yes' for k in keys[1:4])))
            f['verdict'] = 'accept' if ok else 'reject'
            f['verdict_reply'] = str(v.get('verdict') or '')
            f['rationale'] = str(v.get('rationale') or '')[:600]
            f['served'], f['provider'] = r.model, r.provider
            x['frontier'] = f

    # S7 + S8 ------------------------------------------------------------
    def select(self, split, cands_by_anchor, anchors, lists):
        """Quotas for one split: per type, the first candidate of each
        anchor that passed every check, no gold-span overlap; rejected by
        the verifier -> the next reserve of the type."""
        chosen, used_spans = {t: [] for t in TYPES}, []
        def overlaps(x):
            for g in x['gold_spans']:
                for u in used_spans:
                    if (u['doc_id'] == g['doc_id'] and u['char_start'] <
                            g['char_end'] and g['char_start'] < u['char_end']):
                        return True
            return False
        reserves = {t: [] for t in TYPES}
        for a in anchors:
            if a['split'] != split:
                continue
            ok = [x for x in cands_by_anchor.get(a['aid'], [])
                  if x.get('s4_pass') and x.get('s5_pass')]
            if ok:
                reserves[a['type']].append(ok[0])
        verified = {}
        rounds = 0
        while True:
            rounds += 1
            need = []
            for t in TYPES:
                quota = QUOTA[t]
                while len(chosen[t]) < quota and reserves[t]:
                    x = reserves[t].pop(0)
                    if overlaps(x):
                        self.stats['s7_skip_overlap'][t] += 1
                        continue
                    chosen[t].append(x)
                    used_spans.extend(x['gold_spans'])
            # shortfall: fill with single items
            short = {t: QUOTA[t] - len(chosen[t]) for t in TYPES
                     if t != 'single' and len(chosen[t]) < QUOTA[t]}
            fill_n = sum(short.values())
            fills = [x for x in chosen['single'] if 'type_shortfall_fill' in
                     x.get('flags', [])]
            while len(fills) < fill_n and reserves['single']:
                x = reserves['single'].pop(0)
                if overlaps(x):
                    self.stats['s7_skip_overlap']['single'] += 1
                    continue
                x.setdefault('flags', []).append('type_shortfall_fill')
                chosen['single'].append(x)
                fills.append(x)
                used_spans.extend(x['gold_spans'])
            need = [x for t in TYPES for x in chosen[t]
                    if x['cid'] not in verified]
            if not need or self.args.no_verify:
                break
            self.verify(need, lists)
            if any(x.get('frontier') is None for x in need):
                raise SystemExit('verifier calls failed: run again (the '
                                 'answered ones are cached)')
            changed = False
            for x in need:
                verified[x['cid']] = x
                f = x.get('frontier')
                if f is None or f['verdict'] != 'accept':
                    self.stats['s8_reject'][x['type']] += 1
                    for t in TYPES:
                        if x in chosen[t]:
                            chosen[t].remove(x)
                    used_spans[:] = [g for t in TYPES for y in chosen[t]
                                     for g in y['gold_spans']]
                    changed = True
                else:
                    self.stats['s8_accept'][x['type']] += 1
            if not changed:
                break
        self.notes[f'select_{split}'] = {
            'rounds': rounds, 'shortfall': {t: QUOTA[t] - len(chosen[t])
                                            for t in TYPES if t != 'single'},
            'filled_with_single': sum('type_shortfall_fill' in x.get(
                'flags', []) for x in chosen['single']),
            'reserves_left': {t: len(v) for t, v in reserves.items()}}
        return [x for t in TYPES for x in chosen[t]], list(verified.values())

    # S9 ---------------------------------------------------------------
    def snapshot(self, model_id):
        """The dated snapshot (canonical slug) of a served model id in the
        listing snapshot; the id itself if the listing does not name it."""
        row = self.api.listing['models'].get(model_id) or {}
        return row.get('canonical_slug') or model_id

    def validate(self, out):
        """The split rules: train items anchored and spanned in train
        documents; dev/test spans and alternatives in documents of the item's
        split; at most one dev/test item per anchor chunk; quotas met."""
        c = self.book.c
        for s, rows in out.items():
            seen = set()
            for r in rows:
                docs = ([c[i]['doc_id'] for i in r['anchor_ids']] +
                        [g['doc_id'] for g in r['gold_spans'] +
                         r['alt_spans']])
                assert all(c.split_of(d) == s for d in docs), r['qid']
                if s != 'train':
                    assert not seen & set(r['anchor_ids']) or r[
                        'type'] == 'unanswerable', r['qid']
                    seen |= set(r['anchor_ids'])
                    assert r['verified_by'], r['qid']
            if s != 'train':
                n = collections.Counter(r['type'] for r in rows)
                assert all(n[t] == QUOTA[t] for t in TYPES), (s, n)

    def record(self, x, qid):
        flags = list(x.get('flags', []))
        if x['checks'].get('general_knowledge'):
            flags.append('general_knowledge')
        if identifier_flag(x):
            flags.append('identifier')
        f = x.get('frontier')
        checks = dict(x['checks'])
        checks['general_knowledge'] = 'yes' if checks[
            'general_knowledge'] else 'no'
        for k in ('self_contained', 'reference_supported',
                  'key_facts_complete', 'unique_answer'):
            checks[k] = 'yes' if checks[k] else 'no'
        checks['lexical_ok'] = True
        checks['neighbours_ok'] = (True if x['split'] in ('dev', 'test')
                                   else None)
        checks['second_check'] = None
        if f is not None and x['split'] in ('dev', 'test'):
            checks['frontier'] = {k: v for k, v in f.items()
                                  if k not in ('served', 'provider')}
        un = x['type'] == 'unanswerable'
        return {
            'qid': qid, 'split': x['split'], 'type': x['type'],
            'question': x['question'],
            'reference': ('The book does not answer this.' if un
                          else x['reference']),
            'key_facts': [] if un else x['key_facts'],
            'answerable': not un,
            'gold_spans': [] if un else x['gold_spans'],
            'alt_spans': (x.get('alt_spans') or []) if x['split'] in (
                'dev', 'test') and not un else [],
            'corpus_version': CORPUS_VERSION,
            'drafter': x['drafter'], 'checker': x['checker'],
            'verified_by': (self.snapshot(f['served']) if f is not None and
                            x['split'] in ('dev', 'test') else None),
            'checks': checks,
            'closed_book_correct': x['closed_book_correct'],
            **({'closed_book_correct_v1': x['closed_book_correct_v1']}
               if 'closed_book_correct_v1' in x else {}),
            'flags': sorted(set(flags)),
            'anchor_ids': list(x['anchor']['chunks'])}


def kappa(a, b):
    """Cohen's kappa of two equal-length boolean lists."""
    n = len(a)
    if n == 0:
        return None
    po = sum(x == y for x, y in zip(a, b)) / n
    pa, pb = sum(a) / n, sum(b) / n
    pe = pa * pb + (1 - pa) * (1 - pb)
    return None if pe == 1 else (po - pe) / (1 - pe)


def write_zip(path, members):
    """A deterministic zip: sorted members, fixed dates, deflated."""
    with zipfile.ZipFile(path, 'w', zipfile.ZIP_DEFLATED) as z:
        for name in sorted(members):
            info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o644 << 16
            z.writestr(info, members[name])


README = """# lm-book-qa-v1: the question set over the book corpus

Questions about *Dive into Deep Learning*, built over the book corpus
lm-book-r3-v1 (corpus version r3.1) for chapter 19 ("Encoders, Embeddings and
Retrieval"), section "Retrieval-Augmented Generation". Licence: CC BY-SA 4.0,
as the book's text.

Files: qa_train.jsonl, qa_dev.jsonl, qa_test.jsonl (one item per line; the
schema is in the chapter and in manifest.json), manifest.json (models,
prompts, seeds, counts per stage, rejections, spend).

Drafted by DeepSeek-V4.1-Flash-1017; filtered by automatic checks and by
Qwen3.8-27B; every dev and test item verified by a frontier model
({verifier}). Not verified by people. Splits are by document: no dev or test
passage sources a train question. Built by tools/build_book_qa.py; its API
responses and local-stage outputs are the artifact lm-book-qa-api-cache-v1,
from which --mode replay rebuilds this set.
"""


README_V2 = """
## Version 2 (2026-11-14)

Items, splits, references and gold spans are those of lm-book-qa-v1. Only
the closed-book flag changed: `closed_book_correct` is now the verdict of the
judge chosen for section 19.5 against frontier-model labels
(lm-rag-judge-labels-v1): Qwen3.8-27B (FP8 weights, thinking on, prompt
RAG_JUDGE_V1). The v1 judge (Qwen3.5-4B, thinking off) marked vague answers
correct; its flag stays in `closed_book_correct_v1`. The judge's verdicts per
candidate are in closed_book_judge_v2.jsonl: `--mode replay --set-version 2
--judge-v2-file closed_book_judge_v2.jsonl` with the cache
lm-book-qa-api-cache-v1 rebuilds this set.
"""


def build(args):
    b = Build(args)
    d2l = b.d2l
    if args.smoke:
        return smoke(b)
    splits = ('dev', 'test', 'train')
    anchors = [a for a in b.anchors(splits) if a['split'] in args.splits]
    log('S1 anchors:', dict(sorted(b.stats['anchors'].items())))
    order = {s: i for i, s in enumerate(splits)}
    if args.stop_after == 's1':
        return
    # S2 (dev and test first, then train)
    cands, failed = b.draft(sorted(anchors, key=lambda a: order[a['split']]))
    if failed and args.mode == 'record':
        raise SystemExit(f'{failed} draft calls failed: run again to retry')
    log(f'S2: {len(cands)} candidates')
    # S3
    kept = b.filters_abce(cands)
    kept = b.near_duplicates(kept)
    log(f'S3: {len(kept)} candidates pass')
    # S4
    kept = b.check(kept)
    log(f'S4: {len(kept)} candidates pass')
    if args.stop_after == 's4':
        report_stats(b)
        return
    by_anchor = collections.defaultdict(list)
    for x in cands:
        by_anchor[x['anchor']['aid']].append(x)
    # S5: first S4-passing candidate per dev/test anchor, then the second
    # of the anchors whose first was rejected
    dt = [x for x in kept if x['split'] in ('dev', 'test')]
    lists = b.neighbour_lists(dt)
    first, second = [], []
    for aid, xs in by_anchor.items():
        ok = [x for x in xs if x.get('s4_pass') and x['split'] != 'train']
        if ok:
            first.append(ok[0])
            second += ok[1:2]
    b.neighbours(first, lists)
    retry = [x for x in second if not any(
        y.get('s5_pass') for y in by_anchor[x['anchor']['aid']])]
    b.neighbours(retry, lists)
    if args.stop_after == 's5':
        report_stats(b)
        return
    # S7 + S8 per split
    final, verified_all = [], []
    for s in ('dev', 'test'):
        chosen, verified = b.select(s, by_anchor, anchors, lists)
        final += chosen
        verified_all += verified
    if args.stop_after == 's7':
        log(json.dumps({k: v for k, v in b.notes.items()
                        if k.startswith('select')}, indent=1))
        report_stats(b)
        return
    # kappa sample: S4-rejected dev/test candidates, seen by the verifier too
    rejected = [x for x in cands if x['split'] in ('dev', 'test')
                and x.get('s4_pass') is False and x['type'] != 'unanswerable']
    rng = random.Random(f'kappa|{SEED}')
    sample = sorted(rejected, key=lambda x: x['cid'])
    rng.shuffle(sample)
    sample = sample[:KAPPA_SAMPLE]
    if not args.no_verify and sample:
        b.verify(sample, lists)
    # train: every candidate that passed S3 and S4
    train = [x for x in kept if x['split'] == 'train']
    # S6 on the final items
    b.closed_book(final + train)
    # S9
    out = {}
    type_rank = {t: i for i, t in enumerate(TYPES)}
    for s, xs in (('train', train), ('dev', [x for x in final if
                                             x['split'] == 'dev']),
                  ('test', [x for x in final if x['split'] == 'test'])):
        xs = sorted(xs, key=lambda x: (type_rank[x['type']], x['cid']))
        out[s] = [b.record(x, f'{s}-{n:04d}') for n, x in enumerate(xs)]
    b.validate(out)
    manifest = make_manifest(b, out, verified_all, sample, cands)
    members = {f'qa_{s}.jsonl': ''.join(dumps(r) + '\n' for r in out[s])
               for s in out}
    members['manifest.json'] = json.dumps(manifest, indent=1,
                                          ensure_ascii=False) + '\n'
    members['README.md'] = README.format(verifier=VERIFIER).replace(
        'lm-book-qa-v1', KEY) + (README_V2 if args.set_version == 2 else '')
    if args.set_version == 2:  # the verdicts replay reads
        with open(args.judge_v2_file or local_path(
                b.folder, 's6_judge_v2.jsonl'), encoding='utf-8') as f:
            members['closed_book_judge_v2.jsonl'] = f.read()
    os.makedirs(args.out, exist_ok=True)
    for name, text in members.items():
        with open(os.path.join(args.out, name), 'w', encoding='utf-8') as f:
            f.write(text)
    write_zip(os.path.join(args.out, f'{KEY}.zip'),
              {k: v.encode('utf-8') for k, v in members.items()})
    report_stats(b)
    log('counts:', {s: dict(collections.Counter(r['type'] for r in rs))
                    for s, rs in out.items()})
    if args.review:
        write_review(b, out, verified_all, sample, args.review)
    log('wrote', os.path.join(args.out, f'{KEY}.zip'))


def make_manifest(b, out, verified, sample, cands):
    d2l = b.d2l
    c = b.book.c
    fr = [x for x in verified if x.get('frontier')]
    # agreement of verifier and checker on the per-check verdicts
    kap = {}
    pool = fr + [x for x in sample if x.get('frontier')]
    for k in ('self_contained', 'reference_supported', 'key_facts_complete',
              'unique_answer', 'general_knowledge'):
        rows = [(bool(x['checks'][k]), x['frontier'].get(k) == 'yes')
                for x in pool if k in x['frontier'] and k in x['checks']]
        kap[k] = {'n': len(rows), 'agree': (sum(a == v for a, v in rows) /
                                            len(rows)) if rows else None,
                  'kappa': kappa([a for a, _ in rows], [v for _, v in rows])}
    over = [(all(x['checks'][k] for k in ('self_contained',
                                           'reference_supported',
                                           'key_facts_complete',
                                           'unique_answer')),
             x['frontier']['verdict'] == 'accept')
            for x in pool if x['type'] != 'unanswerable']
    kap['overall_pass'] = {'n': len(over), 'agree': (sum(
        a == v for a, v in over) / len(over)) if over else None,
        'kappa': kappa([a for a, _ in over], [v for _, v in over]),
        'note': ('pool = verified dev/test items (all passed the checker) '
                 f'plus {len(sample)} checker-rejected dev/test candidates '
                 'sampled with seed 0')}
    spend = {arm: b.api.recorded(arm) for arm in SHARES
             if os.path.exists(os.path.join(b.folder, arm + '.calls.jsonl'))}
    per_model = collections.defaultdict(float)
    for arm, r in spend.items():
        per_model[ARM_MODEL[arm]] += r['usd']
    stats = {k: dict(sorted(v.items())) for k, v in sorted(b.stats.items())}
    served = collections.defaultdict(collections.Counter)
    for s, rs in out.items():
        for r in rs:
            served['drafter'][f'{r["drafter"]["served"]}@'
                              f'{r["drafter"]["provider"]}'] += 1
            served['checker'][f'{r["checker"]["served"]}@'
                              f'{r["checker"]["provider"]}'] += 1
            if r['verified_by']:
                served['verifier'][r['verified_by']] += 1
    prompts = {k: sha256(json.dumps(v, sort_keys=True))
               for k, v in P.PROMPTS.items()}
    prompts['RAG_JUDGE_V1'] = sha256(json.dumps(d2l.RAG_JUDGE_V1,
                                                sort_keys=True))
    return {
        'name': KEY, 'corpus_version': CORPUS_VERSION,
        'book_commit': c.manifest['book_commit'],
        'created': datetime.datetime.now(datetime.timezone.utc).strftime(
            '%Y-%m-%dT%H:%M:%SZ'),
        'listing_retrieved': b.api.listing.get('retrieved'),
        'snapshots': {m: b.snapshot(m) for m in (DRAFTER, CHECKER, VERIFIER)},
        'models': {
            'drafter': {'requested': DRAFTER, 'provider': DRAFTER_AT,
                        'params': DRAFT_PARAMS,
                        'params_multihop': MULTIHOP_DRAFT_PARAMS,
                        'reasoning': {'effort': 'low'}},
            'checker': {'requested': CHECKER, 'provider': CHECKER_AT,
                        'params': CHECK_PARAMS},
            'verifier': {'requested': VERIFIER, 'provider': VERIFIER_AT,
                         'params': VERIFY_PARAMS,
                         'reasoning': {'effort': 'low'}},
            'closed_book_generator': {'id': GEN, 'revision':
                                      d2l.HF_REVISIONS[GEN]},
            'judge': {'id': JUDGE, 'revision': d2l.HF_REVISIONS[JUDGE],
                      'prompt': 'RAG_JUDGE_V1'},
            'near_duplicates': {'id': 'Qwen/Qwen3-Embedding-0.6B',
                                'revision': d2l.HF_ENCODER_REVISIONS[
                                    'Qwen/Qwen3-Embedding-0.6B']},
            'neighbours': 'd2l.BookRetriever hybrid (BM25 + Qwen3-Embedding-'
                          '0.6B, RRF k=60), top 10 without gold chunks',
            'served_counts': {k: dict(v) for k, v in served.items()}},
        'prompts_sha256': prompts, 'seed': SEED, 'quota': QUOTA,
        'factor': FACTOR, 'train_multihop': TRAIN_MULTIHOP,
        'train_unanswerable': TRAIN_UNANSWERABLE,
        'filters': {'min_anchor_tokens': MIN_TOKENS, 'jaccard_max': 0.5,
                    'shared_5gram': 'none outside identifiers',
                    'near_duplicate_cos': NEAR_DUP,
                    'relevant_chunk_min_overlap': 0.5},
        'counts': {s: dict(collections.Counter(r['type'] for r in rs))
                   for s, rs in out.items()},
        'stats': stats, 'notes': b.notes,
        'verifier_vs_checker': kap,
        'spend_usd': {'per_stage': spend, 'per_model': {
            m: round(v, 6) for m, v in per_model.items()},
            'caps': CAP_USD, 'shares': SHARES},
        'human_verified': False,
        **({'version': 2, 'previous': 'lm-book-qa-v1', 'closed_book_judge': {
            k: v for k, v in JUDGE_V2.items() if k != 'default_params'},
            'changes': 'closed_book_correct re-judged by JUDGE_V2; the v1 '
                       'flag is kept as closed_book_correct_v1'}
           if b.args.set_version == 2 else {})}


def report_stats(b):
    for k, v in sorted(b.stats.items()):
        log(f'  {k}: {dict(sorted(v.items()))}')
    for arm in SHARES:
        if os.path.exists(os.path.join(b.folder, arm + '.calls.jsonl')):
            log(f'  spend {arm}: {b.api.recorded(arm)}')


def write_review(b, out, verified, sample, path):
    """S10: the author's review sheet for the dev and test items."""
    c = b.book.c
    L = ['# lm-book-qa-v1: review sheet (dev and test)', '',
         'Every dev and test item, grouped by document. Each passed the '
         'automatic filters, the Qwen3.8-27B checklist and the neighbour '
         f'check, and was accepted by the frontier verifier ({VERIFIER}). '
         'Nothing waits on this sheet; it is for an optional spot check '
         '(about 1-2 minutes per item).', '']
    counts = {s: collections.Counter(r['type'] for r in out[s])
              for s in out}
    L.append('| split | ' + ' | '.join(TYPES) + ' | total |')
    L.append('|---' * (len(TYPES) + 2) + '|')
    for s in ('train', 'dev', 'test'):
        L.append(f'| {s} | ' + ' | '.join(str(counts[s][t]) for t in TYPES)
                 + f' | {sum(counts[s].values())} |')
    L.append('')
    rej = [x for x in verified if x.get('frontier') and
           x['frontier']['verdict'] != 'accept']
    L += [f'## Items the frontier verifier rejected ({len(rej)}; replaced '
          'from reserves)', '']
    for x in rej:
        L.append(f'- `{x["type"]}` {x["question"]}  \n  reference: '
                 f'{x["reference"]}  \n  verifier: '
                 f'{x["frontier"]["rationale"]}')
    L.append('')
    for s in ('dev', 'test'):
        L += [f'## {s}', '']
        by_doc = collections.defaultdict(list)
        for r in out[s]:
            d = (r['gold_spans'][0]['doc_id'] if r['gold_spans'] else
                 c[r['anchor_ids'][0]]['doc_id'])
            by_doc[d].append(r)
        for d in sorted(by_doc):
            L += [f'### {d}', '']
            for r in by_doc[d]:
                L.append(f'**{r["qid"]}** `{r["type"]}`'
                         + (f' flags: {", ".join(r["flags"])}' if r['flags']
                            else '')
                         + f'; closed-book correct: {r["closed_book_correct"]}')
                L.append(f'- Q: {r["question"]}')
                L.append(f'- A: {r["reference"]}')
                if r['key_facts']:
                    L.append('- key facts: ' + '; '.join(r['key_facts']))
                for g in r['gold_spans']:
                    t = b.book.doc_text(g['doc_id'])[g['char_start']:
                                                     g['char_end']]
                    t = re.sub(r'\s+', ' ', t)
                    L.append(f'- gold ({g["doc_id"]}): '
                             f'"{t[:400]}{"..." if len(t) > 400 else ""}"')
                if r['alt_spans']:
                    L.append(f'- alternative passages: {len(r["alt_spans"])}')
                fv = r['checks'].get('frontier') or {}
                L.append(f'- verifier: {fv.get("rationale", "")}')
                L.append('')
    with open(path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(L) + '\n')
    log('review sheet:', path)


def smoke(b):
    """Ten calls per model into the smoke caches: served ids, providers,
    JSON parse rate, cost per call; S3 and S4 on the drafts."""
    anchors = []
    for s in ('dev',):
        a, _ = anchors_for_split(b.book, s)
        by_t = collections.defaultdict(list)
        for x in a:
            by_t[x['type']].append(x)
        pick = (['single'] * 3 + ['explain', 'explain', 'numeric', 'code',
                                   'code', 'multihop', 'unanswerable'])
        for t in pick:
            anchors.append(by_t[t].pop(0))
    cands, _ = b.draft(anchors, arm='smoke-draft')
    res_d = b.api.clients['smoke-draft']
    log('drafts parsed:', len(cands), 'of', 2 * len(anchors))
    kept = b.filters_abce(cands)
    log('S3 (a-c, e) pass:', len(kept), 'of', len(cands))
    kept = kept[:10]
    passed = b.check(kept, arm='smoke-check')
    log('S4 pass:', len(passed), 'of', len(kept), collections.Counter(
        x['type'] for x in passed), 'of', collections.Counter(
        x['type'] for x in kept))
    lists = {x['cid']: [] for x in kept}
    ver = [x for x in kept if x['type'] != 'unanswerable'][:10]
    b.verify(ver, lists, arm='smoke-verify')
    for x in ver:
        log('  verify', x['type'], (x.get('frontier') or {}).get('verdict'),
            '|', x['question'][:100], '|',
            (x.get('frontier') or {}).get('rationale', '')[:160])
    for arm in ('smoke-draft', 'smoke-check', 'smoke-verify'):
        r = b.api.recorded(arm)
        log(f'{arm}: {r}; per call {r["usd"] / max(1, r["calls"]):.5f} USD')
    report_stats(b)
    for x in cands[:6]:
        log(json.dumps({k: x[k] for k in ('type', 'question', 'reference',
                                           'key_facts', 'evidence')},
                       ensure_ascii=False)[:700])


CALIB_N, CALIB_KEY = 100, 'lm-rag-judge-labels-v1'
# Labellers of the judge-calibration pairs (2026-11-14 judge choice): the
# frontier verifier of S8 first and a second frontier model of another
# family; Qwen3.8-27B is a judge candidate, so it is not a labeller.
SECOND, SECOND_AT = 'google/gemini-4-argon', 'google-ai-studio/flex'
LABELLERS = {  # arm: (model, provider, request parameters, reasoning)
    'label-frontier': (VERIFIER, VERIFIER_AT, dict(
        max_tokens=1500, response_format={'type': 'json_object'}),
        {'effort': 'low'}),
    'label-second': (SECOND, SECOND_AT, dict(
        max_tokens=4000, temperature=0, seed=0,
        response_format={'type': 'json_object'}), {'effort': 'medium'})}
CALIB_CAP_USD = {VERIFIER: 4.5, SECOND: 2.5, CHECKER: 1.0}
CALIB_SHARES = {'label-frontier': 1.0, 'smoke-second': 0.2,
                'label-second': 0.8}
CALIB_README = """# lm-rag-judge-labels-v1: labels for choosing and validating the RAG judge

200 (question, answer) pairs of section 19.5 ("Retrieval-Augmented
Generation"): `base` answers of Qwen3-4B (thinking off, greedy) to dev items
of lm-book-qa-v1, from the hybrid BookRetriever (sections, K=5, budget 3,000
tokens, the fixed prompt). Two sets of 100: `calibration` (drawn uniformly
from the 240 dev generations sorted by qid, random.Random('calib|0')) and
`selection` (100 of the other 140 dev items with the calibration's count per
question type, random.Random('select|0')). The judge was chosen on
`selection`; its agreement on `calibration` is the book's claim C0.

Each pair is labelled `correct` and `faithful` (prompt CALIB_V1 of
tools/lm/book_qa_prompts.py) by two frontier models of different families:
{frontier} (first) and {second} (second). Not labelled by people. `correct`
and `faithful` hold the consensus where both labellers agree (label_source
"both"), else null ("disagree" or "missing"). Licence CC BY-SA 4.0.

Files: pairs.jsonl (the 200 generations), dev_judge_labels.jsonl (labels),
manifest.json (models, served ids, parameters, kappas, spend), listing.json
(OpenRouter listing snapshot), api/<arm>.jsonl and api/<arm>.calls.jsonl (every
recorded response; `python tools/build_book_qa.py --mode replay --calibrate
pairs.jsonl --cache-dir <unpacked zip>` rebuilds the labels from them),
including the recorded responses of the judge candidates (judge-*: Qwen3.8-27B
on OpenRouter, thinking off and on; judge-local-*: the same model's FP8
weights served by vLLM on two RTX 4090, and Qwen3.5-9B and Qwen3.5-4B, thinking
on). candidates/<name>.jsonl holds every candidate's verdicts on the 200 pairs
and candidates/table.json the comparison (agreement, Cohen's kappa with
bootstrap intervals, false-correct rate, flip rate, time and cost per 1,000
judgments); the chosen judge is Qwen3.8-27B with thinking on.
"""


def calib_pairs(gens, items):
    """(calibration, selection) generations: see CALIB_README."""
    gens = sorted((r for r in gens if r.get('config') == 'base' and
                   r.get('split', 'dev') == 'dev'), key=lambda r: r['qid'])
    if len(gens) < 2 * CALIB_N:
        raise SystemExit(f'{len(gens)} base dev generations, need '
                         f'{2 * CALIB_N}')
    pick = set(random.Random(f'calib|{SEED}').sample(range(len(gens)),
                                                     CALIB_N))
    calib = [g for i, g in enumerate(gens) if i in pick]
    rest = [g for i, g in enumerate(gens) if i not in pick]
    want = collections.Counter(items[g['qid']]['type'] for g in calib)
    rng, sel = random.Random(f'select|{SEED}'), []
    for t in sorted(want):
        sel += rng.sample([g for g in rest if items[g['qid']]['type'] == t],
                          want[t])
    return calib, sorted(sel, key=lambda r: r['qid'])


def calib_messages(corpus, item, gen):
    """CALIB_V1 messages for one (item, generation) pair."""
    C, c = P.CALIB_V1, corpus
    passages = '\n\n'.join(
        f'[{n}] {c.header(c[i])}\n{c[i]["text"]}'
        for n, i in enumerate(gen['context_ids'], start=1)) or '(none)'
    user = C['user'].format(
        question=item['question'], reference=item['reference'],
        key_facts='\n'.join(f'- {k}' for k in item['key_facts']) or
        '- (none)', passages=passages, answer=gen['answer'])
    return [{'role': 'system', 'content': C['system']},
            {'role': 'user', 'content': user}]


def calib_kappas(rows):
    """Labeller agreement and kappa per label, overall and per set."""
    out = {}
    for s in ('all', 'calibration', 'selection'):
        rs = [r for r in rows if s == 'all' or r['set'] == s]
        for k in ('correct', 'faithful'):
            both = [(r[f'{k}_frontier'], r[f'{k}_second']) for r in rs
                    if None not in (r[f'{k}_frontier'], r[f'{k}_second'])]
            out[f'{s}/{k}'] = {
                'n': len(both), 'agree': round(sum(a == v for a, v in both)
                                               / max(1, len(both)), 4),
                'kappa': kappa([a for a, _ in both], [v for _, v in both]),
                'consensus': sum(r[f'{k}_label_source'] == 'both'
                                 for r in rs)}
    return out


def calibrate(args):
    """The judge-calibration labels lm-rag-judge-labels-v1 (19.5): 200
    `base` dev generations (calibration and selection sets, see
    CALIB_README) labelled `correct` and `faithful` by two frontier models
    through d2l.ChatAPI (recorded; --mode replay rebuilds the labels from
    the caches); consensus where they agree; Cohen's kappa between them.
    Writes the zip lm-rag-judge-labels-v1.zip to --out."""
    folder = os.path.abspath(args.cache_dir or os.path.join(
        W, 'data', 'lm-api', 'record', CALIB_KEY))
    gens_path, out_dir = (os.path.abspath(args.calibrate),
                          os.path.abspath(args.out))
    os.environ.setdefault('HF_HOME', '/home/smola/.cache/hf-lm')
    os.chdir(os.path.join(W, 'chapter_lm-encoders'))
    from d2l import torch as d2l
    os.makedirs(os.path.join(folder, 'api'), exist_ok=True)
    lp = os.path.join(folder, 'listing.json')
    if not os.path.exists(lp):
        if args.mode == 'replay':
            raise SystemExit(f'{lp} missing')
        rec = d2l.APIRecording('record', os.path.dirname(folder), 1.0, {})
        rec.snapshot_listing([VERIFIER, SECOND, CHECKER], {},
                             key=os.path.basename(folder))
    with open(lp) as f:
        listing = json.load(f)
    recs = {}
    for m, at in ((VERIFIER, VERIFIER_AT), (SECOND, SECOND_AT),
                  (CHECKER, CHECKER_AT)):
        recs[m] = d2l.APIRecording(args.mode, folder, CALIB_CAP_USD[m],
                                   {m: at})
        recs[m].listing = listing

    def client(arm, m, reasoning):
        return recs[m].hosted(m, os.path.join(folder, 'api', arm + '.jsonl'),
                              share=CALIB_SHARES[arm], reasoning=reasoning,
                              timeout=300.0, max_retries=10)

    def run(api, reqs, arm):
        import asyncio
        import time
        out = asyncio.run(d2l.chat_batch(api, reqs, concurrency=5))
        for _ in range(6 if args.mode == 'record' else 0):
            todo = [i for i, r in enumerate(out)
                    if isinstance(r, BaseException)]
            if not todo:
                break
            log(f'{arm}: {len(todo)} failed ({out[todo[0]]!r:.120}); '
                f'retrying in 60 s')
            time.sleep(60)
            again = asyncio.run(d2l.chat_batch(
                api, [reqs[i] for i in todo], concurrency=5))
            for i, r in zip(todo, again):
                out[i] = r
        calls = d2l.api_calls(api.calls_path)
        log(f'{arm}: {sum(not isinstance(r, BaseException) for r in out)}/'
            f'{len(out)} ok; recorded spend '
            f'{sum(c.get("cost") or 0 for c in calls):.4f} USD')
        return out

    corpus = d2l.book_corpus(root=os.path.join(W, 'data', 'lm'))
    # the dev items of lm-book-qa-v1 (v2 changes only the closed-book flag)
    with zipfile.ZipFile(d2l.download_keyed('lm-book-qa-v1', os.path.join(
            W, 'data', 'lm'))) as z:
        name = next(n for n in z.namelist() if n.endswith('qa_dev.jsonl'))
        items = {q['qid']: q for q in map(json.loads, z.read(
            name).decode('utf-8').splitlines()) if q}
    calib, sel = calib_pairs(read_jsonl(gens_path), items)
    pairs = [dict(g, set='calibration') for g in calib] + [
        dict(g, set='selection') for g in sel]
    msgs = [calib_messages(corpus, items[g['qid']], g) for g in pairs]
    if args.smoke:
        m, at, params, reasoning = LABELLERS['label-second']
        res = run(client('smoke-second', m, reasoning),
                  [dict(messages=x, **params) for x in msgs[:10]],
                  'smoke-second')
        for r in res:
            log(repr(r)[:300] if isinstance(r, BaseException) else
                f'{r.model} {r.provider} {r.usage} {r.text[:200]!r}')
        return
    labels = {}
    for arm, (m, at, params, reasoning) in LABELLERS.items():
        res = run(client(arm, m, reasoning),
                  [dict(messages=x, **params) for x in msgs], arm)
        out = []
        for x in res:
            v = None if isinstance(x, BaseException) else parse_json_obj(
                x.text)
            ok = isinstance(v, dict) and all(isinstance(v.get(k), bool) for k
                                             in ('correct', 'faithful'))
            out.append({'correct': v['correct'], 'faithful': v['faithful'],
                        'rationale': str(v.get('rationale', ''))[:400],
                        'served': x.model, 'provider': x.provider}
                       if ok else None)
        labels[arm] = out
    rows = []
    for g, f, q in zip(pairs, labels['label-frontier'],
                       labels['label-second']):
        it = items[g['qid']]
        row = {'qid': g['qid'], 'set': g['set'], 'type': it['type'],
               'answerable': it['answerable'], 'config': 'base',
               'answer': g['answer'], 'context_ids': g['context_ids'],
               'cited_ids': g.get('cited_ids', [])}
        for k in ('correct', 'faithful'):
            a = None if f is None else f[k]
            b = None if q is None else q[k]
            row[f'{k}_frontier'], row[f'{k}_second'] = a, b
            row[k] = a if a is not None and a == b else None
            row[f'{k}_label_source'] = ('both' if row[k] is not None else
                                        'disagree' if None not in (a, b)
                                        else 'missing')
        row['rationale_frontier'] = None if f is None else f['rationale']
        row['rationale_second'] = None if q is None else q['rationale']
        rows.append(row)
    kap = calib_kappas(rows)
    spend = {}
    for arm in ('smoke-second', *LABELLERS):
        calls = d2l.api_calls(os.path.join(folder, 'api',
                                           arm + '.calls.jsonl'))
        ok = [c for c in calls if c.get('status') == 200]
        spend[arm] = {'calls': len(ok), 'failed_attempts': len(calls) -
                      len(ok), 'usd': round(sum(c.get('cost') or 0.0
                                                for c in ok), 6),
                      'served': sorted({str(c.get('served_model'))
                                        for c in ok}),
                      'providers': sorted({str(c.get('provider'))
                                           for c in ok})}
    snap = lambda m: listing['models'][m].get('canonical_slug', m)
    manifest = {
        'name': CALIB_KEY, 'question_set': KEY, 'n': len(rows),
        'sets': {'calibration': CALIB_N, 'selection': CALIB_N},
        'seed': SEED, 'labellers': {
            arm.split('-')[1]: {'requested': m, 'snapshot': snap(m),
                                'provider': at, 'params': params,
                                'reasoning': reasoning}
            for arm, (m, at, params, reasoning) in LABELLERS.items()},
        'kappa_frontier_vs_second': kap,
        'prompt_sha256': sha256(json.dumps(P.CALIB_V1, sort_keys=True)),
        'listing_retrieved': listing.get('retrieved'),
        'spend_usd': spend, 'note': 'labels from models, not from people'}
    members = {
        'pairs.jsonl': ''.join(dumps(g) + '\n' for g in pairs).encode(),
        'dev_judge_labels.jsonl': ''.join(dumps(r) + '\n'
                                          for r in rows).encode(),
        'manifest.json': (json.dumps(manifest, indent=1) + '\n').encode(),
        'README.md': CALIB_README.format(
            frontier=snap(VERIFIER), second=snap(SECOND)).encode(),
        'listing.json': open(lp, 'rb').read()}
    for sub in ('api', 'candidates'):  # caches; the judge candidates' runs
        if not os.path.isdir(os.path.join(folder, sub)):
            continue
        for name in sorted(os.listdir(os.path.join(folder, sub))):
            with open(os.path.join(folder, sub, name), 'rb') as f:
                members[f'{sub}/{name}'] = f.read()
    os.makedirs(out_dir, exist_ok=True)
    write_zip(os.path.join(out_dir, f'{CALIB_KEY}.zip'), members)
    log('kappa:', json.dumps(kap), 'wrote',
        os.path.join(out_dir, f'{CALIB_KEY}.zip'))


CALIB2_KEY = 'lm-rag-judge-labels-v2'
# Version 2 (2026-11-24, author ruling 4b): the calibration set of 100 pairs
# drawn as in version 1 (random.Random('calib|0') over the 240 base dev
# answers sorted by qid), now from the generator of 19.5, Qwen3.5-9B
# (lm-rag-gen-base-v2), labelled once by the same two frontier models with
# the same prompt and parameters. One cost cap of $3.00 for both labellers
# (shares 0.8 and 0.2), checked before every call on its worst case; one
# retry round for requests that failed in transport; no smoke calls.
CALIB2_CAP_USD = 3.0
CALIB2_SHARES = {'label-frontier': 0.8, 'label-second': 0.2}
CALIB2_CONCURRENCY = {'label-frontier': 2, 'label-second': 4}
CALIB2_README = """# lm-rag-judge-labels-v2: labels for checking the RAG judge on 19.5's answers

100 (question, answer) pairs of section 19.5 ("Retrieval-Augmented
Generation"): `base` answers of Qwen3.5-9B (thinking off, greedy), the
section's generator, to dev items of lm-book-qa-v2, as recorded in
lm-rag-gen-base-v2 (hybrid BookRetriever with the model card's embedder
task, section chunks, K=5, budget 3,000 tokens, the fixed prompt). The
pairs are the calibration draw of lm-rag-judge-labels-v1 (uniform over the
240 dev answers sorted by qid, random.Random('calib|0')), so they are the
same 100 questions, answered by the newer generator. There is no selection
set: the judge was chosen on lm-rag-judge-labels-v1.

Each pair is labelled `correct` and `faithful` (prompt CALIB_V1 of
tools/lm/book_qa_prompts.py, unchanged) by the two frontier models of
version 1, with the same parameters: {frontier} (first) and {second}
(second). Not labelled by people. `correct` and `faithful` hold the
consensus where both labellers agree (label_source "both"), else null
("disagree" or "missing"). Licence CC BY-SA 4.0.

Files: pairs.jsonl (the 100 answer records, as in lm-rag-gen-base-v2),
dev_judge_labels.jsonl (labels), manifest.json (models, served ids,
parameters, kappas, spend, cap), listing.json (OpenRouter listing snapshot
with the prices the cap used), api/<arm>.jsonl and api/<arm>.calls.jsonl
(every recorded response and every attempt; `python tools/build_book_qa.py
--mode replay --calibrate <lm-rag-gen-base-v2.jsonl> --calib-version 2
--cache-dir <unpacked zip>` rebuilds the labels from them with no key).
"""


def calib2_setup(args, d2l):
    """(folder, listing, recording, items, corpus, pairs, messages) of the
    version-2 calibration labels."""
    folder = os.path.abspath(args.cache_dir or os.path.join(
        W, 'data', 'lm-api', 'record', CALIB2_KEY))
    os.makedirs(os.path.join(folder, 'api'), exist_ok=True)
    lp = os.path.join(folder, 'listing.json')
    if not os.path.exists(lp):
        if args.mode == 'replay':
            raise SystemExit(f'{lp} missing')
        rec = d2l.APIRecording('record', os.path.dirname(folder), 1.0, {})
        rec.snapshot_listing([VERIFIER, SECOND], {},
                             key=os.path.basename(folder))
    with open(lp) as f:
        listing = json.load(f)
    rec = d2l.APIRecording(args.mode, os.path.dirname(folder),
                           CALIB2_CAP_USD, {VERIFIER: VERIFIER_AT,
                                            SECOND: SECOND_AT})
    rec.listing = listing
    corpus = d2l.book_corpus(root=os.path.join(W, 'data', 'lm'))
    with zipfile.ZipFile(d2l.download_keyed('lm-book-qa-v2', os.path.join(
            W, 'data', 'lm'))) as z:
        name = next(n for n in z.namelist() if n.endswith('qa_dev.jsonl'))
        items = {q['qid']: q for q in map(json.loads, z.read(
            name).decode('utf-8').splitlines()) if q}
    gens = read_jsonl(os.path.abspath(args.calibrate))
    calib, _ = calib_pairs(gens, items)
    pairs = [dict(g, set='calibration') for g in calib]
    msgs = [calib_messages(corpus, items[g['qid']], g) for g in pairs]
    return folder, listing, rec, items, corpus, pairs, msgs


def calib2_estimate(args, d2l, listing, msgs):
    """Expected and worst-case cost of labelling msgs with both labellers,
    from the listing's prices and version 1's recorded tokens per byte and
    completion tokens (no API call)."""
    old = os.path.join(W, 'data', 'lm-api', 'record', CALIB_KEY, 'api')
    out = {}
    for arm, (m, at, params, _) in LABELLERS.items():
        price = d2l.price_from_listing(listing, m, at)
        reqs = {json.loads(l)['hash']: json.loads(l) for l in open(
            os.path.join(old, arm + '.jsonl'))}
        calls = [c for c in d2l.api_calls(os.path.join(
            old, arm + '.calls.jsonl')) if c.get('status') == 200]
        nbytes = lambda ms: len(json.dumps(ms, ensure_ascii=False).encode())
        ratio = (sum(c['usage']['prompt_tokens'] for c in calls) /
                 sum(nbytes(reqs[c['hash']]['request']['messages'])
                     for c in calls))
        comp = [c['usage']['completion_tokens'] for c in calls]
        prompt = [ratio * nbytes(x) for x in msgs]
        exp = sum(d2l.api_cost({'prompt_tokens': p, 'completion_tokens':
                                sum(comp) / len(comp)}, price)
                  for p in prompt)
        hi = sum(d2l.api_cost({'prompt_tokens': p, 'completion_tokens':
                               sorted(comp)[int(0.95 * len(comp))]}, price)
                 for p in prompt)
        worst = sum(d2l.api_cost({'prompt_tokens': nbytes(x),
                                  'completion_tokens': params['max_tokens']},
                                 price) for x in msgs)
        out[arm] = {'model': m, 'provider': at, 'price': (
                        price._asdict() if hasattr(price, '_asdict')
                        else str(price)),
                    'calls': len(msgs), 'tokens_per_byte': round(ratio, 4),
                    'prompt_tokens_mean': round(sum(prompt) / len(prompt)),
                    'completion_tokens_mean_v1': round(sum(comp) /
                                                       len(comp)),
                    'expected_usd': round(exp, 4),
                    'p95_completion_usd': round(hi, 4),
                    'worst_case_usd': round(worst, 4)}
    out['total'] = {k: round(sum(v[k] for v in out.values()), 4) for k in (
        'expected_usd', 'p95_completion_usd', 'worst_case_usd')}
    return out


def calibrate_v2(args):
    """The version-2 calibration labels lm-rag-judge-labels-v2 (19.5; see
    CALIB2_README); with --estimate only the expected cost, no call."""
    os.environ.setdefault('HF_HOME', '/home/smola/.cache/hf-lm')
    os.chdir(os.path.join(W, 'chapter_lm-encoders'))
    from d2l import torch as d2l
    folder, listing, rec, items, corpus, pairs, msgs = calib2_setup(args, d2l)
    est = calib2_estimate(args, d2l, listing, msgs)
    log('estimate:', json.dumps(est, indent=1))
    if args.estimate:
        return
    if args.mode == 'record' and est['total']['expected_usd'] > 2.5:
        raise SystemExit('expected cost above $2.50: stop and ask')
    labels, results = {}, {}
    for arm, (m, at, params, reasoning) in LABELLERS.items():
        api = rec.hosted(m, os.path.join(folder, 'api', arm + '.jsonl'),
                         share=CALIB2_SHARES[arm], reasoning=reasoning,
                         timeout=300.0, max_retries=10)
        log(f'{arm}: cap {api.cost_cap_usd:.4f} USD for this cache')
        reqs = [dict(messages=x, **params) for x in msgs]
        conc = CALIB2_CONCURRENCY[arm]
        first = args.first or len(reqs)  # a staged start: the first pairs
        out = asyncio.run(d2l.chat_batch(api, reqs[:first], concurrency=conc))
        if first < len(reqs):
            log(f'{arm}: first {first}: ' + '; '.join(
                repr(r)[:200] if isinstance(r, BaseException) else
                f'{r.model} {r.provider} {r.usage} {r.text[:120]!r}'
                for r in out))
            continue
        for _ in range(1 if args.mode == 'record' else 0):  # one retry round
            todo = [i for i, r in enumerate(out)
                    if isinstance(r, BaseException)
                    and not isinstance(r, d2l.CostCapExceeded)]
            if not todo:
                break
            import time
            log(f'{arm}: {len(todo)} failed ({out[todo[0]]!r:.160}); '
                'retrying once in 60 s')
            time.sleep(60)
            again = asyncio.run(d2l.chat_batch(
                api, [reqs[i] for i in todo], concurrency=conc))
            for i, r in zip(todo, again):
                out[i] = r
        calls = d2l.api_calls(api.calls_path)
        bad = [r for r in out if isinstance(r, BaseException)]
        log(f'{arm}: {len(out) - len(bad)}/{len(out)} ok; recorded spend '
            f'{sum(c.get("cost") or 0 for c in calls):.4f} USD in '
            f'{len(calls)} attempts')
        if bad:
            log(f'  first failure: {bad[0]!r:.300}')
        results[arm] = out
        res = []
        for x in out:
            v = None if isinstance(x, BaseException) else parse_json_obj(
                x.text)
            ok = isinstance(v, dict) and all(isinstance(v.get(k), bool) for k
                                             in ('correct', 'faithful'))
            res.append({'correct': v['correct'], 'faithful': v['faithful'],
                        'rationale': str(v.get('rationale', ''))[:400],
                        'served': x.model, 'provider': x.provider}
                       if ok else None)
        labels[arm] = res
    if args.first:
        return
    rows = []
    for g, f, q in zip(pairs, labels['label-frontier'],
                       labels['label-second']):
        it = items[g['qid']]
        row = {'qid': g['qid'], 'set': g['set'], 'type': it['type'],
               'answerable': it['answerable'], 'config': 'base',
               'answer': g['answer'], 'context_ids': g['context_ids'],
               'cited_ids': g.get('cited_ids', [])}
        for k in ('correct', 'faithful'):
            a = None if f is None else f[k]
            b = None if q is None else q[k]
            row[f'{k}_frontier'], row[f'{k}_second'] = a, b
            row[k] = a if a is not None and a == b else None
            row[f'{k}_label_source'] = ('both' if row[k] is not None else
                                        'disagree' if None not in (a, b)
                                        else 'missing')
        row['rationale_frontier'] = None if f is None else f['rationale']
        row['rationale_second'] = None if q is None else q['rationale']
        rows.append(row)
    kap = {k: v for k, v in calib_kappas(rows).items()
           if not k.startswith('selection')}
    spend = {}
    for arm in LABELLERS:
        calls = d2l.api_calls(os.path.join(folder, 'api',
                                           arm + '.calls.jsonl'))
        ok = [c for c in calls if c.get('status') == 200]
        spend[arm] = {'calls': len(ok), 'failed_attempts': len(calls) -
                      len(ok), 'usd': round(sum(c.get('cost') or 0.0
                                                for c in ok), 6),
                      'prompt_tokens': sum(c['usage']['prompt_tokens']
                                           for c in ok),
                      'completion_tokens': sum(
                          c['usage']['completion_tokens'] for c in ok),
                      'served': sorted({str(c.get('served_model'))
                                        for c in ok}),
                      'providers': sorted({str(c.get('provider'))
                                           for c in ok})}
    snap = lambda m: listing['models'][m].get('canonical_slug', m)
    manifest = {
        'name': CALIB2_KEY, 'question_set': 'lm-book-qa-v2',
        'answers': 'lm-rag-gen-base-v2', 'generator': 'Qwen/Qwen3.5-9B',
        'n': len(rows), 'sets': {'calibration': CALIB_N}, 'seed': SEED,
        'labellers': {
            arm.split('-')[1]: {'requested': m, 'snapshot': snap(m),
                                'provider': at, 'params': params,
                                'reasoning': reasoning}
            for arm, (m, at, params, reasoning) in LABELLERS.items()},
        'kappa_frontier_vs_second': kap,
        'prompt_sha256': sha256(json.dumps(P.CALIB_V1, sort_keys=True)),
        'listing_retrieved': listing.get('retrieved'),
        'cost_cap_usd': CALIB2_CAP_USD, 'cap_shares': CALIB2_SHARES,
        'estimate_usd': est, 'spend_usd': spend,
        'note': 'labels from models, not from people'}
    members = {
        'pairs.jsonl': ''.join(dumps(g) + '\n' for g in pairs).encode(),
        'dev_judge_labels.jsonl': ''.join(dumps(r) + '\n'
                                          for r in rows).encode(),
        'manifest.json': (json.dumps(manifest, indent=1) + '\n').encode(),
        'README.md': CALIB2_README.format(
            frontier=snap(VERIFIER), second=snap(SECOND)).encode(),
        'listing.json': open(os.path.join(folder, 'listing.json'),
                             'rb').read()}
    for name in sorted(os.listdir(os.path.join(folder, 'api'))):
        with open(os.path.join(folder, 'api', name), 'rb') as f:
            members[f'api/{name}'] = f.read()
    out_dir = os.path.abspath(args.out)
    os.makedirs(out_dir, exist_ok=True)
    write_zip(os.path.join(out_dir, f'{CALIB2_KEY}.zip'), members)
    log('kappa:', json.dumps(kap), 'spend:', json.dumps(
        {a: v['usd'] for a, v in spend.items()}), 'wrote',
        os.path.join(out_dir, f'{CALIB2_KEY}.zip'))


def pack_cache(args):
    """The cache folder (listing, API caches and calls logs of every stage,
    the local-stage outputs) as a deterministic zip, the artifact
    lm-book-qa-api-cache-v1."""
    folder = os.path.abspath(args.cache_dir or os.path.join(
        W, 'data', 'lm-api', 'record', CACHE_KEY))
    members = {}
    for root, _, files in os.walk(folder):
        for name in files:
            path = os.path.join(root, name)
            rel = os.path.relpath(path, folder)
            if rel.startswith('calib-') or rel == 'manifest.json':
                continue  # calibration ships in lm-rag-judge-labels-v1
            with open(path, 'rb') as f:
                members[rel] = f.read()
    write_zip(args.pack_cache, members)
    log(f'{args.pack_cache}: {len(members)} files, '
        f'{os.path.getsize(args.pack_cache)} bytes')


def main():
    global KEY
    p = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    p.add_argument('--mode', choices=('record', 'replay'), required=True)
    p.add_argument('--cache-dir', help='API caches and local outputs '
                   '(default: data/lm-api/record/<cache key> when recording, '
                   'the registered artifact when replaying)')
    p.add_argument('--pack-cache', metavar='ZIP',
                   help='zip the cache folder and exit')
    p.add_argument('--out', default=os.path.join(W, 'data', 'lm',
                                                 'authoring', KEY))
    p.add_argument('--smoke', action='store_true')
    p.add_argument('--stop-after', choices=('s1', 's4', 's5', 's7'))
    p.add_argument('--no-verify', action='store_true')
    p.add_argument('--splits', type=lambda v: tuple(v.split(',')),
                   default=('dev', 'test', 'train'),
                   help='draft only these splits (a pilot with --stop-after)')
    p.add_argument('--review')
    p.add_argument('--calibrate', metavar='GENERATIONS_JSONL',
                   help='label 200 base dev generations of 19.5 (the '
                   'judge selection and calibration sets; with --smoke: 10 '
                   'calls of the second labeller)')
    p.add_argument('--calib-version', type=int, choices=(1, 2), default=1,
                   help='with --calibrate: 2 labels the 100 calibration '
                   'pairs of 19.5\'s generator (lm-rag-judge-labels-v2)')
    p.add_argument('--estimate', action='store_true',
                   help='with --calib-version 2: print the expected cost, '
                   'make no call')
    p.add_argument('--first', type=int, default=0,
                   help='with --calib-version 2: label only the first N '
                   'pairs of each labeller (a staged start)')
    p.add_argument('--set-version', type=int, choices=(1, 2), default=1,
                   help='2: lm-book-qa-v2 (closed-book flag by JUDGE_V2)')
    p.add_argument('--judge-url', help='OpenAI-compatible endpoints '
                   'serving JUDGE_V2, comma-separated (record mode, '
                   '--set-version 2)')
    p.add_argument('--judge-v2-file', help='closed-book v2 verdicts '
                   '(default <cache>/local/s6_judge_v2.jsonl)')
    p.add_argument('--judge-concurrency', type=int, default=32)
    p.add_argument('--concurrency', type=int, default=24)
    p.add_argument('--gen-batch', type=int, default=64)
    p.add_argument('--judge-batch', type=int, default=32)
    args = p.parse_args()
    if args.set_version == 2:
        KEY = 'lm-book-qa-v2'
        if args.out == p.get_default('out'):
            args.out = os.path.join(W, 'data', 'lm', 'authoring', KEY)
    if args.pack_cache:
        return pack_cache(args)
    if args.calibrate and args.calib_version == 2:
        calibrate_v2(args)
    else:
        calibrate(args) if args.calibrate else build(args)


if __name__ == '__main__':
    main()
