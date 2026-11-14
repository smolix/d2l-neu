# Retrieval: the Book Corpus Builder and Embedding Cache (build-only, unlisted)

<!--
This file is not listed in `_quarto.yml`/`CHAPTER_NUMBERING` and is never
rendered or executed as a notebook: `make lib` (`tools/build_lib.py`) only
scans it for `#@save` blocks, and `gen_notebooks.py` excludes it via
`LIB_ONLY_FILES` (appended after the numbered chapters, not early: nothing
uses these symbols at import time). It holds the library code that
:numref:`sec_lm_retrieval` explains in a sentence each but does not show:

- the cleaning of one book source (`clean_d2l_markdown`) and the label table
  its reference rewriting reads (`_r3_label_titles`);
- the block parser, the sentence splitter and the two chunkers
  (`_r3_parse_blocks`, `_r3_sentences`, `chunk_sections`, `chunk_fixed`);
- the corpus build from a git commit (`_r3_quarto_files`, `_r3_build`) that
  `tools/build_r3.py` writes into the artifact `lm-book-r3-v1`, and that
  `book_corpus(source='local')` runs in memory;
- the corpus object (`BookCorpus`) and its views;
- the embedding cache of chunk vectors (`_emb_cache_key`, `_emb_cache_load`,
  `_emb_cache_save`) and the settings of the two released embedders
  (`_EMBEDDERS`, `_R3_EMB_KEYS`).

Every block is tagged `%%tab pytorch` and imports what it needs beyond the
library preamble (`subprocess`, `warnings`, `bisect`, `transformers`) inside
its functions: a module-level import would join the `_preamble` shard and move
the freshness fingerprint of every notebook in the book.
-->

<!-- Not rendered; it sets the "Defined in" suffix of the docstrings. -->
:label:`sec_lm_retrieval`

```{.python .input #retrieval-lib-clean}
%%tab pytorch
#@save
_R3_DIRECTIVES = ('numref', 'eqref', 'exref', 'ref', 'cite', 'citet', 'label',
                  'eqlabel', 'exlabel', 'begin_tab', 'end_tab', 'width',
                  'height', 'maxdepth')

#@save
def _r3_caption_head(caption, width=80):
    """The first sentence of a caption, on one line, at most width chars."""
    text = ' '.join(caption.split())
    head = re.split(r'(?<=[.:;])\s', text, maxsplit=1)[0].rstrip('.:;')
    return head if len(head) <= width else head[:width - 3].rstrip() + '...'

def _r3_label_titles(sources):
    """Pass 1 of the reference rewriting: {label: (kind, title, doc_id)}
    for every :label: in sources ({doc_id: raw markdown}); kind is
    'Chapter', 'Section', 'Figure' or 'Table' by what the label follows."""
    titles = {}
    for doc_id, text in sources.items():
        heading = None
        for m in re.finditer(
                r'^(#{1,6})\s+(.*)$|^\s*:label:`([^`]+)`', text, re.M):
            if m.group(1):
                heading = re.sub(r'\s*\{[^}]*\}\s*$', '', m.group(2)).strip()
                continue
            label, before = m.group(3), text[:m.start()].rstrip()
            last = before.rsplit('\n', 1)[-1]
            image = re.search(r'!\[((?:[^\[\]]|\[[^\[\]]*\])*)\]\([^)\s]*\)'
                              r'(\{[^}]*\})?\s*$', before[-2000:])
            if last.startswith('#'):
                kind = 'Chapter' if label.startswith('chap') else 'Section'
                title = heading
            elif image:
                kind, title = 'Figure', image.group(1)
            elif last.startswith(':') and not last.startswith(':label'):
                kind, title = 'Table', last[1:]
            else:
                kind, title = 'Section', heading
            titles.setdefault(label, (kind, title, doc_id))
    raw = dict(titles)
    for label, (kind, title, doc_id) in raw.items():  # titles as plain text
        title = _r3_rewrite_refs(' '.join((title or '').split()), raw, [])
        if kind in ('Figure', 'Table'):
            title = _r3_caption_head(title)
        titles[label] = (kind, title, doc_id)
    return titles

def _r3_rewrite_refs(line, label_titles, unresolved):
    """Rewrite the reference directives of one prose line (rule 4); a
    kind the text already names ('exercise :exref:', 'Table :numref:') is
    not repeated."""
    def sub(m):
        name, arg = m.group(1), m.group(2)
        before = re.search(r'(\w+)\s*$', m.string[:m.start()])
        before = before.group(1).lower() if before else ''
        if name not in _R3_DIRECTIVES:
            raise ValueError(f'unknown directive :{name}: in {line!r}')
        if name in ('cite', 'citet'):
            return ''
        if name in ('eqref', 'exref'):
            kind = 'Equation' if name == 'eqref' else 'Exercise'
            named = before in ('equation', 'equations', 'exercise',
                               'exercises', 'problem', 'problems')
            return '\x00' if named else kind
        if name in ('numref', 'ref'):
            hit = label_titles.get(arg.strip())
            if hit is None:
                unresolved.append(arg.strip())
                return 'another section'
            kind, title, _ = hit
            named = before.rstrip('s') == kind.lower()
            return (f'"{title}"' if name == 'ref' or named
                    else f'{kind} "{title}"')
        return ''  # label, eqlabel, exlabel, width, height inside a line
    line = re.sub(r'^[ \t]*:(cite|citet):`[^`]*`', '\x01', line)  # join up
    line = re.sub(r'[ \t]*(?<![\w`]):(cite|citet):`[^`]*`', '', line)
    line = re.sub(r'(?<![\w`]):([a-z_]+):`([^`]*)`', sub, line)
    return re.sub(r'[ \t]*\x00', '', line)  # a dropped reference

def clean_d2l_markdown(text, framework='pytorch', keep_exercises=False,
                       label_titles=None):
    """One book source as plain text for retrieval, and its metadata: the
    slide deck cut, framework tabs other than framework dropped, cells as
    plain python fences, references rewritten as titles, citations and
    markup removed, exercises dropped (rules 1-8 of the book corpus).
    Returns (text, meta) with meta keys title, label, headings,
    cell_ids, unresolved_refs, has_exercises."""
    lines = text.split('\n')
    for i, line in enumerate(lines):  # rule 1: cut the slide deck
        if (line.strip() == '<!-- slides -->'
                or re.match(r'^::: \{\.(slide|cover)', line)):
            lines = lines[:i]
            break
    if label_titles is None:
        label_titles = _r3_label_titles({None: '\n'.join(lines)})
    meta = dict(title=None, label=None, headings=[], cell_ids=[],
                unresolved_refs=[], has_exercises=False)
    segs = []  # (kind, text): 'prose' lines, 'code' blocks, 'cell' blocks
    tab_keep, in_comment, last_heading, i = None, False, None, 0
    def emit(kind, s, cell_id=None):
        if tab_keep is not False:
            segs.append((kind, s, cell_id))
    while i < len(lines):
        line = lines[i]
        i += 1
        if in_comment:  # rule 5: HTML comments
            if '-->' in line:
                in_comment, line = False, line.split('-->', 1)[1]
            else:
                continue
        while '<!--' in line:
            head, tail = line.split('<!--', 1)
            if '-->' in tail:
                line = head + tail.split('-->', 1)[1]
            else:
                line, in_comment = head, True
        fence = re.match(r'^(\s*)(`{3,}|~{3,})(.*)$', line)
        if fence:  # rules 3 and 8: a fence and its body
            mark, info, body = fence.group(2), fence.group(3).strip(), []
            while i < len(lines) and not re.match(
                    rf'^\s*{re.escape(mark[0])}{{{len(mark)},}}\s*$', lines[i]):
                body.append(lines[i])
                i += 1
            if i == len(lines):
                raise ValueError(f'unclosed fence {line!r}')
            i += 1
            if info == 'toc':
                continue
            if not info.startswith('{.python'):
                emit('code', '\n'.join([line, *body, lines[i - 1]]))
                continue
            first = next((b for b in body if b.strip()), '')
            tab = re.match(r'^\s*(?:%%tab|#@tab)\s+(.*)$', first)
            if tab:
                fws = {f.strip() for f in tab.group(1).split(',')}
                if not fws & {framework, 'all'}:
                    continue
                body = body[body.index(first) + 1:]
            code = [b for b in body if b.strip()]
            imports = [b for b in code if re.match(  # and d2lbook's tabs
                r'^\s*(import\s|from\s\S+\s+import\s|%matplotlib|'
                r'%load_ext\s|tab\.interact_select\()', b)]
            if not code or len(imports) >= 0.9 * len(code):
                continue
            cell_id = re.search(r'#([\w-]+)', info)
            emit('cell', '```python\n' + '\n'.join(body).strip('\n')
                 + '\n```', cell_id.group(1) if cell_id else None)
            continue
        if in_comment and not line.strip():
            continue
        tab = re.match(r'^\s*:begin_tab:`([^`]*)`\s*$', line)
        if tab:  # rule 2: prose tabs
            if tab_keep is not None:
                raise ValueError('nested :begin_tab:')
            fws = {f.strip() for f in tab.group(1).split(',')}
            tab_keep = bool(fws & {framework, 'all'})
            continue
        if re.match(r'^\s*:end_tab:\s*$', line):
            if tab_keep is None:
                raise ValueError(':end_tab: without :begin_tab:')
            tab_keep = None
            continue
        if re.match(r'^##\s+Exercises\s*$', line):  # rule 7
            meta['has_exercises'] = True
            if not keep_exercises:
                break
        label = re.match(r'^\s*:label:`([^`]+)`\s*$', line)
        if label:  # attach a label to the heading it follows
            prev = next((s for s in reversed(segs) if s[1].strip()), None)
            if last_heading is not None and prev is not None and (
                    prev[1] is last_heading[1]):
                last_heading[0]['label'] = label.group(1)
            continue
        if re.match(r'^\s*:(eqlabel|exlabel|width|height):`[^`]*`\s*$', line):
            continue
        if (re.match(r'^\s*:::', line)
                or re.match(r'^\s*\[Discussions\]\(', line)):
            continue
        caption = re.match(r'^:([^\s:`].*)$', line)
        if caption and not re.match(r'^:[a-z_]+:`', line):  # table caption
            line = 'Table: ' + caption.group(1)
        line = _r3_rewrite_refs(line, label_titles, meta['unresolved_refs'])
        if not line.strip() and tab_keep is not False:
            segs.append(('prose', '', None))
            continue
        heading = re.match(r'^(#{1,6})\s+(.*?)\s*(\{[^}]*\})?\s*$', line)
        line = line.rstrip()
        if heading:  # without attributes such as {.unnumbered}
            line = heading.group(1) + ' ' + heading.group(2)
        if heading and tab_keep is not False:
            h = dict(level=len(heading.group(1)), title=heading.group(2),
                     label=None, char_start=None)
            meta['headings'].append(h)
            last_heading = (h, line)
        emit('prose', line)
    if tab_keep is not None:
        raise ValueError('unclosed :begin_tab:')
    if in_comment:
        raise ValueError('unclosed HTML comment')
    # Images become 'Figure: caption'; runs of blank lines collapse (rule 5)
    out, prose = [], []
    def flush():
        s = '\n'.join(prose)
        s = re.sub(r'!\[((?:[^\[\]]|\[[^\[\]]*\])*)\]\([^)\s]*\)(\{[^}]*\})?',
                   lambda m: 'Figure: ' + ' '.join(m.group(1).split()), s)
        s = re.sub(r'\n\x01(?=[.,;:!?)])', '', s).replace('\x01', '')
        s = re.sub(r'\n\s*\n(\s*\n)+', '\n\n', s)
        if s.strip():
            out.append(s.strip('\n'))
        prose.clear()
    for kind, s, cell_id in segs:
        if kind == 'prose':
            prose.append(s)
            continue
        flush()
        out.append(('cell' if kind == 'cell' else 'code', s, cell_id))
    flush()
    text, parts = '', []
    for part in out:  # prose, code and cells, one blank line apart
        if parts:
            text += '\n\n'
        if isinstance(part, tuple):
            if part[0] == 'cell':
                meta['cell_ids'].append(dict(
                    id=part[2], char_start=len(text),
                    char_end=len(text) + len(part[1])))
            text += part[1]
        else:
            text += part
        parts.append(part)
    text += '\n'
    # Heading offsets in the cleaned text, in order, outside code fences
    spans = [(c.start(), c.end()) for c in re.finditer(
        r'^```.*?^```', text, re.M | re.S)]
    found = [m for m in re.finditer(r'^(#{1,6})\s+(.*?)\s*$', text, re.M)
             if not any(a <= m.start() < b for a, b in spans)]
    if len(found) != len(meta['headings']):
        raise ValueError('heading count changed during cleaning')
    for h, m in zip(meta['headings'], found):
        h['char_start'], h['title'] = m.start(), m.group(2)
    h1 = next((h for h in meta['headings'] if h['level'] == 1), None)
    if h1 is not None:
        meta['title'], meta['label'] = h1['title'], h1['label']
    return text, meta
```

```{.python .input #retrieval-lib-blocks}
%%tab pytorch
#@save
def _r3_sentences(text):
    """Character spans (start, end) of the sentences of a prose text: a
    sentence ends at '.', '!' or '?' followed by whitespace and an upper-case
    letter, a digit, a dollar sign or a backtick."""
    ends = [m.end() for m in re.finditer(
        r'(?<=[.!?])["\')\]]?(?=\s+[A-Z0-9$`(])', text)]
    starts = [0] + [len(text) - len(text[e:].lstrip()) for e in ends]
    return [(s, e) for s, e in zip(starts, ends + [len(text)]) if e > s]

def _r3_parse_blocks(text):
    """The blocks of a cleaned document, as (kind, start, end) character
    spans: 'heading', 'para', 'list', 'table', 'math' ($$ ... $$), 'code'
    (a fence), 'quote'. Blocks are separated by blank lines; a fence or a
    display-math block runs to its closing line; a heading line is a block
    of its own."""
    blocks, pos, n = [], 0, len(text)
    line_starts = [0] + [m.end() for m in re.finditer('\n', text)]
    lines = text.split('\n')
    k = 0
    while k < len(lines):
        line, start = lines[k], line_starts[k]
        if not line.strip():
            k += 1
            continue
        fence = re.match(r'^\s*(`{3,}|~{3,})', line)
        if fence or line.strip().startswith('$$'):
            if fence:
                close = rf'^\s*{re.escape(fence.group(1)[0])}{{{len(fence.group(1))},}}\s*$'
            j = k + 1
            if fence or line.strip() == '$$' or not (
                    line.strip().endswith('$$') and len(line.strip()) > 2):
                while j < len(lines) and not (
                        re.match(close, lines[j]) if fence
                        else lines[j].strip().endswith('$$')):
                    j += 1
                j += 1
            end = line_starts[j - 1] + len(lines[j - 1]) if j <= len(
                lines) else n
            blocks.append(('code' if fence else 'math', start, end))
            k = j
            continue
        if re.match(r'^#{1,6}\s', line):
            blocks.append(('heading', start, start + len(line)))
            k += 1
            continue
        j = k
        while (j + 1 < len(lines) and lines[j + 1].strip()
               and not re.match(r'^(#{1,6}\s|\s*(`{3,}|~{3,})|\s*\$\$)',
                                lines[j + 1])):
            j += 1
        s = line.lstrip()
        kind = ('table' if s.startswith('|') else 'quote' if s.startswith('>')
                else 'list' if re.match(r'^([-*+]|\d+\.)\s', s) else 'para')
        blocks.append((kind, start, line_starts[j] + len(lines[j])))
        k = j + 1
    return blocks
```

```{.python .input #retrieval-lib-chunkers}
%%tab pytorch
#@save
def _r3_ntokens(tokenizer, text):
    """Number of tokens of text, without special tokens."""
    return len(tokenizer(text, add_special_tokens=False)['input_ids'])

def _r3_split_block(text, kind, start, end, tokenizer, max_tokens):
    """Split one block longer than max_tokens into pieces (code at blank
    lines or def/class lines, math at line ends, prose at sentence ends,
    a remainder at token windows), as (kind, start, end) spans."""
    piece = text[start:end]
    if kind == 'code':  # never right after the opening fence line
        cuts = [m.start() for m in re.finditer(
            r'\n(?=\s*\n)|\n(?=(def|class|@)\b)', piece)
            if m.start() > piece.find('\n')]
    elif kind in ('math', 'table', 'list'):
        cuts = [m.start() for m in re.finditer(r'\n', piece)]
    else:
        cuts = [e for _, e in _r3_sentences(piece)][:-1]
    bounds = [0] + sorted(set(cuts)) + [len(piece)]
    units = [(start + a, start + b) for a, b in zip(bounds, bounds[1:])]
    out, cur = [], None
    for a, b in units:
        if cur is not None and _r3_ntokens(
                tokenizer, text[cur[0]:b]) <= max_tokens:
            cur = (cur[0], b)
            continue
        if cur is not None:
            out.append(cur)
        cur = (a, b)
        if _r3_ntokens(tokenizer, text[a:b]) > max_tokens:  # token windows
            for w in chunk_fixed(dict(text=text[a:b]), tokenizer,
                                 size=max_tokens, overlap=0):
                out.append((a + w['char_start'], a + w['char_end']))
            cur = None
    if cur is not None:
        out.append(cur)
    return [(kind, a + len(text[a:b]) - len(text[a:b].lstrip()),
             a + len(text[a:b].rstrip())) for a, b in out
            if text[a:b].strip()]

def _r3_chunk_record(doc, tag, k, strategy, start, end, tokenizer, path,
                     section_label, kinds):
    """One chunk dict of doc over doc['text'][start:end]."""
    text = doc['text'][start:end]
    cells = [c['id'] for c in doc.get('cell_ids', ())
             if c['char_start'] < end and c['char_end'] > start]
    body = set(kinds) - {'heading'}
    kind = ('code' if body == {'code'} else 'table' if body == {'table'}
            else 'mixed' if 'code' in body else 'prose')
    return dict(id=f'{tag}:{doc["doc_id"]}#{k:03d}', doc_id=doc['doc_id'],
                strategy=strategy, text=text, heading_path=list(path),
                char_start=start, char_end=end,
                n_tokens=_r3_ntokens(tokenizer, text), kind=kind,
                cell_ids=[c for c in cells if c], section_label=section_label)

def _r3_heading_path(doc, pos):
    """(H1..H3 titles, deepest label) in force at character pos of doc."""
    path, labels = {}, {}
    for h in doc.get('headings', ()):
        if h['char_start'] > pos:
            break
        if h['level'] <= 3:
            path = {l: t for l, t in path.items() if l < h['level']}
            labels = {l: t for l, t in labels.items() if l < h['level']}
            path[h['level']] = h['title']
            labels[h['level']] = h['label']
    label = next((labels[l] for l in sorted(labels, reverse=True)
                  if labels[l]), None)
    return [path[l] for l in sorted(path)], label

def chunk_sections(doc, tokenizer, max_tokens=512, min_tokens=64,
                   tag='sec512'):
    """Structure-aware chunks of a cleaned document (a dict with doc_id,
    text, headings, cell_ids): the blocks under each H1-H3 heading packed
    greedily up to max_tokens; a display-math block stays with the
    paragraph before it and a following paragraph starting 'where', a code
    cell with the paragraph before it when both fit; a section under
    min_tokens merges into its neighbour under the same H2."""
    text = doc['text']
    size = lambda bl: sum(_r3_ntokens(tokenizer, text[a:b]) for _, a, b in bl)
    units, h2 = [], 0  # [H2 count, path, label, heading block, blocks]
    for kind, a, b in _r3_parse_blocks(text):
        level = len(text[a:b]) - len(text[a:b].lstrip('#'))
        if kind == 'heading' and level <= 3:
            h2 += level <= 2
            units.append([h2, *_r3_heading_path(doc, a), (kind, a, b), []])
        elif units:
            units[-1][4].append((kind, a, b))
        else:
            units.append([h2, *_r3_heading_path(doc, a), None, [(kind, a, b)]])
    merged = []  # a small section joins its neighbour under the same H2
    for u in units:
        if merged and merged[-1][0] == u[0] and (
                size(u[4]) < min_tokens or size(merged[-1][4]) < min_tokens):
            merged[-1][4].extend([u[3]] + u[4])
        else:
            merged.append(u)
    chunks = []
    for _, path, label, _, ublocks in merged:
        groups = []  # atomic runs of blocks
        for kind, a, b in ublocks:
            prev = groups[-1] if groups else None
            glue = prev is not None and (
                (kind == 'math' and prev[-1][0] in ('para', 'list', 'math'))
                or (kind == 'para' and prev[-1][0] == 'math'
                    and text[a:b].lstrip().lower().startswith('where'))
                or (kind == 'code' and prev[-1][0] == 'para' and _r3_ntokens(
                    tokenizer, text[prev[0][1]:b]) <= max_tokens))
            if glue:
                prev.append((kind, a, b))
            else:
                groups.append([(kind, a, b)])
        pieces = []  # groups that fit, else their blocks, else block pieces
        for g in groups:
            if _r3_ntokens(tokenizer, text[g[0][1]:g[-1][2]]) <= max_tokens:
                pieces.append(g)
                continue
            for kind, a, b in g:
                if _r3_ntokens(tokenizer, text[a:b]) <= max_tokens:
                    pieces.append([(kind, a, b)])
                else:
                    pieces.extend([p] for p in _r3_split_block(
                        text, kind, a, b, tokenizer, max_tokens))
        cur = []
        for p in pieces:
            if cur and _r3_ntokens(
                    tokenizer, text[cur[0][1]:p[-1][2]]) > max_tokens:
                chunks.append((cur, path, label))
                cur = []
            cur = cur + p
        if cur:
            chunks.append((cur, path, label))
    out = []
    for k, (bl, path, label) in enumerate(chunks):
        a, b = bl[0][1], bl[-1][2]
        out.append(_r3_chunk_record(doc, tag, k, 'sections', a, b, tokenizer,
                                    path, label, [x[0] for x in bl]))
    return out

def chunk_fixed(doc, tokenizer, size=256, overlap=0, tag=None):
    """Windows of size tokens with overlap tokens shared, over the token
    stream of doc['text'], cut at character boundaries: a window ends at
    its last token whose characters it does not share with the next token
    (a byte-level token can hold part of a character)."""
    text = doc['text']
    enc = tokenizer(text, add_special_tokens=False,
                    return_offsets_mapping=True)
    offs = enc['offset_mapping']
    n = len(offs)
    shared = lambda j: j + 1 < n and offs[j][1] > offs[j + 1][0]
    windows, i = [], 0
    while i < n:
        j = min(i + size, n) - 1
        while j > i and shared(j):
            j -= 1
        windows.append((offs[i][0], offs[j][1]))
        if j == n - 1:
            break
        nxt = max(j + 1 - overlap, i + 1)
        while nxt < n and nxt > 0 and offs[nxt - 1][1] > offs[nxt][0]:
            nxt += 1  # never start inside a character
        i = nxt
    if tag is None:
        return [dict(char_start=a, char_end=b) for a, b in windows]
    out, blocks = [], _r3_parse_blocks(text)
    for k, (a, b) in enumerate(windows):
        path, label = _r3_heading_path(doc, a)
        kinds = [kind for kind, s, e in blocks if s < b and e > a]
        out.append(_r3_chunk_record(doc, tag, k, 'fixed', a, b, tokenizer,
                                    path, label, kinds))
    return out
```

```{.python .input #retrieval-lib-build}
%%tab pytorch
#@save
_R3_EXCLUDED = {  # directory prefix: reason
    'chapter_language-models': 'Language Models part (self-reference)',
    'chapter_lm-': 'Language Models part (self-reference)',
    'chapter_attic': 'Attic', 'chapter_gaussian-processes': 'Attic',
    'chapter_hyperparameter-optimization': 'Attic',
    'chapter_recommender-systems': 'Attic',
    'chapter_natural-language-processing-pretraining': 'retiring NLP chapter',
    'chapter_natural-language-processing-applications': 'retiring NLP chapter',
    'chapter_preface': 'front matter', 'chapter_installation': 'front matter',
    'chapter_notation': 'front matter', 'chapter_references': 'references'}
_R3_SPLIT_SALT = 'r3-split-v1|'
R3_TOKENIZER = 'Qwen/Qwen3-0.6B'
R3_VERSION = 'r3.1'

#@save
def _r3_quarto_files(quarto_yml):
    """[(path without .md, part file or None)] of the chapter files that
    _quarto.yml lists, in book order, part files included."""
    files, part = [], None
    for line in quarto_yml.split('\n'):
        m = re.match(r'^\s*-\s+part:\s*(\S+)\.qmd\s*$', line)
        if m:
            part = m.group(1)
            files.append((part, None))
            continue
        m = re.match(r'^\s*-\s+(chapter_\S+)\.qmd\s*$', line)
        if m:
            files.append((m.group(1), part))
    return files

def _r3_split_docs(doc_ids):
    """{doc_id: 'train'|'dev'|'test'}: within each chapter directory the
    files ordered by sha256 of a salted id; floor(0.15 n + 0.5) to dev, as
    many to test, the rest to train; chapters of fewer than 4 files train."""
    by_chapter = collections.defaultdict(list)
    for d in doc_ids:
        by_chapter[d.split('/')[0]].append(d)
    split = {}
    for docs in by_chapter.values():
        docs = sorted(docs, key=lambda d: hashlib.sha256(
            (_R3_SPLIT_SALT + d).encode()).hexdigest())
        m = math.floor(0.15 * len(docs) + 0.5) if len(docs) >= 4 else 0
        for i, d in enumerate(docs):
            split[d] = 'dev' if i < m else 'test' if i < 2 * m else 'train'
    return split

def _r3_build(read, commit, tokenizer, revision):
    """The book corpus from the sources at a commit: read(path) returns a
    file's text at that commit. Returns (manifest, docs, chunkings), with
    docs a list of document dicts and chunkings {tag: chunk list}."""
    listed = _r3_quarto_files(read('_quarto.yml'))
    parts = {p for p, _ in listed if _ is None}
    included, excluded = [], collections.Counter()
    for path, part in listed:
        reason = next((r for pre, r in _R3_EXCLUDED.items()
                       if path.startswith(pre)), None)
        if part is None and reason is None:
            reason = 'part placeholder'
        if reason is None and path in parts:
            reason = 'part placeholder'
        if reason:
            excluded[(path.split('/')[0], reason)] += 1
        else:
            included.append((path, part))
    raw = {path: read(path + '.md') for path, _ in included}
    titles = _r3_label_titles(raw)
    part_titles = {p: re.search(r'^#\s+(.*?)\s*$', read(p + '.md'),
                                re.M).group(1) for p in parts}
    split = _r3_split_docs(raw)
    docs, unresolved = [], collections.Counter()
    for path, part in sorted(included):
        text, meta = clean_d2l_markdown(raw[path], label_titles=titles)
        unresolved.update(meta['unresolved_refs'])
        docs.append(dict(
            doc_id=path, title=meta['title'], label=meta['label'],
            chapter=path.split('/')[0], part=part_titles[part], text=text,
            headings=meta['headings'], cell_ids=meta['cell_ids'],
            n_tokens=_r3_ntokens(tokenizer, text),
            source_sha256=hashlib.sha256(raw[path].encode()).hexdigest(),
            split=split[path]))
    chapter_titles = {d['chapter']: d['title'] for d in docs
                      if d['doc_id'].endswith('/index')}
    for d in docs:
        d['chapter_title'] = chapter_titles.get(d['chapter'], d['title'])
    chunkings = {'sec512': [], 'fix256o0': [], 'fix256o64': []}
    for d in docs:
        chunkings['sec512'] += chunk_sections(d, tokenizer)
        chunkings['fix256o0'] += chunk_fixed(d, tokenizer, 256, 0,
                                             tag='fix256o0')
        chunkings['fix256o64'] += chunk_fixed(d, tokenizer, 256, 64,
                                              tag='fix256o64')
    def quantiles(xs):
        return {str(q): int(np.quantile(xs, q, method='nearest'))
                for q in (0, 0.1, 0.5, 0.9, 0.99, 1)}
    manifest = dict(
        corpus_version=R3_VERSION, book_commit=commit,
        builder_version='build_r3-1',
        cleaning=dict(framework='pytorch', keep_exercises=False),
        tokenizer=dict(id=R3_TOKENIZER, revision=revision),
        included=dict(collections.Counter(p.split('/')[0]
                                          for p, _ in included)),
        excluded=[dict(directory=d, reason=r, files=n)
                  for (d, r), n in sorted(excluded.items())],
        files={d['doc_id']: d['source_sha256'] for d in docs},
        doc_split={d['doc_id']: d['split'] for d in docs},
        counts=dict(docs=len(docs), **{
            f'docs_{s}': sum(d['split'] == s for d in docs)
            for s in ('train', 'dev', 'test')},
            chars=sum(len(d['text']) for d in docs),
            tokens=sum(d['n_tokens'] for d in docs),
            **{f'chunks_{t}': len(c) for t, c in chunkings.items()}),
        token_quantiles={t: quantiles([c['n_tokens'] for c in cs])
                         for t, cs in chunkings.items()},
        unresolved_refs=dict(count=sum(unresolved.values()),
                             labels=dict(sorted(unresolved.items()))),
        chunking=dict(sec512=dict(strategy='sections', max_tokens=512,
                                  min_tokens=64),
                      fix256o0=dict(strategy='fixed', size=256, overlap=0),
                      fix256o64=dict(strategy='fixed', size=256,
                                     overlap=64)))
    return manifest, docs, chunkings
```

```{.python .input #retrieval-lib-corpus}
%%tab pytorch
#@save
class BookCorpus:
    """The book corpus under one chunking: chunks (a list of dicts in
    document order), docs ({doc_id: dict}), the manifest, and views of the
    chunk texts for a generator ('display'), a dense encoder ('dense') and
    a word index ('sparse')."""
    def __init__(self, manifest, docs, chunks, tag):
        self.manifest, self.tag = manifest, tag
        self.version = manifest['corpus_version']
        self.docs = {d['doc_id']: d for d in docs}
        self.chunks = chunks
        self.ids = [c['id'] for c in chunks]
        self._index = {c['id']: i for i, c in enumerate(chunks)}
        self._by_doc = collections.defaultdict(list)
        for c in chunks:
            self._by_doc[c['doc_id']].append(c)

    def __len__(self):
        return len(self.chunks)

    def __getitem__(self, chunk_id):
        return self.chunks[self._index[chunk_id]]

    def header(self, chunk):
        """'Chapter > Section > Subsection' of a chunk."""
        doc = self.docs[chunk['doc_id']]
        path = list(chunk['heading_path'])
        if not path or path[0] != doc['chapter_title']:
            path = [doc['chapter_title']] + path
        return ' > '.join(path)

    def texts(self, view='display', header=True):
        """Chunk texts, in chunk order, for a view: 'display' (the text),
        'dense' (code cells cut to their first 30 lines) or 'sparse' (math
        removed)."""
        out = []
        for c in self.chunks:
            t = c['text']
            if view == 'dense':
                t = re.sub(r'(^```python\n)(.*?)(^```)',
                           lambda m: m.group(1) + _r3_cut_code(m.group(2))
                           + m.group(3), t, flags=re.M | re.S)
            elif view == 'sparse':
                t = _r3_strip_math(t)
            elif view != 'display':
                raise ValueError(f'unknown view {view!r}')
            out.append(self.header(c) + '\n' + t if header else t)
        return out

    def split_of(self, doc_id):
        """'train', 'dev' or 'test' of a document."""
        return self.manifest['doc_split'][doc_id]

    def chunk_split(self, chunk_id):
        """The split of the document a chunk comes from."""
        return self.split_of(self[chunk_id]['doc_id'])

    def span_to_chunk_ids(self, doc_id, start, end, min_overlap=0.5):
        """Ids of the chunks of doc_id that overlap the character span
        [start, end) by at least min_overlap of the span's or of the
        chunk's characters."""
        out = []
        for c in self._by_doc.get(doc_id, ()):
            ov = min(end, c['char_end']) - max(start, c['char_start'])
            if ov > 0 and (ov >= min_overlap * (end - start) or
                           ov >= min_overlap * (c['char_end'] -
                                                c['char_start'])):
                out.append(c['id'])
        return out

def _r3_cut_code(code, n_lines=30):
    """The first n_lines lines of a code cell's body, '# ...' if cut."""
    lines = code.split('\n')
    return code if len(lines) <= n_lines + 1 else '\n'.join(
        lines[:n_lines] + ['# ...', ''])

def _r3_strip_math(text):
    """text with $$...$$ and $...$ spans removed outside code fences."""
    parts = re.split(r'(^```.*?^```)', text, flags=re.M | re.S)
    for i in range(0, len(parts), 2):
        s = re.sub(r'\$\$.*?\$\$', ' ', parts[i], flags=re.S)
        parts[i] = re.sub(r'(?<!\\)\$[^$\n]+?(?<!\\)\$', ' ', s)
    return ''.join(parts)

def _r3_read_zip(path, tags=None):
    """(manifest, docs, {tag: chunks}) from a corpus zip; tags (default
    all) names the chunkings to read, () none and no documents."""
    with zipfile.ZipFile(path) as z:
        manifest = json.loads(z.read('manifest.json'))
        read = lambda name: [json.loads(l) for l in gzip.decompress(
            z.read(name)).decode('utf-8').splitlines()]
        tags = manifest['chunking'] if tags is None else tags
        docs = read('docs.jsonl.gz') if tags else []
        chunks = {t: read(f'chunks_{t}.jsonl.gz') for t in tags}
    return manifest, docs, chunks

def _r3_local(root, tokenizer, revision):
    """The book corpus rebuilt in memory from a checkout's HEAD."""
    import subprocess
    commit = subprocess.run(['git', '-C', root, 'rev-parse', 'HEAD'],
                            capture_output=True, text=True,
                            check=True).stdout.strip()
    read = lambda p: subprocess.run(
        ['git', '-C', root, 'show', f'{commit}:{p}'], capture_output=True,
        check=True).stdout.decode('utf-8')
    manifest, docs, chunks = _r3_build(read, commit, tokenizer, revision)
    return manifest, docs, chunks
```

```{.python .input #retrieval-lib-cache}
%%tab pytorch
#@save
# The released embedders: pooling, padding side and the query instruction
# format of their model cards; documents get no prefix
_EMBEDDERS = {
    'Qwen/Qwen3-Embedding-0.6B': dict(
        pooling='last', padding_side='left',
        query_format='Instruct: {task}\nQuery:{query}'),
    'microsoft/harrier-oss-v1-0.6b': dict(
        pooling='last', padding_side='left',
        query_format='Instruct: {task}\nQuery: {query}')}
BOOK_TASK = 'Given a question about the book, retrieve passages that answer it'
# Registered chunk vectors (sec512 chunks, 'dense' view, float16) by embedder
_R3_EMB_KEYS = {
    'Qwen/Qwen3-Embedding-0.6B': 'lm-enc-r3-emb-qwen3emb06b-sec512-v1',
    'microsoft/harrier-oss-v1-0.6b': 'lm-enc-r3-emb-harrier06b-sec512-v1'}

#@save
def _emb_cache_key(corpus_version, tag, view, model_id, revision, max_len,
                   pooling, prefix):
    """sha1 naming the chunk vectors of one corpus, view and encoder."""
    return hashlib.sha1('|'.join(map(str, (
        corpus_version, tag, view, model_id, revision, max_len, pooling,
        prefix))).encode()).hexdigest()

def _emb_cache_load(key, root=os.path.join('..', 'data', 'lm', 'cache')):
    """Cached chunk vectors (an array) under key, or None."""
    path = os.path.join(root, f'emb-{key}.npy')
    return np.load(path) if os.path.exists(path) else None

def _emb_cache_save(key, X, root=os.path.join('..', 'data', 'lm', 'cache')):
    """Write chunk vectors under key; return the path."""
    os.makedirs(root, exist_ok=True)
    path = os.path.join(root, f'emb-{key}.npy')
    np.save(path + '.tmp.npy', X)
    os.replace(path + '.tmp.npy', path)
    return path
```
