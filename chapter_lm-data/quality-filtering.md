# Quality Filtering
:label:`sec_lm_quality_filtering`

The previous section turned a quarter of one Common Crawl archive file into
3,288 English documents and ended on a puzzle: about seven in ten of them are
not in FineWeb, although FineWeb was built from the same file. Something
judged them not worth keeping. Anyone who has looked through a crawl will not
need convincing that such a judgment is necessary. Next to manuals and news
articles sit product catalogues, lists of links, pages of keywords written for
search engines, articles run through a program that swaps words for synonyms,
and advertisements for love spells. A model trained on all of it learns all of
it.

Two kinds of filter make the judgment. The older kind is a set of rules,
written by people who had read many bad pages: too few words, too many
repeated lines, too few lines that end in a full stop. The newer kind is a
classifier. A large language model reads a sample of a few hundred thousand
pages and scores each one under a written rubric, a small model learns to
predict the score, and a threshold on the prediction decides for the billions
of pages that no large model will ever read. FineWeb-Edu
:cite:`Penedo.Kydlicek.BenAllal.ea.2024` was built this way, with a rubric
that asks how useful a page would be for teaching at school. Other open
corpora of the same years also select with a cheap classifier, though not
always one trained on a large model's scores: DCLM's learns to tell
instruction data from random web pages, and Nemotron-CC combines several.

Both kinds of filter remove most of the web, and both apply a definition of
quality that somebody wrote down. The rules define it by surface statistics.
The classifier inherits it from a rubric and from the model that applied the
rubric, that model's taste included. The two pages printed below show both
definitions at work. One advertises "death spells" and passes every rule we
implement. The other is a software company's guide to hiring engineers, plain
and informative, and our classifier removes it. Faced with the second page,
the natural instinct is to adjust the threshold. The better question is what
the filter means by quality, and whose pages it removes.

This section builds both kinds of filter, trains a classifier on labels
written by Qwen3.8-27B, chooses its threshold, and audits what it removes by
country, kind of site and language. It ends where the audit points: a filter
keeps what its annotator likes, and only training runs, of the kind that
:numref:`sec_lm_data_mixtures` sets up, can say whether that is what a model
needs.

```{.python .input #quality-filtering-imports}
%%tab pytorch
import math
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from d2l import torch as d2l
```

```{.python .input #quality-filtering-extras}
%%tab pytorch
import collections
import concurrent.futures
import gzip
import hashlib
import io
import json
import multiprocessing
import os
import pathlib
import re
import textwrap
import time
import zlib
import pandas as pd
import transformers
from sentence_transformers import SentenceTransformer
```

Most of what this notebook computes is cheap: the rules run on eight CPU
processes, a 0.6B-parameter model scores perplexity, an embedder encodes the
crawl documents, and linear classifiers train in seconds, so a build takes a
few minutes on one GPU. Two steps are expensive, and a build reads their
results from registered files: labelling 22,000 documents with Qwen3.8-27B
took about an hour on four GPUs, and the review of 200 removed documents a few
minutes more. The vectors of the labelled documents and of the audit stream
are registered too, with the released FineWeb-Edu classifier's scores.
Without a GPU the notebook reads every registered result and scores
perplexity on a small sample only. Setting `D2L_RERUN` to names from `ARTIFACTS` regenerates those files under
new names, and the last cell compares whatever this run computed with the
registered results.

```{.python .input #quality-filtering-config}
%%tab pytorch
def say(*args, sep=' '):
    """print, each line wrapped to the page's 100 columns."""
    print('\n'.join(l if len(l) <= 100 else textwrap.fill(
        l, 100, subsequent_indent='    ', break_long_words=False,
        break_on_hyphens=False) for l in sep.join(map(str, args)).split('\n')))


LABELLER = d2l.EDU_LABELLER_LOCAL['model']  # the rubric's scores (vLLM)
EMBEDDER = 'Qwen/Qwen3-Embedding-0.6B'
PPL_MODEL = 'Qwen/Qwen3-0.6B-Base'   # the perplexity filter's reference
RELEASED = 'HuggingFaceFW/fineweb-edu-classifier'
MAX_LEN = 512             # tokens of a document that every model reads
THRESH_LABEL = 3          # "educational": a score of 3 or more
SCORE_PREFIX = 'Educational score: '  # the labeller's own spacing (pilot)
EXTRA_SPACE = True  # also count a second space before the digit
LABEL_CONCURRENCY = 1  # requests in flight per labeller server: see below
JUDGE_MAX_TOKENS = 12_000  # the review's reasoning judge: no thinking cap
GPU = torch.cuda.is_available()
SMALL = not GPU  # the CPU path: registered labels, vectors and scores
DTYPE = torch.bfloat16 if GPU else torch.float32
ARTIFACTS = {  # name: (registered key, file a regeneration writes)
    'edu-label-pilot': ('lm-edu-annot-22k-v1', 'edu-label-pilot.parquet'),
    'edu-label-0': ('lm-edu-annot-22k-v1', 'edu-label-0.parquet'),
    'edu-label-1': ('lm-edu-annot-22k-v1', 'edu-label-1.parquet'),
    'quality-emb': ('lm-data-emb-qwen3emb-v1', 'lm-data-emb-qwen3emb-v1.npz'),
    'quality-results': ('lm-data-quality-results-v1',
                        'lm-data-quality-results-v1.json'),
    'quality-flags': ('lm-c1-docs-v1', 'lm-c1-docs-v1.jsonl.gz'),
    'removal-labels': ('lm-data-removal-labels-v1',
                       'lm-data-removal-labels-v1.jsonl')}
RERUN = d2l.lm_rerun(ARTIFACTS)
GPU_NAMES = {'edu-label-pilot', 'edu-label-0', 'edu-label-1', 'quality-emb',
             'removal-labels'}
assert GPU or not RERUN & GPU_NAMES, 'these regenerations need a GPU'
RERUN_DIR = os.path.join('..', 'data', 'lm', 'rerun')
R = {}  # what this run measures; written or compared in the last cell
pins = {m: d2l.HF_REVISIONS.get(m) or d2l.HF_ENCODER_REVISIONS.get(m)
        for m in (EMBEDDER, PPL_MODEL)}
pins[LABELLER] = d2l.EDU_LABELLER_LOCAL['revision']
pins[RELEASED] = d2l.LM_DATA_SOURCES['fineweb-edu-classifier'][2]
print(f'GPU={GPU}, SMALL={SMALL}, regenerating: '
      f'{", ".join(sorted(RERUN)) or "none"}')
for m, rev in pins.items():
    print(f'  {m}@{rev[:8]}')
```

The registry adds seven files. The 22,000 labelled documents and the audit
stream are FineWeb text, which its ODC-By licence lets us host with
attribution, and every cell that prints their text prints the attribution
too. Nothing derived from our own crawl text is hosted. The flags of the crawl
documents hold ids, reason codes and scores; their vectors, computed from text
we may not redistribute, are cached on the machine that computed them; and the
review file holds document ids and the judge's one-sentence rationales.

<!-- Registrations: the audit stream (2026-11-20) and the six artifacts of the 2026-11-21 regeneration (four RTX 4090s, vLLM 0.33.0, RedHatAI/Qwen3.8-27B-INT4 at 71dface3); a new regeneration writes new versions (-v2) and never overwrites these. -->

```{.python .input #quality-filtering-data-hub}
%%tab pytorch
#@save
d2l.DATA_HUB['lm-data-audit-stream-v1'] = (  # FineWeb text, ODC-By: the
    # audit stream, six FineWeb-2 slices, same-host documents (20.3);
    # tools/build_lm_data_sources.py audit-stream
    d2l.LM_DATA_URL + 'lm-data-audit-stream-v1.parquet',
    '231fc997824464048a579b133d235f6103dbf957')
d2l.DATA_HUB['lm-edu-annot-22k-v1'] = (  # FineWeb text, ODC-By: 22,000
    # documents with Llama-3-70B's and Qwen3.8-27B's scores (20.2)
    d2l.LM_DATA_URL + 'lm-edu-annot-22k-v1.parquet',
    '4a1c7d61d13569c381eacf0421c9be071c06b201')
d2l.DATA_HUB['lm-data-emb-qwen3emb-v1'] = (  # embeddings and released
    # classifier scores of the annotations and the stream (20.2)
    d2l.LM_DATA_URL + 'lm-data-emb-qwen3emb-v1.npz',
    'dbd67d20099a04c590c1cfaaa7f19a9ea2256550')
d2l.DATA_HUB['lm-data-quality-results-v1'] = (  # 20.2's numbers
    d2l.LM_DATA_URL + 'lm-data-quality-results-v1.json',
    'e7bb769dcf59c793d5a45b586fa4fcd7c798bef2')
d2l.DATA_HUB['lm-quality-head-v1'] = (  # the regression head (20.2)
    d2l.LM_DATA_URL + 'lm-quality-head-v1.npz',
    '4f646fc8adb2503267b2904ecdff2521604e06fd')
d2l.DATA_HUB['lm-c1-docs-v1'] = (  # c0 ids with rule codes and edu_score,
    # no text (20.2)
    d2l.LM_DATA_URL + 'lm-c1-docs-v1.jsonl.gz',
    'd820ecb7bebfbcc9ca5e88e4e7f9edca8a4432de')
d2l.DATA_HUB['lm-data-removal-labels-v1'] = (  # Qwen3.8-27B's keep or drop
    # on 200 removed stream documents, ids and rationales (20.2)
    d2l.LM_DATA_URL + 'lm-data-removal-labels-v1.jsonl',
    '5d9dcdb686c6747538cbd512f2e1dfb881805f1c')

QUALITY_KEYS = ('lm-edu-annot-22k-v1', 'lm-data-audit-stream-v1',
                'lm-data-emb-qwen3emb-v1', 'lm-quality-head-v1',
                'lm-c1-docs-v1', 'lm-data-quality-results-v1',
                'lm-data-removal-labels-v1')
say('registered:', ', '.join(k for k in QUALITY_KEYS if k in d2l.DATA_HUB)
    or 'none', '| pending:', ', '.join(
        k for k in QUALITY_KEYS if k not in d2l.DATA_HUB) or 'none')
```

The first cell computes how much of this dump FineWeb-Edu keeps, from the row
counts of the two datasets, and prints the two pages. Both come from the audit
stream, a sample of FineWeb's output introduced below, and both are chosen by
rule, not by us. The first is the page with the lowest score from our
classifier among the pages that pass every rule and have at least 1,500
characters. The second is the page with the lowest score among those our
classifier removes and the review at the end of the section would keep.

```{.python .input #quality-filtering-opening}
%%tab pytorch
ATTRIBUTION = ('Text: FineWeb, HuggingFaceFW/fineweb, CC-MAIN-2025-05 '
               '(ODC-By 1.0)')
kept, total = 19_287_119, 265_662_695  # the two dataset cards' row counts
print(f'FineWeb-Edu keeps {kept:,} of FineWeb\'s {total:,} documents of '
      f'CC-MAIN-2025-05: {kept / total:.2%}')
STREAM = pd.read_parquet(d2l.lm_artifact('lm-data-audit-stream-v1'))
S = STREAM[STREAM.role == 'stream'].reset_index(drop=True)
REG = d2l.lm_registered('lm-data-quality-results-v1')
REG = None if REG is None else json.load(open(REG))
if REG is None:
    print('the two excerpts are chosen by rules later in this notebook; '
          'they appear once its results are registered')
else:
    by_id = S.set_index('id').text
    phone = re.compile(r'\+?\d[\d ()./-]{7,}\d')  # spam prints numbers
    for title, key in (
            ('passes every rule, at least 1,500 characters, lowest score '
             'of our classifier', 'low_score_id'),
            ('removed by our classifier, labelled "keep" in the review, '
             'lowest score', 'kept_label_id')):
        doc = REG['opening'].get(key)
        if doc is None:
            continue
        say(f'\n{title} ({doc}):')
        text = phone.sub(lambda m: '<PHONE>' if sum(map(str.isdigit, m[0]))
                         >= 9 else m[0], d2l.scrub_pii(by_id[doc][:600])[0])
        print(textwrap.fill(text.replace('\n', ' ') + ' …',
                            100, initial_indent='  ',
                            subsequent_indent='  '))
    print('\n' + ATTRIBUTION)
```

FineWeb-Edu keeps about one FineWeb document in fourteen. The first page shows
why rules alone cannot make that cut. It is long, its sentences end in full
stops, it uses enough common English words and repeats no line, so it passes
every rule; only a reader, human or model, notices that it sells spells and
prints a phone number twice (masked here). The second page shows the opposite
failure. It opens with a sales pitch for an interview platform and then turns
into a guide to hiring software engineers, coherent and useful to the right
reader. A rubric that asks for material to teach schoolchildren gives it
almost nothing, and the classifier that learned the rubric removes it. Keep
both pages in mind. Everything below is a way of deciding between pages like
these, at the scale of billions.

## Filters Without Labels
:label:`subsec_lm_quality_rules`

The oldest filters need no labelled data at all. Either people write down what
a bad page looks like, or a language model of clean text says how surprising a
page is.

### Heuristic Rules

Rules came first because they cost almost nothing and say why they fired.
DeepMind's Gopher models were trained on text filtered by two families of
them :cite:`rae2021scaling`. The *quality* rules bound simple statistics of a
document: between 50 and 100,000 words, a mean word length between 3 and 10
characters, few `#` symbols and ellipses, at most 90% of lines starting with a
bullet and at most 30% ending with an ellipsis, at least 80% of words
containing a letter, and at least two of eight common English words such as
*the*, *of* and *with*. The *repetition* rules catch pages that say the same
thing many times: duplicated lines and paragraphs, and too many characters
inside the most frequent 2-, 3- and 4-grams or inside 5- to 10-grams that occur
more than once. C4 :cite:`raffel2020exploring` added a rule about sentences:
once short lines, JavaScript warnings and cookie notices are set aside, a
document needs at least five sentences, and none of its lines may contain
"lorem ipsum" or a curly brace.

FineWeb :cite:`Penedo.Kydlicek.BenAllal.ea.2024` kept both families and added
rules of its own, chosen by training small models on the output of each
candidate filter: too few lines ending in terminal punctuation, too many lines
under 30 characters, too many characters in duplicated lines, and too many
line breaks per word. Its paper reports that the first three together remove
about 22% of the tokens and that it dropped C4's rule that removes every line
without terminal punctuation, which alone removed about 30%. Our thresholds
are those of datatrove, the code that produced the corpus, and
:numref:`tab_lm_quality_rules` lists the fifteen rules with their reason
codes.

:The heuristic rules, in the order `heuristic_filters` applies them. The thresholds are those of datatrove 0.10.1, the code FineWeb ran.
:label:`tab_lm_quality_rules`

| family | source | reason codes | what it catches |
|:--|:--|:--|:--|
| Gopher repetition | Rae et al. (2021), Table A1 | `gopher_dup_text`, `gopher_top_ngram`, `gopher_dup_ngram` | repeated lines and paragraphs; text dominated by one phrase or by repeated 5- to 10-grams |
| Gopher quality | Rae et al. (2021) | `gopher_words`, `gopher_word_len`, `gopher_symbols`, `gopher_bullets`, `gopher_ellipsis`, `gopher_alpha`, `gopher_stop_words` | fragments and walls of text, keyword lists, tables of numbers, hashtag clouds, text without English function words |
| C4 | Raffel et al. (2020), as FineWeb ran it | `c4_quality` | fewer than five sentences, placeholder text, code |
| FineWeb | Penedo et al. (2024) | `fw_line_punct`, `fw_short_lines`, `fw_dup_chars`, `fw_list_ratio` | menus, lists of links, navigation, repeated lines |

Each rule has a stable reason code, and `heuristic_filters` returns, in this
order, the codes of every rule that fires; an empty list keeps the document.
A stage that removes documents charges each one to the first rule that fires,
the charging rule of :numref:`sec_lm_crawl`, so the manifest can say how many
documents each rule removed while the cells can still count every rule that
fired.

```{.python .input #quality-filtering-rules-def}
%%tab pytorch
#@save
GOPHER_CFG = {  # reason code: thresholds, in the order FineWeb applies them
    # Gopher repetition (Rae et al., 2021, Table A1)
    'gopher_dup_text': dict(para=0.3, para_chars=0.2, line=0.3,
                            line_chars=0.2),
    'gopher_top_ngram': dict(top={2: 0.2, 3: 0.18, 4: 0.16}),
    'gopher_dup_ngram': dict(dup={5: 0.15, 6: 0.14, 7: 0.13, 8: 0.12,
                                  9: 0.11, 10: 0.10}),
    # Gopher quality
    'gopher_words': dict(min=50, max=100_000),  # words with a letter or digit
    'gopher_word_len': dict(min=3, max=10),     # their mean length
    'gopher_symbols': dict(max=0.1),     # '#' or ellipses per token
    'gopher_bullets': dict(max=0.9),     # lines starting with a bullet
    'gopher_ellipsis': dict(max=0.3),    # lines ending with an ellipsis
    'gopher_alpha': dict(min=0.8),       # tokens with a letter
    'gopher_stop_words': dict(min=2, words=('the', 'be', 'to', 'of', 'and',
                                            'that', 'have', 'with')),
    # C4 (Raffel et al., 2020) as FineWeb ran it: no line-punctuation rule
    'c4_quality': dict(min_sentences=5, min_line_words=3, max_word_len=1000),
    # FineWeb's own rules (Penedo et al., 2024; datatrove 0.10.1 values)
    'fw_line_punct': dict(min=0.12),     # lines ending in . ! ? (and CJK)
    'fw_short_lines': dict(max=0.67, chars=30),
    'fw_dup_chars': dict(max=0.01),      # characters in repeated lines
    'fw_list_ratio': dict(max=0.3)}      # newlines per token

#@save
def heuristic_filters(text, cfg=None):
    """The reason codes of the rules in cfg (default GOPHER_CFG) that fire
    on text, in cfg's order; an empty list keeps the document. Thresholds
    and their direction follow datatrove's filters; words are word and
    punctuation tokens (d2l._rule_words), so an unspaced script has few,
    very long words."""
    cfg = d2l.GOPHER_CFG if cfg is None else cfg
    words = d2l._rule_words(text)
    alnum = [w for w in words if not d2l._rule_is_symbol(w)]
    n_tok, n_chr = max(len(words), 1), max(len(text), 1)
    all_lines = text.splitlines() or ['']
    lines = [l for l in text.split('\n') if l.strip()] or ['']
    paras = re.split(r'\n{2,}', text.strip())
    breaks = re.split(r'\n+', text)
    stops = lambda c: len(set(words) & set(c['words']))
    checks = {
        'gopher_dup_text': lambda c: (
            d2l._rule_dups(paras)[0] / len(paras) > c['para'] or
            d2l._rule_dups(paras)[1] / n_chr > c['para_chars'] or
            d2l._rule_dups(breaks)[0] / len(breaks) > c['line'] or
            d2l._rule_dups(breaks)[1] / n_chr > c['line_chars']),
        'gopher_top_ngram': lambda c: any(
            d2l._rule_top_ngram(words, n) / n_chr > f
            for n, f in c['top'].items()),
        'gopher_dup_ngram': lambda c: any(
            d2l._rule_dup_ngram(words, n) / n_chr > f
            for n, f in c['dup'].items()),
        'gopher_words': lambda c: not c['min'] <= len(alnum) <= c['max'],
        'gopher_word_len': lambda c: bool(alnum) and not (
            c['min'] <= np.mean([len(w) for w in alnum]) <= c['max']),
        'gopher_symbols': lambda c: max(
            text.count('#'), text.count('...') + text.count('…')
        ) / n_tok > c['max'],
        'gopher_bullets': lambda c: np.mean([
            l.lstrip().startswith(('•', '-')) for l in all_lines]) > c['max'],
        'gopher_ellipsis': lambda c: np.mean([
            l.rstrip().endswith(('...', '…')) for l in all_lines]) > c['max'],
        'gopher_alpha': lambda c: np.mean([
            any(ch.isalpha() for ch in w) for w in words or ['']]) < c['min'],
        'gopher_stop_words': lambda c: stops(c) < c['min'],
        'c4_quality': lambda c: d2l._rule_c4(text, c),
        'fw_line_punct': lambda c: np.mean([
            l.endswith(d2l._TERMINAL_PUNCT) for l in lines]) < c['min'],
        'fw_short_lines': lambda c: np.mean([
            len(l) <= c['chars'] for l in lines]) > c['max'],
        'fw_dup_chars': lambda c: d2l._rule_dups(lines)[1] / max(
            len(text.replace('\n', '')), 1) > c['max'],
        'fw_list_ratio': lambda c: text.count('\n') / n_tok > c['max']}
    return [code for code, c in cfg.items() if checks[code](c)]
```

Each rule comes with a fixture, a short hand-written document that fires that
rule and no other. A plain paragraph about a river valley fires nothing, and
each fixture changes it just enough to break one rule: a sentence repeated
four times, every sentence turned into a bullet, a run of year numbers
inserted. A threshold that drifts, or a new rule that overlaps an old one,
fails this table at once.

```{.python .input #quality-filtering-rules-fixtures}
%%tab pytorch
#@save
_FIXTURE_BASE = (  # a plain paragraph that every rule keeps
    'The river rises in the hills to the north and flows south through '
    'the valley for about forty kilometres.\nFarmers along its banks have '
    'used the water to irrigate their fields of wheat and barley for '
    'centuries.\nIn spring the snow melts and the river floods the low '
    'meadows, which is why the old villages stand on higher ground.\nToday '
    'a small dam near the town holds back some of that water, and the lake '
    'behind it is popular with anglers and families with children.\nThe '
    'local museum has a fine exhibition that explains how the valley '
    'changed over the last two hundred years.')

#@save
QUALITY_FIXTURES = {  # reason code: a document that fires exactly that code
    'keep': _FIXTURE_BASE,
    'gopher_dup_text': '\n\n*\n\n'.join(_FIXTURE_BASE.split('\n')),
    'gopher_top_ngram': _FIXTURE_BASE.replace(
        'about forty kilometres.', 'about forty kilometres. ' +
        'cheap flights cheap flights cheap flights cheap flights '
        'cheap flights to the coast.'),
    'gopher_dup_ngram': _FIXTURE_BASE.replace(
        'centuries.', 'centuries. ' + ' '.join(
            ['Each autumn they store the grain in wooden barns near the '
             'mill.'] * 4)),
    'gopher_words': (
        'Rain fell on the hills overnight.\nThe river rose by morning.\n'
        'Farmers moved their sheep uphill.\nSchools stayed open as usual.\nBy '
        'noon the water began to drop.'),
    'gopher_word_len': (
        'Telecommunications infrastructure modernization necessitates '
        'comprehensive interdisciplinary collaboration.\nInternationalization '
        'requirements complicate straightforward implementation strategies '
        'considerably.\nSophisticated electromechanical instrumentation '
        'guarantees extraordinarily reproducible measurements.\nUnprecedented '
        'computational capabilities revolutionized pharmaceutical development '
        'methodologies.\nEnvironmental sustainability considerations '
        'increasingly influence architectural decisionmaking.\nParliamentary '
        'representatives overwhelmingly supported comprehensive '
        'transportation legislation.\nMicroelectronics manufacturers '
        'anticipate extraordinary semiconductor shortages throughout '
        'Southeast Asia.\nThe '
        'characteristically understated documentation and the '
        'multidisciplinary acknowledgements impressed reviewers.'),
    'gopher_symbols': _FIXTURE_BASE.replace(
        'centuries.', 'centuries. ' + ' '.join(
            '#' + t for t in ('farm wheat barley water valley river spring '
                              'snow flood dam lake museum anglers families '
                              'meadows villages hills history fishing').split()
        ) + '.'),
    'gopher_bullets': '\n'.join(
        '- ' + s.strip() + '.' for s in _FIXTURE_BASE.split('.') if s.strip()),
    'gopher_ellipsis': _FIXTURE_BASE.replace('.\n', ' and more...\n'),
    'gopher_alpha': _FIXTURE_BASE.replace(
        'centuries.', 'centuries. Yearly yields were ' +
        ' '.join(str(v) for v in range(1903, 1990, 2)) + '.'),
    'gopher_stop_words': (
        'Fresh bread comes out from our ovens every morning at six.\nOur '
        'bakers use stone mills for whole grain rye flour.\nCustomers love '
        'almond croissants filled from pastry cream made in house.\nWe bake '
        'wedding cakes for small or large parties on request.\nGluten free '
        'loaves are available on Fridays at both shops.\nGift vouchers make '
        'great presents for friends who enjoy good food.\nPlease ask staff '
        'about seasonal specials before you order online.'),
    'c4_quality': _FIXTURE_BASE.replace(
        'centuries.', 'centuries. Lorem ipsum dolor sit amet, consectetur.'),
    'fw_line_punct': _FIXTURE_BASE.replace('.\n', '\n').replace(
        'years.', 'years'),
    'fw_short_lines': (
        'The river rises in the hills\nto the north and flows south\nthrough '
        'the valley for about\nforty kilometres.\nFarmers along its banks '
        'have\nused the water to irrigate\ntheir fields of wheat and\nbarley '
        'for centuries.\nIn spring the snow melts and\nthe river floods the '
        'low\nmeadows, which is why the\nold villages stand on higher\n'
        'ground.\nToday a small dam near the\ntown holds back some of that\n'
        'water, and the lake behind\nit is popular with anglers\nand '
        'families with children.\nThe local museum has a fine\nexhibition '
        'that explains how\nthe valley changed over the\nlast two hundred '
        'years.'),
    'fw_dup_chars': _FIXTURE_BASE + '\nRead more about the valley.\n'
                    'Read more about the valley.',
    'fw_list_ratio': (
        'Blacksmiths tapestries fishermen\nOld town\nCathedrals architects '
        'musicians\nWaterfalls farmhouses sculptors\nNew bridge\nCarpenters '
        'woodcarvers gardeners\nStorytellers travellers lanterns.\nQuiet '
        'lanes\nMonasteries shepherdess chestnuts\nInnkeepers fishermen '
        'surveyors\nRiver mills\nTapestries musicians engineers\nArchitects '
        'sculptors vineyards\nStone walls\nFarmhouses gardeners orchards.\n'
        'Woodcarvers lanterns pilgrims\nOver the hills and\nTravellers '
        'chestnuts blacksmiths\nShepherdess surveyors cathedrals\nGreen '
        'fields\nFishermen engineers waterfalls\nMusicians vineyards '
        'carpenters.\nBright mornings\nSculptors orchards storytellers\n'
        'Gardeners pilgrims monasteries\nCold winters\nLanterns blacksmiths '
        'innkeepers\nChestnuts cathedrals tapestries\nLong summers\nSurveyors '
        'waterfalls architects.')}

rows = []
for code, text in d2l.QUALITY_FIXTURES.items():
    fired = d2l.heuristic_filters(text)
    rows.append(dict(fixture=code, fired=', '.join(fired) or '(keep)',
                     ok=fired == ([] if code == 'keep' else [code])))
print(pd.DataFrame(rows).to_string(index=False))
assert all(r['ok'] for r in rows), 'a fixture fires another rule'
assert set(d2l.QUALITY_FIXTURES) - {'keep'} == set(d2l.GOPHER_CFG)
```

Our rules are a reimplementation, and the code FineWeb actually ran is
datatrove's. A table computed once with datatrove 0.10.1, in an environment of
its own, compares the two on 2,000 of our crawl documents, for each family of
filters and for the whole sequence as FineWeb ran it.

```{.python .input #quality-filtering-rules-parity}
%%tab pytorch
PARITY_FILE = os.path.join('..', 'data', 'lm', 'datatrove-parity-v1.json')
if 'quality-results' in RERUN:  # computed once by the sources builder
    assert os.path.exists(PARITY_FILE), (
        f'{PARITY_FILE} is missing: run tools/build_lm_data_sources.py '
        'before regenerating quality-results')
    PARITY = json.load(open(PARITY_FILE))
else:
    PARITY = REG.get('parity') if REG else None
R['parity'] = PARITY
if PARITY is None:  # results registered before the table existed
    print('datatrove parity table pending (tools/build_lm_data_sources.py '
          f'writes {PARITY_FILE})')
else:
    say(f'{PARITY["n_docs"]:,} c0 documents, datatrove '
        f'{PARITY["datatrove_version"]}; per filter, the share of '
        'documents on which both implementations decide alike:')
    print(pd.DataFrame(PARITY['filters']).to_string(index=False))
    for reason, count in PARITY['disagreements'].items():
        print(f'  {count:4d}  {reason}')
```

The two implementations decide alike on more than 99% of the documents for
the repetition, C4 and FineWeb families, and on about 96% for the Gopher
quality rules and for the whole sequence. The disagreements have ordinary
causes. The largest, 58 documents that datatrove removes for too few
alphabetic words and we keep, comes from the word tokenizer: datatrove and our
regular expression split text into words differently, so a document near the
80% threshold can fall on either side. C4's 11 come from sentence splitting,
which datatrove does with NLTK's sentence splitter and we do at a full stop
followed by a space. In the sequence, 27 documents fail our line-punctuation
rule and pass datatrove's, because datatrove's C4 filter deletes lines before
FineWeb's rules see the text, and because datatrove accepts many more
characters as terminal punctuation. We keep our implementation, which is short
enough to read, and the table records where it departs from FineWeb's.

Now the rules on two corpora. The first is c0, the crawl documents of
:numref:`sec_lm_crawl`, which no quality filter has touched. The second is the
*audit stream*: 20,000 documents of FineWeb's output for the same dump, every
45th row of one of its files, without the rows of our archive file. FineWeb has
already applied these rules to the stream, so a rule that still fires there
marks a place where our implementation and FineWeb's disagree. The cell also
applies the rules to 500 documents in each of six languages from FineWeb-2,
FineWeb's multilingual successor, which filtered each language with rules
adapted to it.

```{.python .input #quality-filtering-rules-run}
%%tab pytorch
def run_rules(texts):
    with concurrent.futures.ProcessPoolExecutor(
            min(8, os.cpu_count()),
            mp_context=multiprocessing.get_context('spawn')) as ex:
        return list(ex.map(d2l.heuristic_filters, texts, chunksize=100))

t0 = time.time()
C0 = d2l.c0_documents()  # rebuilt from the pinned range (cached)
C0_CODES = dict(zip([d.id for d in C0], run_rules([d.text for d in C0])))
STREAM['codes'] = run_rules(STREAM.text.tolist())
S['codes'] = STREAM.codes[STREAM.role == 'stream'].to_list()
print(f'rules on {len(C0):,} c0 and {len(STREAM):,} stream documents: '
      f'{time.time() - t0:.0f} s')

def histogram(codes):
    first = collections.Counter(c[0] for c in codes if c)
    fired = collections.Counter(x for c in codes for x in c)
    return first, fired

rows = []
for name, codes in (('c0', list(C0_CODES.values())), ('stream', S.codes)):
    first, fired = histogram(codes)
    for code in d2l.GOPHER_CFG:
        rows.append(dict(corpus=name, code=code, first=first[code],
                         fired=fired[code]))
hist = pd.DataFrame(rows).pivot(index='code', columns='corpus',
                                values=['first', 'fired'])
hist = hist.reindex(list(d2l.GOPHER_CFG)).fillna(0).astype(int)
print(hist.to_string())
for name, codes in (('c0', list(C0_CODES.values())), ('stream', S.codes)):
    n_drop = sum(bool(c) for c in codes)
    print(f'{name}: {len(codes):,} in, {n_drop:,} removed '
          f'({n_drop / len(codes):.1%}), {len(codes) - n_drop:,} kept')
pairs = collections.Counter(
    (a, b) for c in C0_CODES.values() for i, a in enumerate(c)
    for b in c[i + 1:])
say('c0, rules that fire together most often:',
    '; '.join(f'{a} + {b} ({n:,})' for (a, b), n in pairs.most_common(3)))

FW2_LANGS = ('deu_Latn', 'swh_Latn', 'hin_Deva', 'cmn_Hani', 'jpn_Jpan',
             'tha_Thai')
rows = []
for lang in FW2_LANGS:
    codes = STREAM.codes[STREAM.role == f'fw2_{lang}']
    k, n = sum(bool(c) for c in codes), len(codes)
    lo, hi = d2l.wilson_ci(k, n)
    first = collections.Counter(c[0] for c in codes if c).most_common(1)
    other = sum(bool(set(c) - {'gopher_stop_words'}) for c in codes)
    rows.append(dict(language=lang, n=n, removed=k, rate=k / n, lo=lo,
                     hi=hi, top_rule=first[0][0] if first else '-',
                     rate_without_stop_words=other / n))
LANG_TABLE = pd.DataFrame(rows)
print('FineWeb-2 (each already filtered with its own language rules), '
      'under these English rules:')
print(LANG_TABLE.round(3).to_string(index=False))
R['rules'] = dict(hist=hist.reset_index().to_dict('list'),
                  c0_removed=sum(bool(c) for c in C0_CODES.values()),
                  stream_removed=int(S.codes.map(bool).sum()),
                  pairs=[[a, b, n] for (a, b), n in pairs.most_common(3)],
                  languages=LANG_TABLE.to_dict('records'))
```

On c0 the rules remove 2,256 of 3,288 documents, about 69%; on the stream they
remove 317 of 20,000, or 1.6%. The column `first` says which rule each removal
is charged to: duplicated text leads with 571 documents, followed by too few
alphabetic words (403) and too few words (322). The column `fired` tells a
different story. The line-punctuation rule fires on 1,150 documents and the
duplicated-character rule on 937, but both come late in the order, so their
documents are usually charged to an earlier rule. The pairs that fire together
most often, alphabetic words with line punctuation and line punctuation with
short lines, describe one kind of page: a column of short lines without full
stops, such as a menu, a list of links or a table of products. The drop
counts of a manifest therefore depend on the order of the rules, while the set
of documents kept does not.

On the stream, FineWeb's four rules never fire, which is what a faithful
reimplementation should give. What does fire are the word-based Gopher rules
(alphabetic words 108 times, word count 104), the same rules on which the
parity table found differences: most of the stream's 1.6% are documents near a
word-count threshold that the two tokenizers count differently.

The six languages show what English rules do elsewhere. German and Swahili
lose 96% and 93% of their documents, almost all to the stop-word rule, which
asks for two of eight English function words; without that rule they would
lose 28% and 16%. Chinese, Japanese and Thai lose every document. These
scripts do not separate words with spaces, so a run of characters between two
punctuation marks counts as one "word", a page has few long words and many
punctuation tokens, and the word-count or alphabetic-word rule fires on every
page. Hindi does put spaces between words and loses every document too, but
the cause lies in our code, not in the rules. Our word pattern uses Python's
`\w`, which does not count the vowel signs of Devanagari as word characters,
so each word breaks into fragments of one or two letters and the mean word
length falls below 3. Datatrove's word tokenizer keeps the signs with their
letters, so the 100% for Hindi measures our reimplementation, not the rules as
FineWeb ran them. Every one of these documents had already passed FineWeb-2's
own filters for its language. A removal rate of 100% on Thai measures the
rules, not the Thai web.

### Perplexity Filters
:label:`subsec_lm_quality_ppl`

A different cheap filter replaces the rules by a language model. CCNet
(Wenzek et al., 2020) trained a small n-gram model on Wikipedia in each
language, scored every crawled paragraph with it, and split the crawl into
thirds of low, medium and high perplexity; the low third resembles Wikipedia
most. The idea is appealing: a model of clean text should find clean text
predictable. We test it with a small released decoder, Qwen3-0.6B-Base, as
the model of clean text. The cell computes each document's mean negative
log-likelihood per token over its first 512 tokens, for 3,000 stream
documents, and prints the ten most and the ten least predictable.

```{.python .input #quality-filtering-ppl}
%%tab pytorch
N_PPL = 60 if SMALL else 3000  # stream documents scored
model, tok = d2l.load_hf(PPL_MODEL, dtype=DTYPE)
dev = next(model.parameters()).device
sample = S.sample(n=N_PPL, random_state=0).reset_index(drop=True)
ids = tok(sample.text.tolist(), truncation=True, max_length=MAX_LEN,
          add_special_tokens=False)['input_ids']
nll, t0 = np.empty(len(ids)), time.time()
for b in d2l._edu_batches([len(r) for r in ids], 8, 8 * MAX_LEN):
    T = len(ids[b[0]])  # right padding: the mask hides the tail
    X = torch.tensor([ids[i] + [0] * (T - len(ids[i])) for i in b],
                     device=dev)
    mask = torch.tensor([[1] * len(ids[i]) + [0] * (T - len(ids[i]))
                         for i in b], device=dev)
    with torch.inference_mode():
        z = model(input_ids=X, attention_mask=mask).logits[:, :-1].float()
        ce = F.cross_entropy(z.transpose(1, 2), X[:, 1:], reduction='none')
        nll[b] = ((ce * mask[:, 1:]).sum(1) / mask[:, 1:].sum(1)).cpu()
print(f'{len(ids):,} documents in {time.time() - t0:.0f} s, {d2l.lm_peak()}')
del model
d2l.lm_free()
sample['nll'] = nll
q = np.quantile(nll, [0.05, 0.25, 0.5, 0.75, 0.95])
print('NLL per token (nats), 5/25/50/75/95th percentiles:',
      ' '.join(f'{v:.2f}' for v in q))
order = sample.sort_values('nll')
for title, part in (('lowest', order.head(10)), ('highest', order.tail(10))):
    print(f'\n10 {title}:')
    for _, r in part.iterrows():
        print(f'  {r.nll:5.2f}  {r.text[:80]!r}')
print('\n' + ATTRIBUTION)
if not SMALL:
    R['ppl'] = dict(n=len(ids), quantiles=q.tolist(),
                    lowest=order.id.head(10).tolist(),
                    highest=order.id.tail(10).tolist())
```

The median document costs about 2.9 nats per token. The most predictable
documents are not the best ones. They are a forum's boilerplate ("Please be
mindful of our community standards") and a run of fluent, generic explainers:
"Operating systems are the software that manages ...", "In today's digital
age, a strong online presence is crucial ...", "Greetings, readers! I am
excited to delve into the topic of ...". Much of this reads as if a language
model had written it, and a language model finds such text predictable for the
obvious reason. The least predictable documents are mostly spun text ("As have
a peek at these guys you can see regarding the a lot more than pokies"),
together with a few honest informal pages, such as a comment on a garden blog
and a site of railway photographs. Perplexity measures resemblance to the
reference model's training data. Very low values flag boilerplate and
machine-written prose, very high values flag spun text and also unusual but
genuine writing, and neither end is a definition of quality.

## Classifiers on Model Labels
:label:`subsec_lm_quality_classifier`

The rules and the perplexity filter define quality by statistics that are easy
to compute and far from what we care about. The classifier recipe moves the
definition into words. Somebody writes a rubric, a large model applies it to a
sample of documents, a small model learns to predict the large model's score
from a document, and a threshold on the small model's prediction selects the
corpus. The large model reads perhaps half a million documents; the small one
reads billions. FineWeb-Edu is the best-known instance. Llama-3-70B-Instruct
scored 450,000 FineWeb documents under an additive rubric, a regression head
on a frozen encoder of 109M parameters learned the scores, and the corpus
keeps the documents whose rounded prediction is at least 3. DCLM (Li et al.,
2024) instead trained a fastText classifier to separate instruction-style text
and well-rated answers from a forum of explanations from random web pages, and
kept the top tenth of its pool by that classifier's score. Nemotron-CC (Su et
al., 2024) combines several classifiers and sorts documents into quality
buckets, which a mixture can weight separately instead of cutting at one
threshold.

The rubric is the definition. FineWeb-Edu's asks whether "the page has a high
educational value and could be useful in an educational setting for teaching
from primary school to grade school levels", and awards points one at a time:
a first for "some basic information relevant to educational topics", a third
if the extract "is appropriate for educational use and introduces key concepts
relevant to school curricula", a fifth only if it "is outstanding in its
educational value". A page about hiring engineers earns at most the first
point, however well it is written. `EDU_RUBRIC` holds the released prompt word
for word; the extract it scores is a document's first 1,500 characters.

```{.python .input #quality-filtering-label-def}
%%tab pytorch
#@save
EDU_RUBRIC = (  # the released annotation prompt (HuggingFaceFW/fineweb-edu-
    # llama3-annotations, ODC-By); the extract is the text's first 1,500
    # characters (the released prompts hold such a cut)
    'Below is an extract from a web page. Evaluate whether the page has a '
    'high educational value and could be useful in an educational setting '
    'for teaching from primary school to grade school levels using the '
    'additive 5-point scoring system described below. Points are '
    'accumulated based on the satisfaction of each criterion:\n\n'
    '- Add 1 point if the extract provides some basic information relevant '
    'to educational topics, even if it includes some irrelevant or '
    'non-academic content like advertisements and promotional material.\n'
    '- Add another point if the extract addresses certain elements '
    'pertinent to education but does not align closely with educational '
    'standards. It might mix educational content with non-educational '
    'material, offering a superficial overview of potentially useful '
    'topics, or presenting information in a disorganized manner and '
    'incoherent writing style.\n'
    '- Award a third point if the extract is appropriate for educational '
    'use and introduces key concepts relevant to school curricula. It is '
    'coherent though it may not be comprehensive or could include some '
    'extraneous information. It may resemble an introductory section of a '
    'textbook or a basic tutorial that is suitable for learning but has '
    'notable limitations like treating concepts that are too complex for '
    'grade school students. \n'
    '- Grant a fourth point if the extract highly relevant and beneficial '
    'for educational purposes for a level not higher than grade school, '
    'exhibiting a clear and consistent writing style. It could be similar '
    'to a chapter from a textbook or a tutorial, offering substantial '
    'educational content, including exercises and solutions, with minimal '
    'irrelevant information, and the concepts aren\'t too advanced for '
    'grade school students. The content is coherent, focused, and valuable '
    'for structured learning.\n'
    '- Bestow a fifth point if the extract is outstanding in its '
    'educational value, perfectly suited for teaching either at primary '
    'school or grade school. It follows detailed reasoning, the writing '
    'style is easy to follow and offers profound and thorough insights '
    'into the subject matter, devoid of any non-educational or complex '
    'content.\n\nThe extract:\n{extract}\n\nAfter examining the extract: \n'
    '- Briefly justify your total score, up to 100 words.\n'
    '- Conclude with the score using the format: “Educational score:  '
    '<total points>”\n\nRemember to assess from the AI Assistant '
    'perspective, utilizing web search knowledge as necessary. To evaluate '
    'the response in alignment with this additive scoring model, we’ll '
    'systematically attribute points based on the outlined criteria.')

#@save
def edu_label_scores(model, tokenizer, prompts, batch_size=16, *,
                     max_batch_tokens=24_576, prefix='Educational score: ',
                     extra_space=False):
    """(len(prompts), 6) float64: the probabilities of the next tokens
    '0' ... '5' after each prompt as a user turn of the chat template
    (non-thinking), the assistant's answer started with prefix. Prompts are
    sorted by length and left-padded into batches of at most batch_size
    rows and max_batch_tokens tokens with padding (a longer prompt runs
    alone); only the last position's logits are computed; rows come back
    in input order. The prompt is tokenized as if the digit followed the
    prefix, so a prefix ending in spaces splits as in the model's own text.
    With extra_space, the probability of digit d also counts the answers
    that write one more space first: p(d) + p(space) p(d | space). The six
    probabilities need not sum to one: the rest is what the model puts on
    other tokens."""
    ids, digits, space = d2l._edu_label_ids(tokenizer, prompts, prefix,
                                            extra_space)
    want = digits + ([space] if extra_space else [])
    pad = tokenizer.pad_token_id
    pad = tokenizer.eos_token_id if pad is None else pad
    device = next(model.parameters()).device

    def last_probs(ids):
        out = np.empty((len(ids), len(want)))
        for b in d2l._edu_batches([len(r) for r in ids], batch_size,
                                  max_batch_tokens):
            T = len(ids[b[0]])
            X = torch.tensor([[pad] * (T - len(ids[i])) + ids[i] for i in b],
                             device=device)
            mask = torch.tensor([[0] * (T - len(ids[i])) + [1] * len(ids[i])
                                 for i in b], device=device)
            pos = (mask.cumsum(-1) - 1).clamp(min=0)
            with torch.inference_mode():
                z = model(input_ids=X, attention_mask=mask, position_ids=pos,
                          use_cache=False, logits_to_keep=1).logits[:, -1]
                out[b] = z.float().log_softmax(-1)[:, want].exp().cpu().numpy()
        return out

    P = last_probs(ids)
    if not extra_space:
        return P
    Q = last_probs([x + [space] for x in ids])
    return P[:, :6] + P[:, 6:7] * Q[:, :6]

```

The released protocol asks the model to write a justification of up to 100
words and then "Educational score: N". Generating a hundred words per document
is the slow part, and parsing them is fragile, so we skip it.
`edu_label_scores` starts the model's answer with the text
`Educational score: ` and reads, from one forward pass, the probabilities of
the six digits as the next token. The label is the most probable digit, and
the *expected score* $\sum_d d\,p(d) / \sum_d p(d)$ keeps the model's
uncertainty as a number between 0 and 5. One detail matters more than it
looks. The released format puts two spaces before the digit, so a model may
write a second space and then the digit. With `extra_space` the probability
of digit $d$ is $p(d) + p(\textrm{space})\, p(d \mid \textrm{space})$; for
Qwen3.8-27B that second path holds about 15% of the probability, and ignoring
it would change about one label in a hundred.

The labeller is Qwen3.8-27B in RedHat's W4A16 build, with weights stored in 4
bits and activations computed in 16 (:numref:`sec_lm_gptq`), which fits on
one 24 GB RTX 4090 where the 54 GB of its bf16 weights would not. A vLLM
server holds one copy per GPU, and `edu_label_scores_served` sends it the same
token ids that `edu_label_scores` would feed a local model. Each server takes
one request at a time. That is slower than it needs to be, and it is
deliberate. A server that batches concurrent requests changes the order of its
floating-point additions with the batch, so the probabilities of a prompt
depend on which other prompts share its batch, and a near tie can flip its
label. One request at a time, every server returns the same bits for the same
prompt
(:numref:`sec_lm_deterministic_inference` explains why). A build reads the
labels, checks the provenance stored with them, and prints the pilot's check:
32 prompts scored again on another server, in reverse order.

```{.python .input #quality-filtering-label-live}
%%tab pytorch
LABEL_KEY = 'lm-edu-annot-22k-v1'
PROVENANCE = dict(labeller=f'{LABELLER}@{pins[LABELLER]}',
                  server=f'vLLM {d2l.EDU_LABELLER_LOCAL["vllm"]}',
                  score_prefix=SCORE_PREFIX, extra_space=EXTRA_SPACE)
if LABEL_KEY not in d2l.DATA_HUB:
    print(f'{LABEL_KEY} is not registered yet')
else:  # the labeller runs on four GPUs: a build reads its labels
    reg = pd.read_parquet(d2l.lm_artifact(LABEL_KEY))
    for k, v in PROVENANCE.items():
        assert (reg[k] == v).all(), f'the labels were made with another {k}'
    say('labels by', ', '.join(f'{k} {v!r}' for k, v in PROVENANCE.items()))
    one = reg[reg.qwen_probs_one.notna()]  # the pilot's check, stored
    P1, P = (np.stack(one[c].to_numpy()) for c in ('qwen_probs_one',
                                                    'qwen_probs'))
    print(f'{len(one)} prompts scored twice (another server, another '
          f'order): largest |dp| {np.abs(P1 - P).max():.4f}, same label '
          f'{np.mean(P1.argmax(1) == P.argmax(1)):.3f}')
```

The check agrees to the last printed digit. The labels were made once, in
three parts that `D2L_RERUN` can regenerate: `edu-label-pilot` labels the
2,000 test rows, and `edu-label-0` and `edu-label-1` the 18,000 training and
2,000 validation rows. Four servers, one per RTX 4090, labelled all 22,000
rows in about an hour. The pilot does one more thing. It asks the model for a
justification-first answer on each test row, under the released protocol, and
counts what the model writes between the colon and the digit: 1,970 of the
2,000 answers use one space and 30 use two, which fixed `SCORE_PREFIX` with
one space and kept the correction for a second. The same answers are a second
annotation of the test rows by the same model, which the label-noise ceiling
below needs.

```{.python .input #quality-filtering-label-regen}
%%tab pytorch
ANNOT_FILE = os.path.join(RERUN_DIR, 'lm-edu-annot-22k-v1.parquet')
LABEL_PARTS = {  # name: (splits, positions in the sample's order)
    'edu-label-pilot': ('test', range(20000, 22000)),
    'edu-label-0': ('train', range(0, 9000)),
    'edu-label-1': ('train+val', range(9000, 20000))}
# the released format "Educational score:  N", and the markdown forms that
# instruction models write ("**Educational score:** 3", "...: **3**")
SCORE_RE = re.compile(r'Educational score:([\s*]*)(\d)', re.IGNORECASE)
MERGED = False  # whether this run wrote ANNOT_FILE

def with_servers(work):
    """work(urls) with one labeller server per visible GPU, stopped after."""
    servers = d2l.edu_label_servers()
    try:
        t0 = time.time()
        with concurrent.futures.ThreadPoolExecutor(len(servers)) as ex:
            urls = list(ex.map(lambda s: s.start(), servers))
        print(f'{len(urls)} labeller servers ready in '
              f'{time.time() - t0:.0f} s')
        return work(urls)
    finally:
        for s in servers:
            s.stop()

def label_part(name, base, urls, tok):
    splits, positions = LABEL_PARTS[name]
    rows = base.iloc[list(positions)]
    assert rows.split.isin(splits.split('+')).all(), f'{name}: other rows'
    t0 = time.time()
    P = d2l.edu_label_scores_served(urls, tok, rows.prompt.tolist(),
                                    model=LABELLER, prefix=SCORE_PREFIX,
                                    extra_space=EXTRA_SPACE,
                                    concurrency=LABEL_CONCURRENCY)
    print(f'{name}: {len(rows):,} rows in {time.time() - t0:.0f} s on '
          f'{len(urls)} servers')
    part = pd.DataFrame(dict(row=rows.row.to_numpy(), qwen_probs=list(P)))
    part = part.assign(**PROVENANCE)  # checked when the parts are merged
    if name == 'edu-label-pilot':
        # a server batches concurrent requests, which changes the order of
        # the additions and so the bits, so each server takes one at a
        # time; then 32 prompts across the length range give the same bits
        # on the last server, in reverse order
        order = np.argsort(rows.prompt.str.len().to_numpy(), kind='stable')
        pick = order[np.linspace(0, len(rows) - 1, 32).astype(int)]
        one = d2l.edu_label_scores_served(
            urls[-1:], tok, rows.prompt.iloc[pick[::-1]].tolist(),
            model=LABELLER, prefix=SCORE_PREFIX, extra_space=EXTRA_SPACE,
            concurrency=1)[::-1]
        dp = np.abs(one - P[pick]).max()
        same = np.mean(one.argmax(1) == P[pick].argmax(1))
        print(f'32 prompts again, on one server in reverse order: largest '
              f'|dp| {dp:.4f}, same label {same:.3f}')
        assert dp <= 1e-6, 'the scores depend on the server or the order'
        alone = [None] * len(part)
        for j, i in enumerate(pick):
            alone[i] = one[j]
        part['qwen_probs_one'] = alone
        # the released protocol on every pilot row (justify, then score):
        # the second annotation of the test rows
        just = rows
        t0 = time.time()
        with concurrent.futures.ThreadPoolExecutor(16 * len(urls)) as ex:
            gens = list(ex.map(lambda i: d2l._vllm_post(
                urls[i % len(urls)], '/chat/completions', dict(
                    model=LABELLER, temperature=0, seed=0, max_tokens=512,
                    messages=[{'role': 'user',
                               'content': just.prompt.iloc[i]}],
                    chat_template_kwargs={'enable_thinking': False})),
                range(len(just))))
        gens = [g['choices'][0]['message'] for g in gens]
        gens = [g.get('content') or g.get('reasoning_content') or ''
                for g in gens]
        found = [SCORE_RE.search(g) for g in gens]
        seps = collections.Counter(m.group(1) for m in found if m)
        print(f'{len(gens):,} justification-first answers in '
              f'{time.time() - t0:.0f} '
              f's; unparsable {sum(m is None for m in found)}; text between '
              'the colon and the digit: ' + ', '.join(
                  f'{k!r} {n}' for k, n in seps.most_common()))
        part['qwen_just_score'] = pd.array(
            [int(m.group(2)) if m else None for m in found], dtype='Int64')
        part['qwen_just_sep'] = [m.group(1) if m else None for m in found]
        part['qwen_just_text'] = gens
    # a temporary name, then a rename: a part written by another process
    # (the parts may run at once) is either absent or complete
    path = d2l.lm_save_table(f'{name}.{os.getpid()}.tmp.parquet', part,
                             on='row', key=LABEL_KEY)
    os.replace(path, os.path.join(RERUN_DIR, ARTIFACTS[name][1]))

if RERUN & set(LABEL_PARTS):
    base = d2l._edu_annot_sample()  # the 22,000 sampled rows, unlabelled
    tok = transformers.AutoTokenizer.from_pretrained(
        LABELLER, revision=pins[LABELLER])
    with_servers(lambda urls: [label_part(name, base, urls, tok) for name
                               in sorted(RERUN & set(LABEL_PARTS))])
    parts = [os.path.join(RERUN_DIR, ARTIFACTS[n][1]) for n in LABEL_PARTS]
    if all(map(os.path.exists, parts)):  # merge; the same bytes from any run
        lab = pd.concat([pd.read_parquet(p) for p in parts],
                        ignore_index=True)
        for k, v in PROVENANCE.items():  # all parts under this protocol
            assert (lab[k] == v).all(), f'a part has another {k}: rerun it'
        assert lab.row.is_unique and len(lab) == len(base)
        lab['qwen_probs_one'] = [x if isinstance(x, np.ndarray) else None
                                 for x in lab.qwen_probs_one]
        full = base.merge(lab, on='row', validate='1:1')
        P = np.stack(full.qwen_probs.to_numpy())
        full['qwen_score'] = P.argmax(1)
        full['qwen_expected'] = (P * np.arange(6)).sum(1) / P.sum(1)
        tmp = ANNOT_FILE + f'.{os.getpid()}.tmp'
        full.sort_values('row').to_parquet(tmp, index=False,
                                           compression='zstd')
        os.replace(tmp, ANNOT_FILE)
        MERGED = True
        print(f'merged {len(full):,} rows into {ANNOT_FILE}, sha1 '
              f'{hashlib.sha1(open(ANNOT_FILE, "rb").read()).hexdigest()}')
```

Our labels come from a newer model than FineWeb-Edu's. The 22,000 documents
are a sample of the released annotations, so each one also carries the score
that Llama-3-70B-Instruct gave it in 2024 under the same prompt. Comparing the
two is easy and easy to misread. Agreement with one labeller cannot show which
labels are better: where two labellers disagree, either may be wrong, and a
third model would only add a third opinion. Here the comparison is
descriptive. It tells us how the two models use the scale, and whether they
order documents alike; exercise :exref:`ex_lm_quality-two-labellers` compares
the filters that the two labellers produce.

```{.python .input #quality-filtering-agreement}
%%tab pytorch
def load_annotations():
    """The labelled rows, sorted by row: this run's merge, else the
    registered file, else (a pilot run) the sample with the pilot's labels
    only."""
    if MERGED:
        print('labels: this run\'s merge')
        return pd.read_parquet(ANNOT_FILE)
    if LABEL_KEY in d2l.DATA_HUB:
        print(f'labels: {LABEL_KEY}')
        return pd.read_parquet(d2l.lm_artifact(LABEL_KEY))
    print('labels: the pilot\'s rows only')
    lab = pd.read_parquet(os.path.join(RERUN_DIR,
                                       ARTIFACTS['edu-label-pilot'][1]))
    df = d2l._edu_annot_sample().sort_values('row', ignore_index=True)
    df = df.merge(lab, on='row', how='left')
    P = np.stack([p if isinstance(p, np.ndarray) else np.full(6, np.nan)
                  for p in df.qwen_probs])
    df['qwen_score'] = pd.array(np.where(np.isnan(P[:, 0]), np.nan,
                                         np.nan_to_num(P).argmax(1)),
                                dtype='Int64')
    df['qwen_expected'] = (P * np.arange(6)).sum(1) / P.sum(1)
    return df

def boot_ci(stat, n, n_boot=1000, seed=0):
    """stat(rows) and its 95% percentile bootstrap interval over n rows."""
    rng = np.random.default_rng(seed)
    reps = [stat(rng.integers(n, size=n)) for _ in range(n_boot)]
    return stat(np.arange(n)), np.nanquantile(reps, [0.025, 0.975])

def kappa_ci(a, b, n_boot=1000, seed=0):
    """Binary kappa and its percentile bootstrap interval over rows."""
    a, b = np.asarray(a, dtype=bool), np.asarray(b, dtype=bool)
    return boot_ci(lambda i: d2l.cohen_kappa(a[i], b[i]), len(a), n_boot,
                   seed)

def spearman(x, y):
    return pd.Series(x).rank().corr(pd.Series(y).rank())

ANNOT = load_annotations()
lab = ANNOT[ANNOT.qwen_score.notna()]
q, l = lab.qwen_score.astype(int).to_numpy(), lab.llama_score.to_numpy()
e = lab.qwen_expected.to_numpy()
mass = np.stack(lab.qwen_probs.to_numpy()).sum(1)
say(f'{len(lab):,} rows; probability on the six digits (a second space '
    f'counted): median {np.median(mass):.3f}, 1st percentile '
    f'{np.quantile(mass, 0.01):.3f}')
assert np.median(mass) >= 0.9, 'the label is an argmax over little mass'
SHARES = pd.DataFrame({'score >=': range(1, 6),
                       'Qwen3.8-27B': [np.mean(q >= k) for k in range(1, 6)],
                       'Llama-3-70B': [np.mean(l >= k) for k in range(1, 6)]})
print('share of the rows at or above each score:')
print(SHARES.round(4).to_string(index=False))
rho, (rho_lo, rho_hi) = boot_ci(lambda i: spearman(e[i], l[i]), len(e))
kb, (kb_lo, kb_hi) = kappa_ci(q >= THRESH_LABEL, l >= THRESH_LABEL)
qwk = d2l.quadratic_kappa(q, l)
share_l = np.mean(l >= THRESH_LABEL)  # Qwen cut at Llama's share
km = d2l.cohen_kappa(e >= np.quantile(e, 1 - share_l), l >= THRESH_LABEL)
print(f'ranking: Spearman of the expected score against Llama-3-70B\'s '
      f'score {rho:.3f} [{rho_lo:.3f}, {rho_hi:.3f}]')
say(f'labels: exact agreement {np.mean(q == l):.3f}; quadratic-weighted '
    f'kappa {qwk:.3f}; binary (score >= {THRESH_LABEL}) kappa {kb:.3f} '
    f'[{kb_lo:.3f}, {kb_hi:.3f}]; with Qwen cut at Llama\'s share '
    f'({share_l:.3f}) {km:.3f}')
print('rows: Llama-3-70B score, columns: Qwen3.8-27B score')
print(pd.crosstab(pd.Series(l, name='llama'), pd.Series(q, name='qwen'))
      .reindex(index=range(6), columns=range(6), fill_value=0).to_string())

just = []
if 'qwen_just_text' in lab:  # the pilot's justification-first answers
    n_gen = int(lab.qwen_just_text.notna().sum())
    just = lab[lab.qwen_just_score.notna()]
    print(f'\n{n_gen:,} justification-first answers, {n_gen - len(just)} '
          'without a parsable score')
    assert n_gen - len(just) <= 0.1 * n_gen, 'the score format changed'
if len(just):
    jq = just.qwen_just_score.astype(int).to_numpy()
    dq = just.qwen_score.astype(int).to_numpy()
    jk, (jk_lo, jk_hi) = kappa_ci(jq >= THRESH_LABEL, dq >= THRESH_LABEL)
    say(f'the two protocols of Qwen3.8-27B on {len(jq):,} rows: '
        f'exact agreement {np.mean(jq == dq):.3f}, quadratic-weighted '
        f'kappa {d2l.quadratic_kappa(jq, dq):.3f}, binary kappa {jk:.3f} '
        f'[{jk_lo:.3f}, {jk_hi:.3f}]; share >= {THRESH_LABEL}: digit '
        f'probabilities {np.mean(dq >= THRESH_LABEL):.4f}, justification '
        f'first {np.mean(jq >= THRESH_LABEL):.4f}')
    print('rows: digit probabilities, columns: justification first')
    print(pd.crosstab(pd.Series(dq, name='digits'),
                      pd.Series(jq, name='justified'))
          .reindex(index=range(6), columns=range(6), fill_value=0)
          .to_string())
    R['protocol'] = dict(n=len(jq), unparsable=n_gen - len(jq),
                         exact=np.mean(jq == dq),
                         qwk=d2l.quadratic_kappa(jq, dq),
                         kappa_bin=[jk, jk_lo, jk_hi],
                         share_digits=np.mean(dq >= THRESH_LABEL),
                         share_justified=np.mean(jq >= THRESH_LABEL))
R['agreement'] = dict(n=len(lab), exact=np.mean(q == l), qwk=qwk,
                      kappa_bin=[kb, kb_lo, kb_hi], kappa_matched=km,
                      spearman_expected=[rho, rho_lo, rho_hi],
                      shares=SHARES.to_dict('list'),
                      digit_mass=[np.median(mass), np.quantile(mass, 0.01)])
```

The two models use the scale very differently. Llama-3-70B gives a score of 3
or more to about 9% of the documents, Qwen3.8-27B to about 1.4%; at 2 or more
the shares are 31% and 6%. The cross-tabulation shows a shift of about one
point. Of the 1,610 documents that Llama scored 3, Qwen scored 837 as 1, 634
as 2 and 138 as 3. Yet the two order documents alike: the Spearman
correlation of Qwen's expected score with Llama's score is about 0.76, with a
narrow interval. The binary agreement at the threshold, a Cohen's kappa of
0.24, is low because the two cut their scales at different places; cutting
Qwen's expected score so that it keeps Llama's share raises the kappa to 0.65.
The disagreement is mostly one of scale, not of ranking.

We keep the rubric's own cut, a score of 3 or more on Qwen's scale, as this
section's definition of "educational". On the stricter scale that makes a
stricter filter than FineWeb-Edu's, and the threshold cells below say how much
it keeps.

The second protocol measures something else, the stability of one model's
labels. Asked to justify first, the same Qwen3.8-27B gives the same score on
only half of the test rows, mostly by moving 1s to 0s, yet agrees on the
binary decision at 3 with a kappa of 0.66. Its scores run lower: 1.05% of the
rows reach 3, against 1.80% under the digit protocol. The next cell fixes
Qwen's scores as the training labels and splits the rows.

```{.python .input #quality-filtering-labels-choice}
%%tab pytorch
complete = ANNOT.qwen_score.notna().all()
TRAIN_LABEL = 'qwen_score'  # always: the released labels are compared only
OTHER_LABEL = 'llama_score'
print(f'training labels: {TRAIN_LABEL}' + (
    '' if complete else
    ' (Qwen labels cover the pilot rows only: this run is a pilot)'))
if not complete:
    assert not RERUN & {'quality-results', 'quality-flags'}, (
        'regenerate edu-label-0 and edu-label-1 before the results')
    ANNOT['qwen_score'] = ANNOT.qwen_score.fillna(-1).astype(int)
ANNOT['qwen_score'] = ANNOT.qwen_score.astype(int)
SPLIT = {k: ANNOT[ANNOT.split == k].reset_index(drop=True)
         for k in ('train', 'val', 'test')}
print({k: len(v) for k, v in SPLIT.items()})
```

The small model is a frozen embedder with a linear head. `embed_docs` encodes
each document's first 512 tokens with Qwen3-Embedding-0.6B and pools as in
:numref:`sec_lm_text_embeddings`: it takes the vector of the last token, pads
on the left so that the last token is always a real one, and adds no
instruction, which Qwen's embedders reserve for queries. The vectors have unit
length and are stored in 16 bits.

```{.python .input #quality-filtering-embed-def}
%%tab pytorch
#@save
def embed_docs(texts, model_id='Qwen/Qwen3-Embedding-0.6B', max_len=512,
               batch=64, pool='last', device=None):
    """(len(texts), d) float16 unit vectors of documents under a frozen
    embedder, in input order: d2l.load_encoder and d2l.embed_texts with
    the model's padding side (left for last-token pooling), the first
    max_len tokens, batches of similar length; bf16 on a GPU, fp32 on the
    CPU."""
    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
    side = d2l._EMBEDDERS.get(model_id, {}).get(
        'padding_side', 'left' if pool == 'last' else 'right')
    model, tok = d2l.load_encoder(
        model_id, dtype=torch.bfloat16 if device == 'cuda' else torch.float32,
        device=device, padding_side=side)
    Z = d2l.embed_texts(model, tok, list(texts), pooling=pool,
                        batch_size=batch, max_len=max_len, padding_side=side)
    del model
    d2l.lm_free()
    return Z.numpy().astype(np.float16)
```

The vectors of the 22,000 labelled documents and the 20,000 stream documents
are registered, together with the released classifier's scores for the same
documents. The crawl documents' vectors cannot be hosted, so a build computes
them once and caches them locally. Two live checks guard the registered
vectors: 64 test documents are embedded again, by `embed_docs` and by the
sentence-transformers library, and both must match the registered vectors to a
cosine of 0.999.

```{.python .input #quality-filtering-embed}
%%tab pytorch
EMB_KEY = 'lm-data-emb-qwen3emb-v1'
if 'quality-emb' in RERUN:
    t0 = time.time()
    annot = d2l.embed_docs(ANNOT.text, EMBEDDER, MAX_LEN)
    stream = d2l.embed_docs(S.text, EMBEDDER, MAX_LEN)
    t_emb = (time.time() - t0) / (len(ANNOT) + len(S))
    t0 = time.time()
    rel_annot = d2l.released_edu_scores(ANNOT.text)
    rel_stream = d2l.released_edu_scores(S.text)
    t_rel = (time.time() - t0) / (len(ANNOT) + len(S))
    print(f'embedded and scored {len(ANNOT) + len(S):,} documents: '
          f'{1e3 * t_emb:.1f} and {1e3 * t_rel:.1f} ms each, {d2l.lm_peak()}')
    arrays = dict(annot_rows=ANNOT.row.to_numpy(), annot=annot,
                  stream_ids=S.id.to_numpy().astype(str), stream=stream,
                  released_annot=rel_annot, released_stream=rel_stream,
                  ms_per_doc=np.array([1e3 * t_emb, 1e3 * t_rel]))
    def write(path):
        members = {}
        for k, v in arrays.items():
            buf = io.BytesIO()
            np.save(buf, v, allow_pickle=False)
            members[k + '.npy'] = buf.getvalue()
        d2l.lm_write_zip(path, members)
    d2l.lm_save_rerun(ARTIFACTS['quality-emb'][1], write)
EMB = dict(np.load(d2l.lm_artifact(
    EMB_KEY, ARTIFACTS['quality-emb'][1] if 'quality-emb' in RERUN
    else None)))
assert (EMB['annot_rows'] == ANNOT.row.to_numpy()).all()
assert (EMB['stream_ids'] == S.id.to_numpy().astype(str)).all()

n_check = 16 if SMALL else 64  # live: our vectors and sentence-transformers
texts = SPLIT['test'].text[:n_check].tolist()
live = d2l.embed_docs(texts, EMBEDDER, MAX_LEN).astype(np.float32)
reg = EMB['annot'][ANNOT.index[ANNOT.split == 'test'][:n_check]].astype(
    np.float32)
st = SentenceTransformer(EMBEDDER, revision=pins[EMBEDDER],
                         device='cuda' if GPU else 'cpu',
                         model_kwargs=dict(dtype=DTYPE))
st.max_seq_length = MAX_LEN
oracle = st.encode(texts, batch_size=16, normalize_embeddings=True)
del st
d2l.lm_free()
cos_reg, cos_st = (np.sum(live * x, 1) / np.linalg.norm(x, axis=1)
                   / np.linalg.norm(live, axis=1) for x in (reg, oracle))
say(f'{n_check} test texts, cosine of embed_docs to the registered '
    f'vectors: min {cos_reg.min():.4f}; to sentence-transformers: min '
    f'{cos_st.min():.4f}')
assert SMALL or min(cos_reg.min(), cos_st.min()) >= 0.999

C0_EMB_CACHE = pathlib.Path('..', 'data', 'lm', 'c0', 'emb-qwen3emb-v1.npz')
C0_SHA = [hashlib.sha256(d.text.encode('utf-8')).hexdigest() for d in C0]
C0_EMB_KEY = f'{EMBEDDER}@{pins[EMBEDDER]} max_len={MAX_LEN} pool=last'
cache = dict(np.load(C0_EMB_CACHE)) if C0_EMB_CACHE.exists() else {}
if cache and str(cache.get('made_by', '')) != C0_EMB_KEY:
    print(f'{C0_EMB_CACHE} was made by {cache.get("made_by")}: discarded')
    cache = {}
have = dict(zip(cache.get('sha256', []), cache.get('emb', [])))
missing = [i for i, h in enumerate(C0_SHA) if h not in have]
if missing and GPU:  # the c0 vectors: never hosted (rule P1), cached here
    t0 = time.time()
    Z = d2l.embed_docs([C0[i].text for i in missing], EMBEDDER, MAX_LEN)
    have.update(zip([C0_SHA[i] for i in missing], Z))
    np.savez(C0_EMB_CACHE, sha256=np.array(list(have)),
             emb=np.stack(list(have.values())), made_by=np.array(C0_EMB_KEY))
    print(f'embedded {len(missing):,} c0 documents in '
          f'{time.time() - t0:.0f} s (cached in {C0_EMB_CACHE})')
    missing = []
C0_EMB = None if missing else np.stack([have[h] for h in C0_SHA])
print('c0 vectors:', f'{len(C0):,} from the local cache or this run'
      if C0_EMB is not None else
      f'{len(missing):,} of {len(C0):,} not cached and no GPU: c0 scores '
      'are read from the registered flags')
```

Both match to a cosine of 0.9998 or better. The other line of the output
matters as well: 29 of the 64 texts are longer than 512 tokens, so our
classifier, like the released one, judges a long document by its beginning.

On fixed vectors the classifier is a linear model and trains in seconds.
`fit_quality_head` fits either a regression on the score, as the released
classifier does, or a logistic regression on the binary label "score at least
3", the two-class case of the softmax regression of :numref:`sec_softmax`.
Both use full-batch AdamW, starting from zero weights and the best constant.
`HashedBagClassifier` is the model of fastText (Joulin et al., 2017), the
classifier that DCLM used, written in PyTorch. The word unigrams and bigrams of
a document are hashed into a table of about a million embeddings, their
vectors are averaged, and the average is mapped to a logit. It needs neither a
GPU nor a pretrained model, which makes it the path for a reader without
either.

```{.python .input #quality-filtering-heads-def}
%%tab pytorch
#@save
def fit_quality_head(X, y, kind='regression', epochs=200, lr=1e-3,
                     weight_decay=0.0, seed=0):
    """A linear head on fixed features, full-batch AdamW: kind='regression'
    fits the score by squared error, kind='logistic' a 0/1 label by
    logistic loss. Starts at zero weights and the best constant; returns
    the nn.Linear and the loss per epoch."""
    if kind not in ('regression', 'logistic'):
        raise ValueError(f'kind is regression or logistic, not {kind!r}')
    torch.manual_seed(seed)
    X = torch.as_tensor(np.array(X, dtype=np.float32))
    y = torch.as_tensor(np.array(y, dtype=np.float32))
    head = nn.Linear(X.shape[1], 1)
    with torch.no_grad():
        head.weight.zero_()
        m = y.mean()
        head.bias.fill_(m if kind == 'regression' else
                        torch.logit(m.clamp(1e-6, 1 - 1e-6)))
    loss_fn = (nn.MSELoss() if kind == 'regression'
               else nn.BCEWithLogitsLoss())
    opt = torch.optim.AdamW(head.parameters(), lr=lr,
                            weight_decay=weight_decay)
    losses = []
    for _ in range(epochs):
        opt.zero_grad()
        loss = loss_fn(head(X)[:, 0], y)
        loss.backward()
        opt.step()
        losses.append(loss.item())
    return head, losses

#@save
class HashedBagClassifier(nn.Module):
    """fastText's classifier: word n-grams of the lower-cased,
    whitespace-split text, hashed into num_buckets rows (zlib.crc32) of an
    embedding table, averaged, and mapped to num_outputs logits."""
    def __init__(self, num_buckets=2**20, dim=64, ngrams=(1, 2),
                 num_outputs=1):
        super().__init__()
        self.num_buckets, self.ngrams = num_buckets, tuple(ngrams)
        self.bag = nn.EmbeddingBag(num_buckets, dim, mode='mean',
                                   sparse=True)
        self.out = nn.Linear(dim, num_outputs)
        nn.init.uniform_(self.bag.weight, -1 / dim, 1 / dim)  # as fastText
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def hash_ids(self, text):
        import zlib
        w = text.lower().split()
        return [zlib.crc32(' '.join(w[i:i + n]).encode('utf-8'))
                % self.num_buckets for n in self.ngrams
                for i in range(len(w) - n + 1)]

    def featurize(self, texts):
        """(ids, offsets) of a batch of texts for nn.EmbeddingBag."""
        ids = [self.hash_ids(t) for t in texts]
        offsets = np.cumsum([0] + [len(x) for x in ids[:-1]])
        return (torch.tensor([i for x in ids for i in x], dtype=torch.long),
                torch.tensor(offsets, dtype=torch.long))

    def forward(self, ids, offsets):
        return self.out(self.bag(ids, offsets))
```

The next cell trains the two heads on the training rows and the bag on their
texts, and reads the released classifier's registered scores. Each classifier
gets its own threshold $t^*$, the one with the best F1 on the validation rows;
the test rows serve only for the final numbers. Choosing the threshold on the
test labels would report the best of 400 tries as if it were a single one. The
validation and test rows are the released classifier's own hold-out, which we
reproduced from its training code (a stratified 90/10 split with seed 42; the
reproduced hold-out gives the confusion matrix on the classifier's card up to
rounding), so the comparison is out of sample for the released classifier too.

```{.python .input #quality-filtering-heads}
%%tab pytorch
def best_threshold(y, s, name=''):
    """The score threshold with the largest F1 (keep s >= t) on (y, s),
    among quantiles of s that keep 0.5% to 99% of the rows."""
    cand = np.unique(np.quantile(s, np.linspace(0.01, 0.995, 400)))
    f1 = [d2l.binary_f1(y, s >= t) for t in cand]
    i = int(np.argmax(f1))
    if i in (0, len(cand) - 1):
        print(f'WARNING: {name} t* is at the edge of the search grid')
    return float(cand[i])

def train_bag(texts, y, epochs=15, lr=0.5, batch=32, seed=0):
    """SGD on the hashed bag, summed loss (a step per document, as
    fastText), the rate decaying linearly to zero."""
    torch.manual_seed(seed)
    bag = d2l.HashedBagClassifier()
    feats = [torch.tensor(bag.hash_ids(t), dtype=torch.long) for t in texts]
    y = torch.as_tensor(np.asarray(y), dtype=torch.float32)
    opt = torch.optim.SGD(bag.parameters(), lr=lr)
    steps, step = epochs * math.ceil(len(texts) / batch), 0
    for _ in range(epochs):
        for b in torch.randperm(len(texts)).split(batch):
            for g in opt.param_groups:
                g['lr'] = lr * (1 - step / steps)
            ids = torch.cat([feats[i] for i in b])
            offsets = torch.tensor(np.cumsum([0] + [len(feats[i])
                                                   for i in b[:-1]]))
            loss = F.binary_cross_entropy_with_logits(
                bag(ids, offsets)[:, 0], y[b], reduction='sum')
            opt.zero_grad()
            loss.backward()
            opt.step()
            step += 1
    return bag

def bag_scores(bag, texts, batch=256):
    with torch.no_grad():
        return np.concatenate([torch.sigmoid(bag(*bag.featurize(
            texts[i:i + batch]))[:, 0]).numpy()
                               for i in range(0, len(texts), batch)])

X = {k: torch.tensor(EMB['annot'][(ANNOT.split == k).to_numpy()],
                     dtype=torch.float32) for k in SPLIT}
X['stream'] = torch.tensor(EMB['stream'], dtype=torch.float32)
Y = {k: SPLIT[k][TRAIN_LABEL].to_numpy() for k in SPLIT}
t0 = time.time()
reg_head, reg_loss = d2l.fit_quality_head(X['train'], Y['train'],
                                          'regression', epochs=500, lr=1e-2)
log_head, log_loss = d2l.fit_quality_head(
    X['train'], Y['train'] >= THRESH_LABEL, 'logistic', epochs=500, lr=1e-2)
print(f'heads: {time.time() - t0:.0f} s; final training loss: regression '
      f'{reg_loss[-1]:.3f} (MSE), logistic {log_loss[-1]:.3f}')
t0 = time.time()
bag = train_bag(SPLIT['train'].text.tolist(), Y['train'] >= THRESH_LABEL)
print(f'hashed bag: {time.time() - t0:.0f} s on the CPU')
released = {k: EMB['released_annot'][(ANNOT.split == k).to_numpy()]
            for k in SPLIT} | {'stream': EMB['released_stream']}
with torch.no_grad():
    SCORES = {'regression head': {k: reg_head(x)[:, 0].numpy()
                                  for k, x in X.items()},
              'logistic head': {k: torch.sigmoid(log_head(x)[:, 0]).numpy()
                                for k, x in X.items()}}
t0 = time.time()
SCORES['hashed bag'] = {k: bag_scores(bag, SPLIT[k].text.tolist())
                        for k in SPLIT}
bag_ms = 1e3 * (time.time() - t0) / len(ANNOT)
SCORES['hashed bag']['stream'] = bag_scores(bag, S.text.tolist())
SCORES['released'] = released
T_STAR = {m: best_threshold(Y['val'] >= THRESH_LABEL, s['val'], m)
          for m, s in SCORES.items()}
say('thresholds t* (best validation F1):',
    ', '.join(f'{m} {t:.3f}' for m, t in T_STAR.items()))

n_live = 16 if SMALL else 2000  # the released classifier, live
live = d2l.released_edu_scores(SPLIT['test'].text[:n_live].tolist())
gap = np.abs(live - released['test'][:n_live])
print(f'released classifier live on {n_live} test rows: largest difference '
      f'to the registered scores {gap.max():.3f}')
assert gap.max() < (0.05 if GPU else 0.1)
R['thresholds'] = T_STAR
```

The thresholds already say something. The regression head's $t^*$ is about
1.6, far below the label's 3. Each document has one label, but a linear head
on a frozen vector cannot tell every document apart, and under squared loss it
predicts the mean score of the documents it confuses with one another. With
only 1.4% of the labels at 3 or more, a group of such documents hardly ever
has a mean near 3. The head's predictions bunch between 0 and 2, and its
threshold has to be read on that scale. The released
classifier, trained on Llama's more generous scale, gets a threshold of about
2.9 against Qwen's labels. The last line checks the released classifier live
on the 2,000 test rows: its scores differ from the registered ones by at most
0.016.

The table compares the four classifiers on the test rows. The second half of
it has an F1 and an AUC against each labeller's binary label, and an
interval for the F1 against Qwen's.

```{.python .input #quality-filtering-clf-table}
%%tab pytorch
MAIN = 'regression head'  # "our classifier" from here on
test, rows, correct = SPLIT['test'], [], {}
n_params = {'regression head': EMB['annot'].shape[1] + 1,
            'logistic head': EMB['annot'].shape[1] + 1,
            'hashed bag': sum(p.numel() for p in bag.parameters()),
            'released': 109_483_009}
ms = {'regression head': EMB['ms_per_doc'][0],
      'logistic head': EMB['ms_per_doc'][0], 'hashed bag': bag_ms,
      'released': EMB['ms_per_doc'][1]}
for m, s in SCORES.items():
    pred = s['test'] >= T_STAR[m]
    row = dict(classifier=m, threshold=T_STAR[m],
               retained=pred.mean(), params=n_params[m], ms_per_doc=ms[m])
    for lab in (TRAIN_LABEL, OTHER_LABEL):
        y = test[lab].to_numpy() >= THRESH_LABEL
        row[f'F1 {lab}'] = d2l.binary_f1(y, pred)
        row[f'AUC {lab}'] = d2l.rank_auc(y, s['test'])
    if m in ('regression head', 'released'):
        ref = TRAIN_LABEL if m != 'released' else 'llama_score'
        row['MAE'] = np.mean(np.abs(s['test'] - test[ref].to_numpy()))
    correct[m] = pred == (test[TRAIN_LABEL].to_numpy() >= THRESH_LABEL)
    rows.append(row)
y = test[TRAIN_LABEL].to_numpy() >= THRESH_LABEL
pred = {m: s['test'] >= T_STAR[m] for m, s in SCORES.items()}
rng = np.random.default_rng(0)
boot = {m: [] for m in pred} | {'diff': []}
for _ in range(2000):  # resample rows, every classifier on the same rows
    i = rng.integers(len(y), size=len(y))
    for m, p in pred.items():
        boot[m].append(d2l.binary_f1(y[i], p[i]))
    boot['diff'].append(boot[MAIN][-1] - boot['released'][-1])
for row in rows:  # each test F1 on the training labels with its interval
    row['F1 lo'], row['F1 hi'] = np.quantile(boot[row['classifier']],
                                             [0.025, 0.975])
CLF = pd.DataFrame(rows)
say(f'{len(test):,} test rows (parameters: the trained part; the heads '
    f'also run the frozen {EMBEDDER}; F1 lo/hi: bootstrap interval of '
    f'F1 {TRAIN_LABEL})')
wide = CLF.round(3).rename(columns=lambda c: c.replace('_score', ''))
left = ['classifier', 'threshold', 'retained', 'params', 'ms_per_doc', 'MAE']
print(wide[left].to_string(index=False))
print(wide[[c for c in wide if c not in left[1:]]].to_string(index=False))
d_f1 = d2l.binary_f1(y, pred[MAIN]) - d2l.binary_f1(y, pred['released'])
lo, hi = np.quantile(boot['diff'], [0.025, 0.975])
acc = d2l.paired_bootstrap(correct[MAIN], correct['released'])
say(f'{MAIN} minus released, on {TRAIN_LABEL}: F1 {d_f1:+.3f} '
    f'[{lo:+.3f}, {hi:+.3f}] (paired bootstrap over rows); accuracy '
    f'{acc.mean:+.3f} [{acc.ci[0]:+.3f}, {acc.ci[1]:+.3f}]')
R['classifiers'] = CLF.drop(columns='ms_per_doc').to_dict('records')
R['f1_diff'] = [d_f1, lo, hi]
```

The test rows hold only 36 positives among 2,000, so every F1 carries an
interval about 0.12 wide on either side, and the table has to be read with
it. Against Qwen's labels the regression head reaches an F1 of 0.44 and the
logistic head 0.56; their intervals overlap, and we do not rank them. The
hashed bag is clearly worse, at 0.26. The area under the ROC curve, which
asks only whether positives score above negatives, is easier to read: the
three embedding classifiers all reach about 0.98, the bag 0.90. The trained
part of our heads is 1,025 numbers on top of a frozen 0.6B-parameter
embedder; the bag's 67 million parameters are almost all its hash table; the
released classifier's 109 million are a frozen encoder and a small head. The
embedder is the slow part, a few milliseconds per document on a GPU and
several times the released classifier's smaller encoder.

The comparison that matters is our head against the released classifier. The
paired bootstrap over test rows gives an F1 difference of $-0.02$ with an
interval from $-0.14$ to $+0.09$, and no difference in accuracy. On Qwen's
labels, a classifier trained on Llama's labels does as well as one trained on
Qwen's, which is what a shared ranking predicts: each threshold absorbs its
labeller's scale. On Llama's labels the released classifier is ahead (an F1
of 0.54 against 0.42), as it should be on the labels it learned.

### Label Noise and the F1 Ceiling
:label:`subsec_lm_quality_ceiling`

How high could any classifier score on these labels? The labels are noisy, and
a classifier that knew the truth would still be marked wrong wherever the
labeller was. Two short derivations make this precise.

Write $\pi$ for the share of positive labels, $\textrm{TPR}$ for the share of
positives that the classifier keeps (its recall), and $\rho$ for the share of
all documents it keeps, the *retained fraction*. The kept positives make up a
share $\pi\,\textrm{TPR}$ of all documents, so the precision is
$\pi\,\textrm{TPR}/\rho$, and the harmonic mean of precision $P$ and recall
$R$ simplifies to

$$
\textrm{F1} = \frac{2PR}{P + R}
= \frac{2\,(\pi\,\textrm{TPR}/\rho)\,\textrm{TPR}}{\textrm{TPR}\,(\pi + \rho)/\rho}
= \frac{2\pi\,\textrm{TPR}}{\pi + \rho}.
$$
:eqlabel:`eq_lm_quality_f1_retention`

The F1 depends on the threshold only through the recall and the retained
fraction. At a fixed recall it falls as the filter keeps more, and the best a
filter can do is to keep exactly the positives, $\rho = \pi$ with
$\textrm{TPR} = 1$.

Now let each document have a true label $y^*$, positive with probability
$\pi^*$, and let the labeller flip it with probability $\eta$ in either
direction, independently of the document. A classifier that predicts $y^*$
perfectly keeps the true positives, and a share $1 - \eta$ of them carries a
positive label, so its precision against the labels is $1 - \eta$. Its recall
is the share of positive labels that belong to true positives:

$$
R_{\max} = \frac{\pi^*(1-\eta)}{\pi^*(1-\eta) + (1-\pi^*)\,\eta},
\qquad
\textrm{F1}_{\max} = \frac{2\,(1-\eta)\,R_{\max}}{(1-\eta) + R_{\max}}.
$$
:eqlabel:`eq_lm_quality_f1_ceiling`

No classifier does better against these labels: the flips do not depend on
the document, so nothing in the document can predict them. The recall is
where a rare class suffers. Its denominator adds the negatives flipped to
positive, $(1-\pi^*)\eta$, to the true positives, $\pi^*(1-\eta)$, and when
positives are 1% of the documents, a flip rate of 0.5% already makes a third
of the positive labels flipped negatives that no classifier can find.

To use the ceiling we need $\eta$. Two annotations of the same documents that
flip independently, symmetrically and at the same rate disagree with
probability $d = 2\eta(1-\eta)$, so $\eta = (1 - \sqrt{1 - 2d})/2$, and the
observed share of positives, $\pi^*(1 - 2\eta) + \eta$, then gives $\pi^*$.
The same model makes a prediction that can be checked before any estimate:
both annotations have the same expected share of positives. We have two
candidates for the second annotation, and the check applies to both. Llama's
labels fail it by a wide margin, 9.1% positive against Qwen's 1.4%, a
difference of scale rather than of noise. Qwen's own justification-first
answers on the test rows fail it as well, less obviously: 1.05% positive
against 1.80% under the digit protocol. The ceiling that the cell computes
from them is therefore the value under a symmetric model that our data do not
support, and we read it beside a number that needs no model of the noise: the
F1 of one protocol scored against the other. The symmetric estimate is also
optimistic. Errors that both protocols of one model share are invisible to it,
so the true ceiling is lower. A classifier above the ceiling would therefore
be a warning sign: it would mean that the flips do depend on the document, and
that the classifier has learned its labeller's systematic errors.

The cell checks the identity at 50 thresholds and the ceiling against a
simulation of a million documents, then estimates $\eta$ and the ceiling.

```{.python .input #quality-filtering-ceiling}
%%tab pytorch
def f1_ceiling(pi, eta):
    """F1 against noisy labels of a perfect classifier of the true label:
    precision 1 - eta, recall pi(1 - eta) / (pi(1 - eta) + (1 - pi) eta)."""
    p = 1 - eta
    r = pi * (1 - eta) / (pi * (1 - eta) + (1 - pi) * eta)
    return 2 * p * r / (p + r)

s, y = SCORES[MAIN]['test'], test[TRAIN_LABEL].to_numpy() >= THRESH_LABEL
gap = 0.0
for t in np.quantile(s, np.linspace(0.02, 0.98, 50)):  # D1, the identity
    pred = s >= t
    pi, rho, tpr = y.mean(), pred.mean(), pred[y].mean()
    gap = max(gap, abs(d2l.binary_f1(y, pred) - 2 * pi * tpr / (pi + rho)))
print(f'F1 = 2 pi TPR / (pi + rho) at 50 thresholds: largest gap {gap:.1e}')
assert gap < 1e-12

rng, n, rows = np.random.default_rng(0), 10**6, []
for eta in (0.02, 0.05, 0.10):  # D2: closed form against simulation
    pi_true = 0.0926  # the released classifier card's share of scores >= 3
    truth = rng.random(n) < pi_true
    noisy = truth ^ (rng.random(n) < eta)
    prec, rec = noisy[truth].mean(), truth[noisy].mean()
    r_cf = pi_true * (1 - eta) / (pi_true * (1 - eta) + (1 - pi_true) * eta)
    se_p, se_r = (math.sqrt(v * (1 - v) / k) for v, k in (
        (1 - eta, truth.sum()), (r_cf, noisy.sum())))
    ok = abs(prec - (1 - eta)) < 3 * se_p and abs(rec - r_cf) < 3 * se_r
    rows.append(dict(eta=eta, closed_form=f1_ceiling(pi_true, eta),
                     simulated=2 * prec * rec / (prec + rec), within_3se=ok))
print(pd.DataFrame(rows).round(4).to_string(index=False))
assert all(r['within_3se'] for r in rows)

# the second annotation: the same labeller's justification-first scores
q1, q2 = dq >= THRESH_LABEL, jq >= THRESH_LABEL  # on the test rows
def implied(i):
    """eta from the disagreement d = 2 eta (1 - eta), the true prior from
    the observed one, and the ceiling they give."""
    d = np.mean(q1[i] != q2[i])
    eta = (1 - math.sqrt(max(1 - 2 * d, 0))) / 2
    pi_obs = np.mean(np.r_[q1[i], q2[i]])
    pi_true = (pi_obs - eta) / (1 - 2 * eta)
    return eta, f1_ceiling(pi_true, eta)
eta, ceil = implied(np.arange(len(q1)))
rng = np.random.default_rng(0)
boot = np.array([implied(rng.integers(len(q1), size=len(q1)))
                 for _ in range(1000)])
(eta_lo, ceil_lo), (eta_hi, ceil_hi) = np.quantile(boot, [0.025, 0.975], 0)
say(f'the two protocols disagree on {np.mean(q1 != q2):.4f} of '
    f'{len(q1):,} test rows: eta {eta:.4f} [{eta_lo:.4f}, {eta_hi:.4f}], '
    f'ceiling F1 {ceil:.3f} [{ceil_lo:.3f}, {ceil_hi:.3f}]')
p_q, p_l = (np.mean(ANNOT[c].to_numpy() >= THRESH_LABEL)
            for c in ('qwen_score', 'llama_score'))
say('equal symmetric noise implies equal shares of positives; '
    f'Qwen3.8-27B {p_q:.3f} against Llama-3-70B {p_l:.3f}; on the test '
    f'rows, digit protocol {q1.mean():.4f} against justification first '
    f'{q2.mean():.4f}')
up, down = int((q1 & ~q2).sum()), int((~q1 & q2).sum())
n_dis = up + down  # two-sided binomial test of an even split
p_split = min(1.0, 2 * sum(math.comb(n_dis, k) for k in range(
    min(up, down) + 1)) / 2**n_dis)
say(f'of the {n_dis} disagreements, {up} have digits >= 3 and justification '
    f'first < 3, {down} the reverse: an even split gives one this uneven '
    f'with probability {p_split:.4f}')
pp_f1 = lambda i: 2 * np.sum(q1[i] & q2[i]) / (q1[i].sum() + q2[i].sum())
rng2 = np.random.default_rng(1)
pp_lo, pp_hi = np.quantile([pp_f1(rng2.integers(len(q1), size=len(q1)))
                            for _ in range(1000)], [0.025, 0.975])
PP_F1 = pp_f1(np.arange(len(q1)))
say(f'one protocol scored against the other (no noise model): F1 2 x '
    f'{int(np.sum(q1 & q2))} / ({int(q1.sum())} + {int(q2.sum())}) = '
    f'{PP_F1:.3f} [{pp_lo:.3f}, {pp_hi:.3f}]')
f1s = CLF.set_index('classifier')[f'F1 {TRAIN_LABEL}']
say('measured test F1:', ', '.join(f'{m} {v:.3f}' for m, v in f1s.items()),
    '| all below the symmetric-model ceiling:', bool((f1s < ceil).all()),
    '| all below the protocol F1:', bool((f1s < PP_F1).all()))
R['ceiling'] = dict(identity_gap=gap, simulation=rows,
                    eta=[eta, eta_lo, eta_hi], f1=[ceil, ceil_lo, ceil_hi],
                    second='justification first', below=bool((f1s < ceil)
                                                             .all()))
```

The identity holds to rounding error, and the closed form agrees with the
simulation at three noise levels. The two protocols disagree on 19 of the
2,000 test rows, which under the symmetric model gives $\eta \approx 0.005$
and a ceiling F1 of about 0.80, with an interval from 0.68 to 0.89. The
disagreements themselves reject that model. In 17 of the 19, the digit
protocol scores 3 or more and the justification-first protocol less; if the
two directions were equally likely, a split at least this uneven would occur
less than once in a thousand. The number that needs no noise model comes from
the cross-tabulation of the two protocols printed earlier: 36 positives under
the digit protocol, 21 under the justification-first protocol, 19 under both.
One protocol scored against the other therefore reaches an F1 of
$2 \cdot 19 / (36 + 21) \approx 0.67$, which says how well Qwen3.8-27B
reproduces its own labels when it is asked a second way. All four classifiers
are below the 0.80, and their F1 values, 0.26 to 0.56, are below the 0.67 as
well, although the logistic head's interval, which reaches 0.68, does not
exclude it. Label noise thus accounts for part of the classifiers' errors,
and with 36 positives no estimate of it is precise. An exercise asks for the
ceiling under asymmetric noise.

## Choosing a Threshold
:label:`subsec_lm_quality_threshold`

A threshold turns a score into a decision, and the right one depends on what
the corpus is for. On the validation rows, $t^*$ maximizes the F1, which
weighs precision and recall alike. Read through
:eqref:`eq_lm_quality_f1_retention`, every threshold is also a retained
fraction, and the retained fraction is what a pretraining run cares about:
multiplied by the size of the crawl, it gives the tokens left to train on. A
run limited by data, which would otherwise repeat its tokens
(:numref:`sec_lm_budgets`), should keep more and accept a lower precision. A
run limited by compute, which will never read all of its data anyway, can
afford a high threshold and spend its steps on the best documents. The cell
plots precision, recall and F1 against the retained fraction on the test
rows, converts each classifier's $t^*$ into the shares of stream documents
and tokens it keeps, and sweeps the regression head's cut.

```{.python .input #quality-filtering-threshold}
%%tab pytorch
fig, axes = d2l.plt.subplots(1, 3, figsize=(10, 3), sharey=True)
y = test[TRAIN_LABEL].to_numpy() >= THRESH_LABEL
rho_max = max(np.mean(SCORES[m]['test'] >= T_STAR[m]) for m in SCORES)
xmax = min(1.0, max(0.4, rho_max + 0.1))  # every t* marker in view
curves = {}
for m in ('regression head', 'hashed bag', 'released'):
    s = SCORES[m]['test']
    ts = np.quantile(s, np.linspace(0.01, 0.995, 120))
    pred = s[None] >= ts[:, None]
    tp = (pred & y).sum(1)
    curves[m] = dict(rho=pred.mean(1), precision=tp / np.maximum(
        pred.sum(1), 1), recall=tp / y.sum(),
        f1=2 * tp / (y.sum() + pred.sum(1)))
for ax, metric in zip(axes, ('precision', 'recall', 'f1')):
    d2l.plot([c['rho'] for c in curves.values()],
             [c[metric] for c in curves.values()], 'retained fraction',
             metric if metric != 'f1' else 'F1', legend=list(curves)
             if metric == 'recall' else None, xlim=[0, xmax], axes=ax,
             fmts=('-', '--', ':'), legend_loc='lower right')
    for i, m in enumerate(curves):  # each classifier's t*, chosen on val
        s = SCORES[m]['test'] >= T_STAR[m]
        v = dict(precision=y[s].mean(), recall=s[y].mean(),
                 f1=d2l.binary_f1(y, s))[metric]
        ax.plot(s.mean(), v, 'o', color=f'C{i}')
d2l.plt.show()

tokens = S.token_count.to_numpy()  # GPT-2 tokens, as released
print(f'on the {len(S):,}-document stream (FineWeb-Edu kept 7.26% of this '
      'dump):')
rows = []
for m in ('regression head', 'logistic head', 'hashed bag', 'released'):
    keep = SCORES[m]['stream'] >= T_STAR[m]
    rows.append(dict(classifier=m, documents=keep.mean(),
                     tokens=tokens[keep].sum() / tokens.sum()))
keep = np.round(np.clip(SCORES['released']['stream'], 0, 5)) >= 3
rows.append(dict(classifier='released, int_score >= 3 (FineWeb-Edu)',
                 documents=keep.mean(), tokens=tokens[keep].sum()
                 / tokens.sum()))
RETAIN = pd.DataFrame(rows)
print(RETAIN.round(4).to_string(index=False))
yt = test[TRAIN_LABEL].to_numpy() >= THRESH_LABEL
rows = []  # the cut on our head's predicted score: what it keeps and finds
for t in sorted({1.0, 1.5, 2.0, 2.5, 3.0, round(T_STAR[MAIN], 3)}):
    keep, pred = SCORES[MAIN]['stream'] >= t, SCORES[MAIN]['test'] >= t
    rows.append(dict(cut=t, documents=keep.mean(),
                     tokens=tokens[keep].sum() / tokens.sum(),
                     precision=yt[pred].mean() if pred.any() else np.nan,
                     recall=pred[yt].mean(), F1=d2l.binary_f1(yt, pred)))
SWEEP = pd.DataFrame(rows)
say(f'{MAIN}, keep a document whose predicted score is at least the cut '
    f'(t* = {T_STAR[MAIN]:.3f}); stream shares, test precision and recall '
    f'against {TRAIN_LABEL} >= {THRESH_LABEL}:')
print(SWEEP.round(4).to_string(index=False))
print(f'of 1e12 tokens of such a dump, {MAIN} keeps about '
      f'{1e12 * RETAIN.tokens[0]:.2g}')
R['retention'] = RETAIN.to_dict('records')
R['sweep'] = SWEEP.to_dict('records')
```

Look at the retained shares first. FineWeb-Edu's own rule, the released
classifier's rounded score at least 3, keeps 7.1% of our stream's documents
and about 10% of its tokens, close to the 7.26% of documents that FineWeb-Edu
kept of the whole dump. Our head at $t^*$ keeps about 3% of the documents and
about 5% of the tokens, roughly half of FineWeb-Edu's tokens. This is the
strict scale at work: a score of 3 is several times rarer on Qwen's scale than
on Llama's, and a filter that looks for it keeps less. The token shares exceed
the document shares because the kept documents are longer than average.

The sweep shows the trade. A cut at 1.5 keeps 5% of the documents and finds
92% of the educational test rows, at a precision of 0.27; $t^*$ keeps 3% at a
recall of 0.67; a cut at 2.0 keeps 0.3% at a precision of 0.64 but finds only
a fifth of the positives. At 2.5 and above the head keeps nothing at all, for
the reason we met with $t^*$: its predictions never reach the label's own
threshold. Of a trillion tokens of such a dump, $t^*$ keeps about 50 billion.
A run that needs more tokens moves the cut down and accepts that most of what
it adds is, by the rubric, not educational.

The manifest therefore stores the score, not the decision. `lm-c1-docs-v1`
records each crawl document's predicted score to three decimals, and the
threshold is a parameter of its stage, so :numref:`sec_lm_data_mixtures` can
weight bands of the score as separate sources without labelling anything
again.

## Auditing Removals
:label:`subsec_lm_quality_audit`

A filter that removes 97% of the documents will remove some kinds of pages
more than others, and an audit asks which. It has two parts, a table of
removal rates by group, each with an interval, and a reading of removed
documents. We form three groupings. The country-code domain (.uk, .au, .in and
others, against .com) is a proxy for the variety of English, and a weak one:
much British writing lives on .com, and a .in domain says where a site is
registered, not who wrote it. The host class sorts hosts, by rules on their
names, into edu_gov, wiki, forum, news, blog, shop and other. The six
FineWeb-2 languages enter under the rules only, because the classifier was
trained on English and has nothing to say about Thai. All of it is measured
on FineWeb's output, so the audit cannot see what FineWeb's own filters and
deduplication removed before the stream was drawn.

The intervals need care. Documents from one site are alike: they share
templates, authors and topics, and a filter tends to decide them alike. A
bootstrap that resamples documents treats 3,000 pages of one site as 3,000
samples and reports an interval far too narrow. `audit_removals` resamples
whole registered domains within each group, a *cluster bootstrap*, and marks
groups with fewer than 100 documents or fewer than 10 domains as too few to
report.

```{.python .input #quality-filtering-audit-def}
%%tab pytorch
#@save
def audit_removals(df, keep, by, ref=None, n_boot=2000, cluster=None,
                   seed=0):
    """Removal rate per group of df[by] with a 95% percentile bootstrap
    interval, resampling whole clusters (df[cluster], e.g. the registered
    domain; each row its own cluster if None) within each group; with ref,
    the difference of each group's rate to ref's and its interval. keep is
    a boolean array or a column name. Groups with fewer than 100 rows or
    10 clusters are marked too_few."""
    removed = ~np.asarray(df[keep] if isinstance(keep, str) else keep,
                          dtype=bool)
    groups = df[by].to_numpy()
    clusters = (df[cluster].to_numpy() if cluster is not None
                else np.arange(len(df)))
    rng, reps, rows = np.random.default_rng(seed), {}, []
    for g in sorted(set(groups)):
        m = groups == g
        cl = pd.factorize(clusters[m])[0]
        G = cl.max() + 1
        r = np.bincount(cl, removed[m], G)
        n = np.bincount(cl, minlength=G)
        draws = rng.integers(G, size=(n_boot, G))
        reps[g] = r[draws].sum(1) / n[draws].sum(1)
        lo, hi = np.quantile(reps[g], [0.025, 0.975])
        rows.append(dict(group=g, n=int(m.sum()), n_clusters=int(G),
                         removal=r.sum() / n.sum(), lo=lo, hi=hi))
    out = pd.DataFrame(rows)
    if ref is not None:
        base = out.set_index('group').removal[ref]
        out['diff'] = out.removal - base
        d = np.array([np.quantile(reps[g] - reps[ref], [0.025, 0.975])
                      for g in out.group])
        out['diff_lo'], out['diff_hi'] = d[:, 0], d[:, 1]
    out['too_few'] = (out.n < 100) | (out.n_clusters < 10)
    return out
```

The cell applies three decisions to the stream: the rules alone, our
classifier alone at $t^*$, and both. For each group it reports the removal
rate and its difference to the reference group, .com for the country codes
and "other" for the host classes, with a cluster-bootstrap interval. It
repeats the audit on two halves of the stream, split by registered domain so
that no site appears in both, and keeps a finding only if its sign is the same
in both halves. The claim under test is the one an audit exists for: that the
combined filter removes some group at a rate that differs from the reference
by more than the interval, and that the classifier contributes to the
difference beyond the rules. That second part is read from the classifier-alone
arm. On FineWeb's output the rules remove so little that the classifier-alone
and combined arms nearly coincide.

Why should anyone care which groups lose more? A filter keyed on a
classifier's score is a covariate shift by construction: it changes the
distribution of the inputs a model trains on, in the direction of whatever the
classifier was trained to prefer (:numref:`sec_environment-and-distribution-shift`).
Gururangan et al. (2022) found that the filter used for GPT-3, a classifier
trained to tell curated text from random web pages, preferred articles from
the newspapers of American high schools in larger, wealthier and more urban
areas. Dodge et al. (2021) found that C4's list of banned words removed
disproportionately much text in African-American and Hispanic-aligned
English, and text that mentions gay and lesbian people. Neither filter was
designed to do this. Both did it because their notion of quality was learned
from text that was not drawn evenly from the people who write.

```{.python .input #quality-filtering-audit}
%%tab pytorch
CCTLDS = ('.com', '.uk', '.au', '.ca', '.in', '.ng', '.za', '.ph', '.ie',
          '.nz')
S['domain'] = S.url.map(d2l.registered_domain)
S['cctld'] = S.url.map(d2l._cctld)
S['host_class'] = S.url.map(d2l.host_class)
assert S.groupby('domain').half.nunique().max() == 1  # halves share no domain
S['keep_heur'] = ~S.codes.map(bool)
S['keep_clf'] = SCORES[MAIN]['stream'] >= T_STAR[MAIN]
S['keep_both'] = S.keep_heur & S.keep_clf
ARMS = {'heuristics': 'keep_heur', 'classifier': 'keep_clf',
        'both': 'keep_both'}
say('stream halves:', S.half.value_counts().sort_index().to_dict(),
    '| removal rates:', {a: round(float(1 - S[k].mean()), 3)
                         for a, k in ARMS.items()})

def audit(df):
    """The two groupings and three arms on df, one long table."""
    out = []
    for by, ref, sub in (('cctld', '.com', df[df.cctld.isin(CCTLDS)]),
                         ('host_class', 'other', df)):
        for arm, keep in ARMS.items():
            t = d2l.audit_removals(sub, keep, by, ref=ref, cluster='domain')
            out.append(t.assign(grouping=by, arm=arm,
                                ref=t.group == ref))
    return pd.concat(out, ignore_index=True)

AUDIT = {'all': audit(S)} | {h: audit(S[S.half == h]) for h in 'AB'}
full = AUDIT['all']
show = full[~full.ref & ~full.too_few]
for h in 'AB':
    other = AUDIT[h].set_index(['grouping', 'group', 'arm'])
    key = list(zip(show.grouping, show.group, show.arm))
    show = show.assign(**{f'diff_{h}': other['diff'].reindex(key).to_numpy(),
                          f'too_few_{h}': other.too_few.reindex(
                              key).to_numpy()})
show = show.assign(same_sign=np.sign(show.diff_A) == np.sign(show.diff_B))
cols = ['grouping', 'group', 'arm', 'n', 'n_clusters', 'removal', 'diff',
        'diff_lo', 'diff_hi', 'diff_A', 'diff_B', 'same_sign']
print(show[cols].round(3).rename(columns={'n_clusters': 'domains',
                                          'same_sign': 'same'})
      .to_string(index=False))
few = full[full.too_few & (full.arm == 'both')].group.tolist()
print('too few (under 100 documents or 10 domains):', ', '.join(few) or '-')

beyond = show[(show.arm == 'both') & ((show.diff_lo > 0) |
                                      (show.diff_hi < 0))]
clf = show[show.arm == 'classifier'].set_index(['grouping', 'group'])
c4 = [(g, b) for g, b in zip(beyond.grouping, beyond.group)
      if (clf.diff_lo[(g, b)] > 0 or clf.diff_hi[(g, b)] < 0) and
      clf.same_sign[(g, b)] and beyond.set_index(['grouping', 'group'])
      .same_sign[(g, b)]]
say('groups whose combined removal rate differs from the reference '
    'beyond its interval, with the classifier-only interval excluding 0 '
    'and the same sign in both halves:',
    ', '.join(f'{g}={b}' for g, b in c4) or 'none')

fig, (ax1, ax2) = d2l.plt.subplots(1, 2, figsize=(10, 3.5),
                                   gridspec_kw=dict(width_ratios=(2.2, 1)))
groups = show[show.arm == 'both'].group.tolist()
series = {}
for arm in ARMS:
    t = show[show.arm == arm].set_index('group').reindex(groups)
    series[arm] = (100 * t['diff'], 100 * t.diff_lo, 100 * t.diff_hi)
d2l.plot_intervals(groups, series, 'group (country-code domain or host '
                   'class)', 'removal rate minus reference (pp)',
                   reference=0, axes=ax1)
d2l.plot_intervals(list(LANG_TABLE.language), {'heuristics': (
    100 * LANG_TABLE.rate, 100 * LANG_TABLE.lo, 100 * LANG_TABLE.hi)},
    'FineWeb-2 language', 'removal rate (%)', axes=ax2)
d2l.plt.show()
R['audit'] = dict(table=show[cols].to_dict('records'), too_few=few,
                  c4_groups=[f'{g}={b}' for g, b in c4],
                  halves=S.half.value_counts().sort_index().to_dict())
```

The overall rates frame everything else: the rules remove 1.6% of the stream,
the classifier 97.2%. Two groups pass every test. The classifier removes .au
pages at a rate 1.9 points higher than .com pages, with an interval from 0.9 to
2.7 points and the same sign in both halves, and shop hosts at a rate 2.6
points higher than other hosts. A difference of two points matters more than it
seems, because what reaches the corpus is what the filter keeps: about 3% of
.com pages against about 1% of .au pages, 6 of the 544, so Australian sites
enter the corpus at roughly a third of the rate of .com sites. Six pages are
few, so the third is a rough figure; the interval on the difference is the
measured statement. The shops are the rubric working
as written, since product pages teach little. The rules have one finding of
their own: they remove news pages 1.4 points less often than other pages, in
both halves. The other country codes show differences whose intervals include
zero, and .ie, .ng, .ph and the wiki hosts are too few to report. These data
cannot say why .au differs. The country code is a weak proxy, and the finding
is a reason to read .au removals, not a conclusion about Australian English.

Rates say how much each group loses, not whether the losses are mistakes. For
that, someone has to read removed pages. We draw 100 documents that our
classifier removed and the rules kept, all with a score within 0.3 of $t^*$,
where a mistake is most likely, and 100 that the rules removed, whatever the
classifier decided. The labels do not come from people. They come from
Qwen3.8-27B again, this time as a judge with its reasoning switched on, under
a two-line rubric that asks a different question from the training rubric:
would a reader want this page in a general-purpose training corpus? The
reasoning has no budget. A budget of, say, a thousand tokens can stop the
model in mid-thought and force a verdict it has not reached, so each reply may
use up to 12,000 tokens of the 16,384-token context; in practice the replies
were short, and none was cut off. Unlike the labeller, the judge takes ten
requests per server at a time, which makes the review faster but not
reproducible to the bit: batching changes the arithmetic, as we saw above,
so a regenerated review can change the verdicts of pages near a tie. The
review is not independent of the filter it checks, since the same model wrote
the training labels, and a human reading of the same 200 pages would be the
stronger test. The cell reports the share of each arm labelled "drop", the
*removal precision*, with a Wilson interval, and prints five of the pages.

```{.python .input #quality-filtering-review}
%%tab pytorch
REMOVAL_RUBRIC = (  # what the review labels mean (two lines)
    'keep: a reader would want this page in a general-purpose training '
    'corpus (informative, coherent prose, code or data, in any register).\n'
    'drop: spam, boilerplate, link or product lists, machine-generated or '
    'broken text, or nothing a reader could learn from.')
rng = np.random.default_rng(0)
near = np.abs(SCORES[MAIN]['stream'] - T_STAR[MAIN]) <= 0.3
arms = {  # the stream passed FineWeb's rules, so few documents fail ours,
    # and of those the classifier keeps few: the second arm takes them all
    'classifier only': np.flatnonzero(S.keep_heur & ~S.keep_clf & near),
    'heuristics': np.flatnonzero(~S.keep_heur)}
picked = {a: np.sort(rng.choice(i, min(100, len(i)), replace=False))
          for a, i in arms.items()}
say('removed-document pools:', {a: len(i) for a, i in arms.items()},
    '| reviewed:', {a: len(i) for a, i in picked.items()})
REVIEW = pd.concat([S.iloc[i].assign(arm=a) for a, i in picked.items()],
                   ignore_index=True)
REVIEW['score'] = np.concatenate([SCORES[MAIN]['stream'][i]
                                  for i in picked.values()])
LABELS_KEY = 'lm-data-removal-labels-v1'
RUBRIC_SHA = hashlib.sha256(REMOVAL_RUBRIC.encode()).hexdigest()
REVIEW_ASK = (  # the judge's instruction; the document follows it
    'A quality filter removed the web page below from a corpus for '
    'pretraining a language model. Decide with this rubric whether the '
    'page should have been kept:\n\n' + REMOVAL_RUBRIC + '\n\nAnswer with '
    'one line of JSON and nothing else: {"label": "keep" or "drop", '
    '"rationale": "<one sentence>"}\n\nThe page (its first 3,000 '
    'characters):\n\n')

def judge_removals(urls, docs, per_server=10):
    """The judge's label and rationale for each of docs: thinking on,
    greedy, at most JUDGE_MAX_TOKENS new tokens, no reasoning budget."""
    import threading
    gates = [threading.Semaphore(per_server) for _ in urls]
    def one(i):
        body = dict(model=LABELLER, temperature=0, seed=0,
                    max_tokens=JUDGE_MAX_TOKENS,
                    messages=[{'role': 'user', 'content': REVIEW_ASK +
                               docs.text.iloc[i][:3000]}],
                    chat_template_kwargs={'enable_thinking': True})
        with gates[i % len(urls)]:
            d = d2l._vllm_post(urls[i % len(urls)], '/chat/completions',
                               body, timeout=3600.0)
        c = d['choices'][0]
        text = (c['message'].get('content') or '').split('</think>')[-1]
        found = re.findall(r'\{[^{}]*\}', text)
        try:
            v = json.loads(found[-1])
            assert v['label'] in ('keep', 'drop')
        except (IndexError, ValueError, KeyError, AssertionError):
            v = dict(label=None, rationale=text[-300:])
        return dict(id=docs.id.iloc[i], label=v['label'],
                    rationale=str(v.get('rationale', '')),
                    finish=c['finish_reason'],
                    tokens=d['usage']['completion_tokens'])
    with concurrent.futures.ThreadPoolExecutor(per_server * len(urls)) as ex:
        return pd.DataFrame(list(ex.map(one, range(len(docs)))))

if 'removal-labels' in RERUN:
    t0 = time.time()
    got = with_servers(lambda urls: judge_removals(urls, REVIEW))
    print(f'judged {len(got)} removals in {time.time() - t0:.0f} s; reply '
          f'tokens median {got.tokens.median():.0f}, largest '
          f'{got.tokens.max()}; cut off {(got.finish == "length").sum()}, '
          f'without a label {got.label.isna().sum()}')
    assert got.label.notna().all(), 'a reply without a verdict'
    got = got.assign(labeller=f'{LABELLER}@{pins[LABELLER]} thinking, '
                     f'max_tokens {JUDGE_MAX_TOKENS}',
                     rubric_sha256=RUBRIC_SHA)
    d2l.lm_save_rerun(ARTIFACTS['removal-labels'][1], lambda f:
                      d2l.write_jsonl(f, json.loads(got.to_json(
                          orient='records'))))
if LABELS_KEY not in d2l.DATA_HUB and 'removal-labels' not in RERUN:
    print(f'{LABELS_KEY} not registered: removal precision pending')
    R['review'] = None
else:
    labels = d2l.read_jsonl(d2l.lm_artifact(
        LABELS_KEY, ARTIFACTS['removal-labels'][1]
        if 'removal-labels' in RERUN else None))
    assert (labels.rubric_sha256 == RUBRIC_SHA).all(), 'another rubric'
    REVIEW = REVIEW.merge(labels[['id', 'label', 'rationale', 'labeller']],
                          on='id', how='left', validate='1:1')
    assert REVIEW.label.notna().all(), (  # t* or the heads have changed
        f'{REVIEW.label.isna().sum()} sampled documents have no label: '
        'label the new sample and register it as the next version')
    rows = []
    for a, t in REVIEW.groupby('arm'):
        k = int((t.label == 'drop').sum())
        lo, hi = d2l.wilson_ci(k, len(t))
        rows.append(dict(arm=a, n=len(t), drop=k, precision=k / len(t),
                         lo=lo, hi=hi))
    PREC = pd.DataFrame(rows)
    say(f'labeller: {REVIEW.labeller.iloc[0]}; removal precision (share '
        'labelled "drop"):')
    print(PREC.round(3).to_string(index=False))
    for _, r in REVIEW.sample(5, random_state=0).iterrows():
        say(f'\n[{r.arm}, score {r.score:.2f}, label {r.label}] '
            f'{r.rationale}')
        print(textwrap.fill(r.text[:300].replace('\n', ' ') + ' …', 100,
                            initial_indent='  ', subsequent_indent='  '))
    print('\n' + ATTRIBUTION)
    say(f'judge replies: median {labels.tokens.median():.0f} tokens, largest '
        f'{labels.tokens.max()}, cut off at max_tokens '
        f'{(labels.finish == "length").sum()} of {len(labels)}')
    R['review'] = PREC.to_dict('records')
    R['review_judge'] = dict(labeller=REVIEW.labeller.iloc[0],
                             tokens_median=float(labels.tokens.median()),
                             tokens_max=int(labels.tokens.max()),
                             cut_off=int((labels.finish == 'length').sum()))
long = S[S.keep_heur & (S.text.str.len() >= 1500)]
R['opening'] = dict(low_score_id=long.id.iloc[int(np.argmin(
    SCORES[MAIN]['stream'][long.index]))])
if R['review'] is not None:
    kept = REVIEW[(REVIEW.label == 'keep') & (REVIEW.arm == 'classifier only')]
    R['opening']['kept_label_id'] = (kept.sort_values('score').id.iloc[0]
                                     if len(kept) else None)
```

The two arms differ sharply. Of the classifier's removals near its threshold,
the judge would have dropped about 1 in 10 (interval 6 to 19 in 100); of the
rules' removals, about 70% (interval 60% to 78%). Near its threshold the
classifier mostly removes pages that a general corpus would keep, like the
hiring guide that opened the section. The rules mostly remove pages that
nobody would miss, such as the product description of a pair of jeans, though
not always: an encyclopedia entry on Erisichthon of Greek mythology and a
massage therapist's professional profile were removed by rules and judged
worth keeping. The classifier is doing what it was trained to do. It looks for
pages that teach schoolchildren, and by that rubric the pages it removes are
removed correctly; the judge's rubric is broader. The gap between 0.11 and
0.70 is the gap between two definitions of quality, and choosing between them
is a decision about the corpus, not about the classifier.

The last comparison puts our classifier and the released one side by side on
the whole stream, each at its own threshold.

```{.python .input #quality-filtering-agree-stream}
%%tab pytorch
a = SCORES[MAIN]['stream'] >= T_STAR[MAIN]
b = SCORES['released']['stream'] >= T_STAR['released']
k, (k_lo, k_hi) = kappa_ci(a, b)
say(f'on the stream, {MAIN} and the released classifier keep '
    f'{a.mean():.3f} and {b.mean():.3f}; they agree on '
    f'{np.mean(a == b):.3f} of the documents, Cohen\'s kappa {k:.3f} '
    f'[{k_lo:.3f}, {k_hi:.3f}]')
R['stream_kappa'] = [k, k_lo, k_hi]
```

They keep 2.9% and 3.6% of the documents and agree on 96.7% of them, but most
of that agreement is on documents both remove. Cohen's kappa, which discounts
the agreement expected by chance, is about 0.47. Two classifiers with the same
test F1 select noticeably different corpora.

## The Filter in the Pipeline
:label:`subsec_lm_quality_pipeline`

The filters enter the corpus of :numref:`sec_lm_crawl` as two stages of its
manifest. The `heuristics` stage removes documents and charges each one to the
first rule that fires. The `edu_score` stage removes nothing. It attaches the
score and records the threshold, the head's SHA-256 and the embedder's
revision as parameters, so a later stage, or a reader, can apply any threshold
and know exactly which model produced the scores. The flags file
`lm-c1-docs-v1` holds, for each crawl document, its id, its reason codes and
its score, and no text.

```{.python .input #quality-filtering-c1}
%%tab pytorch
FLAGS_KEY = 'lm-c1-docs-v1'
HEAD_KEY, HEAD_FILE = 'lm-quality-head-v1', 'lm-quality-head-v1.npz'
head_arrays = {'weight.npy': reg_head.weight.detach().numpy()[0],
               'bias.npy': reg_head.bias.detach().numpy()}
head_bytes = {}
for k, v in head_arrays.items():
    buf = io.BytesIO()
    np.save(buf, v.astype(np.float32), allow_pickle=False)
    head_bytes[k] = buf.getvalue()
head_sha = hashlib.sha256(b''.join(head_bytes[k] for k in sorted(
    head_bytes))).hexdigest()  # of the registered file's exact weights
if C0_EMB is not None:
    with torch.no_grad():
        c0_score = reg_head(torch.tensor(C0_EMB, dtype=torch.float32)
                            )[:, 0].numpy()
else:  # the CPU path: the scores of the registered flags
    assert FLAGS_KEY in d2l.DATA_HUB and 'quality-flags' not in RERUN, (
        'c0 scores need c0 vectors: run once on a GPU')
    with gzip.open(d2l.lm_artifact(FLAGS_KEY), 'rt') as f:
        reg_rows = {r['id']: r for r in map(json.loads, f)}
    c0_score = np.array([reg_rows[d.id]['edu_score'] for d in C0])
    live = d2l.embed_docs([d.text for d in C0[:8]], EMBEDDER, MAX_LEN)
    with torch.no_grad():
        s8 = reg_head(torch.tensor(live, dtype=torch.float32))[:, 0].numpy()
    gap8 = np.abs(s8 - c0_score[:8]).max()
    print('live check on 8 c0 documents (CPU, fp32): largest score '
          f'difference to the registered flags {gap8:.3f}')
    assert gap8 <= 0.05, 'the registered flags come from other vectors'
FLAG_ROWS = [dict(id=d.id, codes=C0_CODES[d.id],
                  edu_score=round(float(v), 3)) for d, v in zip(C0, c0_score)]
kept = [d for d in C0 if not C0_CODES[d.id]]
first = collections.Counter(C0_CODES[d.id][0] for d in C0 if C0_CODES[d.id])
c0m = json.load(open(d2l.lm_artifact('lm-data-crawl-results-v1')))[
    'c0_manifest']
assert d2l.docs_hash(C0) == c0m['stages'][-1]['out_hash']
m = d2l.CorpusManifest('c1', c0m['sources'], c0m['stages'])
m.add_stage('heuristics', dict(cfg=d2l.GOPHER_CFG, charge='first rule'),
            len(C0), len(kept), first, d2l.docs_hash(C0), d2l.docs_hash(kept),
            impl='d2l.heuristic_filters')
m.add_stage('edu_score', dict(threshold=round(T_STAR[MAIN], 4),
                              head=MAIN, head_sha256=head_sha,
                              embedder=f'{EMBEDDER}@{pins[EMBEDDER]}',
                              max_len=MAX_LEN, labels=TRAIN_LABEL),
            len(kept), len(kept), {}, d2l.docs_hash(kept),
            d2l.docs_hash(kept), impl='d2l.embed_docs + linear head')
for s in m.stages[-2:]:
    say(json.dumps({k: s[k] for k in ('name', 'n_in', 'n_out', 'drops')}))
above = sum(C0_CODES[d.id] == [] and v >= T_STAR[MAIN]
            for d, v in zip(C0, c0_score))
say(f'c0 documents kept by the rules with edu_score >= t*: {above:,} of '
    f'{len(kept):,} (the flag keeps the score; the threshold is a '
    'parameter)')
if 'quality-flags' in RERUN:
    d2l.lm_save_rerun(ARTIFACTS['quality-flags'][1],
                      lambda f: d2l.write_docs(f, FLAG_ROWS))
elif C0_EMB is not None:  # GPU: live scores against the registered flags
    with gzip.open(d2l.lm_artifact(FLAGS_KEY), 'rt') as f:
        reg_rows = {r['id']: r for r in map(json.loads, f)}
    codes_same = all(reg_rows[r['id']]['codes'] == r['codes']
                     for r in FLAG_ROWS)
    diff = np.array([abs(reg_rows[r['id']]['edu_score'] - r['edu_score'])
                     for r in FLAG_ROWS])
    flip = sum((reg_rows[r['id']]['edu_score'] >= T_STAR[MAIN]) !=
               (r['edu_score'] >= T_STAR[MAIN]) for r in FLAG_ROWS)
    say(f'against {FLAGS_KEY}: reason codes '
        f'{"identical" if codes_same else "DIFFER"}; edu_score largest '
        f'difference {diff.max():.3f}; {flip} decisions '
        'flip (each within that difference of t*)')
    assert codes_same and diff.max() <= 0.01  # bf16 tolerance (spec)
    assert flip <= np.sum(np.abs(c0_score - T_STAR[MAIN]) <= diff.max())
R['c1_manifest'] = m.to_dict()
```

The rules keep 1,032 of the 3,288 crawl documents, and 20 of those score at or
above $t^*$, about 2%, in line with the stream. The last cell compares every
number this run computed with the registered results, and the manifest and the
head's weights with their registered copies.

```{.python .input #quality-filtering-results}
%%tab pytorch
def rounded(x):
    """Floats to 4 decimals, numpy values to Python, NaN to None."""
    if isinstance(x, dict):
        return {str(k): rounded(v) for k, v in x.items()}
    if isinstance(x, (list, tuple, np.ndarray)):
        return [rounded(v) for v in x]
    if isinstance(x, (np.integer, np.floating, np.bool_)):
        x = x.item()
    if isinstance(x, float):
        return None if x != x else round(x, 4)
    return x

RES_KEY, RES_FILE = ARTIFACTS['quality-results']
results = rounded(R)
blob = json.dumps(results, sort_keys=True, indent=1).encode('utf-8')
if 'quality-results' in RERUN:
    d2l.lm_save_rerun(RES_FILE, lambda f: pathlib.Path(f).write_bytes(blob))
    d2l.lm_save_rerun(HEAD_FILE, lambda f: d2l.lm_write_zip(f, head_bytes))
else:
    def comparable(x):
        """x without the perplexity results (GPU only) and the head's hash
        (the weights' last bits depend on the CPU's thread count; they are
        compared to 1e-5 below)."""
        if isinstance(x, dict):
            return {k: comparable(v) for k, v in x.items()
                    if k not in ('ppl', 'head_sha256')}
        return [comparable(v) for v in x] if isinstance(x, list) else x
    reg = json.load(open(d2l.lm_artifact(RES_KEY)))
    live, old = comparable(results), comparable(reg)
    a = d2l.json_agreement(live, old)
    same = old['c1_manifest'] == live['c1_manifest']
    say(f'against {RES_KEY}: {a["equal"]}/{a["values"]} values equal, '
        f'largest numeric difference {a["max_diff"]:.2g}; c1 manifest '
        f'{"identical" if same else "DIFFERS"}')
    ALLOWED = set()  # paths whose non-numeric values may differ: none
    bad = [k for k in a['mismatches'] + a['missing'] if k not in ALLOWED]
    for k in bad[:20]:
        print('  differs or one-sided:', '/'.join(map(str, k)))
    head = dict(np.load(d2l.lm_artifact(HEAD_KEY)))
    w_gap = np.abs(head['weight'] - head_arrays['weight.npy']).max()
    bits = reg['c1_manifest'] == results['c1_manifest']  # with the hash
    print(f'head weights against {HEAD_KEY}: largest difference {w_gap:.1e}'
          f' (bit-identical: {bits})')
    assert same and not bad and a['max_diff'] <= 0.005 and w_gap < 1e-5
```

Every filter in this section removes documents because they differ from
something: from English function words, from a reference model's training
data, from what Llama or Qwen think a schoolteacher would use. What survives
leans toward the annotator's taste. The audit can show where it leans, but
not whether a model trained on the result is better or worse for it. That
takes training runs with everything but the data held fixed, the method of
:numref:`sec_lm_data_mixtures`, whose web text is FineWeb-Edu as released,
filtered by the labels of Llama-3-70B.

## Summary

A quality filter decides which part of the web a model learns from, and it
follows a definition of quality that somebody wrote down: rules over surface
statistics, or a rubric that a large model applies and a small classifier
learns. The Gopher, C4 and FineWeb rules remove about two thirds of our crawl
documents and match FineWeb's code on about 96% of them. They remove every
Chinese, Japanese and Thai document, whose words are not separated by spaces,
and our word pattern removes every Hindi one. The documents of lowest
perplexity are boilerplate and machine-written prose. Qwen3.8-27B, scoring
documents under FineWeb-Edu's rubric from its next-token probabilities, uses a
far stricter scale than Llama-3-70B did (about 1.4% against 9% at 3 or more)
yet orders documents alike. A linear head on a frozen embedder learns its
labels about as well as the released classifier does. Both stay below the F1
of about 0.67 at which Qwen reproduces its own labels when asked a second
way. Since $\textrm{F1} = 2\pi\,\textrm{TPR}/(\pi + \rho)$, a threshold is a
choice of retained fraction, here about 3% of the documents. An audit that
resamples whole sites finds that the classifier removes Australian domains and
shops more often, and near its threshold it mostly removes pages that a
general corpus would keep.

## Exercises

1. **Two F1 scores on one card.** The released classifier's card reports an
   F1 of 82% "when converted to a binary classifier using a score threshold of
   3", and a classification report in which score 5 has an F1 of 0.02.
    1. Explain why both can be true, and what each says about using the
       classifier as a filter at 3.
    1. The card's confusion matrix over its 46,867 hold-out rows has these
       rows, true scores 0 to 5, and columns, predicted scores 0 to 5:
       (2791, 2858, 45, 0, 0, 0), (919, 22343, 3180, 69, 1, 0),
       (3, 3225, 6330, 757, 7, 0), (1, 66, 1473, 1694, 173, 0),
       (0, 4, 98, 420, 283, 2), (0, 0, 18, 85, 21, 1). Compute the F1 of the
       positive class "score at least 3". Which average over the two classes
       gives 82%?
    1. [code] Our test rows are drawn from this hold-out. Compute the released
       classifier's F1 against Llama's labels at its rounded score 3, and
       compare it with the table's 0.54 at $t^*$. Why do the two differ?
   :exlabel:`ex_lm_quality-card`

1. **Thresholds on a regression.**
    1. Why is the F1 at the threshold that maximizes F1 on the test rows an
       optimistic estimate of the F1 at deployment? Name two ways to avoid
       the bias.
    1. Among the documents that a head cannot tell apart, a share $q$ is
       labelled 3 and the rest 1. Show that the prediction minimizing the
       expected squared error on them is $1 + 2q$,
       and find the $q$ at which it reaches 2.5. Use this to explain why our
       regression head keeps nothing at a cut of 2.5.
    1. Would the logistic head's probability have the same problem with a
       cut at 0.5? Compare its $t^*$ with 0.5 and explain.
   :exlabel:`ex_lm_quality-regression`

1. **The cost of a false removal.** A crawl holds $T$ tokens in documents of
   equal length, of which a share $\pi$ are labelled educational.
    1. Using :eqref:`eq_lm_quality_f1_retention`, write the educational
       tokens that a filter with retained fraction $\rho$ and score F1
       removes, in terms of $T$, $\pi$, $\rho$ and F1.
    1. Put in this section's numbers ($\pi \approx 0.014$, $\rho \approx
       0.029$, $\textrm{F1} \approx 0.44$) for $T = 10^{12}$. How many
       educational tokens does $t^*$ remove?
    1. A classifier at the noise ceiling of
       :eqref:`eq_lm_quality_f1_ceiling` keeps the same $\rho$. Bound the
       educational tokens it removes, and say when your bound is vacuous.
       Which part of the loss could a better classifier recover, and which
       part is in the labels?
   :exlabel:`ex_lm_quality-false-removal`

1. **Asymmetric noise.** ● The labeller misses a true positive with
   probability $\eta_1$ and invents a positive from a true negative with
   probability $\eta_0$.
    1. Derive the precision, recall and F1 against the labels of a perfect
       classifier of the true label.
    1. With $\pi^* = 0.0926$ and $\eta_1 = 0.02$, find the $\eta_0$ at which
       this ceiling equals the symmetric ceiling at $\eta = 0.05$.
    1. Our two protocols disagree on 19 test rows, 17 of them in one
       direction. Can $\eta_0$ and $\eta_1$ of each protocol be estimated
       from two annotations alone? What would a third annotation add?
   :exlabel:`ex_lm_quality-asymmetric`

1. [code] **Two labellers, two filters.** Train a second regression head on
   Llama's scores, choose its threshold on the validation rows against
   Llama's labels, and apply both heads to the stream. Report the share each
   keeps, their Cohen's kappa with an interval, and the host classes and
   country codes where their decisions differ most. Then predict which of the
   two filtered corpora would train the better model, and say how the proxy
   runs of :numref:`sec_lm_data_mixtures` could test your prediction.
   :exlabel:`ex_lm_quality-two-labellers`

1. [code] **How far behind is the bag?** Before measuring, state the largest
   F1 gap between the hashed bag and the regression head that you would
   accept for a filter that runs on a CPU. Measure the gap with a paired
   bootstrap over the test rows. Then try to close it without a GPU: more
   epochs, character n-grams, or training the bag on the score rather than
   the binary label. Which change helps, and is the improvement larger than
   its interval?
   :exlabel:`ex_lm_quality-bag`

1. [code] **A human review.** Read the 200 reviewed documents yourself and
   label each one under the two-line rubric of the review cell, without
   looking at the judge's labels. Compute the removal precision of each arm
   with its Wilson interval and Cohen's kappa between your labels and the
   judge's. Do your labels change the conclusion about the two arms? Where you
   and the judge disagree, which of you followed the rubric?
   :exlabel:`ex_lm_quality-human-review`

1. [extended] **Quality or script?** Choose one of the six FineWeb-2
   languages that you read. Draw 50 of its documents that the English rules
   remove, label each as worth keeping or not, and record which rule fired
   first. Are the removals quality or script artifacts? Rewrite the rules
   that fire most so that they suit the language (a word tokenizer for the
   script, stop words of the language), and report the removal rate and
   your labels' precision again.
   :exlabel:`ex_lm_quality-script`

<!-- slides -->

::: {.slide}
::: {.cover}
[Dive into Deep Learning · §20.2]{.kicker}

Quality Filtering<br>**heuristic rules · perplexity · classifiers on model labels · label noise · thresholds · audits**
:::
:::

::: {.slide title="A Spell-Selling Page Passes Every Rule; a Hiring Guide Fails the Classifier"}
[The problem]{.kicker}

FineWeb-Edu keeps **7.26%** of FineWeb's 265,662,695 documents of one crawl.

| page, from FineWeb's output | rules | our classifier |
|---|---|---|
| "I NEED REVENGE DEATH SPELL … Call/WhatsApp ‹phone›", 1,500+ characters | keep | lowest score among such pages |
| a company's guide to hiring software engineers, after a sales pitch | keep | remove |

::: {.d2l-note .warn}
Before tuning a threshold: what does the filter mean by quality, and whose
pages does it remove?
:::
:::

::: {.slide title="By the End of This Deck You Can"}
[Goals]{.kicker}

- implement the Gopher, C4 and FineWeb rules with reason codes, and check them
  against FineWeb's code;
- say what perplexity measures, and what it does not;
- label documents with a large model's next-token probabilities, and train a
  linear head on a frozen embedder;
- bound the attainable F1 under label noise;
- turn a threshold into retained tokens, and audit removals with
  site-level intervals.

**Running example:** our 3,288 crawl documents (c0) and 20,000 documents of
FineWeb's output for the same crawl (the *audit stream*).
:::

::: {.slide title="Six Steps From a Crawl Record to a Token on Disk"}
[Where this fits]{.kicker}

::: {.cols .vc}
::: {.col}
| step | section |
|---|---|
| get text | Document Extraction from Web Crawls |
| choose text | **Quality Filtering** |
| | Deduplication and Decontamination |
| mix text | Data Mixtures |
| | Synthetic and Rephrased Data |
| cut text | Tokenizers at Scale |
:::

::: {.col .narrow}
- **Here:** c0 → rules → score → flags in the manifest (c1).
- **Skips:** duplicates, mixtures, and whether the filter helps training.
:::
:::
:::

::: {.slide title="Fifteen Rules, Each With a Reason Code"}
[Heuristic rules]{.kicker}

| family | catches | codes |
|---|---|---|
| Gopher repetition | repeated lines, paragraphs, n-grams | 3 |
| Gopher quality | 50 to 100,000 words; mean length 3 to 10; 80% of words with a letter; 2 of 8 stop words (*the, of, with, …*) | 7 |
| C4 | fewer than 5 sentences, "lorem ipsum", `{` | 1 |
| FineWeb | lines without terminal punctuation, short lines, duplicated characters, line breaks per word | 4 |

A document is charged to the **first** rule that fires.
Thresholds: datatrove 0.10.1, the code that built FineWeb.
:::

::: {.slide title="The Rules Remove Two Thirds of Our Crawl and 1.6% of FineWeb's Output"}
[The rules at work]{.kicker}

| corpus | in | removed |
|---|---|---|
| c0, our crawl documents | 3,288 | 2,256 (69%) |
| audit stream, FineWeb's output | 20,000 | 317 (1.6%) |

- c0, first rule charged: duplicated text 571, alphabetic words 403, word
  count 322.
- Line punctuation *fires* on 1,150 c0 documents but comes late in the order.
- Against datatrove on 2,000 c0 documents: same decision on 96% to 99.9%;
  the gap is word and sentence splitting.

::: {.d2l-note}
On the stream, FineWeb's own four rules never fire.
:::
:::

::: {.slide title="English Rules Remove Every Chinese, Japanese and Thai Page; Our Word Pattern Every Hindi Page"}
[Other languages]{.kicker}

| FineWeb-2, 500 pages each | removed | first rule |
|---|---|---|
| German | 96% | stop words |
| Swahili | 93% | stop words |
| Hindi | 100% | mean word length (our `\w`) |
| Chinese, Thai | 100% | alphabetic words |
| Japanese | 100% | word count |

- Chinese, Japanese, Thai: no spaces, so one "word" per run between
  punctuation marks.
- Hindi has spaces, but our `\w` skips Devanagari vowel signs and words
  shatter; datatrove's tokenizer keeps them. An artifact of our code.

::: {.d2l-note .warn}
Every page had passed FineWeb-2's rules for its language.
:::
:::

::: {.slide title="Check: Drop the Stop-Word Rule. How Much German Is Left?"}
[Check]{.kicker}

The rule asks for 2 of *the, be, to, of, and, that, have, with*.
With it, German loses 96%.

::: {.fragment .fade-up}
::: {.d2l-note}
**Answer.** Without it German loses 28%, Swahili 16% instead of 93%. Chinese,
Japanese, Thai and Hindi still lose 100%: their losses come from the word
rules (for Hindi, from our word pattern), not from English vocabulary.
:::
:::
:::

::: {.slide title="The Most Predictable Pages Read as Machine-Written"}
[Perplexity]{.kicker}

Mean NLL per token, first 512 tokens, Qwen3-0.6B-Base, 3,000 stream pages:
median **2.9 nats**.

| lowest NLL | highest NLL |
|---|---|
| "Please be mindful of our community standards" | "As have a peek at these guys you can see regarding the …" |
| "In today's digital age, a strong online presence is crucial …" | spun text about pokies, loans, Discord |
| "Greetings, readers! I am excited to delve into …" | a garden-blog comment, a railway photo site |

::: {.d2l-note}
Low perplexity means resemblance to the model's training text, not quality.
:::
:::

::: {.slide title="A Large Model Labels a Sample; a Small Model Labels the Web"}
[The recipe]{.kicker}

1. A rubric: "useful … for teaching from primary school to grade school".
2. A large model scores a sample, 0 to 5 points, one criterion per point.
3. A small model learns the score; a threshold selects.

| corpus | labeller | small model | rule |
|---|---|---|---|
| FineWeb-Edu | Llama-3-70B, 450k pages | frozen 109M encoder + regression | rounded score ≥ 3 |
| DCLM | instruction data and explanations vs web | fastText | top 10% |
| Nemotron-CC | several classifiers | ensemble | quality buckets |
:::

::: {.slide title="Read the Score From Six Next-Token Probabilities"}
[Labelling]{.kicker}

Answer prefix `Educational score: `; one forward pass:

$$p(d) \;+\; p(\textrm{space})\,p(d \mid \textrm{space}), \qquad d = 0, \ldots, 5$$

- Label: $\arg\max_d$. Expected score: $\sum_d d\,p(d) / \sum_d p(d)$.
- The second-space path holds about 15% of the probability.
- Qwen3.8-27B, 4-bit weights, one copy per 24 GB GPU; 22,000 pages in about
  an hour on four.

::: {.d2l-note .warn}
Batching changes the arithmetic, and near ties flip. One request at a time:
identical bits on any server.
:::
:::

::: {.slide title="Qwen Is About a Point Stricter Than Llama, but Ranks Pages Alike"}
[Two labellers]{.kicker}

| share of 22,000 pages | score ≥ 1 | ≥ 2 | ≥ 3 | ≥ 4 |
|---|---|---|---|---|
| Llama-3-70B (2024) | 0.88 | 0.31 | **0.091** | 0.018 |
| Qwen3.8-27B | 0.82 | 0.064 | **0.014** | 0.0001 |

- Spearman, Qwen's expected score vs Llama's score: **0.76**.
- Kappa at ≥ 3: 0.24; with Qwen cut at Llama's share: 0.65.

::: {.d2l-note}
Agreement with one labeller cannot show which labels are better; training
runs decide.
:::
:::

::: {.slide title="A Linear Head Matches the Released Classifier on Qwen's Labels"}
[Classifiers]{.kicker}

2,000 test rows, 36 positives (Qwen ≥ 3); $t^*$ from 2,000 validation rows.

| classifier | trained parameters | F1 [95% interval] | AUC |
|---|---|---|---|
| regression head, frozen 0.6B embedder | 1,025 | 0.44 [0.32, 0.55] | 0.98 |
| logistic head, same embedder | 1,025 | 0.56 [0.41, 0.68] | 0.98 |
| hashed bag (fastText), CPU | 67M | 0.26 [0.17, 0.34] | 0.90 |
| released, trained on Llama | head only | 0.47 [0.35, 0.57] | 0.98 |

Paired: regression head − released F1 = −0.02 [−0.14, +0.09].
:::

::: {.slide title="Check: The Regression Head's Best Threshold Is 1.6. Why Not 3?"}
[Check]{.kicker}

The label is "score ≥ 3"; the head fits the score by squared error.

::: {.fragment .fade-up}
::: {.d2l-note}
**Answer.** A linear head on a frozen vector cannot tell every page apart;
under squared error it predicts the mean score of the pages it confuses. Only
1.4% of labels reach 3, so such a group hardly ever has a mean near 3: the
predictions bunch between 0 and 2. A cut at 2.5 keeps nothing.
:::
:::
:::

::: {.slide title="F1 Depends on the Threshold Only Through Recall and Retention"}
[An identity]{.kicker}

$\pi$: share of positive labels; TPR: recall; $\rho$: share of documents kept.

$$\textrm{precision} = \frac{\pi\,\textrm{TPR}}{\rho}
\quad\Rightarrow\quad
\textrm{F1} = \frac{2PR}{P+R} = \frac{2\pi\,\textrm{TPR}}{\pi + \rho}$$

- Best case: keep exactly the positives, $\rho = \pi$, TPR $= 1$.
- At fixed recall, keeping more lowers F1.
- Checked at 50 thresholds: largest gap $6 \times 10^{-17}$.
:::

::: {.slide title="Qwen Reproduces Its Own Labels at an F1 of About 0.67"}
[Label noise]{.kicker}

If labels flip the true label $y^*$ with probability $\eta$ either way, a
perfect classifier of $y^*$ scores

$$P = 1 - \eta, \qquad R = \frac{\pi^*(1-\eta)}{\pi^*(1-\eta) + (1-\pi^*)\eta}$$

- Two Qwen protocols disagree on 19 of 2,000 rows: $\eta \approx 0.005$,
  ceiling **0.80** [0.68, 0.89], *if* the noise is symmetric.
- It is not: 1.80% against 1.05% positive; 17 of the 19 go one way.
- No model needed: one protocol against the other, 36 and 21 positives, 19
  shared: F1 $= 2 \cdot 19/(36+21) \approx$ **0.67**.
- Classifiers 0.26 to 0.56: below both (the logistic head's interval reaches
  0.68).
:::

::: {.slide title="Cutting at the Label's Own 3 Keeps Nothing"}
[The threshold]{.kicker}

Regression head, stream shares and test precision and recall (Qwen ≥ 3):

| cut | documents | tokens | precision | recall |
|---|---|---|---|---|
| 1.5 | 5.1% | 8.0% | 0.27 | 0.92 |
| $t^* = 1.61$ | 2.9% | 4.8% | 0.33 | 0.67 |
| 2.0 | 0.3% | 0.5% | 0.64 | 0.19 |
| 2.5, 3.0 | 0 | 0 | — | 0 |

- FineWeb-Edu's rule on the same stream keeps 7.1% of documents, 9.9% of tokens.
- Data-limited runs cut lower; compute-limited runs cut higher.
:::

::: {.slide title="Resample Sites, Not Pages, and Demand the Same Sign in Both Halves"}
[The audit]{.kicker}

- Groups: country-code domain (a weak proxy for English variety), host class
  by name rules, six languages (rules only).
- Intervals: bootstrap over **registered domains**; 3,000 pages of one site
  are not 3,000 samples.
- Too few: under 100 documents or 10 domains.
- Halves split by domain; a finding needs one sign in both.

::: {.d2l-note}
A filter keyed on a classifier's score is a covariate shift toward what the
classifier prefers.
:::
:::

::: {.slide title="The Classifier Keeps .au Pages at a Third of the .com Rate"}
[Audit results]{.kicker}

| group | arm | removal − reference | 95% interval |
|---|---|---|---|
| .au vs .com | classifier | +1.9 points | [+0.9, +2.7] |
| shop vs other | classifier | +2.6 points | [+1.9, +3.1] |
| news vs other | rules | −1.4 points | [−1.7, −0.9] |
| .uk, .ca, .in, .nz, .za | classifier | intervals include 0 | |

The classifier removes 97.2% of the stream: it keeps about 3% of .com pages
and about 1% of .au pages (6 of 544), so the third is rough.
:::

::: {.slide title="Check: Removal Rates of 97.0% and 98.9%. A Small Difference?"}
[Check]{.kicker}

A filter removes 97.0% of group A and 98.9% of group B.

::: {.fragment .fade-up}
::: {.d2l-note}
**Answer.** No. The corpus receives what is kept: 3.0% of A and 1.1% of B,
so B enters at about a third of A's rate. When a filter keeps little, compare
retention, not removal.
:::
:::
:::

::: {.slide title="Near Its Threshold the Classifier Mostly Removes Pages Worth Keeping"}
[Reading removals]{.kicker}

Judge: Qwen3.8-27B, reasoning on and uncapped, rubric "would a reader want
this page in a general-purpose corpus?"

| 100 removed pages each | judged "drop" | 95% interval |
|---|---|---|
| classifier only, score within 0.3 of $t^*$ | 0.11 | [0.06, 0.19] |
| rules | 0.70 | [0.60, 0.78] |

- Rules removed, judged keep: an entry on a figure of Greek mythology.
- The judge also wrote the labels: not independent.

::: {.d2l-note}
The classifier follows its rubric; the gap is between two definitions of quality.
:::
:::

::: {.slide title="The Manifest Stores the Score; the Threshold Is a Parameter"}
[In the pipeline]{.kicker}

| stage | in | out | recorded |
|---|---|---|---|
| `heuristics` | 3,288 | 1,032 | drops by first reason code |
| `edu_score` | 1,032 | 1,032 | threshold 1.611, head SHA-256, embedder revision |

- Flags per c0 id: reason codes and score, no text.
- 20 of the 1,032 score at or above $t^*$, about 2%.
- Score bands can become mixture sources without relabelling.
:::

::: {.slide title="Recap"}
[Recap]{.kicker}

- Rules: c0 loses 69%, FineWeb's output 1.6%; 96% agreement with datatrove;
  100% loss on unspaced scripts, and on Hindi through our word pattern.
- Lowest perplexity: boilerplate and machine-written prose.
- Qwen: 1.4% at ≥ 3 against Llama's 9%; Spearman 0.76.
- Head F1 0.44, released 0.47: no difference; Qwen against itself 0.67.
- $t^*$ keeps about 3% of documents, 5% of tokens.
- Audit: .au and shops removed more; near $t^*$ the judge keeps 89 in 100.

**Next:** the same page, or nearly the same, appears many times.
:::
