# Quality Filtering
:label:`sec_lm_quality_filtering`

<!-- Author brief, copied verbatim from the Language Models part plan
(docs/llm/language_models_part_plan.md, "3.2 Quality Filtering · T1").
Blueprint: docs/llm/blueprints/ch03_data.md section 3.2.
The plan is a specification, not book text; this comment is replaced
when the section is written.

**Idea:** A quality filter is a classifier trained on model-annotated labels, and its false removals must be audited.
**Opening:** A model rates a page low that a human would keep: a transcript of a town meeting, informative and plain. The reader's instinct is to tune the threshold, but the question is what the filter's definition of quality is and who it removes. The section trains the filter, audits it, and closes on how what the filter keeps is biased toward the annotator's taste.
**Teaches:**
- Heuristic rules (document length, repetition, symbol ratios, stop-word presence) with stable reason codes (`d2l.heuristic_filters`) and why word-based rules fail on unspaced scripts; perplexity filters under a small released decoder and why low perplexity is not quality.
- A classifier on model labels as the current recipe: a frozen embedder (Qwen3-Embedding-0.6B, `d2l.embed_docs`) with a linear head on labels written by Qwen3.5-4B under an educational-value rubric, against a hashed bag-of-n-grams model as the T0 path; the label-noise ceiling from a second annotator.
- The threshold as a precision-recall choice on held-out labels, converted to a retained-token fraction; what a classifier F1 does not say about training (3.4's proxy runs are the yardstick).
- Auditing removals by language, domain and English variety with cluster-bootstrap intervals (`d2l.audit_removals`); the filter as a covariate shift keyed on a proxy label.
**Derivation:** none.
**Notebook:**
1. Label 22,000 FineWeb documents with Qwen3.5-4B under the FineWeb-Edu rubric at authoring time (cached as `lm-edu-annot-22k`, 20,000 train and 2,000 test); embed them with the frozen embedder and fit the linear head; fit the hashed-bag classifier on the same labels; choose the threshold on a validation split carved from train.
2. Apply heuristics, the classifier and both to the 3.1 documents and to a 20,000-document audit stream of the same crawl; write the keep and drop flags to the manifest (`lm-c1-docs`).
3. Audit removal rates by country-code domain, host class and, under heuristics only, six FineWeb-2 languages; compare against 200 human labels on removed documents for precision of removal.
4. Compare the notebook's classifier with the released FineWeb-Edu classifier on the 2,000 test labels with a paired bootstrap, and report their agreement on the audit stream.
5. Claim that survives noise: the removal rate differs by domain group beyond its cluster-bootstrap interval and the classifier contributes to the gap beyond the heuristics; precision of removal against the human labels is stated with its interval; groups with fewer than 100 documents or 10 domains are marked "too few". A second stream seed must give the same sign for every reported group.
6. Reused pieces: `d2l.Doc`, `d2l.read_docs`, `d2l.CorpusManifest`, `d2l.rebuild_c0`, `d2l.detect_lang` (3.1); `d2l.paired_bootstrap` (1.5); the embedder's pooling and padding conventions from 2.3 in one sentence; new `d2l.embed_docs`, `d2l.GOPHER_CFG`, `d2l.heuristic_filters`, `d2l.fit_quality_head`, `d2l.HashedBagClassifier`, `d2l.audit_removals`.
**Figures:** `fig_lm_filter_audit` (plot: a forest plot of removal-rate differences to the reference group with intervals, for heuristics, classifier and both; a second panel of removal rates by language under heuristics only).
**Restate / forward:** `sec_environment-and-distribution-shift` (a classifier that separates two sets measures their shift, and a filter keyed on its score induces one). Forward: 3.4 (the proxy runs that measure what a filter does to training).
**Exercises:** 5 to 7. Load-bearing: measure the agreement between two labellers' classifiers (the notebook's Qwen3.5-4B labels and the released Llama-3 annotations) on the audit stream and locate where they disagree (measure); derive the cost of a false removal in retained tokens as a function of threshold and the label-noise ceiling (derive); check that the hashed-bag classifier's F1 on the test labels is within a stated margin of the embedding head's (check); open: design an audit for one language of FineWeb-2 with a native reader's 50 labels and report whether the heuristics' removals are quality or script artifacts.
**Writer notes:**
- Labeller: labels come from Qwen3.5-4B (the part's judge), produced once at authoring with the FineWeb-Edu additive 0 to 5 rubric and cached; the released Llama-3-70B annotations (`HuggingFaceFW/fineweb-edu-llama3-annotations`, ODC-By) are the comparison labeller, not the training set. State the rubric and the annotator model on the artifact's card; the rubric's prestige bias toward academic style is part of what the audit finds.
- Exports: `d2l.embed_docs(texts, model_id='Qwen/Qwen3-Embedding-0.6B', max_len=512, batch=64, pool='last', device=None)` returning float16 L2-normalized arrays, equal to `sentence_transformers` encoding up to cosine 0.999 on 64 texts (the oracle test); `d2l.heuristic_filters(text, cfg=None)` returning ordered reason codes (empty means keep), checked against Datatrove's filters with the disagreement list reported; `d2l.audit_removals(df, keep, by, ref=None, n_boot=2000, cluster=None, seed=0)`.
- Data: `HuggingFaceFW/fineweb` config `CC-MAIN-2025-05` for the audit stream (shuffle with a seeded buffer or take every 50th row of the first million, because rows are ordered by URL within a file); `HuggingFaceFW/fineweb-2` configs for six languages (500 documents each, ODC-By); the released `HuggingFaceFW/fineweb-edu-classifier` (Apache-2.0; read its card's F1 and class supports at authoring) and `HuggingFaceTB/SmolLM2-135M` for the perplexity filter; the FineWeb-Edu survival rate of this crawl (about 7% of documents) as the reference decision.
- Memory and time: Qwen3-Embedding-0.6B in bf16 is about 1.2 GB plus activations at batch 64 × 512 tokens; embedding about 47,000 documents may exceed 15 minutes on the build box (reduce `audit_docs` and `max_len`, or use a 22M-parameter MiniLM embedder as the fallback); the T0 path is the hashed-bag model, about 3 minutes on CPU.
- Pitfalls to state: selecting the threshold on the test labels; treating model labels as ground truth; evaluating a classifier on documents drawn after the filter (the audit stream is drawn before this notebook's filters but after FineWeb's own, which the text says); a country-code domain is a poor dialect proxy (the editor may substitute a dialect-labelled corpus with a clear licence); 3,000 documents from one site are not 3,000 samples.
- Human labels for the 200-document removal review are produced by the authors; if unavailable, drop the human column and keep the agreement with the released classifier.
- Committed artifacts: `lm-edu-annot-22k` (text truncated to 3,000 characters, scores, split; ODC-By attribution), `lm-quality-head` (the head's weights), `lm-c1-docs` (flags keyed by document id, no text), the audit table as JSON.
- Do not re-teach: logistic regression and thresholds (one sentence to the linear-classification chapters), distribution shift beyond the one restated sentence.
**Sources:**
- Papers: [DataComp-LM](https://arxiv.org/abs/2406.11794) (the fastText quality classifier and the finding that classifier choice dominates other filters in their testbed; read to confirm wording); [Scaling Language Models: Methods, Analysis and Insights from Training Gopher](https://arxiv.org/abs/2112.11446) (the quality and repetition rules and their thresholds, appendix); [The FineWeb Datasets](https://arxiv.org/abs/2406.17557) (the educational classifier recipe and ablations); [Nemotron-CC](https://arxiv.org/abs/2412.02595) (classifier ensembling and quality buckets); [CCNet](https://arxiv.org/abs/1911.00359) (the perplexity-filter pipeline); [Whose Language Counts as High Quality?](https://arxiv.org/abs/2201.10474) (quality filters encode language ideologies; the audit's motivation).
- Code and documentation: [FineWeb-Edu dataset card](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu) (the rubric and survival statistics); [FineWeb-Edu classifier](https://huggingface.co/HuggingFaceFW/fineweb-edu-classifier) and the [Llama-3 annotations](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu-llama3-annotations) (the comparison labeller, ODC-By); [Datatrove](https://github.com/huggingface/datatrove) (the heuristic filters as the parity oracle); [fastText](https://github.com/facebookresearch/fastText) (the hashed-bag model this section reimplements).
- Courses, tutorials and blogs: [FineWeb blog post](https://huggingface.co/spaces/HuggingFaceFW/blogpost-fineweb-v1) (filters and their ablations); [fastText supervised tutorial](https://fasttext.cc/docs/en/supervised-tutorial.html); [CS336](https://cs336.stanford.edu/) (data filtering lecture).
-->

<!-- Placeholder (stage 2, cells step): the cells follow the section
specification's cell table (S/spec/20.2.md part 3); one-line comments mark
where prose goes. Rights rule P1: no cell prints text of our own crawl
extraction (c0); displayed text is FineWeb's (ODC-By) from the audit stream
or the annotation set, everything else about c0 is counts, codes and
scores keyed by id. -->

<!-- Prose: the opening (a plain, informative page the labeller scores low); the question: what is the filter's definition of quality, and who does it remove? -->

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
from sentence_transformers import SentenceTransformer
```

<!-- Prose: what the notebook computes, what runs on a GPU, and what a normal build reads from registered files (the labels, the vectors, the released classifier's scores); D2L_RERUN names the regenerations. -->

```{.python .input #quality-filtering-config}
%%tab pytorch
LABELLER = 'Qwen/Qwen3.5-4B'         # writes the rubric's scores (bf16)
EMBEDDER = 'Qwen/Qwen3-Embedding-0.6B'
PPL_MODEL = 'Qwen/Qwen3-0.6B-Base'   # the perplexity filter's reference
RELEASED = 'HuggingFaceFW/fineweb-edu-classifier'
MAX_LEN = 512             # tokens of a document that every model reads
THRESH_LABEL = 3          # "educational": a score of 3 or more
AGREE_MIN = dict(kappa_bin=0.5, qwk=0.6)  # fixed before any label existed
SCORE_PREFIX = 'Educational score: '  # the labeller's own spacing (pilot)
LABEL_BATCH_TOKENS = 24_576  # padded tokens per labelling batch
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
    'quality-flags': ('lm-c1-docs-v1', 'lm-c1-docs-v1.jsonl.gz')}
RERUN = d2l.lm_rerun(ARTIFACTS)
GPU_NAMES = {'edu-label-pilot', 'edu-label-0', 'edu-label-1', 'quality-emb'}
assert GPU or not RERUN & GPU_NAMES, 'these regenerations need a GPU'
RERUN_DIR = os.path.join('..', 'data', 'lm', 'rerun')
R = {}  # what this run measures; written or compared in the last cell
pins = {m: d2l.HF_REVISIONS.get(m) or d2l.HF_ENCODER_REVISIONS.get(m)
        for m in (LABELLER, EMBEDDER, PPL_MODEL)}
pins[RELEASED] = d2l.LM_DATA_SOURCES['fineweb-edu-classifier'][2]
print(f'GPU={GPU}, SMALL={SMALL}, regenerating: '
      f'{", ".join(sorted(RERUN)) or "none"}')
for m, rev in pins.items():
    print(f'  {m}@{rev[:8]}')
```

<!-- Prose: the registry; the annotation sample and the audit stream are FineWeb text (ODC-By), shipped with attribution; c0 vectors are never hosted. -->

<!-- Pending registrations (cells step, 2026-11-15): the seven keys below join this cell as a `#@save` block once uploaded (W2 builder: `lm-data-audit-stream-v1`; W7/W8 regenerations: the rest). Until then a normal build stops at the opening cell with a KeyError. -->

```{.python .input #quality-filtering-data-hub}
%%tab pytorch
QUALITY_KEYS = ('lm-edu-annot-22k-v1', 'lm-data-audit-stream-v1',
                'lm-data-emb-qwen3emb-v1', 'lm-quality-head-v1',
                'lm-c1-docs-v1', 'lm-data-quality-results-v1',
                'lm-data-removal-labels-v1')
print('registered:', ', '.join(k for k in QUALITY_KEYS if k in d2l.DATA_HUB)
      or 'none', '| pending:', ', '.join(
          k for k in QUALITY_KEYS if k not in d2l.DATA_HUB) or 'none')
```

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
    for title, key in (
            ('passes every rule, at least 1,500 characters, lowest score '
             'of our classifier', 'low_score_id'),
            ('removed by our classifier, labelled "keep" in the review, '
             'lowest score', 'kept_label_id')):
        doc = REG['opening'].get(key)
        if doc is None:
            continue
        print(f'\n{title} ({doc}):')
        print(textwrap.fill(by_id[doc][:600].replace('\n', ' ') + ' …',
                            100, initial_indent='  ',
                            subsequent_indent='  '))
    print('\n' + ATTRIBUTION)
```

## Heuristic Rules
:label:`subsec_lm_quality_rules`

<!-- Prose: Gopher quality and repetition rules, C4's rules, FineWeb's own rules (static table tab_lm_quality_rules below); one reason code per rule; a document is charged to the first rule that fires (20.1's charging rule). FineWeb's char-duplicate threshold: the paper's text says 0.1, the datatrove code that produced the release uses 0.01; the cell follows the code. -->

<!-- Prose (static table, `tab_lm_quality_rules`): family | source | reason codes | what it catches. Gopher repetition (Rae et al., Table A1): gopher_dup_text, gopher_top_ngram, gopher_dup_ngram. Gopher quality: gopher_words, gopher_word_len, gopher_symbols, gopher_bullets, gopher_ellipsis, gopher_alpha, gopher_stop_words. C4 (Raffel et al.; FineWeb's settings): c4_quality. FineWeb (Penedo et al.): fw_line_punct, fw_short_lines, fw_dup_chars, fw_list_ratio. -->

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

<!-- Prose: one hand-written document per reason code, each firing exactly its rule; a plain paragraph fires none. -->

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

<!-- Prose: our rules against datatrove's filters (FineWeb's code) on 2,000 c0 documents: agreement per rule family and the reasons for the disagreements (word tokenizer, sentence splitter, terminal punctuation set, C4's line removal before FineWeb's rules). -->

```{.python .input #quality-filtering-rules-parity}
%%tab pytorch
PARITY_FILE = os.path.join('..', 'data', 'lm', 'datatrove-parity-v1.json')
if 'quality-results' in RERUN:  # computed once by the sources builder
    PARITY = (json.load(open(PARITY_FILE)) if os.path.exists(PARITY_FILE)
              else None)
else:
    PARITY = REG.get('parity') if REG else None
R['parity'] = PARITY
if PARITY is None:
    print('datatrove parity table pending (tools/build_lm_data_sources.py '
          f'writes {PARITY_FILE})')
else:
    print(f'{PARITY["n_docs"]:,} c0 documents, datatrove '
          f'{PARITY["datatrove_version"]}; per filter, the share of '
          'documents on which both implementations decide alike:')
    print(pd.DataFrame(PARITY['filters']).to_string(index=False))
    for reason, count in PARITY['disagreements'].items():
        print(f'  {count:4d}  {reason}')
```

<!-- Prose: the rules on our crawl (c0, before any filter) and on the audit stream (FineWeb's output, so its rules have run: what still fires there is where our implementation and FineWeb's differ); the first-firing funnel; rule pairs that fire together; the six FineWeb-2 languages. -->

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
print('c0, rules that fire together most often:',
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

### Perplexity Filters
:label:`subsec_lm_quality_ppl`

<!-- Prose: CCNet's idea (a model of a clean corpus scores documents; keep the likely buckets); here the mean negative log-likelihood per token under Qwen3-0.6B-Base on the first 512 tokens; the lowest are repetition and boilerplate, the highest code, tables and names; low perplexity is resemblance to the model's training text, not quality. -->

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
        print(f'  {r.nll:5.2f}  {r.text[:120]!r}')
print('\n' + ATTRIBUTION)
if not SMALL:
    R['ppl'] = dict(n=len(ids), quantiles=q.tolist(),
                    lowest=order.id.head(10).tolist(),
                    highest=order.id.tail(10).tolist())
```

## Classifiers on Model Labels
:label:`subsec_lm_quality_classifier`

<!-- Prose: the recipe (a large model scores a sample under a rubric, a cheap model learns to reproduce the score, a threshold selects); FineWeb-Edu's additive rubric, quoted from the released prompt; DCLM's and Nemotron-CC's variants in one clause each. -->

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
                     max_batch_tokens=24_576, prefix='Educational score: '):
    """(len(prompts), 6) float64: the probabilities of the next tokens
    '0' ... '5' after each prompt as a user turn of the chat template
    (non-thinking), the assistant's answer started with prefix. Prompts are
    sorted by length and left-padded into batches of at most batch_size
    rows and max_batch_tokens tokens with padding (a longer prompt runs
    alone); only the last position's logits are computed; rows come back
    in input order. The prompt is tokenized as if the digit followed the
    prefix, so a prefix ending in spaces splits as in the model's own text.
    The six probabilities need not sum to one: the rest is what the model
    puts on other tokens."""
    digits = [tokenizer.encode(str(d), add_special_tokens=False)
              for d in range(6)]
    if any(len(t) != 1 for t in digits):
        raise ValueError(f'the digits are not single tokens: {digits}')
    digits = [t[0] for t in digits]
    ids = []
    for p in prompts:  # tokenized with a digit after the prefix, then cut:
        # a prefix that ends in spaces tokenizes differently on its own
        x = tokenizer(tokenizer.apply_chat_template(
            [{'role': 'user', 'content': p}], tokenize=False,
            add_generation_prompt=True, enable_thinking=False) + prefix + '0',
            add_special_tokens=False)['input_ids']
        if x[-1] != digits[0]:
            raise ValueError(f'the digit after {prefix!r} merges with it')
        ids.append(x[:-1])
    pad = tokenizer.pad_token_id
    pad = tokenizer.eos_token_id if pad is None else pad
    device = next(model.parameters()).device
    out = np.empty((len(ids), 6))
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
            out[b] = z.float().log_softmax(-1)[:, digits].exp().cpu().numpy()
    return out
```

<!-- Prose: the labeller reads the score from its next-token probabilities, without writing a justification first; a live check against the registered labels. -->

```{.python .input #quality-filtering-label-live}
%%tab pytorch
LABEL_KEY = 'lm-edu-annot-22k-v1'
if SMALL or LABEL_KEY not in d2l.DATA_HUB:
    print('live labelling check skipped:',
          'it needs a GPU (about 9 GiB)' if SMALL else
          f'{LABEL_KEY} is not registered yet')
else:
    reg = pd.read_parquet(d2l.lm_artifact(LABEL_KEY))
    rows = reg[reg.split == 'test'].head(16)
    model, tok = d2l.load_hf(LABELLER)
    P = d2l.edu_label_scores(model, tok, rows.prompt.tolist(),
                             max_batch_tokens=LABEL_BATCH_TOKENS,
                             prefix=SCORE_PREFIX)
    print(d2l.lm_peak())
    del model
    d2l.lm_free()
    live, top2 = P.argmax(1), np.sort(P, 1)[:, -2:]
    same = live == rows.qwen_score.to_numpy()
    print(f'live labels equal the registered ones on {same.sum()} of 16 '
          'rows; near ties (top two within 0.05):',
          int(np.sum(top2[:, 1] - top2[:, 0] < 0.05)))
    assert same.sum() >= 15, 'the labeller does not reproduce its labels'
```

<!-- Prose: how the 22,000 labels were made (D2L_RERUN=edu-label-pilot, edu-label-0, edu-label-1; about 25 minutes per part on one GPU): the pilot labels the 2,000 test rows and writes 200 justification-first answers, which fix the score prefix and check the protocol. -->

```{.python .input #quality-filtering-label-regen}
%%tab pytorch
ANNOT_FILE = os.path.join(RERUN_DIR, 'lm-edu-annot-22k-v1.parquet')
LABEL_PARTS = {  # name: rows (in the order of the sampled permutation)
    'edu-label-pilot': ('test', range(0, 2000)),
    'edu-label-0': ('train', range(0, 9000)),
    'edu-label-1': ('train+val', range(9000, 20000))}
SCORE_RE = re.compile(r'Educational score:( *)(\d)')  # the released format

def label_part(name, base, model, tok):
    rows = base.iloc[list(LABEL_PARTS[name][1])]
    t0 = time.time()
    P = d2l.edu_label_scores(model, tok, rows.prompt.tolist(),
                             max_batch_tokens=LABEL_BATCH_TOKENS,
                             prefix=SCORE_PREFIX)
    print(f'{name}: {len(rows):,} rows in {time.time() - t0:.0f} s, '
          f'{d2l.lm_peak()}')
    part = pd.DataFrame(dict(row=rows.row.to_numpy(), qwen_probs=list(P)))
    if name == 'edu-label-pilot':  # the protocol check: justify first
        just = rows.head(200)
        t0 = time.time()
        gens = d2l.generate(model, tok, [[{'role': 'user', 'content': p}]
                                         for p in just.prompt],
                            max_new_tokens=384, chat=True,
                            enable_thinking=False, batch_size=16)
        found = [SCORE_RE.search(g.text) for g in gens]
        spaces = collections.Counter(len(m.group(1)) for m in found if m)
        print(f'200 justification-first answers in {time.time() - t0:.0f} '
              f's; unparsable {sum(m is None for m in found)}; spaces '
              f'before the digit: {dict(spaces)}')
        part['qwen_just_score'] = pd.array(
            [int(m.group(2)) if m else None for m in found] +
            [None] * (len(part) - 200), dtype='Int64')
        part['qwen_just_text'] = [g.text for g in gens] + [None] * (
            len(part) - 200)
    return d2l.lm_save_table(ARTIFACTS[name][1], part, on='row',
                             key=LABEL_KEY)

if RERUN & set(LABEL_PARTS):
    base = d2l._edu_annot_sample()  # the 22,000 sampled rows, unlabelled
    model, tok = d2l.load_hf(LABELLER)
    for name in sorted(RERUN & set(LABEL_PARTS)):
        label_part(name, base, model, tok)
    del model
    d2l.lm_free()
    parts = [os.path.join(RERUN_DIR, ARTIFACTS[n][1]) for n in LABEL_PARTS]
    if all(map(os.path.exists, parts)):  # merge; the same bytes from any run
        lab = pd.concat([pd.read_parquet(p) for p in parts],
                        ignore_index=True)
        assert lab.row.is_unique and len(lab) == len(base)
        full = base.merge(lab, on='row', validate='1:1')
        P = np.stack(full.qwen_probs.to_numpy())
        full['qwen_score'] = P.argmax(1)
        full['qwen_expected'] = (P * np.arange(6)).sum(1) / P.sum(1)
        tmp = ANNOT_FILE + f'.{os.getpid()}.tmp'
        full.sort_values('row').to_parquet(tmp, index=False,
                                           compression='zstd')
        os.replace(tmp, ANNOT_FILE)
        print(f'merged {len(full):,} rows into {ANNOT_FILE}, sha1 '
              f'{hashlib.sha1(open(ANNOT_FILE, "rb").read()).hexdigest()}')
```

<!-- Prose: the author's check: does the small labeller reproduce the large one's judgments? Agreement with the released Llama-3-70B labels on all 22,000 rows, read against thresholds fixed before any label existed; the protocol check on 200 rows. -->

```{.python .input #quality-filtering-agreement}
%%tab pytorch
def load_annotations():
    """The labelled rows: this run's merge, else the registered file, else
    (a pilot run) the sample with the pilot's labels only."""
    if RERUN & set(LABEL_PARTS) and os.path.exists(ANNOT_FILE):
        return pd.read_parquet(ANNOT_FILE)
    if LABEL_KEY in d2l.DATA_HUB:
        return pd.read_parquet(d2l.lm_artifact(LABEL_KEY))
    lab = pd.read_parquet(os.path.join(RERUN_DIR,
                                       ARTIFACTS['edu-label-pilot'][1]))
    df = d2l._edu_annot_sample().merge(lab, on='row', how='left')
    P = np.stack([p if isinstance(p, np.ndarray) else np.full(6, np.nan)
                  for p in df.qwen_probs])
    df['qwen_score'] = pd.array(np.where(np.isnan(P[:, 0]), np.nan,
                                         np.nan_to_num(P).argmax(1)),
                                dtype='Int64')
    df['qwen_expected'] = (P * np.arange(6)).sum(1) / P.sum(1)
    return df

def kappa_ci(a, b, n_boot=1000, seed=0):
    """Binary kappa and its percentile bootstrap interval over rows."""
    a, b = np.asarray(a, dtype=bool), np.asarray(b, dtype=bool)
    idx = np.random.default_rng(seed).integers(len(a), size=(n_boot, len(a)))
    A, B = a[idx], b[idx]
    po = (A == B).mean(1)
    pe = A.mean(1) * B.mean(1) + (1 - A.mean(1)) * (1 - B.mean(1))
    return d2l.cohen_kappa(a, b), np.quantile((po - pe) / (1 - pe),
                                              [0.025, 0.975])

ANNOT = load_annotations()
lab = ANNOT[ANNOT.qwen_score.notna()]
q, l = lab.qwen_score.astype(int).to_numpy(), lab.llama_score.to_numpy()
kb, (kb_lo, kb_hi) = kappa_ci(q >= THRESH_LABEL, l >= THRESH_LABEL)
qwk = d2l.quadratic_kappa(q, l)
mass = np.stack(lab.qwen_probs.to_numpy()).sum(1)
verdict = ('usable' if kb >= AGREE_MIN['kappa_bin'] and qwk >= AGREE_MIN['qwk']
           else 'weak' if kb >= AGREE_MIN['kappa_bin'] - 0.1 and
           qwk >= AGREE_MIN['qwk'] - 0.1 else 'poor')
print(f'{len(lab):,} rows with both labels; probability on the six digits: '
      f'median {np.median(mass):.3f}, 1st percentile '
      f'{np.quantile(mass, 0.01):.3f}')
print(f'exact agreement {np.mean(q == l):.3f}; quadratic-weighted kappa '
      f'{qwk:.3f}; binary (score >= {THRESH_LABEL}) kappa {kb:.3f} '
      f'[{kb_lo:.3f}, {kb_hi:.3f}]')
print(f'share with score >= {THRESH_LABEL}: Qwen3.5-4B '
      f'{np.mean(q >= THRESH_LABEL):.3f}, Llama-3-70B '
      f'{np.mean(l >= THRESH_LABEL):.3f}')
print('rows: Llama-3-70B score, columns: Qwen3.5-4B score')
print(pd.crosstab(pd.Series(l, name='llama'), pd.Series(q, name='qwen'))
      .reindex(index=range(6), columns=range(6), fill_value=0).to_string())
print(f'verdict against kappa >= {AGREE_MIN["kappa_bin"]} and QWK >= '
      f'{AGREE_MIN["qwk"]}: {verdict}' + (
          '  (FLAG for the author: the small labeller does not reproduce '
          'the rubric)' if verdict == 'poor' else ''))

just = lab[lab.qwen_just_score.notna()] if 'qwen_just_score' in lab else []
if len(just):
    jq = just.qwen_just_score.astype(int).to_numpy()
    dq = just.qwen_score.astype(int).to_numpy()
    jl = just.llama_score.to_numpy()
    def both(i):
        return (d2l.cohen_kappa(jq[i] >= 3, jl[i] >= 3) -
                d2l.cohen_kappa(dq[i] >= 3, jl[i] >= 3),
                d2l.quadratic_kappa(jq[i], jl[i]) -
                d2l.quadratic_kappa(dq[i], jl[i]))
    rng = np.random.default_rng(0)
    boot = np.array([both(rng.integers(len(jq), size=len(jq)))
                     for _ in range(1000)])
    d_kb, d_qwk = both(np.arange(len(jq)))
    (kb_l, q_l), (kb_h, q_h) = np.nanquantile(boot, [0.025, 0.975], axis=0)
    print(f'protocol check on {len(jq)} rows (justification first minus '
          f'digit probabilities, against Llama-3-70B): binary kappa '
          f'{d_kb:+.3f} [{kb_l:+.3f}, {kb_h:+.3f}], QWK {d_qwk:+.3f} '
          f'[{q_l:+.3f}, {q_h:+.3f}]')
    R['protocol'] = dict(n=len(jq), d_kappa=[d_kb, kb_l, kb_h],
                         d_qwk=[d_qwk, q_l, q_h])
R['agreement'] = dict(n=len(lab), exact=np.mean(q == l), qwk=qwk,
                      kappa_bin=[kb, kb_lo, kb_hi], verdict=verdict,
                      prevalence=dict(qwen=np.mean(q >= THRESH_LABEL),
                                      llama=np.mean(l >= THRESH_LABEL)),
                      digit_mass_median=np.median(mass))
```

```{.python .input #quality-filtering-labels-choice}
%%tab pytorch
complete = ANNOT.qwen_score.notna().all()
TRAIN_LABEL = 'qwen_score' if verdict != 'poor' and complete else 'llama_score'
OTHER_LABEL = 'llama_score' if TRAIN_LABEL == 'qwen_score' else 'qwen_score'
print(f'training labels: {TRAIN_LABEL} (' + (
    f'agreement {verdict}' if complete else
    'Qwen labels cover the pilot rows only: this run is a pilot') + ')')
if not complete:
    assert not RERUN & {'quality-results', 'quality-flags'}, (
        'regenerate edu-label-0 and edu-label-1 before the results')
    ANNOT['qwen_score'] = ANNOT.qwen_score.fillna(-1).astype(int)
ANNOT['qwen_score'] = ANNOT.qwen_score.astype(int)
SPLIT = {k: ANNOT[ANNOT.split == k].reset_index(drop=True)
         for k in ('train', 'val', 'test')}
print({k: len(v) for k, v in SPLIT.items()})
```

<!-- Prose: a frozen embedder plus a linear head; pooling and padding as in :numref:`sec_lm_text_embeddings` (last token, left padding, no instruction for documents). -->

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

<!-- Prose: registered vectors for the annotation rows and the stream; c0's vectors are computed in each build and cached locally (never hosted: they are derived from crawl text); two live checks. -->

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
print(f'{n_check} test texts, cosine of embed_docs to the registered '
      f'vectors: min {cos_reg.min():.4f}; to sentence-transformers: min '
      f'{cos_st.min():.4f}')
assert SMALL or min(cos_reg.min(), cos_st.min()) >= 0.999

C0_EMB_CACHE = pathlib.Path('..', 'data', 'lm', 'c0', 'emb-qwen3emb-v1.npz')
C0_SHA = [hashlib.sha256(d.text.encode('utf-8')).hexdigest() for d in C0]
cache = dict(np.load(C0_EMB_CACHE)) if C0_EMB_CACHE.exists() else {}
have = dict(zip(cache.get('sha256', []), cache.get('emb', [])))
missing = [i for i, h in enumerate(C0_SHA) if h not in have]
if missing and GPU:  # the c0 vectors: never hosted (rule P1), cached here
    t0 = time.time()
    Z = d2l.embed_docs([C0[i].text for i in missing], EMBEDDER, MAX_LEN)
    have.update(zip([C0_SHA[i] for i in missing], Z))
    np.savez(C0_EMB_CACHE, sha256=np.array(list(have)),
             emb=np.stack(list(have.values())))
    print(f'embedded {len(missing):,} c0 documents in '
          f'{time.time() - t0:.0f} s (cached in {C0_EMB_CACHE})')
    missing = []
C0_EMB = None if missing else np.stack([have[h] for h in C0_SHA])
print('c0 vectors:', f'{len(C0):,} from the local cache or this run'
      if C0_EMB is not None else
      f'{len(missing):,} of {len(C0):,} not cached and no GPU: c0 scores '
      'are read from the registered flags')
```

<!-- Prose: the two heads (a regression on the score, as the released classifier; a logistic regression on score >= 3), full-batch AdamW on the cached vectors (logistic regression and thresholds: one sentence to :numref:`sec_softmax`); fastText's model in PyTorch as the T0 path. -->

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

<!-- Prose: the heads on the training labels, the hashed bag on the training texts, the released classifier's registered scores; each threshold t* chosen on the validation rows by the best F1, never on test. -->

```{.python .input #quality-filtering-heads}
%%tab pytorch
def best_threshold(y, s):
    """The score threshold with the largest F1 (keep s >= t) on (y, s)."""
    cand = np.unique(np.quantile(s, np.linspace(0.5, 0.995, 400)))
    f1 = [d2l.binary_f1(y, s >= t) for t in cand]
    return float(cand[int(np.argmax(f1))])

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
T_STAR = {m: best_threshold(Y['val'] >= THRESH_LABEL, s['val'])
          for m, s in SCORES.items()}
print('thresholds t* (best validation F1):',
      ', '.join(f'{m} {t:.3f}' for m, t in T_STAR.items()))

n_live = 16 if SMALL else 2000  # the released classifier, live
live = d2l.released_edu_scores(SPLIT['test'].text[:n_live].tolist())
gap = np.abs(live - released['test'][:n_live])
print(f'released classifier live on {n_live} test rows: largest difference '
      f'to the registered scores {gap.max():.3f}')
assert gap.max() < (0.05 if GPU else 0.1)
R['thresholds'] = T_STAR
```

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
CLF = pd.DataFrame(rows)
print(f'{len(test):,} test rows (parameters: the trained part; the heads '
      f'also run the frozen {EMBEDDER})')
print(CLF.round(3).to_string(index=False))

y = test[TRAIN_LABEL].to_numpy() >= THRESH_LABEL
a, b = SCORES[MAIN]['test'] >= T_STAR[MAIN], SCORES['released']['test'] >= \
    T_STAR['released']
rng = np.random.default_rng(0)
boot = []
for _ in range(2000):  # resample rows, both classifiers on the same rows
    i = rng.integers(len(y), size=len(y))
    boot.append(d2l.binary_f1(y[i], a[i]) - d2l.binary_f1(y[i], b[i]))
d_f1 = d2l.binary_f1(y, a) - d2l.binary_f1(y, b)
lo, hi = np.quantile(boot, [0.025, 0.975])
acc = d2l.paired_bootstrap(correct[MAIN], correct['released'])
print(f'{MAIN} minus released, on {TRAIN_LABEL}: F1 {d_f1:+.3f} '
      f'[{lo:+.3f}, {hi:+.3f}] (paired bootstrap over rows); accuracy '
      f'{acc.mean:+.3f} [{acc.ci[0]:+.3f}, {acc.ci[1]:+.3f}]')
R['classifiers'] = CLF.drop(columns='ms_per_doc').to_dict('records')
R['f1_diff'] = [d_f1, lo, hi]
```

### Label Noise and the F1 Ceiling
:label:`subsec_lm_quality_ceiling`

<!-- Prose: F1 of a threshold from prior, recall and retention, F1 = 2 pi TPR / (pi + rho) (eq_lm_quality_f1_retention, three lines); the ceiling of a perfect classifier of the true label under symmetric label noise eta (eq_lm_quality_f1_ceiling, short proof); the two labellers' disagreement read as an estimate of eta, under the stated assumption that both flip the true label independently with the same rate. -->

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

q = ANNOT.qwen_score.to_numpy() >= THRESH_LABEL
l = ANNOT.llama_score.to_numpy() >= THRESH_LABEL
def implied(i):
    """eta from the disagreement d = 2 eta (1 - eta), the true prior from
    the observed one, and the ceiling they give."""
    d = np.mean(q[i] != l[i])
    eta = (1 - math.sqrt(max(1 - 2 * d, 0))) / 2
    pi_obs = np.mean(np.r_[q[i], l[i]])
    pi_true = (pi_obs - eta) / (1 - 2 * eta)
    return eta, f1_ceiling(pi_true, eta)
eta, ceil = implied(np.arange(len(q)))
rng = np.random.default_rng(0)
boot = np.array([implied(rng.integers(len(q), size=len(q)))
                 for _ in range(1000)])
(eta_lo, ceil_lo), (eta_hi, ceil_hi) = np.quantile(boot, [0.025, 0.975], 0)
print(f'the labellers disagree on {np.mean(q != l):.3f} of the rows: eta '
      f'{eta:.3f} [{eta_lo:.3f}, {eta_hi:.3f}], ceiling F1 {ceil:.3f} '
      f'[{ceil_lo:.3f}, {ceil_hi:.3f}]')
f1s = CLF.set_index('classifier')[f'F1 {TRAIN_LABEL}']
print('measured test F1:', ', '.join(f'{m} {v:.3f}' for m, v in f1s.items()),
      '| all below the ceiling:', bool((f1s < ceil).all()))
R['ceiling'] = dict(identity_gap=gap, simulation=rows,
                    eta=[eta, eta_lo, eta_hi], f1=[ceil, ceil_lo, ceil_hi],
                    below=bool((f1s < ceil).all()))
```

## Choosing a Threshold
:label:`subsec_lm_quality_threshold`

<!-- Prose: the threshold is a precision-recall choice made on validation labels and read as a retained fraction; retained tokens for a stated corpus size; data-limited budgets favour low thresholds, compute-limited ones high; the manifest keeps the continuous score so 20.4 can use score bands. -->

```{.python .input #quality-filtering-threshold}
%%tab pytorch
fig, axes = d2l.plt.subplots(1, 3, figsize=(10, 3), sharey=True)
y = test[TRAIN_LABEL].to_numpy() >= THRESH_LABEL
curves = {}
for m in ('regression head', 'hashed bag', 'released'):
    s = SCORES[m]['test']
    ts = np.quantile(s, np.linspace(0.3, 0.995, 120))
    pred = s[None] >= ts[:, None]
    tp = (pred & y).sum(1)
    curves[m] = dict(rho=pred.mean(1), precision=tp / np.maximum(
        pred.sum(1), 1), recall=tp / y.sum(),
        f1=2 * tp / (y.sum() + pred.sum(1)))
for ax, metric in zip(axes, ('precision', 'recall', 'f1')):
    d2l.plot([c['rho'] for c in curves.values()],
             [c[metric] for c in curves.values()], 'retained fraction',
             metric if metric != 'f1' else 'F1', legend=list(curves)
             if metric == 'recall' else None, xlim=[0, 0.4], axes=ax,
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
RETAIN = pd.DataFrame(rows)
print(RETAIN.round(4).to_string(index=False))
print(f'of 1e12 tokens of such a dump, {MAIN} keeps about '
      f'{1e12 * RETAIN.tokens[0]:.2g}')
R['retention'] = RETAIN.to_dict('records')
```

## Auditing Removals
:label:`subsec_lm_quality_audit`

<!-- Prose: what an audit is (removal rates by group with intervals, and a reading of examples); groups: English variety by country-code domain (a weak proxy, said so), host class by rule, six FineWeb-2 languages under the heuristics only (the classifier is English-only); cluster bootstrap by registered domain, because 3,000 documents from one site are not 3,000 samples; groups under 100 documents or 10 domains are "too few". -->

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

<!-- Prose: three decisions on the stream (heuristics, our classifier at t*, both); the audit by country-code domain against .com and by host class against "other"; the same audit on each half of the stream, the halves split by registered domain so they share no cluster; the filter as a covariate shift keyed on a proxy label (one sentence restating :numref:`sec_environment-and-distribution-shift`); Gururangan et al. and Dodge et al. -->

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
print('stream halves:', S.half.value_counts().sort_index().to_dict(),
      '| removal rates:', {a: round(1 - S[k].mean(), 3)
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
print(show[cols].round(3).to_string(index=False))
few = full[full.too_few & (full.arm == 'both')].group.tolist()
print('too few (under 100 documents or 10 domains):', ', '.join(few) or '-')

beyond = show[(show.arm == 'both') & ((show.diff_lo > 0) |
                                      (show.diff_hi < 0))]
clf = show[show.arm == 'classifier'].set_index(['grouping', 'group'])
c4 = [(g, b) for g, b in zip(beyond.grouping, beyond.group)
      if (clf.diff_lo[(g, b)] > 0 or clf.diff_hi[(g, b)] < 0) and
      clf.same_sign[(g, b)] and beyond.set_index(['grouping', 'group'])
      .same_sign[(g, b)]]
print('groups whose combined removal rate differs from the reference '
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

<!-- Prose: 200 removed documents read under a two-line rubric: 100 removed by our classifier alone with a score within 0.3 of t*, 100 removed by our heuristics (whatever the classifier decides: on FineWeb's output the rules remove few documents, and the classifier keeps few of those); the labels come from a model (Claude Opus 5.5, the coding agent that built this notebook), not from people; removal precision with Wilson intervals; five excerpts. -->

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
print('removed-document pools:', {a: len(i) for a, i in arms.items()},
      '| reviewed:', {a: len(i) for a, i in picked.items()})
REVIEW = pd.concat([S.iloc[i].assign(arm=a) for a, i in picked.items()],
                   ignore_index=True)
REVIEW['score'] = np.concatenate([SCORES[MAIN]['stream'][i]
                                  for i in picked.values()])
LABELS_KEY = 'lm-data-removal-labels-v1'
if LABELS_KEY not in d2l.DATA_HUB:  # the sample for the labeller
    sample = REVIEW[['id', 'arm', 'url', 'score', 'codes', 'text']].assign(
        text=REVIEW.text.str[:3000], rubric=REMOVAL_RUBRIC)
    d2l.lm_save_rerun('lm-data-removal-sample-v1.jsonl', lambda f:
                      d2l.write_jsonl(f, json.loads(sample.to_json(
                          orient='records'))))
    print(f'{LABELS_KEY} not registered: removal precision pending')
    R['review'] = None
else:
    labels = d2l.read_jsonl(d2l.lm_artifact(LABELS_KEY))
    rubric_sha = hashlib.sha256(REMOVAL_RUBRIC.encode()).hexdigest()
    assert (labels.rubric_sha256 == rubric_sha).all(), 'another rubric'
    REVIEW = REVIEW.merge(labels[['id', 'label', 'rationale', 'labeller']],
                          on='id', validate='1:1')
    rows = []
    for a, t in REVIEW.groupby('arm'):
        k = int((t.label == 'drop').sum())
        lo, hi = d2l.wilson_ci(k, len(t))
        rows.append(dict(arm=a, n=len(t), drop=k, precision=k / len(t),
                         lo=lo, hi=hi))
    PREC = pd.DataFrame(rows)
    print(f'labeller: {REVIEW.labeller.iloc[0]}; removal precision (share '
          'labelled "drop"):')
    print(PREC.round(3).to_string(index=False))
    for _, r in REVIEW.sample(5, random_state=0).iterrows():
        print(f'\n[{r.arm}, score {r.score:.2f}, label {r.label}] '
              f'{r.rationale}')
        print(textwrap.fill(r.text[:300].replace('\n', ' ') + ' …', 100,
                            initial_indent='  ', subsequent_indent='  '))
    print('\n' + ATTRIBUTION)
    R['review'] = PREC.to_dict('records')
long = S[S.keep_heur & (S.text.str.len() >= 1500)]
R['opening'] = dict(low_score_id=long.id.iloc[int(np.argmin(
    SCORES[MAIN]['stream'][long.index]))])
if R['review'] is not None:
    kept = REVIEW[(REVIEW.label == 'keep') & (REVIEW.arm == 'classifier only')]
    R['opening']['kept_label_id'] = (kept.sort_values('score').id.iloc[0]
                                     if len(kept) else None)
```

```{.python .input #quality-filtering-agree-stream}
%%tab pytorch
a = SCORES[MAIN]['stream'] >= T_STAR[MAIN]
b = SCORES['released']['stream'] >= T_STAR['released']
k, (k_lo, k_hi) = kappa_ci(a, b)
print(f'on the stream, {MAIN} and the released classifier keep '
      f'{a.mean():.3f} and {b.mean():.3f}; they agree on '
      f'{np.mean(a == b):.3f} of the documents, Cohen\'s kappa {k:.3f} '
      f'[{k_lo:.3f}, {k_hi:.3f}]')
R['stream_kappa'] = [k, k_lo, k_hi]
```

## The Filter in the Pipeline
:label:`subsec_lm_quality_pipeline`

<!-- Prose: the two stage records on c0 (heuristics: drops by first reason code; edu_score: no drops, the threshold and the head's hash as parameters, the score kept in the flags), the lm-c1-docs-v1 flags (no text), the manifest extended. -->

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
    head_bytes))).hexdigest()
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
    print('live check on 8 c0 documents (CPU, fp32): largest score '
          'difference to the registered flags '
          f'{np.abs(s8 - c0_score[:8]).max():.3f}')
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
    print(json.dumps({k: s[k] for k in ('name', 'n_in', 'n_out', 'drops')}))
above = sum(C0_CODES[d.id] == [] and v >= T_STAR[MAIN]
            for d, v in zip(C0, c0_score))
print(f'c0 documents kept by the rules with edu_score >= t*: {above:,} of '
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
    print(f'against {FLAGS_KEY}: reason codes '
          f'{"identical" if codes_same else "DIFFER"}; edu_score largest '
          f'difference {diff.max():.3f}; {flip} decisions '
          'flip (each within that difference of t*)')
    assert codes_same and flip <= np.sum(np.abs(c0_score - T_STAR[MAIN])
                                         <= diff.max())
R['c1_manifest'] = m.to_dict()
```

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
    reg = json.load(open(d2l.lm_artifact(RES_KEY)))
    live = {k: v for k, v in results.items() if k != 'ppl'}
    a = d2l.json_agreement(live, reg)
    same = reg['c1_manifest'] == results['c1_manifest']
    print(f'against {RES_KEY}: {a["equal"]}/{a["values"]} values equal, '
          f'largest numeric difference {a["max_diff"]:.2g}; c1 manifest '
          f'{"identical" if same else "DIFFERS"}')
    head = dict(np.load(d2l.lm_artifact(HEAD_KEY)))
    w_gap = np.abs(head['weight'] - head_arrays['weight.npy']).max()
    print(f'head weights against {HEAD_KEY}: largest difference {w_gap:.1e}')
    assert same and a['max_diff'] <= 0.005 and w_gap < 1e-5
```

## Summary

<!-- Prose: Summary (3-6 sentences). -->

## Exercises

<!-- Prose: Exercises (7; spec part 8). -->
