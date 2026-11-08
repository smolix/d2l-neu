#!/usr/bin/env python3
"""Build source-attested East Asian term shards from pinned D2L prose."""
import json
import re
from collections import defaultdict, Counter
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from urllib.request import Request, urlopen
from html.parser import HTMLParser

BASE = Path(__file__).resolve().parents[1]
CORPUS = Path('/tmp/d2l-glossary-corpora')
LOCALES = {'zh-Hans': 'zh', 'ja': 'ja', 'ko': 'ko', 'vi': 'vi'}
REPOS = {'zh': ('d2l-ai/d2l-zh', 'e6b18ccea71451a55fcd861d7b96fddf2587b09a'),
         'ja': ('d2l-ai/d2l-ja', 'f84fb229198b49f2dc532ca72ce72b5611a124cb'),
         'ko': ('d2l-ai/d2l-ko', '9857ea9899409b1e17003dcce7859d60aa892559'),
         'vi': ('d2l-ai/d2l-vi', 'c775d6b4998e6243ec5d11f950e67679555a2c74')}
GOOGLE_TOPICS = ('fundamentals','metrics','clustering','df','generative',
                 'googlecloud','responsible-ai','tensorflow','agent')
GOOGLE_LOCALES = {'zh':'zh-cn','ja':'ja','ko':'ko','vi':'vi'}

class HeadingParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.headings=[]; self.capture=False; self.parts=[]; self.anchor=None
    def handle_starttag(self,tag,attrs):
        if tag=='h2':
            attrs=dict(attrs); self.anchor=attrs.get('id');self.capture=True;self.parts=[]
    def handle_endtag(self,tag):
        if tag=='h2' and self.capture:
            self.headings.append((self.anchor,clean(''.join(self.parts))))
            self.capture=False
    def handle_data(self,data):
        if self.capture:self.parts.append(data)

def fetch_google(topic,locale):
    cache=Path('/tmp/d2l-glossary-google')/f'east_asian-{topic}-{locale}.html'
    if not cache.exists():
        url=f'https://developers.google.com/machine-learning/glossary/{topic}?hl={locale}'
        data=urlopen(Request(url,headers={'User-Agent':'Mozilla/5.0'}),timeout=30).read()
        cache.write_bytes(data)
    s=cache.read_text(errors='replace')
    parser=HeadingParser();parser.feed(s)
    return {(anchor,term) for anchor,term in parser.headings if anchor and term},len(s)

def read_jsonl(path):
    return [json.loads(x) for x in path.read_text().splitlines() if x.strip()]

def norm(s):
    return re.sub(r'\s+', ' ', re.sub(r'[*`_]+', '', s)).strip().casefold()

def clean(s):
    return re.sub(r'\s+', ' ', re.sub(r'[*`_]+', '', s)).strip(' \t\u3000。，、:：;；')

def headings(lines):
    return [(i, len(m.group(1)), clean(m.group(2))) for i, line in enumerate(lines, 1)
            if (m := re.match(r'^(#{1,6})\s+(.+)', line))]

def heading_pairs(orig, target):
    """Pair headings within chapter by order, with section labels as guardrails."""
    eh, th = headings(orig), headings(target)
    if not eh or not th:
        return []
    # Most translated chapters preserve heading order and level. A mismatch is
    # a reason to drop candidate evidence, not to guess the correspondence.
    if len(eh) != len(th) or [h[1] for h in eh] != [h[1] for h in th]:
        return []
    return list(zip(eh, th))

def emphasis(lines,start,end):
    found=[]; fenced=False
    for i in range(start,end):
        line=lines[i]
        if line.lstrip().startswith(('```','~~~')):
            fenced=not fenced;continue
        if fenced or line.lstrip().startswith(('![',':label:',':eqlabel:')):continue
        for m in re.finditer(r'(?<!\*)\*([^*\n]{2,90})\*(?!\*)',line):
            found.append((i+1,clean(m.group(1))))
    return found

def add_candidate(out, concept, term, lang, path, line, method, context):
    term = clean(term)
    if not term or len(term) > 100 or norm(term) == norm(concept['en']):
        return
    if re.search(r'[`$<>\\\[\]{}]', term):
        return
    out[lang][concept['id']].append(dict(term=term, path=path, line=line,
                                            method=method, context=context))

def main():
    concepts = read_jsonl(BASE/'concepts.jsonl')
    by_en = defaultdict(list)
    for c in concepts:
        for alias in [c['en'], *c.get('aliases_en', [])]:
            by_en[norm(alias)].append(c)
    out = defaultdict(lambda: defaultdict(list))
    for lang in LOCALES.values():
        root = CORPUS/('d2l-'+lang)
        for file in root.rglob('*.md'):
            if file.name.endswith('_origin.md') or not file.relative_to(root).parts[0].startswith('chapter_'):
                continue
            path = str(file.relative_to(root))
            target = file.read_text(errors='replace').splitlines()
            origin_path = file.with_name(file.stem+'_origin.md')
            if origin_path.exists():
                origin=origin_path.read_text(errors='replace').splitlines()
                pairs=heading_pairs(origin,target)
                for (oi, level, en), (ti, _, translated) in pairs:
                    for c in by_en.get(norm(en), []):
                        add_candidate(out,c,translated,lang,path,ti,'aligned_heading',f'Translated section heading for English "{en}" in the paired original at line {oi}.')
                # When the paired section preserves the number and order of
                # emphasized expressions, introductions align positionally.
                # Still candidate-only until the translated passage is read.
                for k,pair in enumerate(pairs):
                    (oi,_,en),(ti,_,_)=pair
                    oe=pairs[k+1][0][0]-1 if k+1<len(pairs) else len(origin)
                    te=pairs[k+1][1][0]-1 if k+1<len(pairs) else len(target)
                    src=emphasis(origin,oi,oe);dst=emphasis(target,ti,te)
                    if not src or len(src)!=len(dst) or len(src)>40:continue
                    for (eli,et),(tli,tt) in zip(src,dst):
                        for c in by_en.get(norm(et),[]):
                            add_candidate(out,c,tt,lang,path,tli,'aligned_emphasis',f'Emphasized expression aligned with English "{et}" at paired original line {eli}, section "{en}".')
            # D2L Chinese translators explicitly introduce a Chinese phrase
            # followed by its English technical equivalent in parentheses.
            if lang == 'zh':
                for i, line in enumerate(target, 1):
                    if line.lstrip().startswith(('```', '![', ':label:', ':eqlabel:')):
                        continue
                    for m in re.finditer(r'\*([^*]{1,60})\*\s*[（(]([^()（）]{2,90})[）)]', line):
                        for c in by_en.get(norm(m.group(2)), []):
                            add_candidate(out,c,m.group(1),lang,path,i,'bilingual_introduction',f'Chinese term introduced with English "{m.group(2)}" in the same prose sentence.')
        # Explicit editorial terminology tables in the D2L repositories.
        terminology = root/'TERMINOLOGY.md'
        if terminology.exists():
            for i,line in enumerate(terminology.read_text().splitlines(),1):
                if lang == 'zh' and '，' in line and not line.startswith('#'):
                    term,en = line.split('，',1)
                    en = re.sub(r'\s*\([^)]*\)\s*$', '', en)
                    for c in by_en.get(norm(en), []):
                        add_candidate(out,c,term,lang,'TERMINOLOGY.md',i,'terminology_list',f'D2L Chinese English–Chinese terminology entry for "{en}".')
                elif lang == 'ko' and line.startswith('|'):
                    parts=[x.strip() for x in line.strip('|').split('|')]
                    if len(parts)==2 and parts[1] and not set(parts[0]) <= set('- '):
                        for c in by_en.get(norm(parts[0]), []):
                            add_candidate(out,c,parts[1],lang,'TERMINOLOGY.md',i,'terminology_list',f'D2L Korean English–Korean terminology entry for "{parts[0]}".')
        if lang=='vi':
            glossary=root/'glossary.md'
            if glossary.exists():
                special={'bias (model parameter)':'bias',
                         'bias (of an estimator)':'statistical-bias',
                         'variance (of an estimator)':'variance',
                         'policy (in reinforcement learning)':'policy',
                         'precision (vs accuracy metric)':'precision',
                         'pixel (component of digital images)':'pixel',
                         'overfit':'overfitting','underfit':'underfitting'}
                by_id={c['id']:c for c in concepts}
                for i,line in enumerate(glossary.read_text().splitlines(),1):
                    if not line.startswith('|'):continue
                    parts=[x.strip() for x in line.strip('|').split('|')]
                    if len(parts)<2 or not parts[0] or not parts[1]:continue
                    en,term=parts[:2]
                    if norm(en) in special:
                        matches=[by_id[special[norm(en)]]] if special[norm(en)] in by_id else []
                    else:
                        matches=by_en.get(norm(en),[])
                        if not matches:
                            base=re.sub(r'\s*\([^)]*\)','',en).strip()
                            matches=by_en.get(norm(base),[])
                    for c in matches:
                        # Intra-field explanatory glosses are not part of the
                        # standard term; preserve the raw line in evidence.
                        pref=re.sub(r'\s*\(hoặc [^)]+\)','',term).strip()
                        add_candidate(out,c,pref,lang,'glossary.md',i,
                                      'editorial_glossary',f'Vietnamese D2L editorial glossary: "{en}" → "{term}".')
    # Topic-specific Google pages, unlike the monolithic glossary shell, have
    # localized heading AND definition bodies for these locales. Align by the
    # publisher's stable same-page anchor, then inspect sense at curation.
    google = {}
    tasks=[(topic,loc) for topic in GOOGLE_TOPICS for loc in ('en',*GOOGLE_LOCALES.values())]
    with ThreadPoolExecutor(max_workers=8) as pool:
        futures={pool.submit(fetch_google,*pair):pair for pair in tasks}
        for future in as_completed(futures):
            pair=futures[future]
            try:google[pair]=future.result()[0]
            except Exception as e:print('FETCH FAILED',pair,e)
    for topic in GOOGLE_TOPICS:
        en={a:t for a,t in google.get((topic,'en'),set())}
        for lang,locale in GOOGLE_LOCALES.items():
            local={a:t for a,t in google.get((topic,locale),set())}
            for anchor,en_term in en.items():
                if anchor not in local:continue
                for c in by_en.get(norm(en_term),[]):
                    add_candidate(out,c,local[anchor],lang,f'google:{topic}#{anchor}',0,
                                  'google_topic_heading',f'Localized Google ML teaching glossary heading aligned to English "{en_term}" by anchor; definition body requires sense check.')
    candidate_file = BASE/'candidates'/'east_asian-d2l.jsonl'
    candidate_file.parent.mkdir(exist_ok=True)
    with candidate_file.open('w') as f:
        for lang in LOCALES.values():
            for c in concepts:
                for x in out[lang].get(c['id'],[]):
                    f.write(json.dumps({'locale':lang,'concept_id':c['id'],**x},ensure_ascii=False)+'\n')
    for lang in LOCALES.values():
        counts=Counter(x['method'] for candidates in out[lang].values() for x in candidates)
        print(lang,len(out[lang]),sum(counts.values()),dict(counts))

if __name__ == '__main__':
    main()
