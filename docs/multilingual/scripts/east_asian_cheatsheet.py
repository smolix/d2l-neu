#!/usr/bin/env python3
"""Extract direct bilingual term introductions in CS229/230 teaching translations."""
import json
import re
from collections import defaultdict
from pathlib import Path

BASE=Path(__file__).resolve().parents[1]
ROOT=Path('/tmp/d2l-glossary-corpora/cheatsheet-translation')

def clean(s):
    return re.sub(r'\s+',' ',re.sub(r'\*\*|\*','',s)).strip(' :：.。-–― ')

def norm(s):return clean(s).casefold()

def main():
    concepts=[json.loads(x) for x in (BASE/'concepts.jsonl').read_text().splitlines() if x]
    by_en=defaultdict(list)
    for c in concepts:
        for v in [c['en'],*c.get('aliases_en',[])]:by_en[norm(v)].append(c)
    out=[]
    for lang in ('zh','ja','ko','vi'):
        for file in (ROOT/lang).glob('*.md'):
            lines=file.read_text(errors='replace').splitlines()
            for i,line in enumerate(lines):
                match=re.match(r'^\*{0,2}\s*(\d+)\.\s+(.+)',line)
                if not match:continue
                en=clean(match.group(2).split('―',1)[0].split('—',1)[0])
                if len(en)>70:continue
                translated=None;ti=None
                for j in range(i+1,min(i+7,len(lines))):
                    if '&#10230;' in lines[j] or '⟶' in lines[j]:
                        tail=re.split(r'&#10230;|⟶',lines[j],maxsplit=1)[-1].strip()
                        if not tail and j+1<len(lines):tail=lines[j+1].strip();ti=j+2
                        else:ti=j+1
                        translated=clean(re.split(r'\s+[―—-]\s+',tail,maxsplit=1)[0]);break
                if not translated:continue
                variants=[(en,translated)]
                if en.startswith('[') and translated.startswith('['):
                    es=[clean(x) for x in en.strip('[]').split(',')]
                    ts=[clean(x) for x in translated.strip('[]').split(',')]
                    if len(es)==len(ts):variants=list(zip(es,ts))
                for eng,target in variants:
                    for c in by_en.get(norm(eng),[]):
                        if target.casefold()==eng.casefold() or not target:continue
                        out.append({'locale':lang,'concept_id':c['id'],'en':eng,
                                    'term':target,'path':f'{lang}/{file.name}',
                                    'line':ti,'source_english_line':i+1,
                                    'context':f'Numbered bilingual teaching statement {match.group(1)} directly pairs "{eng}" with "{target}".',
                                    'candidate_only':True})
    dest=BASE/'candidates'/'east_asian-cheatsheet.jsonl'
    dest.write_text(''.join(json.dumps(x,ensure_ascii=False)+'\n' for x in out))
    for lang in ('zh','ja','ko','vi'):
        d=[x for x in out if x['locale']==lang]
        print(lang,len(d),len({x['concept_id'] for x in d}))

if __name__=='__main__':main()
