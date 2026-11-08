#!/usr/bin/env python3
"""Find proposal phrases in aligned translated sections (review queue only)."""
import json
import re
from collections import defaultdict,Counter
from pathlib import Path
from build_east_asian import heading_pairs, CORPUS, BASE

def rows(path):
    return [json.loads(x) for x in path.read_text().splitlines() if x]

def main():
    concepts={c['id']:c for c in rows(BASE/'concepts.jsonl')}
    existing={(c['locale'],c['concept_id']) for c in rows(BASE/'candidates'/'east_asian-d2l.jsonl')}
    props=rows(BASE/'candidates'/'east_asian-proposals.jsonl')
    bylang=defaultdict(list)
    for p in props:
        loc=p['locale'];lang={'zh-Hans':'zh','ja':'ja','ko':'ko','vi':'vi'}.get(loc)
        c=concepts.get(p['concept_id']);term=p.get('term')
        if not lang or not c or not term or len(term)<3 or term.casefold()==c['en'].casefold():continue
        if (lang,c['id']) in existing:continue
        bylang[lang].append((c,term))
    out=[];seen=set()
    for lang,entries in bylang.items():
        root=CORPUS/('d2l-'+lang)
        for f in root.rglob('*.md'):
            if f.name.endswith('_origin.md') or not f.relative_to(root).parts[0].startswith('chapter_'):continue
            origin=f.with_name(f.stem+'_origin.md')
            if not origin.exists():continue
            o=origin.read_text(errors='replace').splitlines();t=f.read_text(errors='replace').splitlines()
            pairs=heading_pairs(o,t)
            for k,((oi,_,heading),(ti,_,_)) in enumerate(pairs):
                oe=pairs[k+1][0][0]-1 if k+1<len(pairs) else len(o)
                te=pairs[k+1][1][0]-1 if k+1<len(pairs) else len(t)
                if oe-oi>180 or te-ti>180:continue
                ost=' '.join(o[oi:oe]);target=t[ti:te]
                for c,term in entries:
                    if not re.search(r'(?<![A-Za-z])'+re.escape(c['en'])+r'(?![A-Za-z])',ost,re.I):continue
                    for n,line in enumerate(target,ti+1):
                        if term in line and not line.lstrip().startswith(('```','![',':label:')):
                            key=(lang,c['id'],str(f.relative_to(root)),n)
                            if key in seen:continue
                            seen.add(key)
                            out.append({'locale':lang,'concept_id':c['id'],'en':c['en'],
                                        'term':term,'path':str(f.relative_to(root)),
                                        'line':n,'section':heading,'context':line[:350],
                                        'candidate_only':True})
                            break
    path=BASE/'candidates'/'east_asian-context.jsonl'
    path.write_text(''.join(json.dumps(x,ensure_ascii=False)+'\n' for x in out))
    for lang in bylang:
        d=[x for x in out if x['locale']==lang]
        print(lang,len(d),len({x['concept_id'] for x in d}))

if __name__=='__main__':main()
