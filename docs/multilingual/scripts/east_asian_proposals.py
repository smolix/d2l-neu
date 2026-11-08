#!/usr/bin/env python3
"""Cache machine suggestions as unverified editorial candidates only."""
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import Request, urlopen

BASE=Path(__file__).resolve().parents[1]
OUT=BASE/'candidates'/'east_asian-proposals.jsonl'
LOCALES={'zh-Hans':'zh-CN','ja':'ja','ko':'ko','vi':'vi'}

def main():
    concepts=[json.loads(x) for x in (BASE/'concepts.jsonl').read_text().splitlines() if x]
    have={}
    if OUT.exists():
        for x in OUT.read_text().splitlines():
            try:
                d=json.loads(x);have[d['locale'],d['concept_id']]=d
            except Exception:pass
    def one(job):
        locale,c=job
        u='https://translate.googleapis.com/translate_a/single?'+urlencode({'client':'gtx','sl':'en','tl':LOCALES[locale],'dt':'t','q':c['en']})
        for attempt in range(3):
            try:
                r=json.loads(urlopen(Request(u,headers={'User-Agent':'Mozilla/5.0'}),timeout=20).read())
                term=''.join(z[0] for z in r[0] if z and z[0]).strip()
                return {'locale':locale,'concept_id':c['id'],'term':term,'candidate_only':True,'method':'machine_translation_suggestion'}
            except Exception as e:
                if attempt==2:return {'locale':locale,'concept_id':c['id'],'term':None,'error':str(e),'candidate_only':True}
                time.sleep(0.5*(attempt+1))
    jobs=[(loc,c) for loc in LOCALES for c in concepts if (loc,c['id']) not in have]
    print('fetch',len(jobs),flush=True)
    with ThreadPoolExecutor(max_workers=16) as ex, OUT.open('a') as f:
        for future in as_completed([ex.submit(one,j) for j in jobs]):
            row=future.result();f.write(json.dumps(row,ensure_ascii=False)+'\n');f.flush()
    print('done',len(jobs))

if __name__=='__main__':main()
