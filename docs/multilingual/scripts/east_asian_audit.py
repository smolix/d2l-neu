#!/usr/bin/env python3
"""Repeatable exact-passage spot checks for East Asian semantic review."""
import json
from pathlib import Path
from collections import Counter
from east_asian_finalize import BASE,CORPUS,LOCALES

SAMPLES={
 'zh-Hans':['linear-regression','bias','precision','accuracy','loss-function','objective-function',
            'gradient','gradient-descent','stochastic-gradient-descent','probability',
            'random-variable','norm','kernel','neural-network','state','action','reward',
            'query','value','backpropagation-through-time'],
 'ja':['linear-regression','feature','weight','bias','accuracy','precision','recall',
       'loss-function','objective-function','gradient','gradient-descent','likelihood',
       'kernel','statistical-bias','state','reward','query','value','jacobian','transposed-convolution'],
 'ko':['linear-regression','feature','weight','bias','accuracy','precision','recall',
       'loss-function','objective-function','gradient','gradient-descent','kernel',
       'statistical-bias','state','reward','query','value','computational-graph',
       'central-limit-theorem','reset-gate'],
 'vi':['linear-regression','feature','bias','statistical-bias','accuracy','loss-function',
       'objective-function','chain-rule','backpropagation','automatic-differentiation',
       'probability','kernel','state','reward','neural-network','overfitting',
       'underfitting','covariance','joint-distribution','instance-segmentation']}

def main():
    sources={x['id']:x for x in map(json.loads,(BASE/'sources'/'group-east-asian.jsonl').open())}
    for locale,ids in SAMPLES.items():
        assert len(ids)==20 and len(set(ids))==20
        data={x['concept_id']:x for x in map(json.loads,(BASE/'terms'/f'{locale}.jsonl').open())}
        print('\n##',locale)
        for cid in ids:
            d=data[cid];assert d['status'].startswith('verified'),(locale,cid,d['status'])
            e=d['evidence'][0];s=sources[e['source_id']]
            if s.get('path'):
                n=int(e['locator'].split()[-1])
                path=((CORPUS/'cheatsheet-translation'/s['path'])
                      if s.get('repository')=='shervinea/cheatsheet-translation'
                      else (CORPUS/('d2l-'+LOCALES[locale])/s['path']))
                line=path.read_text().splitlines()[n-1]
                assert e['source_term'] in line.replace('*','').replace('`','').replace('_',''),(locale,cid)
                finding=line[:180].replace('|','/')
            else:finding=e['context']
            print(f"| {cid} | {d['preferred']} | {d['status']} | {s['url']} ({e['locator']}) | {finding} |")
        print('passage checks:',len(ids))

if __name__=='__main__':main()
