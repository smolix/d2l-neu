#!/usr/bin/env python3
"""Join reviewed Romance terms, bilingual candidates, and all English concepts.

The editorial seed is embedded below. Only individually selected bilingual
pairs are verified; other phrases remain proposed. An empty cell stays unresolved.
"""
import json,re,unicodedata
from pathlib import Path
P=Path(__file__).resolve().parents[1]
ROOT=Path('/tmp/d2l-glossary-corpora/cheatsheet-translation')
COMMIT='99f45b6d65b8e60f637d8ac0db15f76327197c23'
SEED=Path(__file__).with_name('romance-seed.tsv')
CAND=P/'candidates'/'romance-cheatsheet.jsonl'
concepts=[json.loads(x) for x in (P/'concepts.jsonl').read_text().splitlines() if x]
source_path=P/'sources'/'group-es-fr-pt.jsonl'
sources={x['id']:x for x in map(json.loads,source_path.read_text().splitlines())}
base_terms={}
for line in SEED.read_text().splitlines():
 parts=line.split('|')
 if len(parts)==4:
  for i,loc in enumerate(('es','fr','pt-BR'),1):base_terms[(loc,parts[0])]=parts[i]
candidates=[json.loads(x) for x in CAND.read_text().splitlines()]
# Selected individually against the numbered English and target source lines.
# Ambiguous kernel/bias/hypothesis/value and malformed or questionable wording excluded.
safe=set('matrix identity-matrix diagonal-matrix transpose trace determinant norm gradient hessian regression loss-function logistic-loss hinge-loss cross-entropy linear-regression logistic-regression neural-network gradient-descent normal-equations softmax-regression exponential-family lagrangian training-error vc-dimension conditional-probability independence random-variable variance standard-deviation covariance correlation random-sample sample-mean central-limit-theorem sample-variance expectation mean jensen-s-inequality convergence principal-component-analysis confusion-matrix model-selection vocabulary training-set validation-set cross-validation regularization hidden-layer activation-function sigmoid cross-entropy-loss learning-rate backpropagation batch-normalization input-gate forget-gate output-gate policy value-function vector dynamic-programming policy-evaluation optimal-policy value-iteration data-augmentation random-crop cosine-similarity language-model perplexity negative-sampling machine-translation beam-search beam-width length-normalization feature-vector margin residual squared-loss stochastic-gradient-descent hypothesis-class dataset unsupervised-learning objective-function rectified-linear-unit receptive-field object-detection image-classification intersection-over-union query gibbs-sampling conditional-independence maximum-likelihood convolution stride padding'.split())
blocked={('fr','loss-function'),('fr','logistic-loss'),('fr','cross-entropy'),('es','hinge-loss'),('es','policy'),('pt-BR','policy'),('pt-BR','covariance'),('pt-BR','sample-variance'),('pt-BR','jensen-s-inequality'),('pt-BR','learning-rate'),('pt-BR','forget-gate'),('pt-BR','query'),('fr','query'),('fr','machine-translation'),('fr','gradient-descent'),('es','learning-rate')}
# Reviewed cases where the same English heading has more than one sense.
blocked |= {(loc,id) for loc in ('es','fr','pt-BR') for id in ('vocabulary','kernel','bias','statistical-bias','hypothesis','value','precision','accuracy','return','score-function','likelihood')}
for loc in ('es','fr','pt-BR'):
 dest=P/'terms'/f'{loc}.jsonl'; existing={x['concept_id']:x for x in map(json.loads,dest.read_text().splitlines())}
 selected={}
 for c in candidates:
  cl='pt-BR' if c['locale']=='pt' else c['locale']
  if cl!=loc or c['concept_id'] not in safe or (loc,c['concept_id']) in blocked or c['concept_id'] in selected:continue
  if '(' in c['term'] or len(c['term'])>70 or 'apredizado' in c['term'].lower():continue
  selected[c['concept_id']]=c
 rows=[]
 for e in concepts:
  id=e['id'];old=existing.get(id)
  if old and old['status'] in ('verified_d2l','verified_external','invariant') and not (id == 'vocabulary' and loc in ('es','fr')):
   row=old
  elif id in selected:
   c=selected[id];term=c['term'];src=f"stanford-paired-{loc}-{Path(c['path']).stem}"; url=f"https://github.com/shervinea/cheatsheet-translation/blob/{COMMIT}/{c['path']}"
   sources[src]={'id':src,'title':f"Bilingual translation: {Path(c['path']).stem}",'locale':loc,'kind':'translated educational notes','url':url,'accessed':'2026-11-07','publisher':'Shervine Amidi and contributors','repository':'https://github.com/shervinea/cheatsheet-translation','commit':COMMIT,'path':c['path'],'license':'MIT','notes':'Numbered English and translated teaching statements appear consecutively. Portuguese repository targets Portuguese; Brazilian-specific usage should be reviewed.'}
   term=term[0].lower()+term[1:];row={'concept_id':id,'preferred':term,'alternatives':[],'status':'verified_external','source_ids':[src],'evidence':[{'source_id':src,'locator':f"line {c['line']}",'source_term':c['term'],'context':f"Numbered bilingual teaching entry pairs this term with {c['en']}; English source is line {c['source_english_line']}."}],'notes':'Initial capitalization normalized. Bilingual statement inspected; regional and contextual terminology review remains useful.'}
  elif (loc,id) in base_terms:
   row={'concept_id':id,'preferred':base_terms[(loc,id)],'alternatives':[],'status':'proposed','source_ids':[],'evidence':[],'notes':'Editorial proposal; not yet independently attested in this sense.'}
  else:row={'concept_id':id,'preferred':None,'alternatives':[],'status':'unresolved','source_ids':[],'evidence':[],'notes':'No reviewed rendering or defensible editorial proposal supplied yet.'}
  if row['preferred'] is not None:row['preferred']=unicodedata.normalize('NFC',row['preferred'])
  rows.append(row)
 dest.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows))
 print(loc,len(rows),{s:sum(r['status']==s for r in rows) for s in ('verified_d2l','verified_external','proposed','unresolved','invariant')})
source_path.write_text(''.join(json.dumps(x,ensure_ascii=False)+'\n' for x in sources.values()))
