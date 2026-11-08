# Manually reviewed localized D2L passages: concept, exact source term, path, line.
records='''linear-regression|regressão linear|chapter_linear-networks/linear-regression-scratch.md|4
training-set|conjunto de treinamento|chapter_computer-vision/kaggle-cifar10.md|52
test-set|conjunto de teste|chapter_computer-vision/kaggle-cifar10.md|52
gradient-descent|gradiente descendente|chapter_introduction/index.md|421
regularization|regularização|chapter_attention-mechanisms/attention-scoring-functions.md|329
transfer-learning|aprendizado por transferência|chapter_computer-vision/fine-tuning.md|14
fine-tuning|ajuste fino|chapter_computer-vision/fine-tuning.md|14
semantic-segmentation|segmentação semântica|chapter_computer-vision/fcn.md|5
reward|recompensa|chapter_introduction/index.md|1119
policy|política|chapter_introduction/index.md|1123
expectation|expectativa|chapter_multilayer-perceptrons/dropout.md|161
vector|vetor|chapter_attention-mechanisms/self-attention-and-positional-encoding.md|70
tensor|tensor|chapter_attention-mechanisms/attention-scoring-functions.md|165
probability|probabilidade|chapter_attention-mechanisms/attention-scoring-functions.md|15
object-detection|detecção de objeto|chapter_computer-vision/rcnn.md|126
overfitting|sobreajuste|chapter_computer-vision/image-augmentation.md|363
vocabulary|vocabulário|chapter_recurrent-neural-networks/text-preprocessing.md|103'''
import json,re
from pathlib import Path
p=Path('docs/multilingual/terms/pt-BR.jsonl'); es=list(map(json.loads,p.read_text().splitlines()));D={x['concept_id']:x for x in es};s=Path('docs/multilingual/sources/group-es-fr-pt.jsonl');S={x['id']:x for x in map(json.loads,s.read_text().splitlines())};root=Path('/tmp/d2l-glossary-corpora/d2l-pt');commit='ee8a9f0a88a4791f277a55e32cb6c83fd00a7cac'
for line in records.splitlines():
 id,term,path,n=line.split('|');n=int(n);text=(root/path).read_text().splitlines()[n-1];m=re.search(re.escape(term),text,re.I);assert m,(id,text);actual=m.group();src='d2l-pt-'+path.replace('/','-').replace('.md','')
 S[src]={'id':src,'title':f'Dive into Deep Learning Portuguese: {path}','locale':'pt-BR','kind':'prior D2L translation','url':f'https://github.com/d2l-ai/d2l-pt/blob/{commit}/{path}','accessed':'2026-11-07','publisher':'D2L contributors','repository':'https://github.com/d2l-ai/d2l-pt','commit':commit,'path':path,'license':'CC BY-SA 4.0','notes':'Brazilian Portuguese D2L translated source; individual prose line inspected.'}
 D[id].update(preferred=term,status='verified_d2l',source_ids=[src],evidence=[{'source_id':src,'locator':f'line {n}','source_term':actual,'context':'The translated D2L prose uses this expression in the same technical sense.'}],notes='D2L prose occurrence inspected; capitalization normalized.')
p.write_text(''.join(json.dumps(e,ensure_ascii=False)+'\n' for e in es));s.write_text(''.join(json.dumps(e,ensure_ascii=False)+'\n' for e in S.values()));print('verified D2L',len(records.splitlines()))
