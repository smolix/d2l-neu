#!/usr/bin/env python3
"""Curate East Asian term candidates and write canonical shards plus provenance."""
import json
import re
import unicodedata
from collections import defaultdict, Counter
from pathlib import Path
from urllib.parse import quote

BASE=Path(__file__).resolve().parents[1]
CORPUS=Path('/tmp/d2l-glossary-corpora')
LOCALES={'zh-Hans':'zh','ja':'ja','ko':'ko','vi':'vi'}
REPOS={'zh':('d2l-ai/d2l-zh','e6b18ccea71451a55fcd861d7b96fddf2587b09a'),
       'ja':('d2l-ai/d2l-ja','f84fb229198b49f2dc532ca72ce72b5611a124cb'),
       'ko':('d2l-ai/d2l-ko','9857ea9899409b1e17003dcce7859d60aa892559'),
       'vi':('d2l-ai/d2l-vi','c775d6b4998e6243ec5d11f950e67679555a2c74')}
RANK={'editorial_glossary':0,'terminology_list':1,'bilingual_introduction':2,
      'manual_context':2.5,'google_topic_heading':3,'aligned_emphasis':4,
      'aligned_heading':5,'manual_body':1,'stanford_bilingual':1.5}
BAD={
 'zh':{'action':['观察'],'policy':['动作'],'observation':['Q网络'],
       'state':['州'],'implicit-bias':['隐性偏差'],
       'neural-network':['输出表示'],'layer':['图层'],'agent':['代理'],
       'in-context-learning':['上下文学习'],
       'lagrangian':['拉格朗日乘数'],'teacher-forcing':['强制教学']},
 'ja':{'feature':['機能'],'gradient':['グラデーション'],'weight-decay':['体重減衰'],
       'neural-network':['ニューラル ネットワークの'],'activation-function':['アクティベーション機能'],
       'transformer':['トランス'],'masked-language-modeling':['仮面言語モデリング'],
       'continuous-bag-of-words':['連続した単語の袋'],'machine-translation':['シーケンス変換'],
       'sequence-transduction':['機械翻訳'],'regression':['リグレッション'],
       'statistical-bias':['バイアス'],'implicit-bias':['暗黙のバイアス'],
       'discriminator':['弁別者'],'vocabulary':['用語'],
       'rectified-linear-unit':['正規化線形ユニット']},
 'ko':{'weight':['무게'],'prediction':['추론'],'gradient-descent':['그라데이션 하강'],
       'automatic-differentiation':['자동 차별화'],'rank':['순위'],
       'likelihood':['가능성'],'maximum-likelihood':['최대화 가능성'],
       'state':['주'],'non-maximum-suppression':['최대가 아닌 억제'],
       'ridge-regression':['능선 회귀'],'transformer':['변압기'],
       'partial-derivative':['편미함'],'intersection-over-union':['('],
       'neural-network':['출력은'],'weight-decay':['웨이트 디케이'],
       'regularization':['정규화'],'statistical-bias':['편향'],
       'implicit-bias':['내재적 편향'],
       'fine-tuning':['세부 조정'],'in-context-learning':['컨텍스트 학습'],
       'sequence-transduction':['기계 변환'],'residual-block':['Resnet'],
       'perplexity':['당혹'],'squared-error':['제곱 오류']},
 'vi':{'weight':['cân nặng'],'loss-function':['Chức năng mất','loss','mất'],
       'mean-squared-error':['Lỗi bình phương trung bình'],'epoch':['thời gian bắt đầu của hệ thống'],
       'learning-rate':['Tỷ lệ học tập'],'automatic-differentiation':['Sự khác biệt tự động'],
       'matrix':['matri'],'dot-product':['dot của họ'],'rank':['thứ hạng'],
       'maximum-likelihood':['Khả năng tối đa'],'neural-network':['mạng nơron'],
       'regularization':['regarization','chuẩn hóa'],'weight-decay':['Trọng lượng phân rã'],
       'dropout':['Bỏ học'],'convolution':['độ lượng'],'pooling':['bộ'],
       'average-pooling':['trung bình'],'computer-vision':['Tầm nhìn máy tính'],
       'hidden-state':['ẩn','hidden','Nhà nước ẩn'],'tokenization':['Mã hóa'],
       'teacher-forcing':['giáo viên forcing'],'perplexity':['Bối rối','độ hỗn loạn'],
       'transformer':['Máy biến áp'],'generator':['Máy phát điện'],
       'discriminator':['Phân biệt đối xử'],'state':['tiểu bang'],
       'sentiment-analysis':['Phân tích tâm học'],'cross-entropy-loss':['Mất Cross-Entropy','cross-entropy','cross-entropy mất'],
       'intersection-over-union':['giao lộ qua công đoàn'],'observation':['quan sách'],
       'non-maximum-suppression':['triệt phi cực đại'],'one-hot-encoding':['một'],
       'projection':['dự áo'],'feature-map':['ánh xạ đặc trưng'],
       'precision':['độ chính xác'],'recall':['mức độ ghi nhớ'],
       'language-model':['Hô hình ngôn ngữ'],
       'overfitting':['khái quát hoá kém'],'underfitting':['khớp dữ liệu kém'],
       'in-context-learning':['học tập trong bối cảnh'],
       'implicit-bias':['thiên kiến ngầm']}
}
# An explicitly checked passage in the D2L chapter may fix an erroneous
# machine-rendered heading. Each line was inspected with its surrounding prose.
BODY={
 'ja':{'feature':('特徴','chapter_linear-networks/softmax-regression.md',20),
       'batch-normalization':('バッチ正規化','chapter_convolutional-modern/batch-norm.md',1),
       'weight':('重み','chapter_linear-networks/softmax-regression.md',30),
       'gradient':('勾配','chapter_optimization/sgd.md',37),
       'weight-decay':('重み減衰','chapter_multilayer-perceptrons/weight-decay.md',240),
       'kernel':('畳み込みカーネル','chapter_convolutional-neural-networks/why-conv.md',55),
       'statistical-bias':('統計的バイアス','chapter_appendix-mathematics-for-deep-learning/statistics.md',124),
       'state':('状態','chapter_introduction/index.md',293),
       'reward':('報酬','chapter_introduction/index.md',282),
       'observation':('観測','chapter_introduction/index.md',282),
       'query':('クエリ','chapter_attention-mechanisms/attention-cues.md',26),
       'key':('キー','chapter_attention-mechanisms/attention-cues.md',26),
       'value':('値','chapter_attention-mechanisms/attention-cues.md',26)},
 'ko':{'gradient-descent':('경사하강법','chapter_optimization/sgd.md',1),
       'kernel':('컨볼루션 커널','chapter_convolutional-neural-networks/why-conv.md',55),
       'statistical-bias':('통계적 편향','chapter_appendix-mathematics-for-deep-learning/statistics.md',124),
       'state':('상태','chapter_introduction/index.md',293),
       'reward':('보상','chapter_introduction/index.md',282),
       'observation':('관찰','chapter_introduction/index.md',282),
       'query':('쿼리','chapter_attention-mechanisms/attention-cues.md',26),
       'key':('키','chapter_attention-mechanisms/attention-cues.md',26),
       'value':('값','chapter_attention-mechanisms/attention-cues.md',26)},
 'zh':{'kernel':('卷积核','chapter_convolutional-neural-networks/why-conv.md',77),
       'state':('状态','chapter_introduction/index.md',542),
       'neural-network':('神经网络','chapter_multilayer-perceptrons/mlp.md',12),
       'agent':('智能体','chapter_introduction/index.md',514),
       'return':('回报','chapter_introduction/index.md',547)},
 'vi':{'kernel':('hạt nhân','chapter_convolutional-neural-networks/why-conv.md',55),
       'state':('trạng thái','chapter_introduction/index.md',293),
       'reward':('phần thưởng','chapter_introduction/index.md',291)}
}
# Final basic concepts: passages inspected in their original pinned files.
BODY['zh'].update({
 'model':('模型','chapter_linear-networks/linear-regression.md',25),
 'training':('训练','chapter_multilayer-perceptrons/mlp.md',8),
 'activation':('激活值','chapter_multilayer-perceptrons/dropout.md',67),
})
BODY['ja'].update({
 'model':('モデル','chapter_linear-networks/linear-regression.md',16),
 'training':('学習','chapter_multilayer-perceptrons/mlp.md',18),
 'mask':('マスク','chapter_multilayer-perceptrons/dropout.md',57),
})
BODY['ko'].update({
 'model':('모델','chapter_linear-networks/linear-regression.md',16),
 'training':('훈련','chapter_multilayer-perceptrons/dropout.md',14),
 'mask':('마스크','chapter_multilayer-perceptrons/dropout.md',57),
})
BODY['vi'].update({
 'model':('mô hình','glossary.md',324),
 'training':('huấn luyện','glossary.md',159),
 'mask':('mặt nạ','chapter_multilayer-perceptrons/dropout.md',57),
})
EXTERNAL={
 'ja':{'precision':('適合率','https://atmarkit.itmedia.co.jp/ait/articles/2010/27/news020.html','### 適合率（Precision）','Article defines TP/(TP+FP) as precision.'),
       'accuracy':('正解率','https://atmarkit.itmedia.co.jp/ait/articles/2010/27/news020.html','section 正解率','Article identifies proportion of correct classifications as accuracy.'),
       'recall':('再現率','https://atmarkit.itmedia.co.jp/ait/articles/2010/27/news020.html','section 再現率','Article defines true-positive recall separately from precision.')},
 'zh':{'recall':('召回率','https://developers.google.com/machine-learning/crash-course/classification/accuracy-precision-recall?hl=zh-CN','section 召回率','Teaching page defines TP/(TP+FN) as recall.')},
 'ko':{'precision':('정밀도','https://developers.google.com/machine-learning/crash-course/classification/accuracy-precision-recall?hl=ko','section 정밀도','Teaching page defines TP/(TP+FP) as precision.'),
       'recall':('재현율','https://developers.google.com/machine-learning/crash-course/classification/accuracy-precision-recall?hl=ko','section 재현율','Teaching page defines TP/(TP+FN) as recall.')},
 'vi':{'matrix':('ma trận','https://stanford.edu/~shervine/l/vi/teaching/cs-229/refresher-algebra-calculus/','Definition: Ma trận','Translated CS229 review defines matrix by rows and columns.'),
       'identity-matrix':('ma trận đơn vị','https://stanford.edu/~shervine/l/vi/teaching/cs-229/refresher-algebra-calculus/','Ma trận chính: Ma trận đơn vị','Translated CS229 review defines the identity matrix by its unit diagonal.'),
       'transpose':('chuyển vị','https://stanford.edu/~shervine/l/vi/teaching/cs-229/refresher-algebra-calculus/','Các phép toán ma trận: Chuyển vị','Translated CS229 review defines transpose by exchanging rows and columns.'),
       'trace':('truy vết','https://stanford.edu/~shervine/l/vi/teaching/cs-229/refresher-algebra-calculus/','Các phép toán ma trận: Truy vết','Translated CS229 review defines trace as the sum of diagonal entries.'),
       'determinant':('định thức','https://stanford.edu/~shervine/l/vi/teaching/cs-229/refresher-algebra-calculus/','Các phép toán ma trận: Định thức','Translated CS229 review defines the determinant of a square matrix.'),
       'rank':('hạng ma trận','https://stanford.edu/~shervine/l/vi/teaching/cs-229/refresher-algebra-calculus/','Những tính chất của ma trận: Hạng ma trận','Translated CS229 review defines rank as dimension of the column span.'),
       'linear-independence':('độc lập tuyến tính','https://stanford.edu/~shervine/l/vi/teaching/cs-229/refresher-algebra-calculus/','Những tính chất của ma trận: Sự phụ thuộc tuyến tính','Translated CS229 review contrasts linear dependence with linear independence.'),
       'positive-definite':('xác định dương','https://stanford.edu/~shervine/l/vi/teaching/cs-229/refresher-algebra-calculus/','Những tính chất của ma trận: Ma trận bán xác định dương','Translated CS229 review gives the strict positive definiteness condition.'),
       'singular-value-decomposition':('phân tích giá trị suy biến','https://stanford.edu/~shervine/l/vi/teaching/cs-229/refresher-algebra-calculus/','Phân tích giá trị suy biến','Translated CS229 review states the singular value decomposition.')}
}
PROPOSAL_OVERRIDE={
 'zh':{'statistical-bias':'统计偏差','implicit-bias':'隐式偏置','precision':'查准率',
       'covariance-kernel':'协方差核'},
 'ja':{'return':'リターン','implicit-bias':'暗黙的バイアス','covariance-kernel':'共分散カーネル'},
 'ko':{'return':'누적 보상','implicit-bias':'암묵적 편향','covariance-kernel':'공분산 커널',
       'rank':'계수'},
 'vi':{'return':'tổng phần thưởng tích lũy','implicit-bias':'thiên lệch ngầm định',
       'covariance-kernel':'hàm hiệp phương sai','rank':'hạng'}
}
INVARIANT={'vi':{'precision':('precision','glossary.md',403,
                              'The D2L Vietnamese editorial glossary deliberately leaves this classifier metric in English to distinguish it from accuracy.'),
                 'recall':('recall','glossary.md',425,
                           'The D2L Vietnamese editorial glossary retains the metric in English.'),
                 'perplexity':('perplexity','glossary.md',389,
                               'The D2L Vietnamese editorial glossary explicitly retains the language-model metric in English.')},
           'ja':{},'ko':{},'zh':{}}
NAMED_METHODS={'word2vec','gpu','bert','adam','adagrad','rmsprop','bleu','adamw',
               'shampoo','muon','lasso','gelu','ddpm','ddim','hyperband',
               'infonce','rmsnorm','alibi','mamba','adaln-zero'}
# Same-section search candidates whose target expression and surrounding
# technical passage have been reviewed. All other hits stay provisional.
CONTEXT_ALLOW={
 'zh':set('memory-bandwidth probability-distribution numerical-stability log-likelihood negative-log-likelihood statistical-learning maximum-entropy matrix-multiplication generalization-gap gaussian-noise uniform-distribution image-classification initialization super-resolution nonlinearity markov-chain correlation relative-position data-augmentation autoregressive-model eigenvector pretraining computational-efficiency cosine-similarity computational-complexity likelihood-function maximum-likelihood-estimation maximum-likelihood identity-matrix covariance probability-density image-recognition inductive-bias attention throughput mixed-precision named-entity-recognition sequence-length embedding-matrix dot-product-attention text-classification semantic-similarity global-minimum taylor-expansion line-search generative-model principal-component bernoulli-distribution fisher-information state-space parallelism'.split()),
 'ja':set('memory-bandwidth floating-point dataset probability-distribution softmax numerical-stability minibatch optimization scalar image-classification pixel linear-algebra logit computational-efficiency squared-error maximum-likelihood-estimation token gaussian-distribution dot-product sampling probability-density gaussian-noise exponential-distribution tensor test-set automatic-differentiation model-parameter initialization distribution-shift conditional-distribution empirical-risk maximum-entropy computational-graph random-variable uniform-distribution sigmoid nonlinearity euclidean-distance normalization-layer pooling-layer vector feature-extraction image-recognition throughput taylor-expansion covariance-matrix recommender-system nearest-neighbor embedding autoencoder denoising relative-position strided-convolution language-model autoregressive-model latent-variable recurrent-neural-network eigenvector corpus word-frequency generative-model subword text-classification likelihood-function identity-matrix mixed-precision quantization probability-mass-function bernoulli-distribution convex-function cumulative-distribution-function central-limit-theorem singular-value-decomposition weight-initialization discretization monte-carlo positional-encoding hessian positive-definite condition-number mean-value-theorem convex-optimization unsupervised-learning deep-reinforcement-learning fisher-information state-space'.split()),
 'ko':set('memory-bandwidth floating-point probability-distribution minibatch optimization numerical-stability image-classification computational-efficiency maximum-likelihood-estimation initialization maximum-entropy gaussian-noise cross-validation gamma-function sigmoid nonlinearity euclidean-distance normalization-layer pooling-layer feature-extraction image-recognition throughput data-augmentation image-resolution computational-complexity covariance-matrix recommender-system autoencoder law-of-large-numbers convolutional-neural-network relative-position pretraining language-model latent-variable-model recurrent-neural-network robustness word-frequency sequence-length generative-model subword text-classification likelihood-function max-pooling mixed-precision quantization probability-mass-function cumulative-distribution-function hypothesis-testing eigenvalue singular-value-decomposition weight-initialization eigenvector discretization monte-carlo sequence-to-sequence convex-optimization variance-reduction mean-value-theorem principal-component-analysis deep-reinforcement-learning bernoulli-distribution state-space'.split()),
 'vi':set('environment probability-distribution image-classification vector cross-entropy linear-transformation sampling probability-density statistical-learning model-parameter initialization empirical-risk nonlinearity euclidean-distance residual-connection embedding autoencoder time-step value-function word-embedding self-attention layer-normalization variance-reduction positive-definite'.split())
}

def rows(path):
    if not path.exists():return []
    return [json.loads(s) for s in path.read_text().splitlines() if s.strip()]

def key(s):return re.sub(r'\s+','',unicodedata.normalize('NFC',s).casefold())

def usable(lang,cid,term,method):
    if cid=='statistical-bias' and method not in ('editorial_glossary','manual_body','stanford_bilingual'):
        return False
    if cid=='implicit-bias' and method=='google_topic_heading':
        # Google uses the fairness sense; our concept is optimizer preference.
        return False
    if not term or term in BAD.get(lang,{}).get(cid,[]):return False
    if re.search(r'[\uFFFD\x00-\x1f]',term):return False
    if len(term)>90 or not term.strip('() '):return False
    if method=='stanford_bilingual':
        if len(term)>48 or cid in {'kernel','bias'}:return False
        if term.casefold().startswith(('data augmentation','leaky relu','sigmoid(')):return False
        if re.search(r'[\[\]{}⟶∇=]',term):return False
        if lang=='vi' and cid in {'perplexity','regularization','cross-entropy-loss','maximum-likelihood-estimator'}:return False
    if lang=='zh' and not re.search(r'[\u4e00-\u9fff]',term):return False
    if lang=='ja' and not re.search(r'[\u3040-\u30ff\u4e00-\u9fff]',term):return False
    if lang=='ko' and not re.search(r'[\uac00-\ud7a3]',term):return False
    if lang=='vi' and method not in ('editorial_glossary','google_topic_heading','manual_body','manual_context','external_teaching','stanford_bilingual'):
        # The Vietnamese translated chapters often contain unedited literal MT.
        # Accept only independently checked local phrases beyond the editorial list.
        return cid in {'regression','classification','prediction','optimization','probability',
                       'conditional-probability','independence','covariance','correlation',
                       'mutual-information','convergence','distribution-shift','latency',
                       'forward-pass','self-supervised-learning','supervised-learning',
                       'unsupervised-learning','question-answering','image-augmentation'}
    return True

def source_record(lang,c):
    path=c['path']
    if path.startswith('cheatsheet:'):
        rel=path[11:]
        sid='east-stanford-'+re.sub(r'[^a-z0-9]+','-',rel.lower()).strip('-')
        commit='99f45b6d65b8e60f637d8ac0db15f76327197c23'
        return sid,{'id':sid,'title':f'CS229/CS230 bilingual teaching translation: {rel}',
                    'locale':next(k for k,v in LOCALES.items() if v==lang),
                    'kind':'bilingual_teaching_translation',
                    'url':f'https://github.com/shervinea/cheatsheet-translation/blob/{commit}/{quote(rel)}',
                    'accessed':'2026-11-08','publisher':'Afshine and Shervine Amidi translation contributors',
                    'repository':'shervinea/cheatsheet-translation','commit':commit,'path':rel,
                    'license':'MIT','notes':'Numbered English statement is paired with the localized teaching statement in the next line.'},f'line {c["line"]}'
    if path.startswith('external:'):
        url=path[9:]
        sid='east-external-'+lang+'-'+re.sub(r'[^a-z0-9]+','-',url.split('/')[2]+'-'+url.split('/')[-1]).strip('-')
        return sid,{'id':sid,'title':'Published machine learning teaching article',
                    'locale':next(k for k,v in LOCALES.items() if v==lang),
                    'kind':'published_teaching_material','url':url,'accessed':'2026-11-08',
                    'publisher':'＠IT' if 'itmedia' in url else 'Google for Developers'},c['locator']
    if path.startswith('google:'):
        topic,anchor=path[7:].split('#',1)
        hl={'zh':'zh-cn','ja':'ja','ko':'ko','vi':'vi'}[lang]
        sid=f'east-google-ml-{lang}-{topic}'
        return sid,{'id':sid,'title':f'Google Machine Learning Glossary: {topic}',
                    'locale':next(k for k,v in LOCALES.items() if v==lang),
                    'kind':'localized_teaching_glossary','url':f'https://developers.google.com/machine-learning/glossary/{topic}?hl={hl}',
                    'accessed':'2026-11-08','publisher':'Google for Developers',
                    'notes':'Topic page has localized heading and definition body; website discloses machine translation via x-mtfrom-en metadata.'},f'#{anchor}'
    repo,commit=REPOS[lang]
    sid='east-'+lang+'-'+re.sub(r'[^a-z0-9]+','-',path.lower()).strip('-')
    return sid,{'id':sid,'title':f'Dive into Deep Learning ({lang}): {path}',
                'locale':next(k for k,v in LOCALES.items() if v==lang),
                'kind':'translated_textbook_terminology' if path in ('glossary.md','TERMINOLOGY.md') else 'translated_textbook',
                'url':f'https://github.com/{repo}/blob/{commit}/{quote(path)}',
                'accessed':'2026-11-08','publisher':'Dive into Deep Learning translation project',
                'repository':repo,'commit':commit,'path':path,
                'notes':'Pinned repository file. Vietnamese glossary is an editorial terminology list; chapter prose in some repositories is unevenly translated.'},f'line {c["line"]}'

def evidence(lang,c,cid,sources):
    sid,rec,locator=source_record(lang,c)
    if c['path'].startswith('external:'):
        pass
    elif c['path'].startswith('google:'):
        # The heading text itself was captured from the topical page.
        anchor=c['path'].split('#',1)[1]
        html=(Path('/tmp/d2l-glossary-google')/f'east_asian-{c["path"][7:].split("#")[0]}-{dict(zh="zh-cn",ja="ja",ko="ko",vi="vi")[lang]}.html').read_text(errors='replace')
        if f'id="{anchor}"' not in html:raise ValueError((cid,c))
    else:
        source_path=(CORPUS/'cheatsheet-translation'/c['path'][11:]) if c['path'].startswith('cheatsheet:') else (CORPUS/('d2l-'+lang)/c['path'])
        line=source_path.read_text(errors='replace').splitlines()[c['line']-1]
        if c['term'] not in re.sub(r'[*`_]','',line):raise ValueError((cid,c['term'],c['path'],c['line'],line))
    sources[sid]=rec
    return {'source_id':sid,'locator':locator,'source_term':c['term'],
            'context':c['context']}

def main():
    concepts=rows(BASE/'concepts.jsonl')
    candidates=defaultdict(list)
    for c in rows(BASE/'candidates'/'east_asian-d2l.jsonl'):
        candidates[c['locale'],c['concept_id']].append(c)
    context_seen=set()
    for x in rows(BASE/'candidates'/'east_asian-context.jsonl'):
        lang,cid=x['locale'],x['concept_id']
        if cid not in CONTEXT_ALLOW.get(lang,set()) or (lang,cid) in context_seen:continue
        context_seen.add((lang,cid))
        candidates[lang,cid].append({'term':x['term'],'path':x['path'],
             'line':x['line'],'method':'manual_context',
             'context':f'Checked translated section "{x["section"]}" uses this phrase; the paired English section discusses "{x["en"]}" in the same technical context.'})
    stanford_seen=set()
    for x in rows(BASE/'candidates'/'east_asian-cheatsheet.jsonl'):
        lang,cid=x['locale'],x['concept_id']
        if cid=='statistical-bias' and 'probability' not in x['path']:continue
        if (lang,cid,x['term']) in stanford_seen:continue
        stanford_seen.add((lang,cid,x['term']))
        candidates[lang,cid].append({'term':x['term'],'path':'cheatsheet:'+x['path'],
             'line':x['line'],'method':'stanford_bilingual','context':x['context']})
    for lang,items in BODY.items():
        for cid,(term,path,line) in items.items():
            candidates[lang,cid].append({'term':term,'path':path,'line':line,
                                         'method':'manual_body','context':f'Checked D2L prose uses {term} for this concept in its technical explanation.'})
    # The Japanese D2L softmax exercise says 分割関数; the bilingual teaching
    # explanation of the exponential-family normalizer uses 分配関数.
    candidates['ja','partition-function'].append({
        'term':'分配関数','path':'cheatsheet:ja/cs-229-supervised-learning.md',
        'line':201,'method':'stanford_bilingual',
        'context':'The paired English statement calls a(η) the log-partition function; the Japanese statement calls it 対数分配関数, identifying the probability normalizer.'})
    for lang,items in EXTERNAL.items():
        for cid,(term,url,locator,context) in items.items():
            candidates[lang,cid].append({'term':term,'path':'external:'+url,
                                         'line':0,'locator':locator,'method':'external_teaching',
                                         'context':context})
    RANK['external_teaching']=0
    proposals={(p['locale'],p['concept_id']):p for p in rows(BASE/'candidates'/'east_asian-proposals.jsonl')}
    editorial={}
    editorial_file=BASE/'candidates'/'east_asian-editorial.psv'
    if editorial_file.exists():
        for raw in editorial_file.read_text().splitlines():
            if not raw or raw.startswith('#'):continue
            cells=raw.split('|')
            if len(cells)!=5:raise ValueError(('editorial table needs five columns',raw))
            cid,*terms=cells
            for locale,term in zip(LOCALES,terms):
                if term.strip():editorial[locale,cid]=term.strip()
    sources={};(BASE/'terms').mkdir(exist_ok=True);(BASE/'sources').mkdir(exist_ok=True)
    for locale,lang in LOCALES.items():
        terms=[]
        for concept in concepts:
            cid=concept['id']
            if cid in INVARIANT.get(lang,{}):
                term,path,line,reason=INVARIANT[lang][cid]
                c={'term':term,'path':path,'line':line,'method':'manual_body','context':reason}
                ev=[evidence(lang,c,cid,sources)]
                terms.append({'concept_id':cid,'preferred':term,'alternatives':[],
                              'status':'invariant','source_ids':[ev[0]['source_id']],
                              'evidence':ev,'notes':reason})
                continue
            cc=[x for x in candidates.get((lang,cid),[]) if usable(lang,cid,x['term'],x['method'])]
            # Prefer publisher's edited glossary over incidental chapter phrasing.
            cc.sort(key=lambda x:(RANK.get(x['method'],9),len(x['term']),x['path']))
            unique=[];seen=set()
            for x in cc:
                if key(x['term']) not in seen:unique.append(x);seen.add(key(x['term']))
            if unique:
                best=unique[0]
                ev=[evidence(lang,best,cid,sources)]
                status='verified_external' if best['method'] in ('google_topic_heading','external_teaching','stanford_bilingual') else 'verified_d2l'
                alternatives=[x['term'] for x in unique[1:] if x['term']!=best['term']][:5]
                note=f'{best["method"]}; source phrase checked at locator. Alternatives are source-attested variants only; each needs separate review before use.' if alternatives else f'{best["method"]}; source phrase checked at locator.'
                pref=best['term']
                if best['method']=='stanford_bilingual':
                    pref=re.sub(r'\s*[（(][^）)]*[）)]$','',pref).strip()
                if lang=='zh' and cid=='accuracy' and pref=='准确率/精度':
                    pref='准确率';alternatives=list(dict.fromkeys(['精度',*alternatives]))
                row={'concept_id':cid,'preferred':pref,'alternatives':alternatives,
                     'status':status,'source_ids':[ev[0]['source_id']],'evidence':ev,'notes':note}
            else:
                p=proposals.get((locale,cid))
                term=editorial.get((locale,cid)) or PROPOSAL_OVERRIDE.get(lang,{}).get(cid) or (p.get('term') if p else None)
                named_retention=((locale,cid) in editorial and cid in NAMED_METHODS)
                if term and (key(term)!=key(concept['en']) or named_retention) and not re.search(r'[\uFFFD\x00-\x1f]',term):
                    row={'concept_id':cid,'preferred':term,'alternatives':[],
                         'status':'proposed','source_ids':[],'evidence':[],
                         'notes':('Editorial retention of a named method or acronym; local prose usage unverified. Exclude from automatic constraints.'
                                  if named_retention and key(term)==key(concept['en']) else
                                  'Editorial translation proposal; technical sense and local usage need source review. Exclude from automatic constraints.'
                                  if (locale,cid) in editorial else
                                  'Machine translation suggestion only; technical sense and local usage unverified. Exclude from automatic constraints.')}
                else:
                    row={'concept_id':cid,'preferred':None,'alternatives':[],
                         'status':'unresolved','source_ids':[],'evidence':[],
                         'notes':'No defensible localized preferred term attested; search or editorial review needed.'}
            terms.append(row)
        (BASE/'terms'/f'{locale}.jsonl').write_text(''.join(json.dumps(x,ensure_ascii=False)+'\n' for x in terms))
        print(locale,len(terms),dict(Counter(x['status'] for x in terms)))
    (BASE/'sources'/'group-east-asian.jsonl').write_text(''.join(json.dumps(x,ensure_ascii=False)+'\n' for x in sorted(sources.values(),key=lambda x:x['id'])))
    print('sources',len(sources))

if __name__=='__main__':main()
