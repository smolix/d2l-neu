# East Asian terminology review

This shard covers all 1,184 English concepts in Simplified Chinese, Japanese, Korean, and Vietnamese. It translates prose terminology only. Source-backed entries contain an exact expression at a pinned repository line or a localized teaching-page heading. Editorial proposals are useful review candidates and are excluded from automatic translation constraints.

| Locale | D2L verified | External verified | Attested invariant | Editorial proposal | Unresolved |
| --- | ---: | ---: | ---: | ---: | ---: |
| zh-Hans | 295 | 26 | 0 | 863 | 0 |
| ja | 217 | 116 | 0 | 851 | 0 |
| ko | 252 | 88 | 0 | 844 | 0 |
| vi | 212 | 64 | 3 | 905 | 0 |

## Source selection

- First choice: translated D2L editorial term lists (`TERMINOLOGY.md` for Chinese and Korean, `glossary.md` for Vietnamese) and checked chapter prose. All D2L links pin the cloned repository commit. The Vietnamese glossary is more carefully edited than parts of the translated chapter text.
- Supplemental source: numbered bilingual English and target statements in the Amidi CS229/CS230 teaching cheatsheets. Links pin commit `99f45b6d65b8e60f637d8ac0db15f76327197c23` of `shervinea/cheatsheet-translation`. Japanese translation files are present there even where the live teaching site redirects to English.
- Supplemental source: localized Google Machine Learning Glossary topic pages and specific classification teaching pages; the topic pages expose machine-translation provenance. Their translated heading and topic context were inspected. The cached monolithic glossary page fell back to English for several locales and was excluded.
- Where a term has no defensible attestation, a complete localized editorial phrase was proposed from the English concept sense. An additional pass replaced wrong-sense machine suggestions such as Chinese `型号参数` for model parameter, Japanese `派生関数` for derivative, Korean `체중 감소` for weight decay, and Vietnamese `máy biến áp` for Transformer. No raw machine-suggestion rows remain. Proposal rows have no source IDs or evidence. Named methods and acronyms are retained in some proposals with an explicit review note; this does not assert attested local usage.

## Sense decisions and rejected candidates

| Concept | Decision | Reason |
| --- | --- | --- |
| `bias` / `statistical-bias` | Kept model intercept distinct from estimation bias | Vietnamese D2L lists `hệ số điều chỉnh` versus `độ chệch`; Japanese and Korean statistical prose supplied sense-specific expressions. |
| `kernel` / `kernel-function` | Verified convolution kernel in convolution prose; covariance kernel remains a proposal where no same-sense source was found | An isolated `kernel` match would conflate convolution and Gaussian-process senses. |
| `accuracy` / `precision` / `recall` | Kept classification accuracy, positive predictive value, and sensitivity distinct | Japanese @IT and Korean/Chinese teaching pages define the confusion-matrix ratios; Vietnamese D2L deliberately leaves `precision` and `recall` in English. |
| RL `state`, `action`, `reward`, `return` | Used RL chapter prose and rejected unrelated ordinary-word headings | Google headings included literal mistakes such as Chinese `州` for state. |
| `implicit-bias` | Google fairness-page translation rejected | Catalog sense is an optimizer's preference among solutions. |
| `neural-network`, `transformer`, `layer`, `gradient` | Rejected malformed literal headings and used D2L prose or proposals | Examples rejected include Chinese `输出表示`, Vietnamese `Máy biến áp`, and Japanese `グラデーション`. |
| Japanese `batch-normalization` / `partition-function` | Used `バッチ正規化` from the D2L chapter heading and `分配関数` from the bilingual CS229 exponential-family explanation | Earlier emphasis alignment dropped `batch` from the first term and picked literal `分割関数` for the probability normalizer. Korean normalization entries retained their modifiers in the checked sources. |

## Stratified passage and meaning audit

The following 20 verified concepts per locale were checked against the recorded source term at the line or section locator. The sample includes common terms, mathematical terms, less common architecture terms, and known English-sense collisions. Findings show the source expression or context and allow independent review. `east_asian_audit.py` reproduces the checks against the pinned local repository files.


## zh-Hans

| Concept | Preferred | Status | Source | Inspected context |
| --- | --- | --- | --- | --- |
| linear-regression | 线性回归 | verified_d2l | https://github.com/d2l-ai/d2l-zh/blob/e6b18ccea71451a55fcd861d7b96fddf2587b09a/TERMINOLOGY.md (line 237) | 线性回归，linear regression |
| bias | 偏置 | verified_d2l | https://github.com/d2l-ai/d2l-zh/blob/e6b18ccea71451a55fcd861d7b96fddf2587b09a/TERMINOLOGY.md (line 145) | 偏置，bias |
| precision | 查准率 | verified_d2l | https://github.com/d2l-ai/d2l-zh/blob/e6b18ccea71451a55fcd861d7b96fddf2587b09a/TERMINOLOGY.md (line 21) | 查准率，precision |
| accuracy | 准确率 | verified_d2l | https://github.com/d2l-ai/d2l-zh/blob/e6b18ccea71451a55fcd861d7b96fddf2587b09a/TERMINOLOGY.md (line 277) | 准确率/精度，accuracy |
| loss-function | 损失函数 | verified_d2l | https://github.com/d2l-ai/d2l-zh/blob/e6b18ccea71451a55fcd861d7b96fddf2587b09a/TERMINOLOGY.md (line 195) | 损失函数，loss function |
| objective-function | 目标函数 | verified_d2l | https://github.com/d2l-ai/d2l-zh/blob/e6b18ccea71451a55fcd861d7b96fddf2587b09a/chapter_introduction/index.md (line 101) | 1. 一个*目标函数*（objective function），用来量化模型的有效性； |
| gradient | 梯度 | verified_d2l | https://github.com/d2l-ai/d2l-zh/blob/e6b18ccea71451a55fcd861d7b96fddf2587b09a/TERMINOLOGY.md (line 205) | 梯度，gradient |
| gradient-descent | 梯度下降 | verified_external | https://github.com/shervinea/cheatsheet-translation/blob/99f45b6d65b8e60f637d8ac0db15f76327197c23/zh/cs-229-supervised-learning.md (line 93) | &#10230; 梯度下降 - 记学习率为 α∈R，梯度下降的更新规则使用学习率和成本函数 J 表示如下： |
| stochastic-gradient-descent | 随机梯度下降 | verified_d2l | https://github.com/d2l-ai/d2l-zh/blob/e6b18ccea71451a55fcd861d7b96fddf2587b09a/chapter_optimization/sgd.md (line 4) | 在前面的章节中，我们一直在训练过程中使用随机梯度下降，但没有解释它为什么起作用。为了澄清这一点，我们刚在 :numref:`sec_gd`中描述了梯度下降的基本原则。本节继续更详细地说明*随机梯度下降*（stochastic gradient descent）。 |
| probability | 概率 | verified_d2l | https://github.com/d2l-ai/d2l-zh/blob/e6b18ccea71451a55fcd861d7b96fddf2587b09a/chapter_preliminaries/probability.md (line 261) | *概率*（probability）可以被认为是将集合映射到真实值的函数。 |
| random-variable | 随机变量 | verified_d2l | https://github.com/d2l-ai/d2l-zh/blob/e6b18ccea71451a55fcd861d7b96fddf2587b09a/chapter_preliminaries/probability.md (line 278) | 在我们掷骰子的随机实验中，我们引入了*随机变量*（random variable）的概念。 |
| norm | 范数 | verified_d2l | https://github.com/d2l-ai/d2l-zh/blob/e6b18ccea71451a55fcd861d7b96fddf2587b09a/chapter_preliminaries/linear-algebra.md (line 942) | 线性代数中最有用的一些运算符是*范数*（norm）。 |
| kernel | 卷积核 | verified_d2l | https://github.com/d2l-ai/d2l-zh/blob/e6b18ccea71451a55fcd861d7b96fddf2587b09a/chapter_convolutional-neural-networks/why-conv.md (line 77) | 在深度学习研究社区中，$\mathbf{V}$被称为*卷积核*（convolution kernel）或者*滤波器*（filter），亦或简单地称之为该卷积层的*权重*，通常该权重是可学习的参数。 |
| neural-network | 神经网络 | verified_d2l | https://github.com/d2l-ai/d2l-zh/blob/e6b18ccea71451a55fcd861d7b96fddf2587b09a/chapter_multilayer-perceptrons/mlp.md (line 12) | 现在我们可以开始对深度神经网络的探索，这也是本书主要涉及的一类模型。 |
| state | 状态 | verified_d2l | https://github.com/d2l-ai/d2l-zh/blob/e6b18ccea71451a55fcd861d7b96fddf2587b09a/chapter_introduction/index.md (line 542) | 也就是说，当前的观察结果可能无法阐述有关当前状态的所有信息。 |
| action | 动作 | verified_d2l | https://github.com/d2l-ai/d2l-zh/blob/e6b18ccea71451a55fcd861d7b96fddf2587b09a/chapter_introduction/index.md (line 514) | 在每个特定时间点，智能体从环境接收一些*观察*（observation），并且必须选择一个*动作*（action），然后通过某种机制（有时称为执行器）将其传输回环境，最后智能体从环境中获得*奖励*（reward）。 |
| reward | 奖励 | verified_d2l | https://github.com/d2l-ai/d2l-zh/blob/e6b18ccea71451a55fcd861d7b96fddf2587b09a/chapter_introduction/index.md (line 514) | 在每个特定时间点，智能体从环境接收一些*观察*（observation），并且必须选择一个*动作*（action），然后通过某种机制（有时称为执行器）将其传输回环境，最后智能体从环境中获得*奖励*（reward）。 |
| query | 查询 | verified_d2l | https://github.com/d2l-ai/d2l-zh/blob/e6b18ccea71451a55fcd861d7b96fddf2587b09a/chapter_attention-mechanisms/attention-cues.md (line 75) | 在注意力机制的背景下，自主性提示被称为*查询*（query）。 |
| value | 值 | verified_d2l | https://github.com/d2l-ai/d2l-zh/blob/e6b18ccea71451a55fcd861d7b96fddf2587b09a/chapter_attention-mechanisms/attention-cues.md (line 78) | 在注意力机制中，这些感官输入被称为*值*（value）。 |
| backpropagation-through-time | 通过时间反向传播 | verified_d2l | https://github.com/d2l-ai/d2l-zh/blob/e6b18ccea71451a55fcd861d7b96fddf2587b09a/chapter_recurrent-neural-networks/bptt.md (line 22) | *通过时间反向传播*（backpropagation through time，BPTT） |
passage checks: 20

## ja

| Concept | Preferred | Status | Source | Inspected context |
| --- | --- | --- | --- | --- |
| linear-regression | 線形回帰 | verified_external | https://github.com/shervinea/cheatsheet-translation/blob/99f45b6d65b8e60f637d8ac0db15f76327197c23/ja/cs-229-supervised-learning.md (line 129) | &#10230;線形回帰 |
| feature | 特徴 | verified_d2l | https://github.com/d2l-ai/d2l-ja/blob/f84fb229198b49f2dc532ca72ce72b5611a124cb/chapter_linear-networks/softmax-regression.md (line 20) | 足を濡らすために、簡単な画像分類問題から始めましょう。ここで、各入力は $2\times2$ グレースケールイメージで構成されています。各ピクセル値を 1 つのスカラーで表すことができ、4 つの特徴 $x_1, x_2, x_3, x_4$ が得られます。さらに、それぞれの画像が「猫」、「鶏」、「犬」のいずれかのカテゴリに属すると仮定します。  |
| weight | 重み | verified_d2l | https://github.com/d2l-ai/d2l-ja/blob/f84fb229198b49f2dc532ca72ce72b5611a124cb/chapter_linear-networks/softmax-regression.md (line 30) | 考えられるすべてのクラスに関連する条件付き確率を推定するには、クラスごとに 1 つずつ、複数の出力をもつモデルが必要です。線形モデルによる分類に対処するには、出力の数だけアフィン関数が必要になります。各出力は独自のアフィン関数に対応します。この例では、4 つの特徴量と 3 つの可能な出力カテゴリがあるため、重みを表すには 12 個のスカラー (添字付き $w |
| bias | バイアス | verified_d2l | https://github.com/d2l-ai/d2l-ja/blob/f84fb229198b49f2dc532ca72ce72b5611a124cb/chapter_linear-networks/linear-regression.md (line 28) | :eqref:`eq_price-area` では $w_{\mathrm{area}}$ と $w_{\mathrm{age}}$ は*ウェイト* と呼ばれ、$b$ は*バイアス* (*オフセット* または*インターセプト* とも呼ばれる) と呼ばれています。重みは各特徴が予測に及ぼす影響を決定し、バイアスはすべての特徴が値0になったときに予測価格が取るべ |
| accuracy | 正解率 | verified_external | https://atmarkit.itmedia.co.jp/ait/articles/2010/27/news020.html (section 正解率) | Article identifies proportion of correct classifications as accuracy. |
| precision | 適合率 | verified_external | https://atmarkit.itmedia.co.jp/ait/articles/2010/27/news020.html (### 適合率（Precision）) | Article defines TP/(TP+FP) as precision. |
| recall | 再現率 | verified_external | https://atmarkit.itmedia.co.jp/ait/articles/2010/27/news020.html (section 再現率) | Article defines true-positive recall separately from precision. |
| loss-function | 損失関数 | verified_external | https://github.com/shervinea/cheatsheet-translation/blob/99f45b6d65b8e60f637d8ac0db15f76327197c23/ja/cs-229-supervised-learning.md (line 69) | &#10230;損失関数 ― 損失関数とは(z,y)∈R×Y⟼L(z,y)∈Rを満たす関数Lで、予測値zとそれに対応する正解データ値yを入力とし、その誤差を出力するものである。一般的な損失関数は次表に集約される： |
| objective-function | 目的関数 | verified_external | https://developers.google.com/machine-learning/glossary/metrics?hl=ja (#objective-function) | Localized Google ML teaching glossary heading aligned to English "objective function" by anchor; definition body requires sense check. |
| gradient | 勾配 | verified_d2l | https://github.com/d2l-ai/d2l-ja/blob/f84fb229198b49f2dc532ca72ce72b5611a124cb/chapter_optimization/sgd.md (line 37) | $\mathbf{x}$ における目的関数の勾配は次のように計算されます。  |
| gradient-descent | 勾配降下法 | verified_external | https://github.com/shervinea/cheatsheet-translation/blob/99f45b6d65b8e60f637d8ac0db15f76327197c23/ja/cs-229-supervised-learning.md (line 93) | &#10230;勾配降下法 ― 学習率をα∈Rとし、勾配降下法における更新ルールは学習率とコスト関数Jを用いて次のように表される： |
| likelihood | 尤度 | verified_external | https://github.com/shervinea/cheatsheet-translation/blob/99f45b6d65b8e60f637d8ac0db15f76327197c23/ja/cs-229-supervised-learning.md (line 105) | &#10230;尤度 ― パラメータをθとすると、あるモデルの尤度L(θ)を最大にすることにより最適なパラメータを求められる。実際には、最適化しやすい対数尤度ℓ(θ)=log(L(θ))を用いる。すなわち： |
| kernel | 畳み込みカーネル | verified_d2l | https://github.com/d2l-ai/d2l-ja/blob/f84fb229198b49f2dc532ca72ce72b5611a124cb/chapter_convolutional-neural-networks/why-conv.md (line 55) | は、畳み込み層を含むニューラルネットワークの特別なファミリーです。深層学習の研究コミュニティでは、$\mathbf{V}$ は*畳み込みカーネル*、*フィルター*、または単に層の*重み* と呼ばれ、多くの場合学習可能なパラメーターです。ローカルリージョンが小さい場合、完全に接続されたネットワークと比較して大きな違いが生じる可能性があります。以前は、画像処理ネ |
| statistical-bias | 統計的バイアス | verified_d2l | https://github.com/d2l-ai/d2l-ja/blob/f84fb229198b49f2dc532ca72ce72b5611a124cb/chapter_appendix-mathematics-for-deep-learning/statistics.md (line 124) | まず、系統誤差を測定してみましょう。推定器 $\hat{\theta}_n$ の場合、*統計的バイアス* の数学的図解は次のように定義できます。  |
| state | 状態 | verified_d2l | https://github.com/d2l-ai/d2l-ja/blob/f84fb229198b49f2dc532ca72ce72b5611a124cb/chapter_introduction/index.md (line 293) | 強化学習では、部分可観測性の問題にも対処しなければならない場合があります。つまり、現在の観測では、現在の状態に関するすべてがわかるとは限りません。掃除ロボットが家の中の同じクローゼットの一つに閉じ込められていることに気付いたとしましょう。ロボットの正確な位置 (および状態) を推測するには、クローゼットに入る前にロボットの以前の観測を考慮する必要がある場合が |
| reward | 報酬 | verified_d2l | https://github.com/d2l-ai/d2l-ja/blob/f84fb229198b49f2dc532ca72ce72b5611a124cb/chapter_introduction/index.md (line 282) | 強化学習は、エージェントが一連のタイムステップで環境と対話するという、非常に一般的な問題の説明を提供します。各タイムステップで、エージェントは環境から何らかの*観測*を受け取り、*アクション*を選択しなければならず、その後、何らかのメカニズム (アクチュエータとも呼ばれる) を介して環境に送り返されます。最後に、エージェントは環境から報酬を受け取ります。この |
| query | クエリ | verified_d2l | https://github.com/d2l-ai/d2l-ja/blob/f84fb229198b49f2dc532ca72ce72b5611a124cb/chapter_attention-mechanisms/attention-cues.md (line 26) | ## クエリ、キー、値 |
| value | 値 | verified_d2l | https://github.com/d2l-ai/d2l-ja/blob/f84fb229198b49f2dc532ca72ce72b5611a124cb/chapter_attention-mechanisms/attention-cues.md (line 26) | ## クエリ、キー、値 |
| jacobian | ヤコビアン | verified_d2l | https://github.com/d2l-ai/d2l-ja/blob/f84fb229198b49f2dc532ca72ce72b5611a124cb/chapter_appendix-mathematics-for-deep-learning/integral-calculus.md (line 474) | $D\phi$ は $\phi$ の*ヤコビアン* で、$\boldsymbol{\phi} = (\phi_1(x_1, \ldots, x_n), \ldots, \phi_n(x_1, \ldots, x_n))$ の偏微分の行列です。  |
| transposed-convolution | 転置畳み込み | verified_d2l | https://github.com/d2l-ai/d2l-ja/blob/f84fb229198b49f2dc532ca72ce72b5611a124cb/chapter_computer-vision/transposed-conv.md (line 7) | *転置畳み込み*。これは*分数ストライド畳み込み* :cite:`Dumoulin.Visin.2016`とも呼ばれ、  |
passage checks: 20

## ko

| Concept | Preferred | Status | Source | Inspected context |
| --- | --- | --- | --- | --- |
| linear-regression | 선형 회귀 | verified_external | https://github.com/shervinea/cheatsheet-translation/blob/99f45b6d65b8e60f637d8ac0db15f76327197c23/ko/cs-229-supervised-learning.md (line 81) | &#10230; [선형 회귀, 로지스틱 회귀, 서포트 벡터 머신, 인공신경망] |
| feature | 특성 | verified_d2l | https://github.com/d2l-ai/d2l-ko/blob/9857ea9899409b1e17003dcce7859d60aa892559/TERMINOLOGY.md (line 57) | / feature                                          / 특성                          / |
| weight | 가중치 | verified_d2l | https://github.com/d2l-ai/d2l-ko/blob/9857ea9899409b1e17003dcce7859d60aa892559/TERMINOLOGY.md (line 157) | / weight                                           / 가중치                        / |
| bias | 편향 | verified_d2l | https://github.com/d2l-ai/d2l-ko/blob/9857ea9899409b1e17003dcce7859d60aa892559/TERMINOLOGY.md (line 13) | / bias                                             / 편향                          / |
| accuracy | 정확도 | verified_d2l | https://github.com/d2l-ai/d2l-ko/blob/9857ea9899409b1e17003dcce7859d60aa892559/TERMINOLOGY.md (line 6) | / accuracy                                         / 정확도                        / |
| precision | 정밀도 | verified_external | https://developers.google.com/machine-learning/crash-course/classification/accuracy-precision-recall?hl=ko (section 정밀도) | Teaching page defines TP/(TP+FP) as precision. |
| recall | 재현율 | verified_external | https://developers.google.com/machine-learning/crash-course/classification/accuracy-precision-recall?hl=ko (section 재현율) | Teaching page defines TP/(TP+FN) as recall. |
| loss-function | 손실 함수 | verified_d2l | https://github.com/d2l-ai/d2l-ko/blob/9857ea9899409b1e17003dcce7859d60aa892559/TERMINOLOGY.md (line 90) | / loss function                                    / 손실 함수                     / |
| objective-function | 목적 함수 | verified_d2l | https://github.com/d2l-ai/d2l-ko/blob/9857ea9899409b1e17003dcce7859d60aa892559/TERMINOLOGY.md (line 109) | / objective function                               / 목적 함수                     / |
| gradient | 그래디언트 | verified_d2l | https://github.com/d2l-ai/d2l-ko/blob/9857ea9899409b1e17003dcce7859d60aa892559/TERMINOLOGY.md (line 65) | / gradient                                         / 그래디언트                    / |
| gradient-descent | 경사하강법 | verified_d2l | https://github.com/d2l-ai/d2l-ko/blob/9857ea9899409b1e17003dcce7859d60aa892559/chapter_optimization/sgd.md (line 1) | # 확률적 경사하강법 |
| kernel | 컨볼루션 커널 | verified_d2l | https://github.com/d2l-ai/d2l-ko/blob/9857ea9899409b1e17003dcce7859d60aa892559/chapter_convolutional-neural-networks/why-conv.md (line 55) | 는 컨벌루션 계층을 포함하는 특수한 신경망 제품군입니다.딥러닝 연구 커뮤니티에서 $\mathbf{V}$는*컨볼루션 커널*, *필터* 또는 단순히 학습 가능한 파라미터인 레이어의*가중치*라고 합니다.로컬 지역이 작은 경우 완전히 연결된 네트워크와 비교할 때 차이가 클 수 있습니다.이전에는 이미지 처리 네트워크에서 단일 계층 |
| statistical-bias | 통계적 편향 | verified_d2l | https://github.com/d2l-ai/d2l-ko/blob/9857ea9899409b1e17003dcce7859d60aa892559/chapter_appendix-mathematics-for-deep-learning/statistics.md (line 124) | 먼저 체계적인 오차를 측정해 보겠습니다.추정기 $\hat{\theta}_n$의 경우*통계적 편향*의 수학적 그림은 다음과 같이 정의할 수 있습니다.  |
| state | 상태 | verified_d2l | https://github.com/d2l-ai/d2l-ko/blob/9857ea9899409b1e17003dcce7859d60aa892559/chapter_introduction/index.md (line 293) | 강화 학습자는 부분적 관찰 가능성 문제를 다루어야 할 수도 있습니다.즉, 현재 관측치가 현재 상태에 대한 모든 것을 알려주지 못할 수도 있습니다.청소 로봇이 집안의 많은 동일한 옷장 중 하나에 갇혀 있다고 가정 해보십시오.로봇의 정확한 위치 (따라서 상태) 를 추론하려면 옷장에 들어가기 전에 이전 관찰을 고려해야 할 수  |
| reward | 보상 | verified_d2l | https://github.com/d2l-ai/d2l-ko/blob/9857ea9899409b1e17003dcce7859d60aa892559/chapter_introduction/index.md (line 282) | 강화 학습은 에이전트가 일련의 시간 단계에 걸쳐 환경과 상호 작용하는 문제에 대한 매우 일반적인 설명을 제공합니다.각 시간 스텝에서 에이전트는 환경으로부터*관찰*을 받고 이후에 일부 메커니즘 (액추에이터라고도 함) 을 통해 환경으로 다시 전송되는*작업*을 선택해야 합니다.마지막으로 에이전트는 환경으로부터 보상을 받습니다. |
| query | 쿼리 | verified_d2l | https://github.com/d2l-ai/d2l-ko/blob/9857ea9899409b1e17003dcce7859d60aa892559/chapter_attention-mechanisms/attention-cues.md (line 26) | ## 쿼리, 키 및 값 |
| value | 값 | verified_d2l | https://github.com/d2l-ai/d2l-ko/blob/9857ea9899409b1e17003dcce7859d60aa892559/chapter_attention-mechanisms/attention-cues.md (line 26) | ## 쿼리, 키 및 값 |
| computational-graph | 계산 그래프 | verified_d2l | https://github.com/d2l-ai/d2l-ko/blob/9857ea9899409b1e17003dcce7859d60aa892559/chapter_preliminaries/autograd.md (line 6) | 딥 러닝 프레임워크는 도함수 (예: *자동 미분*) 를 자동으로 계산하여 이 작업을 신속하게 처리합니다.실제로 설계된 모델을 기반으로 시스템은*계산 그래프*를 구축하여 어떤 데이터를 결합하여 어떤 작업을 통해 출력을 생성하는지 추적합니다.자동 미분을 통해 시스템은 그라디언트를 역전파할 수 있습니다.여기서*backpropa |
| central-limit-theorem | 중심 극한 정리 | verified_external | https://github.com/shervinea/cheatsheet-translation/blob/99f45b6d65b8e60f637d8ac0db15f76327197c23/ko/cs-229-probability.md (line 321) | &#10230;중심 극한 정리 ― 평균 μ와 분산 σ2를 갖는 주어진 분포를 따르는 랜덤 표본 X1, ..., Xn을 가정해 봅시다 그러면 우리는 다음을 가집니다: |
| reset-gate | 리셋 게이트 | verified_d2l | https://github.com/d2l-ai/d2l-ko/blob/9857ea9899409b1e17003dcce7859d60aa892559/chapter_recurrent-modern/gru.md (line 18) | 가장 먼저 소개해야 할 것은*리셋 게이트*와*업데이트 게이트*입니다.볼록 조합을 수행할 수 있도록 $(0, 1)$의 항목을 가진 벡터로 설계합니다.예를 들어, 리셋 게이트를 사용하면 이전 상태 중 기억하고 싶은 상태를 제어할 수 있습니다.마찬가지로 업데이트 게이트를 사용하면 새 상태가 이전 상태의 복사본인 정도를 제어할  |
passage checks: 20

## vi

| Concept | Preferred | Status | Source | Inspected context |
| --- | --- | --- | --- | --- |
| linear-regression | hồi quy tuyến tính | verified_d2l | https://github.com/d2l-ai/d2l-vi/blob/c775d6b4998e6243ec5d11f950e67679555a2c74/glossary.md (line 293) | / linear regression                  / hồi quy tuyến tính             /                                              / |
| feature | đặc trưng | verified_d2l | https://github.com/d2l-ai/d2l-vi/blob/c775d6b4998e6243ec5d11f950e67679555a2c74/glossary.md (line 179) | / feature                       / đặc trưng                                  /                                              / |
| bias | hệ số điều chỉnh | verified_d2l | https://github.com/d2l-ai/d2l-vi/blob/c775d6b4998e6243ec5d11f950e67679555a2c74/glossary.md (line 53) | / bias (model parameter)                 / hệ số điều chỉnh               / [https://git.io/Jvopx](https://git.io/Jvopx) / |
| statistical-bias | độ chệch | verified_d2l | https://github.com/d2l-ai/d2l-vi/blob/c775d6b4998e6243ec5d11f950e67679555a2c74/glossary.md (line 54) | / bias (of an estimator)                 / độ chệch                       / [https://git.io/JvQxO](https://git.io/JvQxO) / |
| accuracy | độ chính xác | verified_d2l | https://github.com/d2l-ai/d2l-vi/blob/c775d6b4998e6243ec5d11f950e67679555a2c74/glossary.md (line 18) | / accuracy                              / độ chính xác              /                                              / |
| loss-function | hàm mất mát | verified_d2l | https://github.com/d2l-ai/d2l-vi/blob/c775d6b4998e6243ec5d11f950e67679555a2c74/glossary.md (line 300) | / loss function                      / hàm mất mát                    / [https://git.io/Jvojp](https://git.io/Jvojp) / |
| objective-function | hàm mục tiêu | verified_d2l | https://github.com/d2l-ai/d2l-vi/blob/c775d6b4998e6243ec5d11f950e67679555a2c74/glossary.md (line 365) | / objective function     / hàm mục tiêu       / [https://git.io/Jvojp](https://git.io/Jvojp) / |
| chain-rule | quy tắc dây chuyền | verified_d2l | https://github.com/d2l-ai/d2l-vi/blob/c775d6b4998e6243ec5d11f950e67679555a2c74/glossary.md (line 77) | / chain rule                       / quy tắc dây chuyền        / [https://git.io/Jvojk](https://git.io/Jvojk) / |
| backpropagation | lan truyền ngược | verified_d2l | https://github.com/d2l-ai/d2l-vi/blob/c775d6b4998e6243ec5d11f950e67679555a2c74/glossary.md (line 43) | / backpropagation                        / lan truyền ngược               /                                              / |
| automatic-differentiation | tính vi phân tự động | verified_d2l | https://github.com/d2l-ai/d2l-vi/blob/c775d6b4998e6243ec5d11f950e67679555a2c74/glossary.md (line 33) | / automatic differentiation             / tính vi phân tự động      / [https://git.io/JvojU](https://git.io/JvojU) / |
| probability | xác suất | verified_d2l | https://github.com/d2l-ai/d2l-vi/blob/c775d6b4998e6243ec5d11f950e67679555a2c74/chapter_preliminaries/probability.md (line 172) | Về mặt chính thức, *xác suất* có thể được coi là một hàm ánh xạ một tập hợp thành giá trị thực. Xác suất của một sự kiện $\mathcal{A}$ trong không gian mẫu đã cho $\mathcal{S}$, ký |
| kernel | hạt nhân | verified_d2l | https://github.com/d2l-ai/d2l-vi/blob/c775d6b4998e6243ec5d11f950e67679555a2c74/glossary.md (line 266) | / kernel                  / hạt nhân                 / [https://git.io/Jfe1I](https://git.io/Jfe1I), [https://git.io/Jf0vK](https://git.io/Jf0vK) / |
| state | trạng thái | verified_d2l | https://github.com/d2l-ai/d2l-vi/blob/c775d6b4998e6243ec5d11f950e67679555a2c74/chapter_introduction/index.md (line 293) | Người học củng cố cũng có thể phải đối phó với vấn đề quan sát từng phần. Đó là, quan sát hiện tại có thể không cho bạn biết mọi thứ về trạng thái hiện tại của bạn. Nói rằng một ro |
| reward | phần thưởng | verified_d2l | https://github.com/d2l-ai/d2l-vi/blob/c775d6b4998e6243ec5d11f950e67679555a2c74/chapter_introduction/index.md (line 291) | Xem xét ví dụ như các trò chơi của cờ vua. Tín hiệu phần thưởng thực sự duy nhất đến ở cuối trò chơi khi chúng ta giành chiến thắng, mà chúng ta có thể gán phần thưởng là 1, hoặc k |
| neural-network | mạng nơ-ron | verified_d2l | https://github.com/d2l-ai/d2l-vi/blob/c775d6b4998e6243ec5d11f950e67679555a2c74/glossary.md (line 346) | / neural network                       / mạng nơ-ron                       / [https://git.io/JvQxY](https://git.io/JvQxY), [https://git.io/JvQxR](https://git.io/JvQxR) / |
| overfitting | quá khớp | verified_d2l | https://github.com/d2l-ai/d2l-vi/blob/c775d6b4998e6243ec5d11f950e67679555a2c74/glossary.md (line 375) | / overfit                / quá khớp           / [https://git.io/JvQxY](https://git.io/JvQxY) / |
| underfitting | dưới khớp | verified_d2l | https://github.com/d2l-ai/d2l-vi/blob/c775d6b4998e6243ec5d11f950e67679555a2c74/glossary.md (line 541) | / underfit                  / dưới khớp                 / [https://git.io/JvQxY](https://git.io/JvQxY) / |
| covariance | Hiệp phương sai | verified_external | https://github.com/shervinea/cheatsheet-translation/blob/99f45b6d65b8e60f637d8ac0db15f76327197c23/vi/cs-229-probability.md (line 247) | &#10230; Hiệp phương sai - Chúng ta xác định hiệp phương sai của hai biến ngẫu nhiên X và Y, thường được kí hiệu σ2XY hay Cov(X,Y), như sau: |
| joint-distribution | phân phối đồng thời | verified_d2l | https://github.com/d2l-ai/d2l-vi/blob/c775d6b4998e6243ec5d11f950e67679555a2c74/glossary.md (line 260) | / joint distribution / phân phối đồng thời / [https://git.io/Jvohb](https://git.io/Jvohb) / |
| instance-segmentation | phân vùng thực thể | verified_d2l | https://github.com/d2l-ai/d2l-vi/blob/c775d6b4998e6243ec5d11f950e67679555a2c74/glossary.md (line 251) | / instance segmentation                       / phân vùng thực thể           /                                              / |
passage checks: 20

## Final synonym reconciliation

The language review preceded the final 21 synonym merges. The canonical shard now has 1,184 concepts; [COVERAGE.md](../COVERAGE.md) gives current status counts. Audited older IDs resolve through [concept-redirects.json](../concept-redirects.json). Pre-merge variants and their evidence are preserved in `candidates/merged-variant-terms.jsonl`; they are not automatically approved alternatives.
