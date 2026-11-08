#!/usr/bin/env python3
"""Build the reviewed Hindi shard from the English concept inventory.

This script is intentionally deterministic. Google headings are used only for
the explicitly reviewed same-sense allowlist; other rows are editorial
proposals. The component lexicon is an editing aid, not independent evidence.
"""
import json
import re
import unicodedata
from pathlib import Path

from bs4 import BeautifulSoup

BASE = Path(__file__).resolve().parents[1]
CACHE = Path('/tmp/d2l-glossary-corpora/google-topics')
TOPICS = 'fundamentals metrics clustering df generative googlecloud responsible-ai tensorflow agent'.split()
ACCESSED = '2026-11-07'
CBSE_ID = 'cbse-ai-integration-hi'
CBSE_URL = 'https://cbseacademic.nic.in/web_material/Curriculum20/AI_Integration_Manual_Hindi.pdf'
CBSE_ATTEST = {
    # PDF page 125 (printed page 123) visually inspected. The legacy text
    # layer is garbled, so these strings are transcribed from the page image.
    'neural-network':('तंत्रिका नेटवर्क','The Google Lens explanation names the neural network used for image recognition.',True),
    'image-recognition':('छवि पहचान','The Google Lens explanation describes recognition of objects in an image.',True),
    'test-set':('परीक्षण डेटा','The data-acquisition entry partitions acquired data into training and test data.',True),
    'training-set':('प्रशिक्षण डेटा','The data-acquisition entry partitions acquired data into training and test data.',False),
    'prediction':('भविष्यवाणी','The lesson says a trained machine predicts answers from data.',False),
}
GOOGLE_BODY_ATTEST = {
    'accuracy':('fundamentals','accuracy','सटीकता',False,
                'The Hindi definition uses this noun when calculating the fraction of correct classifications.'),
    'epoch':('fundamentals','epoch','epoch',True,
             'The Hindi definition explains one pass across the entire training set.'),
    'prompt':('generative','prompt','प्रॉम्प्ट',False,
              'The Hindi definition uses this loanword for input text supplied to a language model.'),
    'tensor':('tensorflow','tensor','टेंसर',False,
              'The Hindi definition uses this loanword for an N-dimensional TensorFlow data structure.'),
}

# A same-anchor pair only establishes an aligned candidate. These entries were
# checked against the Hindi definition body and the D2L concept sense.
VERIFIED = set('''accuracy activation-function backpropagation batch-size binary-classification
confusion-matrix early-stopping embedding-layer feature-vector generalization
gradient-descent hidden-layer hyperparameter label layer learning-rate linear-regression
logistic-regression log-odds loss-function minibatch multiclass-classification negative-class
neural-network neuron one-hot-encoding overfitting positive-class precision prediction recall
regularization softmax squared-loss training-loss training-set underfitting validation-set
cross-entropy entropy hinge-loss model-capacity objective-function perplexity
autoregressive-model context-window fine-tuning large-language-model latency
machine-translation mixture-of-experts pretraining eager-execution selection-bias'''.split())

# Preferred idiomatic Hindi for key senses. In particular, don't inherit
# Google's "सुविधा" for feature, "राज्य" for RL state, or "ऐक्शन गेम" for action.
PREFERRED = {
 'regression':'प्रतिगमन', 'classification':'वर्गीकरण', 'feature':'विशेषता',
 'training-example':'प्रशिक्षण उदाहरण', 'dataset':'डेटासेट',
 'test-set':'परीक्षण समुच्चय', 'model-parameter':'मॉडल पैरामीटर',
 'weight':'भार', 'bias':'बायस पद', 'squared-error':'वर्ग त्रुटि',
 'mean-squared-error':'माध्य वर्ग त्रुटि', 'empirical-risk':'अनुभवजन्य जोखिम',
 'epoch':'प्रशिक्षण युग', 'stochastic-gradient-descent':'यादृच्छिक प्रवणता अवरोहण',
 'optimization':'अनुकूलन', 'gradient':'प्रवणता', 'derivative':'अवकलज',
 'partial-derivative':'आंशिक अवकलज', 'chain-rule':'श्रृंखला नियम',
 'automatic-differentiation':'स्वचालित अवकलन', 'computational-graph':'संगणना आलेख',
 'vector':'सदिश', 'matrix':'आव्यूह', 'tensor':'टेंसर', 'scalar':'अदिश',
 'transpose':'परिवर्त', 'dot-product':'डॉट गुणनफल',
 'matrix-multiplication':'आव्यूह गुणन', 'eigenvalue':'अभिलक्षणिक मान',
 'eigenvector':'अभिलक्षणिक सदिश', 'singular-value-decomposition':'एकल मान अपघटन',
 'norm':'मानक', 'euclidean-distance':'यूक्लिडीय दूरी',
 'linear-algebra':'रैखिक बीजगणित', 'linear-transformation':'रैखिक रूपांतरण',
 'identity-matrix':'तत्समक आव्यूह', 'determinant':'सारणिक', 'rank':'कोटि',
 'trace':'अनुरेख', 'orthogonal':'लंबकोणीय', 'positive-definite':'धनात्मक निश्चित',
 'probability':'प्रायिकता', 'random-variable':'यादृच्छिक चर',
 'probability-distribution':'प्रायिकता वितरण', 'probability-density':'प्रायिकता घनत्व',
 'conditional-probability':'सशर्त प्रायिकता', 'joint-distribution':'संयुक्त वितरण',
 'marginal-distribution':'सीमांत वितरण', 'independence':'स्वतंत्रता',
 'expectation':'अपेक्षित मान', 'variance':'प्रसरण', 'standard-deviation':'मानक विचलन',
 'covariance':'सहप्रसरण', 'correlation':'सहसंबंध',
 'gaussian-distribution':'गाउसियन वितरण', 'bernoulli-distribution':'बर्नूली वितरण',
 'poisson-distribution':'पॉइसन वितरण', 'bayes-theorem':'बेयस प्रमेय',
 'prior':'पूर्व वितरण', 'posterior':'पश्च वितरण', 'likelihood':'संभाव्यता फलन',
 'maximum-likelihood':'अधिकतम संभाव्यता',
 'maximum-likelihood-estimation':'अधिकतम संभाव्यता आकलन',
 'sampling':'प्रतिचयन', 'monte-carlo':'मोंटे कार्लो विधि',
 'mutual-information':'पारस्परिक सूचना', 'sigmoid':'सिग्मॉइड',
 'cross-entropy-loss':'क्रॉस एंट्रॉपी हानि',
 'nonlinearity':'अरैखिकता', 'multilayer-perceptron':'बहुस्तरीय परसेप्ट्रॉन',
 'rectified-linear-unit':'रेक्टिफाइड लीनियर यूनिट',
 'weight-decay':'भार क्षय', 'dropout':'ड्रॉपआउट', 'model-selection':'मॉडल चयन',
 'cross-validation':'क्रॉस वैलिडेशन', 'initialization':'आरंभीकरण',
 'vanishing-gradient':'लुप्त होती प्रवणता', 'exploding-gradient':'बढ़ती प्रवणता',
 'gradient-clipping':'प्रवणता सीमितकरण',
 'batch-normalization':'बैच सामान्यीकरण', 'layer-normalization':'स्तर सामान्यीकरण',
 'convolution':'संवलन', 'convolutional-neural-network':'संवलनी तंत्रिका नेटवर्क',
 'kernel':'संवलन कर्नेल', 'filter':'फ़िल्टर', 'channel':'चैनल', 'stride':'स्ट्राइड',
 'padding':'पैडिंग', 'pooling':'पूलिंग', 'max-pooling':'अधिकतम पूलिंग',
 'average-pooling':'औसत पूलिंग', 'receptive-field':'ग्रहणशील क्षेत्र',
 'translation-equivariance':'स्थानांतरण समतुल्यता',
 'residual-connection':'अवशिष्ट संयोजन', 'residual-network':'अवशिष्ट नेटवर्क',
 'skip-connection':'स्किप संयोजन', 'transfer-learning':'स्थानांतरण अधिगम',
 'object-detection':'वस्तु पहचान', 'bounding-box':'परिबद्ध बॉक्स',
 'intersection-over-union':'प्रतिच्छेदन अनुपात संघ',
 'semantic-segmentation':'अर्थगत खंडीकरण',
 'instance-segmentation':'इंस्टेंस खंडीकरण', 'computer-vision':'कंप्यूटर दृष्टि',
 'recurrent-neural-network':'पुनरावर्ती तंत्रिका नेटवर्क',
 'hidden-state':'आंतरिक अवस्था', 'time-step':'समय चरण', 'sequence':'अनुक्रम',
 'language-model':'भाषा मॉडल', 'token':'टोकन', 'tokenization':'टोकनीकरण',
 'vocabulary':'शब्दावली', 'embedding':'एम्बेडिंग', 'word-embedding':'शब्द एम्बेडिंग',
 'sequence-to-sequence':'अनुक्रम से अनुक्रम मॉडल', 'encoder':'एन्कोडर',
 'decoder':'डीकोडर', 'teacher-forcing':'टीचर फ़ोर्सिंग', 'beam-search':'बीम खोज',
 'long-short-term-memory':'दीर्घ अल्पकालिक स्मृति',
 'gated-recurrent-unit':'गेटेड पुनरावर्ती इकाई', 'attention':'अटेंशन',
 'self-attention':'स्व अटेंशन', 'multi-head-attention':'बहु हेड अटेंशन',
 'query':'क्वेरी', 'key':'कुंजी', 'value':'मान',
 'positional-encoding':'स्थिति एन्कोडिंग', 'transformer':'ट्रांसफ़ॉर्मर',
 'masked-language-model':'मास्क्ड भाषा मॉडल',
 'generative-model':'जनन मॉडल', 'discriminative-model':'विभेदक मॉडल',
 'generator':'जनरेटर', 'discriminator':'विभेदक', 'autoencoder':'ऑटोएन्कोडर',
 'latent-variable':'गुप्त चर', 'latent-space':'गुप्त स्थान',
 'diffusion-model':'विसरण मॉडल', 'noise-schedule':'शोर अनुसूची',
 'denoising':'शोर निष्कासन', 'score-function':'लॉग घनत्व की प्रवणता',
 'markov-chain':'मार्कोव श्रृंखला', 'markov-decision-process':'मार्कोव निर्णय प्रक्रिया',
 'reinforcement-learning':'प्रबलन अधिगम', 'agent':'एजेंट',
 'environment':'परिवेश', 'state':'अवस्था', 'action':'क्रिया',
 'reward':'पुरस्कार', 'policy':'नीति', 'return':'संचित प्रतिफल',
 'discount-factor':'छूट गुणक', 'value-function':'मान फलन',
 'action-value-function':'क्रिया मान फलन', 'value-iteration':'मान पुनरावृत्ति',
 'exploration':'अन्वेषण', 'policy-gradient':'नीति प्रवणता',
 'covariance-function':'सहप्रसरण फलन', 'kernel-function':'कर्नेल फलन',
 'covariance-kernel':'सहप्रसरण कर्नेल', 'statistical-bias':'आकलक की अभिनति',
 'implicit-bias':'सीखने की अंतर्निहित समाधान प्रवृत्ति', 'score-density':'लॉग घनत्व की प्रवणता',
 'critic-generative':'प्रतिद्वंद्वी जनन मॉडल का आलोचक',
}
PREFERRED.update({
 'adaln-zero':'adaLN-Zero',
 'saddle-point':'काठी बिंदु','brownian-motion':'ब्राउनी गति',
 'conjugate-prior':'संयुग्मी पूर्व वितरण',
 'mixture-of-gaussians':'गाउसियन वितरणों का मिश्रण',
 'log-sum-exp':'लॉग सम एक्सप', 'newton-s-method':'न्यूटन विधि',
 'n-gram':'एन ग्राम','ito-calculus':'इतो कलन',
 'euler-maruyama':'ऑयलर मरुयामा विधि',
 'markov-chain-monte-carlo':'मार्कोव श्रृंखला मोंटे कार्लो',
 'convex-conjugate':'उत्तल संयुग्मी',
 'fundamental-theorem-of-calculus':'कलन का मूल प्रमेय',
 'jensen-s-inequality':'जेनसेन असमानता',
 'probability-simplex':'प्रायिकता सिंप्लेक्स',
 'unbiased-estimator':'निष्पक्ष आकलक',
 'consistent-estimator':'संगत आकलक',
 'rejection-sampling':'अस्वीकृति प्रतिचयन',
 'data-processing-inequality':'डेटा प्रसंस्करण असमानता',
 'receiver-operating-characteristic':'ग्राही प्रचालन अभिलक्षण वक्र',
 'area-under-the-curve':'वक्र के नीचे का क्षेत्रफल',
 'held-out-data':'अलग रखा गया डेटा',
 'closed-form-solution':'स्पष्ट रूप में समाधान',
 'root-mean-squared-error':'मूल माध्य वर्ग त्रुटि',
 'fully-connected-layer':'पूर्ण संयोजित स्तर',
 'zero-padding':'शून्य पैडिंग',
 'ground-truth-bounding-box':'वास्तविक वस्तु का परिबद्ध बॉक्स',
 'region-of-interest':'रुचि का क्षेत्र',
 'single-shot-multibox-detection':'एक चरण बहु बॉक्स पहचान',
 'cold-start':'कोल्ड स्टार्ट',
 'beginning-of-sequence-token':'अनुक्रम आरंभ टोकन',
 'end-of-sequence-token':'अनुक्रम अंत टोकन',
 'co-occurrence':'सहप्रकटता',
 'textual-entailment':'पाठीय निहितार्थ',
 'epistemic-uncertainty':'ज्ञानगत अनिश्चितता',
 'aleatoric-uncertainty':'यादृच्छिक अनिश्चितता',
 'signal-to-noise-ratio':'संकेत शोर अनुपात',
 'reverse-time-sde':'उल्टे समय का यादृच्छिक अवकल समीकरण',
 'variational-autoencoder':'परिवर्ती ऑटोएन्कोडर',
 'frechet-inception-distance':'फ़्रेशे इंसेप्शन दूरी',
 'maximum-mean-discrepancy':'अधिकतम माध्य अंतर',
 'constrained-optimization':'बंधनयुक्त अनुकूलन',
 'projected-gradient-descent':'प्रक्षेपित प्रवणता अवरोहण',
 'backtracking-line-search':'पश्च खोज रेखा खोज',
 'conjugate-gradient':'संयुग्मी प्रवणता',
 'nonconvex-optimization':'अनुत्तल अनुकूलन',
 'exponential-family':'घातांकीय कुल',
 'asymptotic-normality':'सीमांत सामान्यता',
 'successive-halving':'क्रमिक अर्धीकरण',
 'matrix-completion':'आव्यूह पूर्णन',
 'normalized-discounted-cumulative-gain':'सामान्यीकृत छूटयुक्त संचयी लाभ',
 'information-theory':'सूचना सिद्धांत',
 'lossless-compression':'हानिरहित संपीड़न',
 'random-walk':'यादृच्छिक भ्रमण',
 'ito-process':'इतो प्रक्रिया','ito-integral':'इतो समाकल',
 'tokens-per-second':'टोकन प्रति सेकंड',
 'weight-tying':'भार साझा करना',
 'top-k-routing':'शीर्ष k विशेषज्ञों की ओर मार्गन',
 'all-gather':'सभी से संग्रह',
 'partition-function':'विभाजन फलन',
 'convergence-guarantee':'अभिसरण की गारंटी',
 'normalizing-constant':'सामान्यीकरण नियतांक',
 'bias-variance-decomposition':'बायस-प्रसरण अपघटन',
 'label-shift':'लेबल वितरण परिवर्तन',
 'feature-map':'विशेषता मानचित्र',
 'loss-scaling':'हानि का पैमाना बढ़ाना',
 'reduce-scatter':'रिड्यूस स्कैटर',
 'long-tail':'लंबी पूँछ',
 'smoothness':'मृदुता',
 'double-precision':'द्विगुण परिशुद्धता',
 'mixture-of-experts':'विशेषज्ञों का मिश्रण',
 'log-odds':'लॉग ऑड्स',
 'hessian':'हेसियन','jacobian':'जैकोबियन',
 'lagrangian':'लाग्रांजियन',
 'roofline':'रूफ़लाइन मॉडल','softplus':'सॉफ़्टप्लस',
 'ito-integral':'इतो समाकल',
 'word2vec':'वर्डटूवेक',
 'adaptive-overfitting':'अनुकूली अति अनुकूलन',
 'interpolation-threshold':'अंतर्वेशन सीमा',
 'roofline-model':'रूफ़लाइन मॉडल',
 'double-descent':'डबल डिसेंट',
 'training-throughput':'प्रशिक्षण थ्रूपुट',
 'score-estimate':'लॉग घनत्व प्रवणता का अनुमान',
 'denoising-score-matching':'शोर निष्कासन आधारित स्कोर मिलान',
 'implicit-feedback':'अप्रत्यक्ष प्रतिक्रिया',
 'memory-footprint':'स्मृति उपयोग',
 'compute-bound':'गणना सीमित',
 'click-through-rate':'क्लिक दर',
 'posterior-mode':'पश्च वितरण का बहुलक',
 'continuous-normalizing-flow':'सतत सामान्यीकरण प्रवाह',
})

# Whole-phrase semantic review after the 1,205-concept refresh. In these
# entries, independently translating English components gave misleading
# word order, the wrong technical sense, or an unnatural Hindi phrase. All
# remain proposals unless separate term-level evidence is attached below.
PREFERRED.update({
 'translation-invariance':'स्थानांतरण के प्रति अपरिवर्तनीयता',
 'translation-equivariance':'स्थानांतरण के प्रति समपरिवर्तनशीलता',
 'in-distribution':'संदर्भ वितरण के अनुरूप',
 'out-of-distribution':'प्रशिक्षण-वितरण से बाहर',
 'gradient-norm':'प्रवणता का नॉर्म',
 'exploding-gradient':'विस्फोटक प्रवणता',
 'normalizing-flow':'नॉर्मलाइज़िंग फ़्लो',
 'continuous-normalizing-flow':'सतत नॉर्मलाइज़िंग फ़्लो',
 'bias-correction':'आकलक की अभिनति का सुधार',
 'exposure-bias':'एक्सपोज़र बायस',
 'latent-diffusion':'अव्यक्त स्थान में विसरण',
 'posterior-predictive-distribution':'पश्चवर्ती पूर्वानुमान वितरण',
 'kernel-launch':'GPU कर्नेल शुरू करना',
 'neural-ode':'न्यूरल ODE',
 'law-of-large-numbers':'बड़ी संख्याओं का नियम',
 'maximum-a-posteriori':'अधिकतम पश्च प्रायिकता',
 'evidence-lower-bound':'साक्ष्य निम्न सीमा',
 'sufficient-statistic':'पर्याप्त सांख्यिक',
 'condition-number':'कंडीशन संख्या',
 'in-context-learning':'संदर्भ के भीतर अधिगम',
 'scaled-dot-product-attention':'स्केल्ड डॉट-प्रोडक्ट अटेंशन',
 'continuous-bag-of-words':'कॉन्टिन्युअस बैग ऑफ़ वर्ड्स',
 'question-answering':'प्रश्नोत्तर',
 'non-maximum-suppression':'नॉन-मैक्सिमम सप्रेशन',
 'off-policy':'ऑफ़-पॉलिसी',
 'on-policy':'ऑन-पॉलिसी',
 'reverse-process':'उलटी प्रक्रिया',
 'reverse-transition':'उलटा संक्रमण',
 'reward-to-go':'अब से मिलने वाले पुरस्कारों का योग',
 'time-to-first-token':'पहले टोकन तक का समय',
 'just-in-time-compilation':'जस्ट-इन-टाइम संकलन',
 'all-reduce':'ऑल-रिड्यूस',
 'multi-armed-bandit':'मल्टी-आर्म्ड बैंडिट',
 'statistical-power':'सांख्यिकीय परीक्षण की शक्ति',
 'variance-preserving':'प्रसरण संरक्षित रखने वाला',
 'variance-exploding':'प्रसरण बढ़ाने वाला',
 'image-to-image-translation':'एक छवि से दूसरी छवि में रूपांतरण',
 'change-of-variables':'चर परिवर्तन',
 'text-corpus':'पाठ संग्रह',
 'out-of-vocabulary':'शब्दावली से बाहर',
 'attention-map':'अटेंशन मानचित्र',
 'flow-map':'प्रवाह मानचित्र',
 'negative-feedback':'नकारात्मक प्रतिक्रिया',
 'total-variation':'कुल परिवर्तन',
 'total-variation-distance':'कुल परिवर्तन दूरी',
 'posterior-distribution':'पश्चवर्ती वितरण',
 'posterior-probability':'पश्चवर्ती प्रायिकता',
 'posterior-inference':'पश्चवर्ती अनुमिति',
 'posterior-mean':'पश्चवर्ती माध्य',
 'posterior-variance':'पश्चवर्ती प्रसरण',
 'posterior-predictive':'पश्चवर्ती पूर्वानुमान',
 'posterior-predictive-mean':'पश्चवर्ती पूर्वानुमान माध्य',
 'gaussian-posterior':'गाउसियन पश्चवर्ती वितरण',
 'gaussian-process-posterior':'गाउसियन प्रक्रिया का पश्चवर्ती वितरण',
 'gaussian-prior':'गाउसियन पूर्व वितरण',
 'gaussian-process-prior':'गाउसियन प्रक्रिया का पूर्व वितरण',
 'gaussian-likelihood':'गाउसियन संभाव्यता फलन',
 'forward-posterior':'अग्र प्रक्रिया का पश्चवर्ती वितरण',
 'acquisition-function':'अगले परीक्षण का चयन फलन',
 'stop-gradient':'प्रवणता का प्रवाह रोकना',
 'prior-term':'पूर्व वितरण से तुलना का पद',
 'activation-memory':'मध्यवर्ती सक्रियणों की स्मृति',
 'launch-overhead':'कर्नेल शुरू करने की अतिरिक्त लागत',
 'root-mean-square':'वर्गों के माध्य का वर्गमूल',
 'root-mean-squared-error':'माध्य वर्ग त्रुटि का वर्गमूल',
 'quantile':'क्वांटाइल',
 'variational-inference':'वेरिएशनल इन्फ़रेंस',
 'variational-objective':'वेरिएशनल उद्देश्य फलन',
 'variational-autoencoder':'वेरिएशनल ऑटोएन्कोडर',
 'matrix-exponential':'आव्यूह घातांक',
 'backward-pass':'पश्चगामी चरण',
 'forward-pass':'अग्रगामी चरण',
 'forward-mode':'अग्रगामी अवकलन विधि',
 'reverse-mode':'पश्चगामी अवकलन विधि',
 'region-proposal':'संभावित वस्तु वाला प्रस्तावित क्षेत्र',
 'region-proposal-network':'क्षेत्र प्रस्ताव नेटवर्क',
 'singular-matrix':'अव्युत्क्रमणीय आव्यूह',
 'normal-equations':'नॉर्मल समीकरण',
 'memory-bound':'डेटा स्थानांतरण से सीमित',
 'compute-bound':'अंकगणितीय क्षमता से सीमित',
 'latent-factor':'अव्यक्त कारक',
 'total-variation':'पड़ोसी पिक्सेलों का कुल अंतर',
 'score-matching':'लॉग घनत्व की प्रवणता का मिलान',
 'score-network':'लॉग घनत्व प्रवणता का अनुमानक नेटवर्क',
 'log-likelihood':'लॉग संभाव्यता फलन',
 'negative-log-likelihood':'ऋणात्मक लॉग संभाव्यता फलन',
 'marginal-likelihood':'सीमांत संभाव्यता फलन',
 'log-marginal-likelihood':'सीमांत संभाव्यता फलन का लॉग',
 'classifier-free-guidance':'वर्गकारक रहित मार्गदर्शन',
 'mode-collapse':'मोड कोलैप्स',
 'mode-dropping':'कुछ मोड छूट जाना',
 'random-horizontal-flip':'छवि को यादृच्छिक रूप से क्षैतिज पलटना',
 'random-resized-crop':'यादृच्छिक आकार और स्थान पर छवि का क्रॉप',
 'ground-truth':'सत्यापित संदर्भ मान',
 'click-through-rate-prediction':'क्लिक दर का पूर्वानुमान',
 'information-bottleneck':'सूचना बॉटलनेक',
 'induction-head':'इंडक्शन हेड',
 'induction-circuit':'इंडक्शन परिपथ',
 'head-dimension':'हेड की विमा',
 'training-tokens':'प्रशिक्षण में प्रयुक्त टोकन',
 'throughput':'थ्रूपुट',
 'spatial-invariance':'स्थान बदलने पर भी अपरिवर्तनशीलता',
 'class-conditioning':'वर्ग लेबल के आधार पर सशर्तन',
 'latent-code':'अव्यक्त कोड',
 'covariate-shift':'इनपुट वितरण में परिवर्तन',
 'concept-drift':'इनपुट और लक्ष्य के संबंध में बदलाव',
 'representation-learning':'डेटा से उपयोगी निरूपण सीखना',
 'residual-stream':'अवशिष्ट संयोजनों से आगे बढ़ती निरूपण धारा',
 'attention-sink':'अटेंशन का स्थायी आकर्षण बिंदु',
 'training-recipe':'प्रशिक्षण की कार्यविधि',
 'kernel-size':'कर्नेल का आकार',
 'memory-bandwidth':'स्मृति बैंडविड्थ',
 'arithmetic-intensity':'प्रति बाइट अंकगणितीय कार्य की मात्रा',
 'training-distribution':'प्रशिक्षण डेटा का वितरण',
 'test-distribution':'परीक्षण डेटा का वितरण',
 'f1-score':'F1 स्कोर',
 'attention-map':'अटेंशन भारों का मानचित्र',
 'feature-pyramid':'विशेषताओं का पिरामिड',
 'multi-scale':'बहु-पैमाने वाला',
 'multi-fidelity-optimization':'अलग-अलग सटीकता स्तरों वाला अनुकूलन',
 'fidelity':'अनुमान की सटीकता का स्तर',
})

# English headword components for transparent proposals. These are chosen for
# the ML/maths senses in D2L; phrase overrides below correct polysemy.
WORD = {
 'a':'एक','absolute':'निरपेक्ष','accuracy':'सटीकता','action':'क्रिया','activation':'सक्रियण',
 'active':'सक्रिय','adaptive':'अनुकूली','addition':'योग','adversarial':'प्रतिद्वंद्वी',
 'affine':'एफ़ाइन','agent':'एजेंट','aggregation':'समूहन','algebra':'बीजगणित',
 'algorithm':'एल्गोरिदम','aligned':'संरेखित','allocation':'आवंटन','all':'सभी',
 'analysis':'विश्लेषण','analytic':'विश्लेषणात्मक','and':'और','annealing':'अनीलिंग',
 'approximate':'सन्निकट','approximation':'सन्निकटन','arithmetic':'अंकगणितीय',
 'asynchronous':'अतुल्यकालिक','attention':'अटेंशन','attribute':'विशेषता',
 'attribution':'श्रेय निर्धारण','augmented':'संवर्धित','augmentation':'संवर्धन',
 'auto':'स्व','automatic':'स्वचालित','autoregressive':'स्वप्रतिगामी',
 'average':'औसत','backward':'पश्च','balance':'संतुलन','batch':'बैच',
 'bayes':'बेयस','bayesian':'बेयसी','beam':'बीम','bellman':'बेलमैन',
 'bias':'पूर्वाग्रह','binary':'द्विआधारी','bit':'बिट','block':'ब्लॉक',
 'bound':'सीमा','bounded':'सीमाबद्ध','boundary':'सीमा','budget':'बजट',
 'calibration':'अंशशोधन','capacity':'क्षमता','categorical':'श्रेणीबद्ध',
 'causal':'कारणात्मक','causality':'कारणता','chain':'श्रृंखला','change':'परिवर्तन',
 'channel':'चैनल','checkpoint':'चेकपॉइंट','checkpointing':'चेकपॉइंटिंग',
 'class':'वर्ग','classification':'वर्गीकरण','classifier':'वर्गकारक',
 'clipping':'सीमितकरण','code':'कोड','coding':'कोडन','coefficient':'गुणांक',
 'collaborative':'सहयोगी','column':'स्तंभ','combination':'संयोजन',
 'component':'घटक','compression':'संपीड़न','compute':'संगणना',
 'computational':'संगणकीय','condition':'शर्त','conditional':'सशर्त',
 'confidence':'विश्वास','configuration':'विन्यास','connection':'संयोजन',
 'consistency':'संगति','constraint':'बंधन','context':'संदर्भ',
 'continuous':'सतत','contrastive':'व्यतिरेकी','convergence':'अभिसरण',
 'convex':'उत्तल','convexity':'उत्तलता','convolution':'संवलन',
 'convolutional':'संवलनी','correlation':'सहसंबंध','cost':'लागत',
 'covariance':'सहप्रसरण','critic':'आलोचक','cross':'क्रॉस',
 'cumulative':'संचयी','curve':'वक्र','data':'डेटा','dataset':'डेटासेट',
 'decay':'क्षय','decoder':'डीकोडर','decoding':'डीकोडिंग','decision':'निर्णय',
 'decomposition':'अपघटन','deep':'गहन','definite':'निश्चित','denoising':'शोर निष्कासन',
 'density':'घनत्व','derivative':'अवकलज','descent':'अवरोहण',
 'description':'विवरण','detection':'पहचान','determinant':'सारणिक',
 'device':'उपकरण','diagonal':'विकर्ण','difference':'अंतर',
 'differential':'अवकल','differentiation':'अवकलन','diffusion':'विसरण',
 'dimension':'विमा','dimensionality':'विमीयता','direct':'प्रत्यक्ष',
 'discount':'छूट','discrete':'विविक्त','discriminator':'विभेदक',
 'distribution':'वितरण','distillation':'आसवन','distance':'दूरी',
 'distortion':'विरूपण','divergence':'विचलन','domain':'क्षेत्र',
 'dot':'डॉट','double':'दुगुना','downstream':'अनुवर्ती','drift':'बहाव',
 'dropout':'ड्रॉपआउट','dual':'द्वैत','duality':'द्वैतता',
 'dynamics':'गतिकी','early':'शीघ्र','eigenvalue':'अभिलक्षणिक मान',
 'element':'तत्व','embedding':'एम्बेडिंग','empirical':'अनुभवजन्य',
 'encoder':'एन्कोडर','encoding':'एन्कोडिंग','energy':'ऊर्जा',
 'entropy':'एंट्रॉपी','environment':'परिवेश','equation':'समीकरण',
 'equivariance':'समतुल्यता','error':'त्रुटि','estimate':'अनुमान',
 'estimation':'आकलन','estimator':'आकलक','evaluation':'मूल्यांकन',
 'example':'उदाहरण','expectation':'अपेक्षा','expected':'अपेक्षित',
 'expert':'विशेषज्ञ','experts':'विशेषज्ञों','exploration':'अन्वेषण',
 'exponential':'घातांकीय','exponent':'घातांक','factor':'गुणक',
 'factorization':'गुणनखंडन','feature':'विशेषता','feedback':'प्रतिक्रिया',
 'fidelity':'निष्ठा','field':'क्षेत्र','filter':'फ़िल्टर','fine':'सूक्ष्म',
 'first':'प्रथम','flow':'प्रवाह','floating':'चल','forward':'अग्र',
 'form':'रूप','free':'मुक्त','frequency':'आवृत्ति','function':'फलन',
 'gaussian':'गाउसियन','gate':'गेट','gated':'गेटयुक्त',
 'generalization':'सामान्यीकरण','generalized':'व्यापक',
 'generative':'जनन','generator':'जनरेटर','global':'वैश्विक',
 'gradient':'प्रवणता','graph':'आलेख','greedy':'लालची',
 'group':'समूह','guidance':'मार्गदर्शन','half':'अर्ध','head':'हेड',
 'hidden':'छिपी','hierarchical':'पदानुक्रमिक','high':'उच्च',
 'hyperparameter':'हाइपरपैरामीटर','hypothesis':'परिकल्पना',
 'identity':'तत्समक','image':'छवि','implicit':'अंतर्निहित',
 'importance':'महत्त्व','improvement':'सुधार','in':'में',
 'independent':'स्वतंत्र','induction':'आगमन','inference':'अनुमिति',
 'information':'सूचना','initialization':'आरंभीकरण',
 'input':'इनपुट','instance':'उदाहरण','integral':'समाकल',
 'integration':'समाकलन','interaction':'परस्पर क्रिया',
 'interpolation':'अंतर्वेशन','intersection':'प्रतिच्छेदन',
 'interval':'अंतराल','inverse':'प्रतिलोम','iteration':'पुनरावृत्ति',
 'item':'आइटम','joint':'संयुक्त','kernel':'कर्नेल','key':'कुंजी',
 'label':'लेबल','lagrange':'लाग्रांज','langevin':'लांजेविन',
 'language':'भाषा','large':'बड़ा','latent':'गुप्त','layer':'स्तर',
 'learning':'अधिगम','length':'लंबाई','likelihood':'संभाव्यता',
 'limit':'सीमा','linear':'रैखिक','lipschitz':'लिप्शित्स',
 'local':'स्थानीय','log':'लॉग','loss':'हानि','low':'निम्न',
 'machine':'मशीन','marginal':'सीमांत','markov':'मार्कोव',
 'mask':'मास्क','masked':'मास्कित','matching':'मिलान',
 'matrix':'आव्यूह','max':'अधिकतम','maximum':'अधिकतम',
 'mean':'माध्य','median':'माध्यिका','memory':'स्मृति',
 'metric':'मापदंड','min':'न्यूनतम','minimum':'न्यूनतम',
 'minibatch':'मिनीबैच','minimization':'न्यूनीकरण',
 'mixture':'मिश्रण','model':'मॉडल','moment':'आघूर्ण',
 'momentum':'संवेग','monte':'मोंटे','multi':'बहु','mutual':'पारस्परिक',
 'natural':'प्राकृतिक','negative':'ऋणात्मक','network':'नेटवर्क',
 'neural':'तंत्रिका','neuron':'न्यूरॉन','newton':'न्यूटन',
 'next':'अगला','noise':'शोर','non':'गैर','norm':'मानक',
 'normal':'सामान्य','normalization':'सामान्यीकरण',
 'normalizing':'सामान्यीकृत','numerical':'संख्यात्मक',
 'objective':'उद्देश्य','observation':'प्रेक्षण','of':'का',
 'offline':'ऑफ़लाइन','on':'पर','one':'एक','optimal':'इष्टतम',
 'optimality':'इष्टतमता','optimization':'अनुकूलन','optimizer':'अनुकूलक',
 'order':'क्रम','orthogonal':'लंबकोणीय','out':'बाहर',
 'output':'आउटपुट','over':'ऊपर','pair':'युग्म','parallel':'समांतर',
 'parallelism':'समांतरता','parameter':'पैरामीटर',
 'partial':'आंशिक','path':'पथ','pattern':'प्रतिरूप',
 'perceptual':'बोधात्मक','performance':'प्रदर्शन',
 'pixel':'पिक्सेल','point':'बिंदु','policy':'नीति','pooling':'पूलिंग',
 'position':'स्थिति','positional':'स्थितीय','positive':'धनात्मक',
 'posterior':'पश्च','precision':'परिशुद्धता','prediction':'पूर्वानुमान',
 'predictive':'पूर्वानुमानी','prefix':'उपसर्ग','pretraining':'पूर्व प्रशिक्षण',
 'principal':'प्रधान','prior':'पूर्व','probability':'प्रायिकता',
 'process':'प्रक्रिया','product':'गुणनफल','projection':'प्रक्षेपण',
 'proposal':'प्रस्ताव','proximal':'समीपस्थ','q':'क्यू',
 'quadratic':'द्विघात','quantile':'मात्रक','quantization':'परिमाणीकरण',
 'query':'क्वेरी','random':'यादृच्छिक','rank':'कोटि',
 'ranking':'रैंकिंग','rate':'दर','ratio':'अनुपात',
 'reconstruction':'पुनर्निर्माण','recurrent':'पुनरावर्ती',
 'reduction':'न्यूनीकरण','reference':'संदर्भ','region':'क्षेत्र',
 'regression':'प्रतिगमन','regularization':'नियमितीकरण',
 'reinforcement':'प्रबलन','representation':'निरूपण',
 'residual':'अवशिष्ट','resolution':'रिज़ॉल्यूशन',
 'retrieval':'पुनर्प्राप्ति','return':'प्रतिफल','reverse':'उल्टा',
 'reward':'पुरस्कार','risk':'जोखिम','root':'मूल','routing':'मार्गन',
 'rule':'नियम','sample':'नमूना','sampling':'प्रतिचयन',
 'scale':'पैमाना','scaling':'पैमाना परिवर्तन',
 'scatter':'बिखराव','schedule':'अनुसूची','score':'स्कोर',
 'search':'खोज','second':'द्वितीय','selection':'चयन',
 'self':'स्व','semantic':'अर्थगत','segmentation':'खंडीकरण',
 'sequence':'अनुक्रम','set':'समुच्चय','shift':'परिवर्तन',
 'shrinkage':'संकुचन','similarity':'समानता','single':'एकल',
 'singular':'एकल','size':'आकार','skip':'स्किप','soft':'सॉफ़्ट',
 'solver':'समाधानकर्ता','source':'स्रोत','space':'स्थान',
 'sparse':'विरल','sparsity':'विरलता','spectral':'वर्णक्रमीय',
 'stability':'स्थिरता','stable':'स्थिर','standard':'मानक',
 'state':'अवस्था','stationary':'स्थिर','statistic':'सांख्यिकी',
 'statistical':'सांख्यिकीय','step':'चरण','stochastic':'यादृच्छिक',
 'stop':'रोक','stopping':'रोकना','strong':'प्रबल',
 'subgradient':'उपप्रवणता','subword':'उपशब्द',
 'successive':'क्रमिक','supervised':'पर्यवेक्षित',
 'system':'प्रणाली','target':'लक्ष्य','teacher':'शिक्षक',
 'temperature':'तापमान','tensor':'टेंसर','term':'पद',
 'test':'परीक्षण','text':'पाठ','theorem':'प्रमेय',
 'through':'के माध्यम से','throughput':'उत्पादन दर',
 'time':'समय','to':'तक','token':'टोकन','total':'कुल',
 'training':'प्रशिक्षण','trajectory':'प्रक्षेपपथ',
 'transfer':'स्थानांतरण','transformation':'रूपांतरण',
 'transition':'संक्रमण','translation':'अनुवाद',
 'transpose':'परिवर्त','tree':'वृक्ष','tuning':'समायोजन',
 'uncertainty':'अनिश्चितता','under':'नीचे','uniform':'समरूप',
 'union':'संघ','unit':'इकाई','unsupervised':'अपर्यवेक्षित',
 'update':'अद्यतन','user':'उपयोगकर्ता','validation':'सत्यापन',
 'value':'मान','variable':'चर','variance':'प्रसरण',
 'variational':'परिवर्ती','vector':'सदिश','velocity':'वेग',
 'vision':'दृष्टि','vocabulary':'शब्दावली','warmup':'वार्मअप',
 'weight':'भार','width':'चौड़ाई','window':'विंडो','word':'शब्द',
 'zero':'शून्य',
}

# Less frequent components and inflections. Keeping this map separate makes
# the unavoidable editorial nature of compositional proposals easy to audit.
WORD.update({
 'method':'विधि','generation':'जनन','mode':'विधा','transformer':'ट्रांसफ़ॉर्मर',
 'operator':'संक्रियक','softmax':'सॉफ़्टमैक्स','inequality':'असमानता',
 'feasible':'व्यवहार्य','complexity':'जटिलता','gram':'ग्राम','map':'मानचित्रण',
 'behavior':'व्यवहार','bottleneck':'बाधा बिंदु','circuit':'परिपथ',
 'filtering':'फ़िल्टरिंग','sentiment':'भाव','law':'नियम',
 'identically':'समान रूप से','distributed':'वितरित','sufficient':'पर्याप्त',
 'null':'शून्य','sum':'योग','number':'संख्या','line':'रेखा',
 'continuity':'सातत्य','pass':'चरण','landscape':'परिदृश्य',
 'feedforward':'फ़ीडफ़ॉरवर्ड','corpus':'पाठ संग्रह','cosine':'कोज्या',
 'nearest':'निकटतम','neighbor':'पड़ोसी','modeling':'मॉडलिंग',
 'recognition':'पहचान','pointwise':'बिंदुवार','box':'बॉक्स','crop':'क्रॉप',
 'terminal':'अंतिम','bandit':'बैंडिट','calculus':'कलन',
 'effective':'प्रभावी','horizon':'क्षितिज','dynamic':'गतिशील','actor':'कर्ता',
 'ball':'गोला','trust':'विश्वास','problem':'समस्या','power':'घात',
 'independence':'स्वतंत्रता','truncation':'छंटाई','level':'स्तर',
 'grouped':'समूहित','relative':'सापेक्ष','pipeline':'पाइपलाइन',
 'depth':'गहराई','gap':'अंतर','margin':'हाशिया','selective':'चयनात्मक',
 'scan':'स्कैन','backpropagation':'बैकप्रॉपैगेशन','compilation':'संकलन',
 'reduce':'घटाना','advantage':'लाभ','least':'न्यूनतम','squares':'वर्ग',
 'ill':'खराब','basis':'आधार','bilinear':'द्विरैखिक','constant':'नियतांक',
 'population':'समष्टि','type':'प्रकार','true':'सत्य','false':'असत्य',
 'fully':'पूर्णतः','spatial':'स्थानिक','invariance':'अपरिवर्तनशीलता',
 'patch':'पैच','ground':'वास्तविक','truth':'सत्य','style':'शैली',
 'content':'सामग्री','variation':'विविधता','rating':'रेटिंग',
 'click':'क्लिक','pairwise':'युग्मवार','quality':'गुणवत्ता',
 'penalty':'दंड','masking':'मास्किंग','long':'लंबा',
 'conditioning':'सशर्तन','noisy':'शोरयुक्त','tokens':'टोकन',
 'launch':'आरंभ','multiplier':'गुणक','grid':'ग्रिड',
 'recommender':'अनुशंसा','mechanism':'तंत्र','ordinary':'साधारण',
 'motion':'गति','variables':'चर','property':'गुण','testing':'परीक्षण',
 'bootstrap':'बूटस्ट्रैप','central':'केंद्रीय','numbers':'संख्याएँ',
 'robustness':'सुदृढ़ता','privacy':'गोपनीयता','fairness':'निष्पक्षता',
 'bandwidth':'बैंडविड्थ','knowledge':'ज्ञान','mass':'द्रव्यमान',
 'posteriori':'पश्च','seed':'बीज','significance':'महत्त्व',
 'credible':'विश्वसनीय','probabilistic':'प्रायिकतामूलक',
 'evidence':'साक्ष्य','lower':'निचला','underflow':'अल्पप्रवाह',
 'overflow':'अतिप्रवाह','quasi':'अर्ध','coordinate':'निर्देशांक',
 'sharing':'साझाकरण','count':'गणना','propagation':'प्रसार',
 'efficiency':'दक्षता','inductive':'आगमनात्मक','universal':'सार्वत्रिक',
 'extraction':'निष्कर्षण','prompt':'प्रॉम्प्ट','scaled':'पैमाने पर',
 'rotary':'घूर्णी','square':'वर्ग','additive':'योगात्मक','byte':'बाइट',
 'analogy':'उपमा','bag':'थैला','words':'शब्द','sentence':'वाक्य',
 'question':'प्रश्न','answering':'उत्तर देना','named':'नामित',
 'entity':'सत्ता','strided':'स्ट्राइडयुक्त','transposed':'परिवर्तित',
 'dilated':'विस्तारित','depthwise':'गहराईवार','anchor':'एंकर',
 'suppression':'दमन','synthesis':'संश्लेषण','super':'सुपर',
 'pyramid':'पिरामिड','episode':'प्रकरण','off':'ऑफ़',
 'armed':'बाहु','regret':'पछतावा','experience':'अनुभव','replay':'पुनर्चालन',
 'discretization':'विविक्तीकरण','adjoint':'सहयुग्मी',
 'transform':'रूपांतरण','temporal':'कालिक','go':'गो',
 'programming':'प्रोग्रामिंग','contextual':'संदर्भगत','control':'नियंत्रण',
 'variate':'चर','fixed':'स्थिर','backup':'बैकअप','partially':'आंशिकतः',
 'observable':'प्रेक्षणीय','preconditioner':'पूर्वअनुबंधक',
 'sharpness':'तीक्ष्णता','heavy':'भारी','thresholding':'देहलीकरण',
 'subdifferential':'उपअवकलज','shadow':'छाया','price':'मूल्य',
 'ridge':'रिज','radius':'त्रिज्या','nuclear':'नाभिकीय',
 'span':'रैखिक विस्तार','equations':'समीकरण','concatenation':'संयोजन',
 'broadcasting':'प्रसारण','shape':'आकार','slice':'खंड',
 'finite':'परिमित','expansion':'विस्तार','surprisal':'आश्चर्य सूचना',
 'gamma':'गामा','preserving':'संरक्षण','exploding':'विस्फोटक',
 'transport':'परिवहन','collapse':'पतन','cycle':'चक्र','cache':'कैश',
 'sink':'सिंक','stream':'प्रवाह','prefill':'पूर्वभराई',
 'decode':'डीकोड','intensity':'तीव्रता','mixed':'मिश्रित',
 'loader':'लोडर','moving':'गतिमान','inverted':'उलटा',
 'logit':'लॉजिट','recurrence':'पुनरावृत्ति','dependent':'आश्रित',
 'forget':'भूल','reset':'रीसेट','truncated':'छंटा हुआ',
 'transduction':'रूपांतरण','router':'राउटर','load':'भार',
 'balancing':'संतुलन','footprint':'स्मृति पदचिह्न',
 'hierarchy':'पदानुक्रम','access':'अभिगम','just':'केवल',
 'execution':'निष्पादन','synchronization':'समकालिकन',
 'accumulation':'संचय','reproducibility':'पुनरुत्पादकता',
 'rounding':'पूर्णांकन','range':'परास','cloning':'अनुकरण',
 'hacking':'हैकिंग','planning':'योजना','bootstrapping':'बूटस्ट्रैपिंग',
 'bonus':'अधिलाभ','upper':'ऊपरी','multivariate':'बहुचर',
 'semidefinite':'अर्धनिश्चित','symmetric':'सममित','subspace':'उपस्थान',
 'inner':'आंतरिक','outer':'बाहरी','conditioned':'अनुबंधित',
 'invertible':'व्युत्क्रमणीय','pseudoinverse':'छद्म व्युत्क्रम',
 'orthonormal':'लंबमानकीकृत','eigendecomposition':'अभिलक्षणिक अपघटन',
 'left':'बायाँ','right':'दायाँ','triangle':'त्रिभुज',
 'directional':'दिशात्मक','initial':'प्रारंभिक',
 'indefinite':'अनिश्चित','by':'से','parts':'भाग',
 'missingness':'अनुपस्थिति','denoiser':'शोर हटाने वाला',
 'minimax':'मिनिमैक्स','smoothness':'चिकनापन','primal':'प्राथमिक',
 'complementary':'पूरक','slackness':'शिथिलता','curvature':'वक्रता',
 'preconditioning':'पूर्वअनुबंधन','minimizer':'न्यूनक',
 'maximizer':'अधिकतमक','outlier':'अपवाद मान','asymptotic':'अनंतस्पर्शी',
 'identifiability':'पहचानयोग्यता','hyperprior':'हाइपरपूर्व वितरण',
 'autocorrelation':'स्वसहसंबंध','bottleneck':'बाधा बिंदु',
 'launch':'आरंभ','overhead':'अतिरिक्त लागत','batching':'बैच बनाना',
 'reflow':'रीफ़्लो',
 'carlo':'कार्लो','conjugate':'संयुग्मी','padding':'पैडिंग',
 'fundamental':'मूल','samples':'नमूने','unbiased':'निष्पक्ष',
 'consistent':'संगत','rejection':'अस्वीकृति','processing':'प्रसंस्करण',
 'simple':'सरल','modelability':'मॉडलनीयता','logistic':'लॉजिस्टिक',
 'receiver':'ग्राही','operating':'प्रचालन','characteristic':'अभिलक्षण',
 'area':'क्षेत्रफल','the':'यह','imbalance':'असंतुलन',
 'strength':'शक्ति','held':'रखा गया','covariate':'सहचर',
 'concept':'अवधारणा','intercept':'अंतःखंड','predictor':'पूर्वानुमानक',
 'closed':'बंद','solution':'समाधान','odds':'ऑड्स',
 'heteroscedastic':'विषम प्रसरणशील','squared':'वर्ग',
 'regularized':'नियमितीकृत','leaky':'लीकी','hyperbolic':'अतिपरवलयिक',
 'tangent':'स्पर्शज्या','connected':'संयोजित','full':'पूर्ण',
 'decoupled':'पृथक्कृत','correction':'सुधार','receptive':'ग्रहणशील',
 'dilation':'फैलाव','downsampling':'डाउनसैंपलिंग',
 'upsampling':'अपसैंपलिंग','aliasing':'एलियसिंग',
 'horizontal':'क्षैतिज','flip':'पलटना','resized':'पुनः आकारित',
 'color':'रंग','jitter':'परिवर्तन','synthetic':'कृत्रिम',
 'recipe':'विधि','shortcut':'संक्षिप्त मार्ग','bounding':'परिबद्ध',
 'interest':'रुचि','shot':'चरण','multibox':'बहु बॉक्स',
 'recommendation':'अनुशंसा','explicit':'स्पष्ट','personalized':'व्यक्तिगत',
 'start':'आरंभ','tokenization':'टोकनीकरण','unknown':'अज्ञात',
 'beginning':'आरंभ','end':'अंत','center':'केंद्रीय',
 'occurrence':'प्रकटन','vectors':'सदिश','subsampling':'उपप्रतिचयन',
 'alignment':'संरेखण','exposure':'प्रदर्शन','scheduled':'अनुसूचित',
 'brevity':'संक्षिप्तता','bidirectional':'द्विदिश',
 'task':'कार्य','textual':'पाठीय','entailment':'निहितार्थ',
 'premise':'पूर्वधारणा','contradiction':'विरोधाभास',
 'smoothing':'सुगमीकरण','conjugacy':'संयुग्मिता',
 'quantification':'परिमाणीकरण','missing':'अनुपस्थित',
 'based':'आधारित','partition':'विभाजन',
 'unnormalized':'असामान्यीकृत','annealed':'अनीलित',
 'signal':'संकेत','clean':'शुद्ध','ancestral':'पूर्वज आधारित',
 'autoencoder':'ऑटोएन्कोडर','dropping':'छोड़ना',
 'diversity':'विविधता','discrepancy':'अंतर',
 'game':'खेल','critical':'क्रांतिक','strict':'सख्त',
 'constrained':'बंधनयुक्त','weak':'दुर्बल',
 'projected':'प्रक्षेपित','backtracking':'पश्च खोज',
 'direction':'दिशा','lemma':'सहायक प्रमेय',
 'averaging':'औसतीकरण','buffer':'बफ़र',
 'guarantee':'गारंटी','surface':'सतह',
 'nonconvex':'अनुत्तल','family':'कुल','normality':'सामान्यता',
 'acquisition':'अधिग्रहण','halving':'अर्धीकरण',
 'scheduling':'अनुसूचन','completion':'पूर्णन',
 'tail':'पूँछ','hit':'प्राप्ति','normalized':'सामान्यीकृत',
 'discounted':'छूटयुक्त','gain':'लाभ','theory':'सिद्धांत',
 'lossless':'हानिरहित','measure':'माप','detailed':'सूक्ष्म',
 'mixing':'मिश्रण','walk':'भ्रमण','per':'प्रति',
 'tying':'साझाकरण','table':'तालिका','top':'शीर्ष',
 'shared':'साझा','parameters':'पैरामीटर','format':'प्रारूप',
 'gather':'संग्रह',
})

def norm(s):
    return re.sub(r'[^a-z0-9]+', '', s.casefold())

def compact(s):
    return re.sub(r'\s+', ' ', s).strip()

def proposition(c):
    if c['id'] in PREFERRED:
        return PREFERRED[c['id']]
    en = c['en']
    # Names and established abbreviations remain legible in Roman script,
    # while the surrounding descriptive parts are translated.
    parts = re.findall(r"[A-Za-z]+|\d+|[^A-Za-z\d\s-]+", en)
    converted = [WORD.get(p.casefold(), p) for p in parts]
    result = compact(' '.join(converted))
    return unicodedata.normalize('NFC', result)

def heading_context(h):
    chunks=[]
    for sib in h.next_siblings:
        if getattr(sib, 'name', None) == 'h2':
            break
        if getattr(sib, 'name', None) in ('p','ul','ol'):
            t=compact(sib.get_text(' ',strip=True))
            if t and t not in ('#fundamentals','#मेट्रिक'):
                chunks.append(t)
                if len(' '.join(chunks))>180:break
    return compact(' '.join(chunks))[:180]

def google_candidates(concepts):
    by_en={norm(c['en']):c for c in concepts}
    out={}
    sources={}
    for topic in TOPICS:
        ep= CACHE/f'{topic}-en.html'; hp=CACHE/f'{topic}-hi.html'
        if not ep.exists() or not hp.exists():continue
        e=BeautifulSoup(ep.read_text(errors='replace'),'html.parser')
        h=BeautifulSoup(hp.read_text(errors='replace'),'html.parser')
        hi={x['id']:x for x in h.select('h2[id]')}
        sid=f'google-ml-hi-{topic}'
        sources[sid]={
            'id':sid,'title':f'Google Machine Learning Glossary: {topic}',
            'locale':'hi','kind':'localized_teaching_glossary',
            'url':f'https://developers.google.com/machine-learning/glossary/{topic}?hl=hi',
            'accessed':ACCESSED,'publisher':'Google for Developers',
            'notes':'Hindi topic page and definition body inspected for cited entries. HTML lang=hi-x-mtfrom-en discloses machine translation; no human review is claimed.'}
        for x in e.select('h2[id]'):
            c=by_en.get(norm(x.get_text(' ',strip=True)))
            if not c or x['id'] not in hi:continue
            term=unicodedata.normalize('NFC',compact(hi[x['id']].get_text(' ',strip=True)))
            if not re.search(r'[\u0900-\u097f]',term):continue
            out.setdefault(c['id'],(topic,x['id'],term,heading_context(hi[x['id']])))
    return out,sources

def main():
    concepts=[json.loads(x) for x in (BASE/'concepts.jsonl').read_text().splitlines() if x.strip()]
    candidates,sources=google_candidates(concepts)
    sources[CBSE_ID]={
        'id':CBSE_ID,'title':'Artificial Intelligence Integration Manual (Hindi)',
        'locale':'hi','kind':'published_teaching_manual','url':CBSE_URL,
        'accessed':ACCESSED,'publisher':'Central Board of Secondary Education (CBSE)',
        'notes':'PDF page 125, printed page 123, visually inspected. Legacy Hindi text extraction is unreliable; cited terms are transcribed from the page image.'}
    rows=[];used_sources=set()
    for c in concepts:
        cid=c['id']; proposed=proposition(c)
        hit=candidates.get(cid)
        if cid in VERIFIED and hit:
            topic,anchor,term,_context=hit
            # Google uses short labels such as 'सटीक' for accuracy. Keep the
            # attested form as an alternative when a clearer Hindi noun exists.
            preferred=proposed if cid in PREFERRED else term
            if cid=='accuracy':preferred='सटीकता'
            alternatives=([term] if preferred!=term else [])
            sid=f'google-ml-hi-{topic}';used_sources.add(sid)
            rows.append({'concept_id':cid,'preferred':preferred,'alternatives':alternatives,
              'status':'verified_external' if preferred==term else 'proposed',
              'source_ids':[sid],
              'evidence':[{'source_id':sid,'locator':f'{sources[sid]["url"]}#{anchor}',
                 'source_term':term,
                 'context':f'Localized heading aligned with English “{c["en"]}”; inspected Hindi definition describes the D2L sense: {c["sense"]}'}],
              'notes':('Published teaching glossary; publisher discloses machine translation. '
                 'Preferred form still needs Hindi editorial review.' if preferred==term else
                 'Google attests the listed alternative; the clearer preferred form is an editorial proposal.')})
        else:
            notes='Editorial Hindi terminology proposal based on the D2L sense; native-language teaching attestation and review pending.'
            if hit:
                notes+=' A Google localized heading candidate was not adopted because its sense or wording needs review.'
            if re.search(r'[A-Za-z]',proposed):
                notes+=' Roman-script name or unreviewed borrowing remains within this Hindi phrase.'
            rows.append({'concept_id':cid,'preferred':proposed,'alternatives':[],
                         'status':'proposed','source_ids':[],'evidence':[],'notes':notes})
    by_id={x['concept_id']:x for x in rows}
    for cid,(term,context,prefer) in CBSE_ATTEST.items():
        row=by_id[cid];used_sources.add(CBSE_ID)
        if prefer:
            old=row['preferred']
            row['preferred']=term
            if old!=term and old not in row['alternatives']:
                row['alternatives'].append(old)
            row['status']='verified_external'
            row['notes']='Preferred form appears in visually inspected CBSE Hindi teaching prose; alternatives retain their own evidence or remain proposals.'
        elif term!=row['preferred'] and term not in row['alternatives']:
            row['alternatives'].append(term)
            row['notes']+=' CBSE attests the listed alternative in teaching prose.'
        row['source_ids'].append(CBSE_ID)
        ev={'source_id':CBSE_ID,
            'locator':f'{CBSE_URL}#page=125',
            'source_term':term,'context':context}
        if prefer:row['evidence'].insert(0,ev)
        else:row['evidence'].append(ev)
    for cid,(topic,anchor,term,invariant,context) in GOOGLE_BODY_ATTEST.items():
        row=by_id[cid];sid=f'google-ml-hi-{topic}';used_sources.add(sid)
        row['preferred']=term
        row['status']='invariant' if invariant else 'verified_external'
        if sid not in row['source_ids']:row['source_ids'].append(sid)
        row['evidence'].insert(0,{'source_id':sid,
            'locator':f'{sources[sid]["url"]}#{anchor}',
            'source_term':term,'context':context})
        row['notes']=('English term deliberately retained because the Hindi glossary uses it as the heading and explains it in Hindi.'
            if invariant else 'Hindi definition body uses this technical loanword in the D2L sense; the publisher discloses machine translation.')
    for row in rows:
        row['preferred']=unicodedata.normalize('NFC', row['preferred'])
        row['alternatives']=[unicodedata.normalize('NFC', x) for x in row['alternatives']]
        if row['concept_id']=='adaln-zero':
            row['notes']='Named normalization/initialization scheme retained as adaLN-Zero; Hindi usage still needs attestation.'
    assert len(rows)==len(concepts)
    (BASE/'terms'/'hi.jsonl').write_text(''.join(json.dumps(x,ensure_ascii=False)+'\n' for x in rows))
    (BASE/'sources'/'group-hi.jsonl').write_text(''.join(json.dumps(sources[s],ensure_ascii=False)+'\n' for s in sorted(used_sources)))
    print(f'{len(rows)} Hindi rows; {sum(x["status"]=="verified_external" for x in rows)} source-attested preferred forms; {len(used_sources)} source pages')
    print('Roman-only preferred forms:',sum(not re.search(r'[\u0900-\u097f]',x['preferred']) for x in rows))

if __name__=='__main__':main()
