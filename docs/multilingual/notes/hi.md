# Hindi terminology review

The Hindi file covers all 1,184 English concepts. It is a reviewable working glossary, **not** a certified Hindi translation. As of 2026-11-07, 56 preferred forms are attested in inspected Hindi teaching sources (Google topic pages and a CBSE manual), one English term is deliberately retained with source evidence (`invariant`), and 1,127 are marked `proposed`. Google's Hindi pages identify themselves as machine translated (`lang="hi-x-mtfrom-en"`); publication and same-sense usage are recorded, but human editorial review is not assumed. The all-topics Google page had an English body despite its Hindi locale and was not used. No earlier Hindi D2L edition was found in the documented best-effort inventory.

`scripts/build_hi.py` generates the canonical shard. Its explicit preferred-form table and component lexicon are editorial aids. Component translations do not create source verification. Entries with Roman names/acronyms (such as BERT, AdamW, DDPM, and ODE) remain `proposed` unless there is term-level evidence; mixed script is intentional for recognizable model names and standard abbreviations. The script stores source evidence and its page locator for each attested form. Preferred forms that improve on an awkward published heading are still `proposed`, with the published form in `alternatives` and evidence.

## Same-sense source audit

The following 25 Google entries were checked against the Hindi definition following the localized heading, not merely a search snippet or the English anchor. Locators and brief contextual descriptions are in `terms/hi.jsonl`. All these are `verified_external` in the output. They attest published Hindi usage; the publisher's machine-translation disclosure still applies.

| Concept IDs | Definition-body check |
| --- | --- |
| `linear-regression`, `logistic-regression` | The definitions respectively describe a linear model for floating-point prediction and a model for predicting a probability. |
| `label`, `training-set`, `validation-set` | The definitions distinguish the example's target, the fit subset, and the development-evaluation subset. |
| `prediction`, `loss-function`, `objective-function` | The definitions identify model output, a mathematical loss calculation, and the quantity to optimize. |
| `minibatch`, `batch-size` | They identify a randomly selected update subset and the number of examples in a batch. |
| `gradient-descent`, `learning-rate`, `backpropagation` | The definitions describe repeated parameter adjustment against loss, update step size, and backward gradient computation in neural networks. |
| `entropy`, `cross-entropy`, `softmax` | The definitions discuss uncertainty, a log-loss-derived objective, and class probabilities summing to one. |
| `precision`, `recall`, `confusion-matrix` | They are explicitly classification metrics/table; the precision/recall formulas and true/false counts appear in the inspected material. |
| `neural-network`, `hidden-layer`, `activation-function` | Definitions identify a layered network, an interior layer, and a nonlinear transform. |
| `overfitting`, `underfitting`, `generalization` | Definitions distinguish poor unseen-data performance from insufficient training fit and successful unseen-data behavior. |

`accuracy` uses the noun `सटीकता` found in the inspected Hindi definition body; the heading adjective `सटीक` remains an attested alternative. `feature` is preferred as `विशेषता` rather than Google's `सुविधा`. The agent glossary's `ऐक्शन गेम` and `राज्य` were rejected for RL `action` and `state`. Google's `log-odds` definition appears to describe the log of a probability rather than the log of odds, so `लॉग ऑड्स` remains proposed. The Google `convergence` definition is narrower than the mathematical concept in D2L and was not marked verified. Google's `एक्सपर्ट का मिक्सचर` is an attested alternative to the proposed, more natural `विशेषज्ञों का मिश्रण`.

The CBSE Hindi manual was visually inspected at PDF page 125 (printed page 123). It explicitly uses `तंत्रिका नेटवर्क` for a network in image recognition, `छवि पहचान` for identifying image content, and `परीक्षण डेटा` for the held-out test partition; these three are preferred `verified_external` forms. The page also uses `प्रशिक्षण डेटा` for the training partition and `भविष्यवाणी` for model prediction; they are recorded as attested alternatives to the existing preferred forms. The page's Hindi text layer is unreliable, so the page image, not extracted text, was reviewed.

Google's Hindi definition body also uses `प्रॉम्प्ट` for language-model input and `टेंसर` for the N-dimensional TensorFlow structure; these loanwords are `verified_external`. Its `epoch` heading remains in English and its Hindi definition explains one full training-set pass, so that one term is `invariant` with an explicit retention reason.

## Editorial sense and readability audit

The following 39 proposed preferreds were examined against their English concept definitions across mathematics, ML, language, vision, RL, dynamics, performance, and recommendation. These checks are editorial judgments rather than source attestation. Several were revised during the audit; all remain `proposed` pending Hindi teaching-source and native-language review.

| Concept | Preferred | Decision |
| --- | --- | --- |
| `regression` | प्रतिगमन | Numerical target task. |
| `classification` | वर्गीकरण | Class assignment task. |
| `bias` | बायस पद | Additive model offset; distinguished from estimator bias. |
| `statistical-bias` | आकलक की अभिनति | Systematic estimator error; distinguished from additive offset. |
| `bias-variance-decomposition` | बायस-प्रसरण अपघटन | Statistical decomposition, not ethical prejudice. |
| `label-shift` | लेबल वितरण परिवर्तन | Shift in class-label distribution, not a relabeling operation. |
| `feature-map` | विशेषता मानचित्र | Representation, rather than the action of mapping. |
| `partition-function` | विभाजन फलन | Probability/statistical-mechanics normalizer. |
| `normalizing-constant` | सामान्यीकरण नियतांक | Constant *used for* normalization, not one already normalized. |
| `cold-start` | कोल्ड स्टार्ट | Recognizable recommendation-system loanword; not yet attested here. |
| `ito-integral` | इतो समाकल | Fixed accented-name tokenization failure. |
| `kernel` | संवलन कर्नेल | Local convolution weights. |
| `kernel-function` | कर्नेल फलन | GP covariance/similarity function, separated from convolution. |
| `transition-kernel` | संक्रमण कर्नेल | Conditional next-state distribution. |
| `return` | संचित प्रतिफल | Future RL reward sum, not a programming return. |
| `score-function` | लॉग घनत्व की प्रवणता | Probability-density score, not an evaluation score. |
| `score-estimate` | लॉग घनत्व प्रवणता का अनुमान | Carries the same density-score sense. |
| `attention-score` | अटेंशन स्कोर | Query–key compatibility, distinct from density score. |
| `policy` | नीति | RL action-selection rule. |
| `state` | अवस्था | RL state, rather than a political state. |
| `stochastic-gradient-descent` | यादृच्छिक प्रवणता अवरोहण | Optimization method. |
| `tensor` | टेंसर | Established technical loanword preferred over a forced compound. |
| `covariance` | सहप्रसरण | Joint variation of two random variables. |
| `denoising-score-matching` | शोर निष्कासन आधारित स्कोर मिलान | Scores mean log-density gradients. |
| `diffusion-model` | विसरण मॉडल | Generative noising/reversal method. |
| `probability-flow-ode` | प्रायिकता प्रवाह ODE | The abbreviation is intentionally retained. |
| `language-model` | भाषा मॉडल | Language-token sequence probabilities. |
| `tokenization` | टोकनीकरण | Segmentation of text into tokens. |
| `self-attention` | स्व अटेंशन | Attention within the same sequence. |
| `mixture-of-experts` | विशेषज्ञों का मिश्रण | Readability edit; Google heading is recorded only as an alternative. |
| `critic-generative` | प्रतिद्वंद्वी जनन मॉडल का आलोचक | Critic in adversarial generation, not a general reviewer. |
| `computer-vision` | कंप्यूटर दृष्टि | Visual machine-learning field. |
| `bounding-box` | परिबद्ध बॉक्स | Image object rectangle. |
| `training-throughput` | प्रशिक्षण थ्रूपुट | Uses the familiar performance loanword. |
| `loss-scaling` | हानि का पैमाना बढ़ाना | Operation in mixed-precision training. |
| `reduce-scatter` | रिड्यूस स्कैटर | Retains recognizable collective-operation name. |
| `double-descent` | डबल डिसेंट | Familiar technical name; a literal “double descent” compound was rejected. |
| `matrix-completion` | आव्यूह पूर्णन | Filling unknown entries of a matrix. |
| `information-theory` | सूचना सिद्धांत | Uncertainty/compression/communication field. |

Some specialized proposals may still sound unnatural or need different wording in context. They are deliberately excluded from the source-attested automation file. In particular, modern named architectures, rare numerical methods, and some multiword compounds need expert Hindi review before a translator should treat them as fixed terminology.

## Second whole-phrase QA pass

After the English inventory grew to 1,205 concepts, the four added basics (`model`, `training`, `activation`, `mask`) received provisional Hindi preferreds. I inspected the generated list of component-built phrases across the catalog and added 121 explicit whole-phrase decisions in `build_hi.py`. These decisions remain `proposed`; they do not borrow verification from a translated component or from the English source. Existing source-backed decisions and their locators were retained.

The pass corrected technical senses that an English word-by-word mapping obscured: image *translation* is `स्थानांतरण`, while language translation remains `अनुवाद`; distribution-relative phrases now say whether data match or lie outside the training distribution; density-score terms refer to log-density gradients; `bias-correction` specifies estimator bias; and `kernel-launch` describes starting a GPU operation. It also corrected word order in the law of large numbers, image-to-image transformation, reward-to-go, time to first token, posterior predictive expressions, and the square root in root-mean-square. Familiar technical loanwords were retained for normalizing flow, off-policy/on-policy, mode collapse, and distributed collectives where forced literal compounds were misleading.

This is a bounded editorial pass, not a claim that every one of the 1,148 proposals is idiomatic in every Hindi sentence. Translator review must still decide inflection, context, and style. Specialized mathematical vocabulary, newer architectures, and the remaining compositional proposals are the main review queue; none enters the source-attested automation subset solely because this pass improved its wording.

## Source limits and next review queue

- The CBSE [AI Integration Manual](https://cbseacademic.nic.in/web_material/Curriculum20/AI_Integration_Manual_Hindi.pdf) was downloaded. PDF page 125 (printed 123) supports the five entries described above; an earlier inspected lesson page 123 (printed 121) did not. The legacy Hindi text layer is unreliable, so evidence was visually transcribed.
- The CSTT AI/ML and mathematics glossaries and the IGNOU regression/statistics PDFs are documented in `hindi-source-leads.md`. Retrieval attempts failed or yielded only snippets, so no entries were marked verified from them.
- Review first the high-impact proposals `bias`, `statistical-bias`, `kernel`, `kernel-function`, `return`, `score-function`, `label-shift`, and `mixture-of-experts` against accessible Hindi university or government teaching prose. Then review rare modern ML terminology and remaining compositional proposals.

## Final synonym reconciliation

The language review preceded the final 21 synonym merges. The canonical shard now has 1,184 concepts; [COVERAGE.md](../COVERAGE.md) gives current status counts. Audited older IDs resolve through [concept-redirects.json](../concept-redirects.json). Pre-merge variants and their evidence are preserved in `candidates/merged-variant-terms.jsonl`; they are not automatically approved alternatives.
