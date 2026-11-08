# Arabic, Turkish, and German term review

This shard follows the 1,184 English concept IDs in `concepts.jsonl`. Terms were chosen from pinned translated D2L chapters, the Turkish D2L editorial `TERMINOLOGY.md`, Taima's independent Arabic D2L PDF edition, Amidi's bilingual teaching sheets, a KIT bilingual glossary, and Google's explicitly machine-translated glossary topic pages. The exact source, short source expression, and locator are recorded in each term row. The source registry distinguishes independent publications from the official D2L repository. An editorial phrase without a reviewed same-sense bilingual source remains `proposed`, even if a string search found it in teaching prose. Neither a substring hit nor a machine-translated heading by itself establishes semantic correctness.

## Semantic spot checks

The following are **20 concrete checks per locale**, spanning source families and common homographs. The check concerns the concept sense and preferred term, not merely whether a string appears at the cited locator. It does not certify every other row. Citations are in the corresponding `terms/*.jsonl` rows.

| Arabic concept ID | Preferred | Sense decision |
|---|---|---|
| `linear-regression` | الانحدار الخطّي | Bilingual CS229 regression method, rather than literal motion backward. |
| `regression` | الانحدار | Taima uses it in predictive-model context. |
| `test-set` | مجموعة الاختبار | Dataset split, separate from training and validation. |
| `gradient` | التدرج | Taima's function gradient; rejected the Amidi sheet's long paraphrase of a high-dimensional derivative. |
| `derivative` | المشتق | Calculus derivative, distinct from the gradient vector. |
| `partial-derivative` | المشتق الجزئي | Derivative with respect to one variable. |
| `matrix` | مصفوفة | Bilingual linear-algebra heading, mathematical matrix. |
| `expectation` | التوقع | Probabilistic expectation, not everyday anticipation. |
| `sampling` | أخذ العينات | Drawing samples in Taima's probabilistic discussion. |
| `logistic-regression` | الانحدار اللوجستي | CS229 classification model, distinct from generic regression. |
| `regularization` | التنظيم | Taima PDF page 175 explicitly pairs this with “Regularization”; supersedes Amidi's broad ضبط. |
| `overfitting` | الضبط الزائد | Taima PDF page 164 pairs the heading with “Overfitting”; later text also uses فرط التجهيز. |
| `underfitting` | الضبط الناقص | Same page pairs the heading with “Underfitting.” |
| `dropout` | التسرب | Neural-network dropout in Taima, not ordinary leakage. |
| `kernel` | نواة الالتفاف | Taima volume 2 page 60 says نواة الالتفاف beside “convolution kernel”; CS229's similarity kernel was rejected for this CNN concept. |
| `embedding` | التضمين | Learned representation in Taima's NLP material. |
| `attention` | الانتباه | Neural attention mechanism, rather than ordinary attention. |
| `convexity` | التحدب | Mathematical convexity in Taima's optimization material. |
| `statistical-bias` | الانحياز | Estimator bias in CS229 probability, distinct from the neural intercept. |
| `gram-matrix` | مصفوفة جرام | Matrix of feature inner products in Taima's style-transfer material. |

| Turkish concept ID | Preferred | Sense decision |
|---|---|---|
| `linear-regression` | Doğrusal Bağlanım (Regresyon) | Full D2L chapter heading records both labels; do not conflate with general bağlanım. |
| `regression` | regresyon | D2L editorial list explicitly gives bağlanım veya regresyon; concise lemma chosen. |
| `test-set` | test kümesi | Held-out test split in D2L dataset chapter. |
| `bias` | ek girdi | D2L editorial list explicitly marks the neural-network sense; this is not statistical yanlılık. |
| `statistical-bias` | yanlılık | Same list explicitly marks the algorithmic/statistical sense. |
| `gradient` | gradyan | Editorial pair gradyan, eğim; calculus gradient chosen. |
| `derivative` | türev | Inflected source türevi normalized to dictionary form. |
| `partial-derivative` | kısmi türev | Inflected source kısmi türevi normalized. |
| `expectation` | beklenti | Probabilistic expectation, source inflected as beklentisi. |
| `softmax` | softmaks | D2L editorial pair with the Turkish paraphrase; short technical lemma chosen. |
| `logistic-regression` | Lojistik regresyon | Explicit bilingual CS229 classification model. |
| `overfitting` | aşırı öğrenme | D2L editorial equivalent for overfitting. |
| `regularization` | düzenlileştirme | D2L editorial equivalent, distinct from parameter tuning. |
| `dropout` | seyreltme | Amidi's bilingual neural-net sheet explicitly defines random unit removal with Seyreltme; D2L's literal hattan düşme remains an alternative. |
| `kernel` | çekirdek | D2L editorial mathematical/convolutional kernel term. |
| `state` | Durum | Amidi's Markov-decision-process state, not a political state. |
| `policy` | Politika | Amidi's mapping from states to actions. |
| `convexity` | Dışbükeylik | D2L optimization chapter heading. |
| `overflow` | taşma | Numerical overflow in D2L softmax discussion; source verb taşar normalized. |
| `gram-matrix` | Gram matrisi | Style-transfer matrix, source locative Gram matrisinde normalized. |

| German concept ID | Preferred | Sense decision |
|---|---|---|
| `linear-regression` | Lineare Regression | KIT bilingual appendix explicitly pairs the ML model. |
| `classification` | Klassifikation | KIT's ML task label, not a generic classification scheme. |
| `training-set` | Trainingsdaten | KIT model-training data split. |
| `test-set` | Testdaten | KIT held-out evaluation data. |
| `learning-rate` | Lernrate | Amidi bilingual optimization parameter. |
| `backpropagation` | Fehlerrückführung | Amidi bilingual neural gradient algorithm. |
| `automatic-differentiation` | Automatische Differenzierung | German D2L chapter heading on autograd. |
| `conditional-probability` | Bedingte Wahrscheinlichkeit | D2L conditional probability, with mathematical conditioning. |
| `bayes-theorem` | Satz von Bayes | German D2L probability passage. |
| `logistic-regression` | logistische Regression | Google's German definition classifies it as a binary model. |
| `precision` | Präzision | KIT evaluation metric; distinguished from floating-point precision. |
| `recall` | Trefferquote | KIT evaluation metric, distinct from memory recall. |
| `activation-function` | Aktivierungsfunktion | Amidi neural-net function. |
| `overfitting` | Überanpassung | Google German model-generalization definition. |
| `underfitting` | Unteranpassung | Same generalization contrast. |
| `regularization` | Regularisierung | Model complexity control, distinct from ordinary regulation. |
| `policy` | Strategie | Amidi explicitly defines a state-to-action mapping. |
| `value-function` | Wertfunktion | Same RL sheet, function of policy and state. |
| `q-learning` | Q-Lernen | Same RL algorithmic section. |
| `overflow` | Überlauf | Numerical overflow in German D2L softmax chapter. |

During this audit, source-aligned but bad German machine renderings were demoted to proposals: `vanishing-gradient` had *Verschwindende Farbverläufe* (color gradients), `exploding-gradient` had an awkward literal *Explodieren von Verläufen*, `joint-probability` had *Gelenkwahrscheinlichkeit* (anatomical “joint”), and `covariate-shift` / `label-shift` had unnatural literal compounds. A bilingual sheet's *Bündelnormalisierung* was also excluded for `batch-normalization`. Arabic `kernel` was corrected from a CS229 similarity-kernel citation to the CNN convolution kernel in Taima's volume 2. These cases demonstrate why alignment alone is not a verification decision.

## PDF extraction and rights

Taima's volume 1 PDF pages 164 and 175 and volume 2 page 60 were rendered and inspected as images alongside extracted text. Page 164 visibly places the Arabic headings الضبط الناقص and الضبط الزائد beside “Underfitting” and “Overfitting.” Page 175 visibly pairs التنظيم with “Regularization” in the chapter summary. Volume 2 page 60 pairs نواة الالتفاف with “convolution kernel.” A representative early page (volume 1 PDF page 8) was also rendered to check the text direction. **The other PDF-sourced terms were checked against the PDF text layer and local context, not visually checked page by page.** That layer sometimes has duplicated initial alif and bidirectional controls; `source_term` preserves its exact extracted spelling, while `preferred` gives a readable form. Taima's independent edition does not establish a license to redistribute the PDF text.

The KIT thesis appendix A.2 on PDF pages 53–54 was inspected in text and page 53 was rendered. The appendix is a short external bilingual list, so its attested terms carry source attribution rather than any assumption that KIT's glossary is a D2L translation. Amidi's sheets are MIT-licensed community translations. Google topic pages explicitly mark their locale as machine-translated (`x-mtfrom-en`), and the broader glossary landing pages in Arabic and German fall back to English; only localized topic sections with inspected target text were considered.

## Remaining review boundary

`verified_d2l` and `verified_external` rows record an attested target expression at a precise source location; the 60 IDs above received explicit manual sense checks. Other automatically aligned source pairs still deserve editorial spot review, especially machine-translated chapters and Google pages. `proposed` rows are editorial translations suitable for human terminology review. The three D2L editions and the supplemental publications differ in date and style. For Turkish dropout, the independent bilingual teaching source supports `seyreltme` as preferred, with D2L's literal `hattan düşme` retained as an attested alternative. The nearly complete proposal matrix should not be presented as 1,205 independently published translations per language.

## Final synonym reconciliation

The language review preceded the final 21 synonym merges. The canonical shard now has 1,184 concepts; [COVERAGE.md](../COVERAGE.md) gives current status counts. Audited older IDs resolve through [concept-redirects.json](../concept-redirects.json). Pre-merge variants and their evidence are preserved in `candidates/merged-variant-terms.jsonl`; they are not automatically approved alternatives.
