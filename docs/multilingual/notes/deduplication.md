# English concept deduplication

The catalog contains 1,184 concepts after 21 merges from 1,205. `concept-redirects.json` maps each retired ID directly to its retained ID. For each merge, the retained record includes both records' English names and aliases, distinct book references, and the strongest prior `source_review` level. A retained record's review level describes the strongest existing review, not a new audit of every merged reference. English senses were clarified for skip-gram, DDIM, loss landscape, image generation, and hyperparameter optimization to cover the merged usage.

The following retired IDs map to the retained IDs shown:

| Retired | Retained | Rationale |
| --- | --- | --- |
| layer-norm | layer-normalization | Abbreviated form of the same normalization. |
| self-information | surprisal | Both denote negative log probability. |
| position-encoding | positional-encoding | Equivalent names for information representing sequence position. |
| prior-distribution | prior | Both denote a distribution before conditioning on observations. |
| posterior-distribution | posterior | Both denote a distribution after conditioning on observations. |
| skip-gram-model | skip-gram | The named embedding model and its context-prediction objective are covered together. |
| bayes-rule | bayes-theorem | Two names for the same conditional-probability relationship. |
| likelihood-function | likelihood | Both name likelihood as a function of parameters. |
| attention-mechanism | attention | Both describe query-dependent weighting of values. |
| convolutional-kernel | kernel | The retained `kernel` is explicitly the convolution sense. |
| roofline-model | roofline | Equivalent names for the performance model. |
| data-parallel | data-parallelism | Adjectival and noun forms of the same computation pattern. |
| gradient-checkpointing | activation-checkpointing | Names for saving selected activations and recomputing others for backpropagation. |
| forward-propagation | forward-pass | Both describe evaluating a model from inputs to outputs. |
| bleu-score | bleu | Both denote the same translation metric. |
| text-corpus | corpus | Both denote a collection of texts. |
| image-synthesis | image-generation | Both describe producing images using an algorithm or model. |
| feasible-set | feasible-region | Both denote points satisfying the constraints. |
| denoising-diffusion-implicit-model | ddim | The long name and acronym identify the same sampling framework. |
| loss-surface | loss-landscape | Both describe loss as a function of model parameters. |
| hyperparameter-tuning | hyperparameter-optimization | Both describe choosing hyperparameter settings using performance evidence. |

Several related records remain separate deliberately. `information-content` is broader than the specifically defined negative-log-probability `surprisal`; `weight-tying` names sharing between model components, a particular use of broader `weight-sharing`; `posterior-predictive` describes a prediction, while `posterior-predictive-distribution` names the full predictive distribution. `training-dynamics` is limited to training while `learning-dynamics` can describe learning systems more broadly. `maximum-likelihood` names a principle and `maximum-likelihood-estimation` its estimation procedure. `covariance-function` specifies covariance, while `kernel-function` also covers more general similarity kernels. These distinctions should be reconsidered only with evidence that the intended translation sense is identical.

The same English word `bias` remains separate for a model offset and statistical estimator bias; `critic` remains separate for reinforcement-learning evaluation and generative adversarial scoring. All 11 language shards were subsequently reconciled using the redirect map, preserving pre-merge variants in `candidates/merged-variant-terms.jsonl`. The exports were regenerated from the reconciled canonical records.
