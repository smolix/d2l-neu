# Review Findings

## A. Prose Typos and Grammar Errors

### Preface & Introduction

| File | ~Line | Issue | Suggested Fix |
|------|-------|-------|---------------|
| chapter_preface/index.qmd | ~386,403,423,441 | "Below lists dependencies" (×4) | "Below we list the dependencies" |
| chapter_preface/index.qmd | ~395 | "latest stable version TensorFlow" | "latest stable version of TensorFlow" |
| chapter_preface/index.qmd | ~410 | "JIT compliation" | "JIT compilation" |
| chapter_preface/index.qmd | ~621 | "range of  technologies" (double space) | "range of technologies" |
| chapter_introduction/index.qmd | ~108 | "work force" | "workforce" |
| chapter_introduction/index.qmd | ~123 | "smart phone" | "smartphone" |
| chapter_introduction/index.qmd | ~204 | "some of we authors" | "some of us authors" |
| chapter_introduction/index.qmd | ~833 | "they relied" (subject is PageRank) | "it relied" |
| chapter_introduction/index.qmd | ~934 | "10× return 10×" (duplicated) | "10× return" |

### Preliminaries

| File | ~Line | Issue | Suggested Fix |
|------|-------|-------|---------------|
| chapter_preliminaries/calculus.qmd | ~121 | "receiving operating characteristic" | "receiver operating characteristic" |
| chapter_preliminaries/autograd.qmd | ~893 | "focus on less menial." (incomplete) | "focus on less menial tasks." |
| chapter_preliminaries/probability.qmd | ~172 | "underly the data" | "underlie the data" |
| chapter_preliminaries/probability.qmd | ~488 | "rolling a single coin" | "tossing a single coin" |
| chapter_preliminaries/probability.qmd | ~537 | "coming up odds" | "coming up odd" |

### Linear Regression & Classification

| File | ~Line | Issue | Suggested Fix |
|------|-------|-------|---------------|
| chapter_linear-regression/linear-regression.qmd | ~228 | "double-edge sword" | "double-edged sword" |
| chapter_linear-regression/linear-regression.qmd | ~740 | "maximimum" | "maximum" |
| chapter_linear-regression/weight-decay.qmd | ~95 | spurious comma: "*weight decay*, operates" | "*weight decay* operates" |
| chapter_linear-classification/softmax-regression.qmd | ~418 | "not only it is boring" | "not only is it boring" |
| chapter_linear-classification/softmax-regression-concise.qmd | ~80 | "dectorator" | "decorator" |

### Multilayer Perceptrons

| File | ~Line | Issue | Suggested Fix |
|------|-------|-------|---------------|
| chapter_multilayer-perceptrons/dropout.qmd | ~7 | "peform well" | "perform well" |
| chapter_multilayer-perceptrons/dropout.qmd | ~64 | "such an justification" | "such a justification" |
| chapter_multilayer-perceptrons/mlp.qmd | ~117 | "body temperatures drops" | "body temperature drops" |
| chapter_multilayer-perceptrons/mlp-implementation.qmd | ~258 | "softmax regression implementation" (redundant "implementation") | "softmax regression" |
| chapter_multilayer-perceptrons/backprop.qmd | ~162 | "order of calculations are reversed" | "order of calculations is reversed" |
| chapter_multilayer-perceptrons/numerical-stability-and-init.qmd | ~22 | "heuristics that you will find useful" (redundant) | "heuristics that you will find valuable" |
| chapter_multilayer-perceptrons/numerical-stability-and-init.qmd | ~387 | "first author of its creators" | "first author among its creators" |
| chapter_multilayer-perceptrons/generalization-deep.qmd | ~64 | "far the bigger problem" | "by far the bigger problem" |
| chapter_multilayer-perceptrons/generalization-deep.qmd | ~222 | "established deep connection" | "established deep connections" |

### Builders' Guide & CNNs

| File | ~Line | Issue | Suggested Fix |
|------|-------|-------|---------------|
| chapter_builders-guide/use-gpu.qmd | ~30 | "we often refer it as" | "we often refer to it as" |
| chapter_builders-guide/use-gpu.qmd | ~233 | "NIVIDA's latest Ampere" | "NVIDIA's latest Ampere" |
| chapter_convolutional-neural-networks/index.qmd | ~23 | "on the Imagnet collection" | "on the ImageNet collection" |
| chapter_convolutional-neural-networks/pooling.qmd | ~94 | "information aggregation might be aggregated" | "information might be aggregated" |
| chapter_convolutional-modern/vgg.qmd | ~174 | "dimensonality" | "dimensionality" |
| chapter_convolutional-modern/cnn-design.qmd | ~378 | "corresonding CDFs" | "corresponding CDFs" |

### RNNs & Attention

| File | ~Line | Issue | Suggested Fix |
|------|-------|-------|---------------|
| chapter_recurrent-neural-networks/index.qmd | ~22 | "protypical tabular" | "prototypical tabular" |
| chapter_recurrent-neural-networks/sequence.qmd | ~38 | "words that likely to appear" | "words that are likely to appear" |
| chapter_recurrent-neural-networks/language-model.qmd | ~179 | "discuss about how" | "discuss how" |
| chapter_recurrent-neural-networks/language-model.qmd | ~198 | "comes handy here" | "comes in handy here" |
| chapter_recurrent-neural-networks/bptt.qmd | ~112 | "total derivate" | "total derivative" |
| chapter_recurrent-neural-networks/bptt.qmd | ~167 | "backpropgation" | "backpropagation" |
| chapter_recurrent-modern/lstm.qmd | ~739 | "Tranformers" | "Transformers" |
| chapter_recurrent-modern/bi-rnn.qmd | ~7 | "tasks contexts" | "task contexts" |
| chapter_recurrent-modern/bi-rnn.qmd | ~27 | "third sentences" | "third sentence" |
| chapter_recurrent-modern/machine-translation-and-dataset.qmd | ~14 | "two language's" | "two languages'" |
| chapter_attention-mechanisms-and-transformers/large-pretraining-transformers.qmd | ~264 | "GPT pretraining with a Transformer encoder" | "GPT pretraining with a Transformer decoder" (factual) |

### Optimization & Computation

| File | ~Line | Issue | Suggested Fix |
|------|-------|-------|---------------|
| chapter_computational-performance/hardware.qmd | ~29 | "4 0GB/s" (spurious space) | "40 GB/s" |
| chapter_computer-vision/semantic-segmentation-and-dataset.qmd | ~38 | "On of the most important" | "One of the most important" |

### NLP

| File | ~Line | Issue | Suggested Fix |
|------|-------|-------|---------------|
| chapter_natural-language-processing-pretraining/glove.qmd | ~206-209 | "co-occurence" (×4) | "co-occurrence" |
| chapter_natural-language-processing-pretraining/bert-dataset.qmd | ~393 | "to download and WikiText-2 dataset" | "to download the WikiText-2 dataset" |
| chapter_natural-language-processing-pretraining/bert-dataset.qmd | ~466 | "WikiText-2 dateset" | "WikiText-2 dataset" |
| chapter_natural-language-processing-pretraining/subword-embedding.qmd | ~98 | "its variants has been used" | "its variants have been used" |
| chapter_natural-language-processing-pretraining/bert.qmd | ~339 | "special tokens `<seq>`" | "special tokens `<sep>`" (factual) |
| chapter_natural-language-processing-applications/natural-language-inference-and-dataset.qmd | ~6 | "inferred form another" | "inferred from another" |
| chapter_natural-language-processing-applications/natural-language-inference-attention.qmd | ~211 | "For easy of demonstration" | "For ease of demonstration" |
| chapter_natural-language-processing-applications/natural-language-inference-attention.qmd | ~221 | "such as comparing step" | "such a comparing step" |
| chapter_natural-language-processing-applications/finetuning-bert.qmd | ~184 | "receives an query" | "receives a query" |

### Reinforcement Learning & GPs

| File | ~Line | Issue | Suggested Fix |
|------|-------|-------|---------------|
| chapter_reinforcement-learning/mdp.qmd | ~31 | "gridwold" | "gridworld" |
| chapter_reinforcement-learning/value-iter.qmd | ~52 | "next sate" | "next state" |
| chapter_reinforcement-learning/value-iter.qmd | ~56 | "developement" | "development" |
| chapter_reinforcement-learning/value-iter.qmd | ~92 | "enviroment" | "environment" |
| chapter_reinforcement-learning/qlearning.qmd | ~88 | "down" uses → arrow instead of ↓ | Fix arrow symbol |
| chapter_gaussian-processes/index.qmd | ~6 | "ubitiquous" | "ubiquitous" |
| chapter_gaussian-processes/index.qmd | ~8 | "GPs and and deep" | "GPs and deep" |
| chapter_gaussian-processes/index.qmd | ~10 | "tradiational" | "traditional" |
| chapter_gaussian-processes/gp-intro.qmd | ~7 | "seem to varying?" | "seem to vary?" |
| chapter_gaussian-processes/gp-intro.qmd | ~149 | "righly quite large" | "rightly quite large" |
| chapter_gaussian-processes/gp-inference.qmd | ~233 | "playing close attention" | "paying close attention" |
| chapter_gaussian-processes/gp-inference.qmd | ~369 | "marginal lkelihood" | "marginal likelihood" |
| chapter_gaussian-processes/gp-inference.qmd | ~371 | "the only we can" | "the only way we can" |

### HPO & GANs

| File | ~Line | Issue | Suggested Fix |
|------|-------|-------|---------------|
| chapter_hyperparameter-optimization/hyperopt-intro.qmd | ~178 | "we use the use the" | "we use the" |
| chapter_hyperparameter-optimization/hyperopt-api.qmd | ~239 | "We being by defining" | "We begin by defining" |
| chapter_hyperparameter-optimization/rs-async.qmd | ~17 | "layers of filters" | "layers or filters" |
| chapter_hyperparameter-optimization/rs-async.qmd | ~226 | "by distribution" | "by distributing" |
| chapter_hyperparameter-optimization/sh-async.qmd | ~30 | "ideling time" | "idling time" |
| chapter_hyperparameter-optimization/sh-async.qmd | ~32 | "implementaitons" | "implementations" |
| chapter_generative-adversarial-networks/gan.qmd | ~530 | "GANs composes of" | "GANs are composed of" |

### Recommender Systems

| File | ~Line | Issue | Suggested Fix |
|------|-------|-------|---------------|
| chapter_recommender-systems/recsys-intro.qmd | ~4 | "two-folds" | "twofold" |
| chapter_recommender-systems/movielens.qmd | ~214 | "It is public available" | "They are publicly available" |
| chapter_recommender-systems/ranking.qmd | ~5 | "approaches considers" | "approaches consider" |
| chapter_recommender-systems/neumf.qmd | ~338 | "check the its impact" | "check its impact" |
| chapter_recommender-systems/fm.qmd | ~32 | "complexity are decreased" | "complexity is decreased" |
| chapter_recommender-systems/deepfm.qmd | ~96 | "with the a pyramid" | "with a pyramid" |

### Appendix

| File | ~Line | Issue | Suggested Fix |
|------|-------|-------|---------------|
| chapter_appendix-tools-for-deep-learning/utils.qmd | ~123,169 | "enviroment" (×2) | "environment" |
| chapter_appendix-mathematics-for-deep-learning/information-theory.qmd | ~452 | "how does it related" | "how does it relate" |

---

## B. Cross-Reference Issues

**Note**: An earlier audit reported ~230 broken references, but most were false
positives from a grep pattern that didn't match uppercase letters in labels.
After correction, only **3 real issues** remain plus some Citeproc warnings.

### Only 3 truly broken cross-references

All are **range references** where two `@sec-` refs are joined by an en-dash,
which the preprocessor didn't handle:

1. `@sec-alexnet--@sec-googlenet` (2 files) — should be `@sec-alexnet--@sec-googlenet`
   with proper rendering as "Sections AlexNet–GoogLeNet"
   - chapter_computer-vision/semantic-segmentation-and-dataset.qmd:278
   - chapter_computer-vision/bounding-box.qmd:4

2. `@sec-bbox--@sec-object-detection-dataset` and `@sec-bbox--@sec-rcnn` (2 files)
   - chapter_computer-vision/ssd.qmd:3
   - chapter_computer-vision/semantic-segmentation-and-dataset.qmd:4

3. `@sec-resnet---batch normalization` — an em-dash after a reference got merged
   - chapter_convolutional-modern/batch-norm.qmd:8
   - Should be: `@sec-resnet---batch` → `@sec-resnet — batch`

**Root cause**: The preprocessor converts `:numref:` refs individually, but when
two refs are separated by `--` (en-dash) in the source, the result is
`@sec-foo--@sec-bar` where Quarto sees `@sec-foo--` as one malformed ref.
Fix: the preprocessor should detect `--` between refs and insert a space.

### Cross-references misinterpreted as Citeproc citations (rendering warnings)

These are labels without standard Quarto type prefixes (sec-/fig-/tbl-/eq-).
The preprocessor's `convert_label_id()` didn't recognize their prefix, so
they became bare `@something` which Citeproc treats as bibliography keys:

- `@oo-design-data`, `@oo-design-training`, `@oo-design-utilities`
- `@rnn-h-with-state`, `@rnn-h-without-state`
- `@gd-hot-taylor`, `@gd-taylor-2`, `@gru-tilde-H`
- `@img-conv-pad`, `@img-conv-reuse`, `@img-conv-stride`, `@img-lenet`, `@img-lenet-vert`
- `@img-samples-lc`, `@img-waldo`, `@field-visual`
- `@table-latency-numbers`, `@table-latency-numbers-tesla`
- `@eqref-sgd-xt-diff`, `@subsec-connection-to-mat-transposition`
- `@asha`, `@distributed-scheduling`, `@synchronous-sh`

**Root cause**: The original d2l labels used non-standard prefixes like
`img_`, `table_`, `oo-design-`, `rnn-h-`, etc. The preprocessor only
recognizes `sec_`, `fig_`, `tab_`, `eq_`, `chap_`, `subsec_` prefixes.

### Citation keys with trailing period

These generate Citeproc warnings because `@Key.` doesn't match `Key` in the bib:
- `@Csiszar.2008.`, `@Fechner.1860.`, `@Hochreiter.Schmidhuber.1997.`
- `@Krizhevsky.Sutskever.Hinton.2012.`, `@Polyak.1964.`
- And ~15 more

**Root cause**: The preprocessor's citation regex captures a trailing sentence
period as part of the citation key.

---

## C. Missing Images

None — all 296 referenced images exist in `img/`.

---

## D. Stray Conversion Artifacts in Code Blocks

These are `%%tab` and `tab.selected()` patterns inside code cells. They are **intentional** for the HTML multi-framework tabs and are handled by the downstream generators (gen_pdf.py, gen_notebooks.py). Do NOT remove them from the .qmd source files.

- `chapter_appendix-tools-for-deep-learning/utils.qmd`: contains `#@tab` markers in code
- `chapter_appendix-tools-for-deep-learning/d2l.qmd`: contains `:begin_tab:` (conversion missed)
- `chapter_convolutional-modern/resnet.qmd`: `# %%tab jax` in a code cell
- `chapter_linear-regression/linear-regression.qmd`: `%%tab pytorch, tensorflow, jax` + `tab.selected()` in code
- Multiple files in chapter_attention-mechanisms-and-transformers/: `tab.selected()` in code cells
