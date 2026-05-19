# Slide Quality Audit

This report audits authored `<!-- slides -->` blocks for likely teaching and layout issues.
It does not treat framework-specific missing code variants as failures;
those warnings require semantic review against the source code because
different frameworks often teach the same point with different cells.

## Summary

| Metric | Count |
|--------|-------|
| Markdown files with slides | 140 |
| Authored slides | 1264 |
| Markdown files without slides | 51 |
| Untitled or weak first slides | 0 |
| Likely code-wall slides | 60 |
| Sparse placeholder slides | 304 |
| Sparse output-only slides | 17 |
| Likely overflow slides | 0 |
| Math-heavy decks with math-light slides | 33 |

## Likely Overflow

| Slide | Title | Words | Placeholders | Outputs | Code Lines | Max Cell | IDs |
|-------|-------|-------|--------------|---------|------------|----------|-----|

## Code Walls

| Slide | Title | Words | Placeholders | Outputs | Code Lines | Max Cell | IDs |
|-------|-------|-------|--------------|---------|------------|----------|-----|
| `chapter_computer-vision/anchor.md:1570` | Non-maximum suppression (NMS) | 44 | 2 | 0 | 64 | 40 | anchor-predicting-bounding-boxes-with-non-maximum-suppression-2, anchor-predicting-bounding-boxes-with-non-maximum-suppression-3 |
| `chapter_computational-performance/multiple-gpus.md:892` | One step of multi-GPU training | 10 | 2 | 0 | 68 | 38 | multiple-gpus-training-1, multiple-gpus-training-2 |
| `chapter_natural-language-processing-applications/natural-language-inference-bert.md:1059` | Encoding sentence pairs | 21 | 2 | 0 | 63 | 47 | natural-language-inference-bert-the-dataset-for-fine-tuning-bert-1, natural-language-inference-bert-the-dataset-for-fine-tuning-bert-2 |
| `chapter_generative-adversarial-networks/gan.md:689` | Training loop | 31 | 1 | 0 | 59 | 59 | gan-training-3 |
| `chapter_computer-vision/ssd.md:1825` | Training | 33 | 1 | 0 | 58 | 58 | ssd-training-the-model |
| `chapter_computer-vision/neural-style.md:1058` | Optimization loop | 20 | 1 | 0 | 60 | 60 | neural-style-training-1 |
| `chapter_appendix-mathematics-for-deep-learning/multivariable-calculus.md:980` | Chain rule and backprop | 13 | 3 | 0 | 60 | 24 | multivariable-calculus-the-backpropagation-algorithm-1, multivariable-calculus-the-backpropagation-algorithm-2, multivariable-calculus-the-backpropagation-algorithm-3 |
| `chapter_recommender-systems/seqrec.md:469` | Sequential dataset | 16 | 1 | 0 | 58 | 58 | seqrec-sequential-dataset-with-negative-sampling |
| `chapter_attention-mechanisms-and-transformers/bahdanau-attention.md:674` | Seq2SeqAttentionDecoder | 62 | 1 | 0 | 50 | 50 | bahdanau-attention-defining-the-decoder-with-attention-2 |
| `chapter_linear-regression/linear-regression-scratch.md:789` | The whole epoch | 20 | 2 | 0 | 54 | 50 | linear-regression-scratch-training-2, linear-regression-scratch-training-3 |
| `chapter_computer-vision/fcn.md:924` | Pretrained backbone | 24 | 1 | 0 | 52 | 52 | fcn-the-model-1 |
| `chapter_linear-regression/oo-design.md:769` | `Module`: models | 22 | 1 | 0 | 52 | 52 | oo-design-models |
| `chapter_linear-regression/oo-design.md:784` | `Trainer`: the loop | 20 | 1 | 0 | 52 | 52 | oo-design-training |
| `chapter_natural-language-processing-applications/natural-language-inference-attention.md:872` | Training | 19 | 2 | 0 | 52 | 45 | natural-language-inference-attention-training-and-evaluating-the-model-2-1, natural-language-inference-attention-training-and-evaluating-the-model-2-2 |
| `chapter_natural-language-processing-pretraining/bert-pretraining.md:637` | Training loop | 36 | 2 | 0 | 49 | 46 | bert-pretraining-pretraining-bert-2-3, bert-pretraining-pretraining-bert-2-4 |
| `chapter_attention-mechanisms-and-transformers/transformer.md:1640` | Decoder block | 32 | 1 | 0 | 49 | 49 | transformer-decoder-1 |
| `chapter_natural-language-processing-pretraining/word-embedding-dataset.md:726` | Reusable loader | 17 | 3 | 0 | 51 | 27 | word-embedding-dataset-putting-it-all-together-1, word-embedding-dataset-putting-it-all-together-2, word-embedding-dataset-putting-it-all-together-3 |
| `chapter_recommender-systems/autorec.md:267` | Training | 44 | 1 | 0 | 46 | 46 | autorec-training-and-evaluating-the-model |
| `chapter_optimization/minibatch-sgd.md:954` | Concise: framework optimizer | 13 | 2 | 0 | 51 | 47 | minibatch-sgd-concise-implementation-1, minibatch-sgd-concise-implementation-2 |
| `chapter_natural-language-processing-pretraining/word2vec-pretraining.md:716` | Training loop | 27 | 2 | 0 | 48 | 45 | word2vec-pretraining-defining-the-training-loop-1, word2vec-pretraining-defining-the-training-loop-2 |
| `chapter_natural-language-processing-pretraining/bert-dataset.md:690` | Generating Masked LM labels | 29 | 2 | 0 | 47 | 28 | bert-dataset-generating-the-masked-language-modeling-task-1, bert-dataset-generating-the-masked-language-modeling-task-2 |
| `chapter_natural-language-processing-applications/natural-language-inference-bert.md:1083` | Fine-tuning | 45 | 1 | 0 | 44 | 44 | natural-language-inference-bert-fine-tuning-bert-3 |
| `chapter_natural-language-processing-applications/sentiment-analysis-cnn.md:750` | Training | 25 | 3 | 0 | 47 | 37 | sentiment-analysis-cnn-training-and-evaluating-the-model-1, sentiment-analysis-cnn-training-and-evaluating-the-model-2, sentiment-analysis-cnn-training-and-evaluating-the-model-3 |
| `chapter_natural-language-processing-applications/sentiment-analysis-rnn.md:568` | Training | 24 | 2 | 0 | 46 | 38 | sentiment-analysis-rnn-training-and-evaluating-the-model-1, sentiment-analysis-rnn-training-and-evaluating-the-model-2 |
| `chapter_recommender-systems/seqrec.md:462` | Implementation | 10 | 1 | 0 | 48 | 48 | seqrec-model-implementation |
| `chapter_computer-vision/kaggle-cifar10.md:1096` | Organizing the dataset | 21 | 3 | 0 | 45 | 27 | kaggle-cifar10-organizing-the-dataset-1, kaggle-cifar10-organizing-the-dataset-2, kaggle-cifar10-organizing-the-dataset-3 |
| `chapter_recommender-systems/neumf.md:507` | Hit@50 and AUC | 42 | 2 | 0 | 41 | 31 | neumf-evaluator-1, neumf-evaluator-2 |
| `chapter_natural-language-processing-pretraining/similarity-analogy.md:385` | Loading GloVe | 16 | 3 | 0 | 45 | 32 | similarity-analogy-loading-pretrained-word-vectors-1, similarity-analogy-loading-pretrained-word-vectors-2, similarity-analogy-loading-pretrained-word-vectors-3 |
| `chapter_attention-mechanisms-and-transformers/transformer.md:1661` | Decoder stack | 28 | 1 | 0 | 43 | 43 | transformer-decoder-3 |
| `chapter_optimization/minibatch-sgd.md:911` | Generic training loop | 20 | 1 | 0 | 43 | 43 | minibatch-sgd-implementation-from-scratch-2 |
| `chapter_computer-vision/kaggle-cifar10.md:1158` | Framework model contract | 23 | 1 | 0 | 42 | 42 | kaggle-cifar10-defining-the-model-3 |
| `chapter_reinforcement-learning/qlearning.md:247` | Q-learning training loop | 28 | 2 | 0 | 41 | 36 | qlearning-implementation-of-q-learning-2, qlearning-implementation-of-q-learning-3 |
| `chapter_computer-vision/kaggle-dog.md:1044` | Head loss and validation | 26 | 1 | 0 | 41 | 41 | kaggle-dog-fine-tuning-a-pretrained-model-2 |
| `chapter_natural-language-processing-applications/natural-language-inference-attention.md:825` | Step 1: Attend | 18 | 2 | 0 | 42 | 25 | natural-language-inference-attention-attending-1, natural-language-inference-attention-attending-2 |
| `chapter_convolutional-modern/batch-norm.md:1026` | Wrapping as a `Module` | 12 | 1 | 0 | 43 | 43 | batch-norm-implementation-from-scratch-2 |
| `chapter_recommender-systems/ctr.md:201` | Dataset wrapper | 16 | 2 | 0 | 42 | 39 | ctr-dataset-wrapper-1, ctr-dataset-wrapper-2 |
| `chapter_convolutional-modern/densenet.md:611` | The DenseNet model | 13 | 2 | 0 | 42 | 24 | densenet-densenet-model-1, densenet-densenet-model-2 |
| `chapter_computer-vision/anchor.md:1459` | Generating anchors | 46 | 1 | 0 | 35 | 35 | anchor-generating-multiple-anchor-boxes-1 |
| `chapter_computational-performance/multiple-gpus.md:849` | Toy network | 15 | 1 | 0 | 40 | 40 | multiple-gpus-a-toy-network |
| `chapter_computer-vision/semantic-segmentation-and-dataset.md:891` | Reusable loader factory | 0 | 1 | 0 | 42 | 42 | semantic-segmentation-and-dataset-putting-it-all-together |
| `chapter_attention-mechanisms-and-transformers/multihead-attention.md:504` | Per-head dimension trick | 47 | 1 | 0 | 34 | 34 | multihead-attention-implementation-1 |
| `chapter_attention-mechanisms-and-transformers/transformer.md:1628` | Encoder stack | 26 | 2 | 0 | 37 | 31 | transformer-encoder-3, transformer-encoder-4 |
| `chapter_recommender-systems/neumf.md:525` | Training helper | 27 | 1 | 0 | 36 | 36 | neumf-training-and-evaluating-the-model-1 |
| `chapter_recommender-systems/mf.md:340` | Training | 18 | 1 | 0 | 37 | 37 | mf-training-and-evaluating-the-model-1 |
| `chapter_computer-vision/anchor.md:1529` | Labeling implementation | 0 | 1 | 0 | 40 | 40 | anchor-labeling-classes-and-offsets-2 |
| `chapter_attention-mechanisms-and-transformers/vision-transformer.md:629` | Putting it together | 24 | 1 | 0 | 36 | 36 | vision-transformer-putting-it-all-together |
| `chapter_convolutional-modern/batch-norm.md:1017` | From scratch | 17 | 2 | 0 | 37 | 30 | batch-norm-batch-normalization, batch-norm-implementation-from-scratch-1 |
| `chapter_computer-vision/rcnn.md:468` | RoI pooling output | 28 | 1 | 0 | 35 | 35 | rcnn-fast-r-cnn-3 |
| `chapter_convolutional-modern/resnet.md:887` | ResNeXt: width via cardinality | 21 | 1 | 0 | 34 | 34 | resnet-resnext-1 |
| `chapter_recurrent-modern/seq2seq.md:1245` | Greedy prediction | 43 | 1 | 0 | 30 | 30 | seq2seq-prediction |
| `chapter_recurrent-modern/seq2seq.md:1200` | Decoder: context-conditioned RNN | 30 | 1 | 0 | 32 | 32 | seq2seq-decoder-1 |
| `chapter_natural-language-processing-applications/sentiment-analysis-cnn.md:726` | textCNN model | 15 | 1 | 0 | 34 | 34 | sentiment-analysis-cnn-defining-the-model-1 |
| `chapter_natural-language-processing-pretraining/bert-dataset.md:713` | Custom Dataset class | 13 | 1 | 0 | 34 | 34 | bert-dataset-transforming-text-into-the-pretraining-dataset-2 |
| `chapter_recurrent-modern/seq2seq.md:1216` | Putting it together | 24 | 1 | 0 | 32 | 32 | seq2seq-encoder-decoder-for-sequence-to-sequence-learning |
| `chapter_natural-language-processing-pretraining/bert.md:1058` | Putting it together | 23 | 1 | 0 | 32 | 32 | bert-putting-it-all-together |
| `chapter_natural-language-processing-applications/sentiment-analysis-rnn.md:532` | BiRNN classifier | 26 | 1 | 0 | 31 | 31 | sentiment-analysis-rnn-representing-single-text-with-rnns-1 |
| `chapter_recurrent-neural-networks/text-sequence.md:395` | Vocabulary | 19 | 1 | 0 | 30 | 30 | text-sequence-vocabulary-1 |
| `chapter_natural-language-processing-pretraining/bert.md:997` | BERTEncoder | 19 | 1 | 0 | 30 | 30 | bert-input-representation-2 |
| `chapter_computer-vision/fcn.md:990` | Training | 19 | 1 | 0 | 30 | 30 | fcn-training |
| `chapter_optimization/rmsprop.md:264` | From-scratch RMSProp | 16 | 4 | 0 | 27 | 11 | rmsprop-implementation-from-scratch-1, rmsprop-implementation-from-scratch-2, rmsprop-implementation-from-scratch-3, rmsprop-implementation-from-scratch-4 |

## Sparse Output Slides

| Slide | Title | Words | Placeholders | Outputs | Code Lines | Max Cell | IDs |
|-------|-------|-------|--------------|---------|------------|----------|-----|
| `chapter_recurrent-modern/machine-translation-and-dataset.md:405` | Length distribution | 16 | 2 | 1 | 11 | 11 | machine-translation-and-dataset-tokenization-3, machine-translation-and-dataset-tokenization-4 |
| `chapter_appendix-mathematics-for-deep-learning/integral-calculus.md:649` | Definite integral | 24 | 1 | 1 | 0 | 0 | integral-calculus-geometric-interpretation-2 |
| `chapter_optimization/lr-scheduler.md:713` | Constant-LR baselines | 23 | 1 | 1 | 0 | 0 | lr-scheduler-schedulers-1 |
| `chapter_optimization/lr-scheduler.md:754` | Multi-step training | 22 | 1 | 1 | 0 | 0 | lr-scheduler-multi-factor-scheduler-2 |
| `chapter_optimization/lr-scheduler.md:722` | Constant-LR baselines (cont.) | 22 | 1 | 1 | 0 | 0 | lr-scheduler-schedulers-3 |
| `chapter_natural-language-processing-applications/natural-language-inference-bert.md:1051` | Instantiate pretrained BERT | 22 | 1 | 1 | 0 | 0 | natural-language-inference-bert-loading-pretrained-bert-3 |
| `chapter_computer-vision/image-augmentation.md:1188` | Train it | 21 | 1 | 1 | 0 | 0 | image-augmentation-multi-gpu-training-4 |
| `chapter_appendix-mathematics-for-deep-learning/distributions.md:1100` | Poisson | 21 | 1 | 1 | 0 | 0 | distributions-poisson-1 |
| `chapter_optimization/lr-scheduler.md:732` | Square-root schedule training | 19 | 1 | 1 | 0 | 0 | lr-scheduler-schedulers-4 |
| `chapter_optimization/lr-scheduler.md:771` | Cosine training | 18 | 1 | 1 | 0 | 0 | lr-scheduler-cosine-scheduler-2 |
| `chapter_appendix-mathematics-for-deep-learning/distributions.md:1108` | Poisson CDF | 18 | 1 | 1 | 0 | 0 | distributions-poisson-2 |
| `chapter_optimization/lr-scheduler.md:747` | Multi-step decay | 16 | 1 | 1 | 0 | 0 | lr-scheduler-multi-factor-scheduler-1 |
| `chapter_optimization/lr-scheduler.md:740` | Polynomial / factor decay | 16 | 1 | 1 | 0 | 0 | lr-scheduler-factor-scheduler |
| `chapter_appendix-mathematics-for-deep-learning/distributions.md:1117` | Poisson samples | 16 | 1 | 1 | 0 | 0 | distributions-poisson-3 |
| `chapter_appendix-mathematics-for-deep-learning/integral-calculus.md:660` | Riemann approximation | 13 | 1 | 1 | 0 | 0 | integral-calculus-geometric-interpretation-3 |
| `chapter_appendix-mathematics-for-deep-learning/integral-calculus.md:643` | Geometric interpretation | 8 | 1 | 1 | 0 | 0 | integral-calculus-geometric-interpretation-1 |
| `chapter_recurrent-neural-networks/text-sequence.md:440` | Word-frequency plot | 0 | 1 | 1 | 0 | 0 | text-sequence-exploratory-language-statistics-2 |

## Sparse Placeholder Slides

| Slide | Title | Words | Placeholders | Outputs | Code Lines | Max Cell | IDs |
|-------|-------|-------|--------------|---------|------------|----------|-----|
| `chapter_computational-performance/multiple-gpus.md:892` | One step of multi-GPU training | 10 | 2 | 0 | 68 | 38 | multiple-gpus-training-1, multiple-gpus-training-2 |
| `chapter_appendix-mathematics-for-deep-learning/multivariable-calculus.md:980` | Chain rule and backprop | 13 | 3 | 0 | 60 | 24 | multivariable-calculus-the-backpropagation-algorithm-1, multivariable-calculus-the-backpropagation-algorithm-2, multivariable-calculus-the-backpropagation-algorithm-3 |
| `chapter_recommender-systems/seqrec.md:469` | Sequential dataset | 16 | 1 | 0 | 58 | 58 | seqrec-sequential-dataset-with-negative-sampling |
| `chapter_natural-language-processing-pretraining/word-embedding-dataset.md:726` | Reusable loader | 17 | 3 | 0 | 51 | 27 | word-embedding-dataset-putting-it-all-together-1, word-embedding-dataset-putting-it-all-together-2, word-embedding-dataset-putting-it-all-together-3 |
| `chapter_optimization/minibatch-sgd.md:954` | Concise: framework optimizer | 13 | 2 | 0 | 51 | 47 | minibatch-sgd-concise-implementation-1, minibatch-sgd-concise-implementation-2 |
| `chapter_recommender-systems/seqrec.md:462` | Implementation | 10 | 1 | 0 | 48 | 48 | seqrec-model-implementation |
| `chapter_natural-language-processing-pretraining/similarity-analogy.md:385` | Loading GloVe | 16 | 3 | 0 | 45 | 32 | similarity-analogy-loading-pretrained-word-vectors-1, similarity-analogy-loading-pretrained-word-vectors-2, similarity-analogy-loading-pretrained-word-vectors-3 |
| `chapter_convolutional-modern/batch-norm.md:1026` | Wrapping as a `Module` | 12 | 1 | 0 | 43 | 43 | batch-norm-implementation-from-scratch-2 |
| `chapter_recommender-systems/ctr.md:201` | Dataset wrapper | 16 | 2 | 0 | 42 | 39 | ctr-dataset-wrapper-1, ctr-dataset-wrapper-2 |
| `chapter_convolutional-modern/densenet.md:611` | The DenseNet model | 13 | 2 | 0 | 42 | 24 | densenet-densenet-model-1, densenet-densenet-model-2 |
| `chapter_computational-performance/multiple-gpus.md:849` | Toy network | 15 | 1 | 0 | 40 | 40 | multiple-gpus-a-toy-network |
| `chapter_computer-vision/semantic-segmentation-and-dataset.md:891` | Reusable loader factory | 0 | 1 | 0 | 42 | 42 | semantic-segmentation-and-dataset-putting-it-all-together |
| `chapter_computer-vision/anchor.md:1529` | Labeling implementation | 0 | 1 | 0 | 40 | 40 | anchor-labeling-classes-and-offsets-2 |
| `chapter_convolutional-modern/batch-norm.md:1017` | From scratch | 17 | 2 | 0 | 37 | 30 | batch-norm-batch-normalization, batch-norm-implementation-from-scratch-1 |
| `chapter_recommender-systems/movielens.md:347` | DataLoader | 0 | 2 | 0 | 39 | 23 | movielens-loading-the-data-1, movielens-loading-the-data-2 |
| `chapter_computer-vision/ssd.md:1745` | Five-block pyramid | 16 | 2 | 0 | 36 | 25 | ssd-the-complete-model-1, ssd-the-complete-model-2 |
| `chapter_hyperparameter-optimization/sh-intro.md:360` | Implementation (cont.) | 0 | 2 | 0 | 37 | 29 | sh-intro-successive-halving-3, sh-intro-successive-halving-4 |
| `chapter_natural-language-processing-applications/sentiment-analysis-cnn.md:726` | textCNN model | 15 | 1 | 0 | 34 | 34 | sentiment-analysis-cnn-defining-the-model-1 |
| `chapter_natural-language-processing-pretraining/bert-dataset.md:713` | Custom Dataset class | 13 | 1 | 0 | 34 | 34 | bert-dataset-transforming-text-into-the-pretraining-dataset-2 |
| `chapter_computer-vision/neural-style.md:1002` | Feature extractor (cont.) | 0 | 2 | 0 | 35 | 23 | neural-style-extracting-features-4, neural-style-extracting-features-5 |
| `chapter_recurrent-neural-networks/rnn-concise.md:320` | The model | 16 | 3 | 0 | 32 | 15 | rnn-concise-concise-implementation-of-recurrent-neural-networks, rnn-concise-defining-the-model-1, rnn-concise-defining-the-model-2 |
| `chapter_convolutional-modern/resnet.md:821` | Block in code | 16 | 2 | 0 | 32 | 27 | resnet-residual-networks-resnet-and-resnext, resnet-residual-blocks-1 |
| `chapter_optimization/minibatch-sgd.md:856` | Setup | 0 | 2 | 0 | 34 | 22 | minibatch-sgd-vectorization-and-caches-1, minibatch-sgd-vectorization-and-caches-2 |
| `chapter_appendix-mathematics-for-deep-learning/random-variables.md:1066` | Covariance and correlation | 16 | 2 | 0 | 31 | 16 | random-variables-covariance, random-variables-correlation |
| `chapter_convolutional-modern/googlenet.md:638` | The four branches | 14 | 2 | 0 | 31 | 26 | googlenet-multi-branch-networks-googlenet, googlenet-inception-blocks |
| `chapter_convolutional-modern/alexnet.md:547` | The architecture in code | 12 | 2 | 0 | 31 | 26 | alexnet-deep-convolutional-neural-networks-alexnet, alexnet-capacity-control-and-preprocessing-1 |
| `chapter_computer-vision/semantic-segmentation-and-dataset.md:871` | Custom Dataset class | 15 | 1 | 0 | 29 | 29 | semantic-segmentation-and-dataset-custom-semantic-segmentation-dataset-class |
| `chapter_computer-vision/kaggle-cifar10.md:1131` | Data loaders | 5 | 2 | 0 | 30 | 16 | kaggle-cifar10-reading-the-dataset-1, kaggle-cifar10-reading-the-dataset-2 |
| `chapter_hyperparameter-optimization/sh-intro.md:352` | Implementation | 0 | 2 | 0 | 30 | 17 | sh-intro-successive-halving-1, sh-intro-successive-halving-2 |
| `chapter_optimization/rmsprop.md:264` | From-scratch RMSProp | 16 | 4 | 0 | 27 | 11 | rmsprop-implementation-from-scratch-1, rmsprop-implementation-from-scratch-2, rmsprop-implementation-from-scratch-3, rmsprop-implementation-from-scratch-4 |
| `chapter_computer-vision/anchor.md:1481` | Anchors at one pixel | 14 | 2 | 0 | 27 | 20 | anchor-generating-multiple-anchor-boxes-4, anchor-generating-multiple-anchor-boxes-5 |
| `chapter_natural-language-processing-applications/natural-language-inference-and-dataset.md:478` | Reading SNLI | 13 | 3 | 0 | 27 | 19 | natural-language-inference-and-dataset-reading-the-dataset-1, natural-language-inference-and-dataset-reading-the-dataset-2, natural-language-inference-and-dataset-reading-the-dataset-3 |
| `chapter_computer-vision/kaggle-dog.md:1028` | Data loaders | 0 | 2 | 0 | 29 | 15 | kaggle-dog-reading-the-dataset-1, kaggle-dog-reading-the-dataset-2 |
| `chapter_natural-language-processing-applications/natural-language-inference-bert.md:1071` | Classifier head | 15 | 2 | 0 | 26 | 14 | natural-language-inference-bert-fine-tuning-bert-1, natural-language-inference-bert-fine-tuning-bert-2 |
| `chapter_recommender-systems/deepfm.md:224` | Training | 11 | 1 | 0 | 26 | 26 | deepfm-training-and-evaluating-the-model |
| `chapter_natural-language-processing-pretraining/bert-dataset.md:720` | Loader factory | 8 | 1 | 0 | 25 | 25 | bert-dataset-transforming-text-into-the-pretraining-dataset-3 |
| `chapter_natural-language-processing-applications/natural-language-inference-and-dataset.md:493` | Custom Dataset | 8 | 1 | 0 | 25 | 25 | natural-language-inference-and-dataset-defining-a-class-for-loading-the-dataset |
| `chapter_recommender-systems/mf.md:322` | The model | 6 | 2 | 0 | 25 | 19 | mf-the-matrix-factorization-model, mf-model-implementation |
| `chapter_recommender-systems/deepfm.md:220` | Implementation | 0 | 1 | 0 | 26 | 26 | deepfm-implementation-of-deepfm |
| `chapter_natural-language-processing-applications/sentiment-analysis-and-dataset.md:367` | Data loaders | 0 | 2 | 0 | 26 | 20 | sentiment-analysis-and-dataset-creating-data-iterators, sentiment-analysis-and-dataset-putting-it-all-together |
| `chapter_multilayer-perceptrons/kaggle-house-price.md:594` | The transforms in code | 6 | 2 | 0 | 25 | 23 | kaggle-house-price-data-preprocessing-2, kaggle-house-price-data-preprocessing-3 |
| `chapter_natural-language-processing-applications/natural-language-inference-and-dataset.md:499` | Loader factory | 0 | 3 | 0 | 25 | 17 | natural-language-inference-and-dataset-putting-it-all-together-1, natural-language-inference-and-dataset-putting-it-all-together-2, natural-language-inference-and-dataset-putting-it-all-together-3 |
| `chapter_convolutional-modern/batch-norm.md:1033` | LeNet + BatchNorm | 11 | 1 | 0 | 23 | 23 | batch-norm-lenet-with-batch-normalization-1 |
| `chapter_attention-mechanisms-and-transformers/self-attention-and-positional-encoding.md:595` | PositionalEncoding class | 17 | 1 | 0 | 22 | 22 | self-attention-and-positional-encoding-positional-encoding-1 |
| `chapter_computer-vision/object-detection-dataset.md:456` | Reading the dataset | 16 | 1 | 0 | 22 | 22 | object-detection-dataset-reading-the-dataset-1 |
| `chapter_optimization/adam.md:329` | From-scratch Adam | 14 | 1 | 0 | 22 | 22 | adam-implementation-1 |
| `chapter_computer-vision/kaggle-cifar10.md:1120` | Augmentation pipelines | 12 | 2 | 0 | 22 | 17 | kaggle-cifar10-image-augmentation-1, kaggle-cifar10-image-augmentation-2 |
| `chapter_appendix-mathematics-for-deep-learning/distributions.md:1085` | Binomial | 11 | 3 | 0 | 22 | 13 | distributions-binomial-1, distributions-binomial-2, distributions-binomial-3 |
| `chapter_natural-language-processing-applications/sentiment-analysis-and-dataset.md:341` | Reading IMDb | 14 | 2 | 0 | 21 | 17 | sentiment-analysis-and-dataset-reading-the-dataset-1, sentiment-analysis-and-dataset-reading-the-dataset-2 |
| `chapter_hyperparameter-optimization/hyperopt-api.md:442` | Tuner | 8 | 1 | 0 | 22 | 22 | hyperopt-api-tuner |
| `chapter_multilayer-perceptrons/kaggle-house-price.md:636` | K-fold in code | 0 | 2 | 0 | 23 | 14 | kaggle-house-price-k-fold-cross-validation-1, kaggle-house-price-k-fold-cross-validation-2 |
| `chapter_linear-classification/image-classification-dataset.md:340` | Dataset setup | 6 | 2 | 0 | 22 | 12 | image-classification-dataset-the-image-classification-dataset, image-classification-dataset-loading-the-dataset-1 |
| `chapter_convolutional-neural-networks/lenet.md:465` | LeNet initialization | 0 | 1 | 0 | 23 | 23 | lenet-2 |
| `chapter_linear-regression/synthetic-regression-data.md:412` | Synthetic data module | 11 | 2 | 0 | 21 | 12 | synthetic-regression-data, synthetic-regression-data-generating-the-dataset-1 |
| `chapter_convolutional-neural-networks/padding-and-strides.md:450` | Shape-preserving padding | 17 | 2 | 0 | 20 | 15 | padding-and-strides-padding-1, padding-and-strides-padding-2 |
| `chapter_convolutional-modern/googlenet.md:675` | Head and assembly | 17 | 2 | 0 | 20 | 11 | googlenet-googlenet-model-5, googlenet-googlenet-model-6 |
| `chapter_convolutional-modern/densenet.md:576` | Conv block | 16 | 2 | 0 | 20 | 15 | densenet-densely-connected-networks-densenet, densenet-dense-blocks-1 |
| `chapter_computer-vision/image-augmentation.md:1138` | Combined color jitter | 15 | 1 | 0 | 20 | 20 | image-augmentation-changing-colors-3 |
| `chapter_appendix-mathematics-for-deep-learning/multivariable-calculus.md:995` | Hessians | 13 | 1 | 0 | 20 | 20 | multivariable-calculus-hessians |
| `chapter_recurrent-neural-networks/text-sequence.md:376` | Read | 0 | 3 | 0 | 22 | 10 | text-sequence-converting-raw-text-into-sequence-data, text-sequence-reading-the-dataset-1, text-sequence-reading-the-dataset-2 |
| `chapter_multilayer-perceptrons/dropout.md:632` | MLP with dropout | 0 | 1 | 0 | 22 | 22 | dropout-defining-the-model |
| `chapter_recurrent-modern/gru.md:597` | Concise: nn.GRU | 11 | 3 | 0 | 20 | 13 | gru-concise-implementation-1, gru-concise-implementation-2, gru-concise-implementation-3 |
| `chapter_natural-language-processing-pretraining/word2vec-pretraining.md:693` | Masked binary cross-entropy | 17 | 3 | 0 | 19 | 10 | word2vec-pretraining-binary-cross-entropy-loss-1, word2vec-pretraining-binary-cross-entropy-loss-2, word2vec-pretraining-binary-cross-entropy-loss-3 |
| `chapter_computer-vision/fcn.md:945` | The class & upsampling head | 17 | 1 | 0 | 19 | 19 | fcn-the-model-4 |
| `chapter_recurrent-neural-networks/rnn-scratch.md:926` | The RNN cell | 16 | 2 | 0 | 19 | 12 | rnn-scratch-recurrent-neural-network-implementation-from-scratch, rnn-scratch-rnn-model-1 |
| `chapter_recurrent-modern/machine-translation-and-dataset.md:405` | Length distribution | 16 | 2 | 1 | 11 | 11 | machine-translation-and-dataset-tokenization-3, machine-translation-and-dataset-tokenization-4 |
| `chapter_recommender-systems/movielens.md:317` | Downloading | 0 | 2 | 0 | 21 | 14 | movielens-getting-the-data-1, movielens-getting-the-data-2 |
| `chapter_builders-guide/use-gpu.md:767` | What hardware do we have? | 0 | 3 | 0 | 21 | 8 | use-gpu-gpus, use-gpu-computing-devices-1, use-gpu-computing-devices-2 |
| `chapter_attention-mechanisms-and-transformers/attention-pooling.md:466` | Four kernels | 12 | 2 | 0 | 19 | 10 | attention-pooling-kernels-and-data-1, attention-pooling-kernels-and-data-2 |
| `chapter_computer-vision/fine-tuning.md:941` | From-scratch baseline | 17 | 1 | 0 | 18 | 18 | fine-tuning-fine-tuning-the-model-3 |
| `chapter_linear-classification/softmax-regression-scratch.md:505` | Implementing it | 16 | 2 | 0 | 18 | 12 | softmax-regression-scratch-the-cross-entropy-loss-2, softmax-regression-scratch-the-cross-entropy-loss-3 |
| `chapter_hyperparameter-optimization/hyperopt-api.md:422` | Searcher base class | 3 | 2 | 0 | 20 | 14 | hyperopt-api-searcher-1, hyperopt-api-searcher-2 |
| `chapter_computer-vision/ssd.md:1834` | Inference | 15 | 2 | 0 | 18 | 12 | ssd-prediction-1, ssd-prediction-2 |
| `chapter_computational-performance/multiple-gpus.md:882` | Distribute the minibatch | 6 | 2 | 0 | 19 | 10 | multiple-gpus-distributing-data-1, multiple-gpus-distributing-data-2 |
| `chapter_recurrent-modern/lstm.md:748` | From scratch: parameters | 16 | 1 | 0 | 17 | 17 | lstm-initializing-model-parameters-1 |
| `chapter_linear-classification/softmax-regression-concise.md:303` | The model | 10 | 2 | 0 | 18 | 11 | softmax-regression-concise-concise-implementation-of-softmax-regression, softmax-regression-concise-defining-the-model |
| `chapter_gaussian-processes/gp-inference.md:415` | Visualizing predictions | 16 | 1 | 0 | 17 | 17 | gp-inference-worked-example-from-scratch-4 |
| `chapter_computer-vision/neural-style.md:980` | Preprocessing | 10 | 1 | 0 | 18 | 18 | neural-style-preprocessing-and-postprocessing |
| `chapter_computational-performance/multiple-gpus-concise.md:603` | Multi-GPU initialization | 16 | 3 | 0 | 17 | 7 | multiple-gpus-concise-network-initialization-1, multiple-gpus-concise-network-initialization-2, multiple-gpus-concise-network-initialization-3 |
| `chapter_attention-mechanisms-and-transformers/transformer.md:1670` | Training: tiny config | 16 | 1 | 0 | 17 | 17 | transformer-training-1 |

## Weak First Slides

| Slide | Title | Words | Placeholders | Outputs | Code Lines | Max Cell | IDs |
|-------|-------|-------|--------------|---------|------------|----------|-----|

## Math-Light Decks

| File | Slides | Source Math Markers | Slide Math Markers |
|------|--------|---------------------|--------------------|
| `chapter_appendix-mathematics-for-deep-learning/random-variables.md` | 7 | 307 | 9 |
| `chapter_appendix-mathematics-for-deep-learning/information-theory.md` | 10 | 298 | 15 |
| `chapter_preliminaries/probability.md` | 6 | 217 | 10 |
| `chapter_appendix-mathematics-for-deep-learning/multivariable-calculus.md` | 6 | 209 | 7 |
| `chapter_optimization/convexity.md` | 7 | 196 | 19 |
| `chapter_appendix-mathematics-for-deep-learning/geometry-linear-algebraic-ops.md` | 9 | 164 | 4 |
| `chapter_preliminaries/linear-algebra.md` | 19 | 171 | 18 |
| `chapter_appendix-mathematics-for-deep-learning/statistics.md` | 6 | 123 | 6 |
| `chapter_appendix-mathematics-for-deep-learning/single-variable-calculus.md` | 7 | 126 | 10 |
| `chapter_appendix-mathematics-for-deep-learning/integral-calculus.md` | 6 | 118 | 9 |
| `chapter_gaussian-processes/gp-inference.md` | 14 | 110 | 10 |
| `chapter_computer-vision/anchor.md` | 17 | 111 | 12 |
| `chapter_appendix-mathematics-for-deep-learning/naive-bayes.md` | 9 | 83 | 4 |
| `chapter_convolutional-modern/resnet.md` | 13 | 80 | 3 |
| `chapter_recurrent-neural-networks/sequence.md` | 7 | 83 | 9 |
| `chapter_convolutional-modern/cnn-design.md` | 10 | 68 | 0 |
| `chapter_optimization/adagrad.md` | 6 | 76 | 9 |
| `chapter_recurrent-neural-networks/rnn.md` | 7 | 71 | 7 |
| `chapter_recurrent-modern/seq2seq.md` | 15 | 67 | 3 |
| `chapter_appendix-mathematics-for-deep-learning/maximum-likelihood.md` | 4 | 68 | 5 |
| `chapter_convolutional-modern/batch-norm.md` | 8 | 43 | 4 |
| `chapter_convolutional-modern/googlenet.md` | 10 | 36 | 0 |
| `chapter_computer-vision/transposed-conv.md` | 7 | 40 | 5 |
| `chapter_recommender-systems/neumf.md` | 9 | 36 | 3 |
| `chapter_natural-language-processing-applications/natural-language-inference-attention.md` | 12 | 32 | 3 |
| `chapter_recommender-systems/seqrec.md` | 7 | 29 | 2 |
| `chapter_computer-vision/rcnn.md` | 9 | 24 | 3 |
| `chapter_computer-vision/fcn.md` | 14 | 23 | 2 |
| `chapter_linear-classification/softmax-regression-concise.md` | 5 | 18 | 1 |
| `chapter_convolutional-modern/nin.md` | 6 | 15 | 0 |
| `chapter_convolutional-modern/alexnet.md` | 6 | 15 | 0 |
| `chapter_convolutional-neural-networks/pooling.md` | 10 | 15 | 1 |
| `chapter_recommender-systems/autorec.md` | 6 | 13 | 2 |

