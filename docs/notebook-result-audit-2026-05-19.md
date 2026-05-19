# Notebook Result Audit

This report audits current `_notebooks` outputs and `.executed` stamps.

## Coverage

| Framework | Notebooks | Stamps | Current Stamps | Notebooks With Outputs |
|-----------|-----------|--------|----------------|------------------------|
| pytorch | 139 | 139 | 5 | 5 |
| tensorflow | 123 | 123 | 3 | 3 |
| jax | 123 | 123 | 2 | 2 |
| mxnet | 128 | 86 | 86 | 80 |

## Result

| Severity | Count | Meaning |
|----------|-------|---------|
| fail | 1 | current outputs contain explicit bad/error metrics |
| warn | 1 | current outputs look suspicious and need review |
| unverified | 417 | existing stamp cannot prove current notebook output quality |

## Issues

| Severity | Framework | Notebook | Metrics | Issue |
|----------|-----------|----------|---------|-------|
| unverified | pytorch | `chapter_appendix-mathematics-for-deep-learning/distributions.ipynb` |  | stale .executed stamp; notebook was regenerated after passing; stamp exists but current notebook has no executed cells |
| unverified | pytorch | `chapter_appendix-mathematics-for-deep-learning/eigendecomposition.ipynb` |  | stale .executed stamp; notebook was regenerated after passing; stamp exists but current notebook has no executed cells |
| unverified | pytorch | `chapter_appendix-mathematics-for-deep-learning/geometry-linear-algebraic-ops.ipynb` |  | stale .executed stamp; notebook was regenerated after passing; stamp exists but current notebook has no executed cells |
| unverified | pytorch | `chapter_appendix-mathematics-for-deep-learning/information-theory.ipynb` |  | stale .executed stamp; notebook was regenerated after passing; stamp exists but current notebook has no executed cells |
| unverified | pytorch | `chapter_appendix-mathematics-for-deep-learning/integral-calculus.ipynb` |  | stale .executed stamp; notebook was regenerated after passing; stamp exists but current notebook has no executed cells |
| unverified | pytorch | `chapter_appendix-mathematics-for-deep-learning/maximum-likelihood.ipynb` |  | stale .executed stamp; notebook was regenerated after passing; stamp exists but current notebook has no executed cells |
| unverified | pytorch | `chapter_appendix-mathematics-for-deep-learning/multivariable-calculus.ipynb` |  | stale .executed stamp; notebook was regenerated after passing; stamp exists but current notebook has no executed cells |
| unverified | pytorch | `chapter_appendix-mathematics-for-deep-learning/naive-bayes.ipynb` |  | stale .executed stamp; notebook was regenerated after passing; stamp exists but current notebook has no executed cells |
| unverified | pytorch | `chapter_appendix-mathematics-for-deep-learning/random-variables.ipynb` |  | stale .executed stamp; notebook was regenerated after passing; stamp exists but current notebook has no executed cells |
| unverified | pytorch | `chapter_appendix-mathematics-for-deep-learning/single-variable-calculus.ipynb` |  | stale .executed stamp; notebook was regenerated after passing; stamp exists but current notebook has no executed cells |
| unverified | pytorch | `chapter_appendix-mathematics-for-deep-learning/statistics.ipynb` |  | stale .executed stamp; notebook was regenerated after passing; stamp exists but current notebook has no executed cells |
| unverified | pytorch | `chapter_appendix-tools-for-deep-learning/utils.ipynb` |  | stale .executed stamp; notebook was regenerated after passing; stamp exists but current notebook has no executed cells |
| unverified | pytorch | `chapter_attention-mechanisms-and-transformers/attention-pooling.ipynb` |  | stale .executed stamp; notebook was regenerated after passing; stamp exists but current notebook has no executed cells |
| unverified | pytorch | `chapter_attention-mechanisms-and-transformers/attention-scoring-functions.ipynb` |  | stale .executed stamp; notebook was regenerated after passing; stamp exists but current notebook has no executed cells |
| unverified | pytorch | `chapter_attention-mechanisms-and-transformers/bahdanau-attention.ipynb` |  | stale .executed stamp; notebook was regenerated after passing; stamp exists but current notebook has no executed cells |
| unverified | pytorch | `chapter_attention-mechanisms-and-transformers/multihead-attention.ipynb` |  | stale .executed stamp; notebook was regenerated after passing; stamp exists but current notebook has no executed cells |
| unverified | pytorch | `chapter_attention-mechanisms-and-transformers/queries-keys-values.ipynb` |  | stale .executed stamp; notebook was regenerated after passing; stamp exists but current notebook has no executed cells |
| unverified | pytorch | `chapter_attention-mechanisms-and-transformers/self-attention-and-positional-encoding.ipynb` |  | stale .executed stamp; notebook was regenerated after passing; stamp exists but current notebook has no executed cells |
| unverified | pytorch | `chapter_attention-mechanisms-and-transformers/transformer.ipynb` |  | stale .executed stamp; notebook was regenerated after passing; stamp exists but current notebook has no executed cells |
| unverified | pytorch | `chapter_attention-mechanisms-and-transformers/vision-transformer.ipynb` |  | stale .executed stamp; notebook was regenerated after passing; stamp exists but current notebook has no executed cells |
| unverified | pytorch | `chapter_builders-guide/custom-layer.ipynb` |  | stale .executed stamp; notebook was regenerated after passing; stamp exists but current notebook has no executed cells |
| unverified | pytorch | `chapter_builders-guide/init-param.ipynb` |  | stale .executed stamp; notebook was regenerated after passing; stamp exists but current notebook has no executed cells |
| unverified | pytorch | `chapter_builders-guide/lazy-init.ipynb` |  | stale .executed stamp; notebook was regenerated after passing; stamp exists but current notebook has no executed cells |
| unverified | pytorch | `chapter_builders-guide/model-construction.ipynb` |  | stale .executed stamp; notebook was regenerated after passing; stamp exists but current notebook has no executed cells |
| unverified | pytorch | `chapter_builders-guide/parameters.ipynb` |  | stale .executed stamp; notebook was regenerated after passing; stamp exists but current notebook has no executed cells |
| unverified | pytorch | `chapter_builders-guide/read-write.ipynb` |  | stale .executed stamp; notebook was regenerated after passing; stamp exists but current notebook has no executed cells |
| unverified | pytorch | `chapter_computational-performance/hybridize.ipynb` |  | stale .executed stamp; notebook was regenerated after passing; stamp exists but current notebook has no executed cells |
| unverified | pytorch | `chapter_computer-vision/anchor.ipynb` |  | stale .executed stamp; notebook was regenerated after passing; stamp exists but current notebook has no executed cells |
| unverified | pytorch | `chapter_computer-vision/bounding-box.ipynb` |  | stale .executed stamp; notebook was regenerated after passing; stamp exists but current notebook has no executed cells |
| unverified | pytorch | `chapter_computer-vision/fcn.ipynb` |  | stale .executed stamp; notebook was regenerated after passing; stamp exists but current notebook has no executed cells |
| unverified | pytorch | `chapter_computer-vision/fine-tuning.ipynb` |  | stale .executed stamp; notebook was regenerated after passing; stamp exists but current notebook has no executed cells |
| unverified | pytorch | `chapter_computer-vision/image-augmentation.ipynb` |  | stale .executed stamp; notebook was regenerated after passing; stamp exists but current notebook has no executed cells |
| unverified | pytorch | `chapter_computer-vision/kaggle-cifar10.ipynb` |  | stale .executed stamp; notebook was regenerated after passing; stamp exists but current notebook has no executed cells |
| unverified | pytorch | `chapter_computer-vision/kaggle-dog.ipynb` |  | stale .executed stamp; notebook was regenerated after passing; stamp exists but current notebook has no executed cells |
| unverified | pytorch | `chapter_computer-vision/multiscale-object-detection.ipynb` |  | stale .executed stamp; notebook was regenerated after passing; stamp exists but current notebook has no executed cells |
| unverified | pytorch | `chapter_computer-vision/neural-style.ipynb` |  | stale .executed stamp; notebook was regenerated after passing; stamp exists but current notebook has no executed cells |
| unverified | pytorch | `chapter_computer-vision/object-detection-dataset.ipynb` |  | stale .executed stamp; notebook was regenerated after passing; stamp exists but current notebook has no executed cells |
| unverified | pytorch | `chapter_computer-vision/rcnn.ipynb` |  | stale .executed stamp; notebook was regenerated after passing; stamp exists but current notebook has no executed cells |
| unverified | pytorch | `chapter_computer-vision/semantic-segmentation-and-dataset.ipynb` |  | stale .executed stamp; notebook was regenerated after passing; stamp exists but current notebook has no executed cells |
| unverified | pytorch | `chapter_computer-vision/ssd.ipynb` |  | stale .executed stamp; notebook was regenerated after passing; stamp exists but current notebook has no executed cells |
| unverified | pytorch | `chapter_computer-vision/transposed-conv.ipynb` |  | stale .executed stamp; notebook was regenerated after passing; stamp exists but current notebook has no executed cells |
| unverified | pytorch | `chapter_convolutional-modern/alexnet.ipynb` |  | stale .executed stamp; notebook was regenerated after passing; stamp exists but current notebook has no executed cells |
| unverified | pytorch | `chapter_convolutional-modern/batch-norm.ipynb` |  | stale .executed stamp; notebook was regenerated after passing; stamp exists but current notebook has no executed cells |
| unverified | pytorch | `chapter_convolutional-modern/cnn-design.ipynb` |  | stale .executed stamp; notebook was regenerated after passing; stamp exists but current notebook has no executed cells |
| unverified | pytorch | `chapter_convolutional-modern/densenet.ipynb` |  | stale .executed stamp; notebook was regenerated after passing; stamp exists but current notebook has no executed cells |
| unverified | pytorch | `chapter_convolutional-modern/googlenet.ipynb` |  | stale .executed stamp; notebook was regenerated after passing; stamp exists but current notebook has no executed cells |
| unverified | pytorch | `chapter_convolutional-modern/nin.ipynb` |  | stale .executed stamp; notebook was regenerated after passing; stamp exists but current notebook has no executed cells |
| unverified | pytorch | `chapter_convolutional-modern/resnet.ipynb` |  | stale .executed stamp; notebook was regenerated after passing; stamp exists but current notebook has no executed cells |
| unverified | pytorch | `chapter_convolutional-modern/vgg.ipynb` |  | stale .executed stamp; notebook was regenerated after passing; stamp exists but current notebook has no executed cells |
| unverified | pytorch | `chapter_convolutional-neural-networks/channels.ipynb` |  | stale .executed stamp; notebook was regenerated after passing; stamp exists but current notebook has no executed cells |
| unverified | pytorch | `chapter_convolutional-neural-networks/conv-layer.ipynb` |  | stale .executed stamp; notebook was regenerated after passing; stamp exists but current notebook has no executed cells |
| unverified | pytorch | `chapter_convolutional-neural-networks/lenet.ipynb` |  | stale .executed stamp; notebook was regenerated after passing; stamp exists but current notebook has no executed cells |
| unverified | pytorch | `chapter_convolutional-neural-networks/padding-and-strides.ipynb` |  | stale .executed stamp; notebook was regenerated after passing; stamp exists but current notebook has no executed cells |
| unverified | pytorch | `chapter_convolutional-neural-networks/pooling.ipynb` |  | stale .executed stamp; notebook was regenerated after passing; stamp exists but current notebook has no executed cells |
| unverified | pytorch | `chapter_gaussian-processes/gp-inference.ipynb` |  | stale .executed stamp; notebook was regenerated after passing; stamp exists but current notebook has no executed cells |
| unverified | pytorch | `chapter_gaussian-processes/gp-priors.ipynb` |  | stale .executed stamp; notebook was regenerated after passing; stamp exists but current notebook has no executed cells |
| unverified | pytorch | `chapter_generative-adversarial-networks/dcgan.ipynb` |  | stale .executed stamp; notebook was regenerated after passing; stamp exists but current notebook has no executed cells |
| unverified | pytorch | `chapter_generative-adversarial-networks/gan.ipynb` |  | stale .executed stamp; notebook was regenerated after passing; stamp exists but current notebook has no executed cells |
| unverified | pytorch | `chapter_hyperparameter-optimization/hyperopt-api.ipynb` |  | stale .executed stamp; notebook was regenerated after passing; stamp exists but current notebook has no executed cells |
| unverified | pytorch | `chapter_hyperparameter-optimization/hyperopt-intro.ipynb` |  | stale .executed stamp; notebook was regenerated after passing; stamp exists but current notebook has no executed cells |
| unverified | pytorch | `chapter_hyperparameter-optimization/rs-async.ipynb` |  | stale .executed stamp; notebook was regenerated after passing; stamp exists but current notebook has no executed cells |
| unverified | pytorch | `chapter_hyperparameter-optimization/sh-async.ipynb` |  | stale .executed stamp; notebook was regenerated after passing; stamp exists but current notebook has no executed cells |
| unverified | pytorch | `chapter_hyperparameter-optimization/sh-intro.ipynb` |  | stale .executed stamp; notebook was regenerated after passing; stamp exists but current notebook has no executed cells |
| unverified | pytorch | `chapter_linear-classification/classification.ipynb` |  | stale .executed stamp; notebook was regenerated after passing; stamp exists but current notebook has no executed cells |
| unverified | pytorch | `chapter_linear-classification/image-classification-dataset.ipynb` |  | stale .executed stamp; notebook was regenerated after passing; stamp exists but current notebook has no executed cells |
| unverified | pytorch | `chapter_linear-classification/softmax-regression-concise.ipynb` |  | stale .executed stamp; notebook was regenerated after passing; stamp exists but current notebook has no executed cells |
| unverified | pytorch | `chapter_linear-classification/softmax-regression-scratch.ipynb` |  | stale .executed stamp; notebook was regenerated after passing; stamp exists but current notebook has no executed cells |
| unverified | pytorch | `chapter_linear-regression/linear-regression-concise.ipynb` |  | stale .executed stamp; notebook was regenerated after passing; stamp exists but current notebook has no executed cells |
| unverified | pytorch | `chapter_linear-regression/linear-regression-scratch.ipynb` |  | stale .executed stamp; notebook was regenerated after passing; stamp exists but current notebook has no executed cells |
| unverified | pytorch | `chapter_linear-regression/linear-regression.ipynb` |  | stale .executed stamp; notebook was regenerated after passing; stamp exists but current notebook has no executed cells |
| unverified | pytorch | `chapter_linear-regression/oo-design.ipynb` |  | stale .executed stamp; notebook was regenerated after passing; stamp exists but current notebook has no executed cells |
| unverified | pytorch | `chapter_linear-regression/synthetic-regression-data.ipynb` |  | stale .executed stamp; notebook was regenerated after passing; stamp exists but current notebook has no executed cells |
| unverified | pytorch | `chapter_linear-regression/weight-decay.ipynb` |  | stale .executed stamp; notebook was regenerated after passing; stamp exists but current notebook has no executed cells |
| unverified | pytorch | `chapter_multilayer-perceptrons/dropout.ipynb` |  | stale .executed stamp; notebook was regenerated after passing; stamp exists but current notebook has no executed cells |
| unverified | pytorch | `chapter_multilayer-perceptrons/kaggle-house-price.ipynb` |  | stale .executed stamp; notebook was regenerated after passing; stamp exists but current notebook has no executed cells |
| unverified | pytorch | `chapter_multilayer-perceptrons/mlp-implementation.ipynb` |  | stale .executed stamp; notebook was regenerated after passing; stamp exists but current notebook has no executed cells |
| unverified | pytorch | `chapter_multilayer-perceptrons/mlp.ipynb` |  | stale .executed stamp; notebook was regenerated after passing; stamp exists but current notebook has no executed cells |
| unverified | pytorch | `chapter_multilayer-perceptrons/numerical-stability-and-init.ipynb` |  | stale .executed stamp; notebook was regenerated after passing; stamp exists but current notebook has no executed cells |
| unverified | pytorch | `chapter_natural-language-processing-applications/natural-language-inference-and-dataset.ipynb` |  | stale .executed stamp; notebook was regenerated after passing; stamp exists but current notebook has no executed cells |
| unverified | pytorch | `chapter_natural-language-processing-applications/natural-language-inference-attention.ipynb` |  | stale .executed stamp; notebook was regenerated after passing; stamp exists but current notebook has no executed cells |
| unverified | pytorch | `chapter_natural-language-processing-applications/natural-language-inference-bert.ipynb` |  | stale .executed stamp; notebook was regenerated after passing; stamp exists but current notebook has no executed cells |
| unverified | pytorch | `chapter_natural-language-processing-applications/sentiment-analysis-and-dataset.ipynb` |  | stale .executed stamp; notebook was regenerated after passing; stamp exists but current notebook has no executed cells |
| unverified | pytorch | `chapter_natural-language-processing-applications/sentiment-analysis-cnn.ipynb` |  | stale .executed stamp; notebook was regenerated after passing; stamp exists but current notebook has no executed cells |
| unverified | pytorch | `chapter_natural-language-processing-applications/sentiment-analysis-rnn.ipynb` |  | stale .executed stamp; notebook was regenerated after passing; stamp exists but current notebook has no executed cells |
| unverified | pytorch | `chapter_natural-language-processing-pretraining/bert-dataset.ipynb` |  | stale .executed stamp; notebook was regenerated after passing; stamp exists but current notebook has no executed cells |
| unverified | pytorch | `chapter_natural-language-processing-pretraining/bert-pretraining.ipynb` |  | stale .executed stamp; notebook was regenerated after passing; stamp exists but current notebook has no executed cells |
| unverified | pytorch | `chapter_natural-language-processing-pretraining/bert.ipynb` |  | stale .executed stamp; notebook was regenerated after passing; stamp exists but current notebook has no executed cells |
| unverified | pytorch | `chapter_natural-language-processing-pretraining/similarity-analogy.ipynb` |  | stale .executed stamp; notebook was regenerated after passing; stamp exists but current notebook has no executed cells |
| unverified | pytorch | `chapter_natural-language-processing-pretraining/subword-embedding.ipynb` |  | stale .executed stamp; notebook was regenerated after passing; stamp exists but current notebook has no executed cells |
| unverified | pytorch | `chapter_natural-language-processing-pretraining/word-embedding-dataset.ipynb` |  | stale .executed stamp; notebook was regenerated after passing; stamp exists but current notebook has no executed cells |
| unverified | pytorch | `chapter_natural-language-processing-pretraining/word2vec-pretraining.ipynb` |  | stale .executed stamp; notebook was regenerated after passing; stamp exists but current notebook has no executed cells |
| unverified | pytorch | `chapter_optimization/adadelta.ipynb` |  | stale .executed stamp; notebook was regenerated after passing; stamp exists but current notebook has no executed cells |
| unverified | pytorch | `chapter_optimization/adagrad.ipynb` |  | stale .executed stamp; notebook was regenerated after passing; stamp exists but current notebook has no executed cells |
| unverified | pytorch | `chapter_optimization/adam.ipynb` |  | stale .executed stamp; notebook was regenerated after passing; stamp exists but current notebook has no executed cells |
| unverified | pytorch | `chapter_optimization/convexity.ipynb` |  | stale .executed stamp; notebook was regenerated after passing; stamp exists but current notebook has no executed cells |
| unverified | pytorch | `chapter_optimization/gd.ipynb` |  | stale .executed stamp; notebook was regenerated after passing; stamp exists but current notebook has no executed cells |
| unverified | pytorch | `chapter_optimization/lr-scheduler.ipynb` |  | stale .executed stamp; notebook was regenerated after passing; stamp exists but current notebook has no executed cells |
| unverified | pytorch | `chapter_optimization/minibatch-sgd.ipynb` |  | stale .executed stamp; notebook was regenerated after passing; stamp exists but current notebook has no executed cells |
| unverified | pytorch | `chapter_optimization/momentum.ipynb` |  | stale .executed stamp; notebook was regenerated after passing; stamp exists but current notebook has no executed cells |
| unverified | pytorch | `chapter_optimization/optimization-intro.ipynb` |  | stale .executed stamp; notebook was regenerated after passing; stamp exists but current notebook has no executed cells |
| unverified | pytorch | `chapter_optimization/rmsprop.ipynb` |  | stale .executed stamp; notebook was regenerated after passing; stamp exists but current notebook has no executed cells |
| unverified | pytorch | `chapter_optimization/sgd.ipynb` |  | stale .executed stamp; notebook was regenerated after passing; stamp exists but current notebook has no executed cells |
| unverified | pytorch | `chapter_preface/index.ipynb` |  | stale .executed stamp; notebook was regenerated after passing; stamp exists but current notebook has no executed cells |
| unverified | pytorch | `chapter_preliminaries/autograd.ipynb` |  | stale .executed stamp; notebook was regenerated after passing; stamp exists but current notebook has no executed cells |
| unverified | pytorch | `chapter_preliminaries/calculus.ipynb` |  | stale .executed stamp; notebook was regenerated after passing; stamp exists but current notebook has no executed cells |
| unverified | pytorch | `chapter_preliminaries/linear-algebra.ipynb` |  | stale .executed stamp; notebook was regenerated after passing; stamp exists but current notebook has no executed cells |
| unverified | pytorch | `chapter_preliminaries/lookup-api.ipynb` |  | stale .executed stamp; notebook was regenerated after passing; stamp exists but current notebook has no executed cells |
| unverified | pytorch | `chapter_preliminaries/ndarray.ipynb` |  | stale .executed stamp; notebook was regenerated after passing; stamp exists but current notebook has no executed cells |
| unverified | pytorch | `chapter_preliminaries/pandas.ipynb` |  | stale .executed stamp; notebook was regenerated after passing; stamp exists but current notebook has no executed cells |
| unverified | pytorch | `chapter_preliminaries/probability.ipynb` |  | stale .executed stamp; notebook was regenerated after passing; stamp exists but current notebook has no executed cells |
| unverified | pytorch | `chapter_recommender-systems/autorec.ipynb` |  | stale .executed stamp; notebook was regenerated after passing; stamp exists but current notebook has no executed cells |
| unverified | pytorch | `chapter_recommender-systems/ctr.ipynb` |  | stale .executed stamp; notebook was regenerated after passing; stamp exists but current notebook has no executed cells |
| unverified | pytorch | `chapter_recommender-systems/deepfm.ipynb` |  | stale .executed stamp; notebook was regenerated after passing; stamp exists but current notebook has no executed cells |
| unverified | pytorch | `chapter_recommender-systems/fm.ipynb` |  | stale .executed stamp; notebook was regenerated after passing; stamp exists but current notebook has no executed cells |
| unverified | pytorch | `chapter_recommender-systems/mf.ipynb` |  | stale .executed stamp; notebook was regenerated after passing; stamp exists but current notebook has no executed cells |
| unverified | pytorch | `chapter_recommender-systems/movielens.ipynb` |  | stale .executed stamp; notebook was regenerated after passing; stamp exists but current notebook has no executed cells |
| unverified | pytorch | `chapter_recommender-systems/neumf.ipynb` |  | stale .executed stamp; notebook was regenerated after passing; stamp exists but current notebook has no executed cells |
| unverified | pytorch | `chapter_recommender-systems/ranking.ipynb` |  | stale .executed stamp; notebook was regenerated after passing; stamp exists but current notebook has no executed cells |
| unverified | pytorch | `chapter_recommender-systems/seqrec.ipynb` |  | stale .executed stamp; notebook was regenerated after passing; stamp exists but current notebook has no executed cells |
| unverified | pytorch | `chapter_recurrent-modern/bi-rnn.ipynb` |  | stale .executed stamp; notebook was regenerated after passing; stamp exists but current notebook has no executed cells |
| unverified | pytorch | `chapter_recurrent-modern/deep-rnn.ipynb` |  | stale .executed stamp; notebook was regenerated after passing; stamp exists but current notebook has no executed cells |
| unverified | pytorch | `chapter_recurrent-modern/encoder-decoder.ipynb` |  | stale .executed stamp; notebook was regenerated after passing; stamp exists but current notebook has no executed cells |
| unverified | pytorch | `chapter_recurrent-modern/gru.ipynb` |  | stale .executed stamp; notebook was regenerated after passing; stamp exists but current notebook has no executed cells |
| unverified | pytorch | `chapter_recurrent-modern/lstm.ipynb` |  | stale .executed stamp; notebook was regenerated after passing; stamp exists but current notebook has no executed cells |
| unverified | pytorch | `chapter_recurrent-modern/machine-translation-and-dataset.ipynb` |  | stale .executed stamp; notebook was regenerated after passing; stamp exists but current notebook has no executed cells |
| unverified | pytorch | `chapter_recurrent-modern/seq2seq.ipynb` |  | stale .executed stamp; notebook was regenerated after passing; stamp exists but current notebook has no executed cells |
| unverified | pytorch | `chapter_recurrent-neural-networks/language-model.ipynb` |  | stale .executed stamp; notebook was regenerated after passing; stamp exists but current notebook has no executed cells |
| unverified | pytorch | `chapter_recurrent-neural-networks/rnn-concise.ipynb` |  | stale .executed stamp; notebook was regenerated after passing; stamp exists but current notebook has no executed cells |
| unverified | pytorch | `chapter_recurrent-neural-networks/rnn-scratch.ipynb` |  | stale .executed stamp; notebook was regenerated after passing; stamp exists but current notebook has no executed cells |
| unverified | pytorch | `chapter_recurrent-neural-networks/rnn.ipynb` |  | stale .executed stamp; notebook was regenerated after passing; stamp exists but current notebook has no executed cells |
| unverified | pytorch | `chapter_recurrent-neural-networks/sequence.ipynb` |  | stale .executed stamp; notebook was regenerated after passing; stamp exists but current notebook has no executed cells |
| unverified | pytorch | `chapter_recurrent-neural-networks/text-sequence.ipynb` |  | stale .executed stamp; notebook was regenerated after passing; stamp exists but current notebook has no executed cells |
| unverified | pytorch | `chapter_reinforcement-learning/qlearning.ipynb` |  | stale .executed stamp; notebook was regenerated after passing; stamp exists but current notebook has no executed cells |
| unverified | pytorch | `chapter_reinforcement-learning/value-iter.ipynb` |  | stale .executed stamp; notebook was regenerated after passing; stamp exists but current notebook has no executed cells |
| unverified | tensorflow | `chapter_appendix-mathematics-for-deep-learning/distributions.ipynb` |  | stale .executed stamp; notebook was regenerated after passing; stamp exists but current notebook has no executed cells |
| unverified | tensorflow | `chapter_appendix-mathematics-for-deep-learning/eigendecomposition.ipynb` |  | stale .executed stamp; notebook was regenerated after passing; stamp exists but current notebook has no executed cells |
| unverified | tensorflow | `chapter_appendix-mathematics-for-deep-learning/geometry-linear-algebraic-ops.ipynb` |  | stale .executed stamp; notebook was regenerated after passing; stamp exists but current notebook has no executed cells |
| unverified | tensorflow | `chapter_appendix-mathematics-for-deep-learning/information-theory.ipynb` |  | stale .executed stamp; notebook was regenerated after passing; stamp exists but current notebook has no executed cells |
| unverified | tensorflow | `chapter_appendix-mathematics-for-deep-learning/integral-calculus.ipynb` |  | stale .executed stamp; notebook was regenerated after passing; stamp exists but current notebook has no executed cells |
| unverified | tensorflow | `chapter_appendix-mathematics-for-deep-learning/maximum-likelihood.ipynb` |  | stale .executed stamp; notebook was regenerated after passing; stamp exists but current notebook has no executed cells |
| unverified | tensorflow | `chapter_appendix-mathematics-for-deep-learning/multivariable-calculus.ipynb` |  | stale .executed stamp; notebook was regenerated after passing; stamp exists but current notebook has no executed cells |
| unverified | tensorflow | `chapter_appendix-mathematics-for-deep-learning/naive-bayes.ipynb` |  | stale .executed stamp; notebook was regenerated after passing; stamp exists but current notebook has no executed cells |
| unverified | tensorflow | `chapter_appendix-mathematics-for-deep-learning/random-variables.ipynb` |  | stale .executed stamp; notebook was regenerated after passing; stamp exists but current notebook has no executed cells |
| unverified | tensorflow | `chapter_appendix-mathematics-for-deep-learning/single-variable-calculus.ipynb` |  | stale .executed stamp; notebook was regenerated after passing; stamp exists but current notebook has no executed cells |
| unverified | tensorflow | `chapter_appendix-mathematics-for-deep-learning/statistics.ipynb` |  | stale .executed stamp; notebook was regenerated after passing; stamp exists but current notebook has no executed cells |
| unverified | tensorflow | `chapter_appendix-tools-for-deep-learning/utils.ipynb` |  | stale .executed stamp; notebook was regenerated after passing; stamp exists but current notebook has no executed cells |
| unverified | tensorflow | `chapter_attention-mechanisms-and-transformers/attention-pooling.ipynb` |  | stale .executed stamp; notebook was regenerated after passing; stamp exists but current notebook has no executed cells |
| unverified | tensorflow | `chapter_attention-mechanisms-and-transformers/attention-scoring-functions.ipynb` |  | stale .executed stamp; notebook was regenerated after passing; stamp exists but current notebook has no executed cells |
| unverified | tensorflow | `chapter_attention-mechanisms-and-transformers/bahdanau-attention.ipynb` |  | stale .executed stamp; notebook was regenerated after passing; stamp exists but current notebook has no executed cells |
| unverified | tensorflow | `chapter_attention-mechanisms-and-transformers/multihead-attention.ipynb` |  | stale .executed stamp; notebook was regenerated after passing; stamp exists but current notebook has no executed cells |
| unverified | tensorflow | `chapter_attention-mechanisms-and-transformers/queries-keys-values.ipynb` |  | stale .executed stamp; notebook was regenerated after passing; stamp exists but current notebook has no executed cells |
| unverified | tensorflow | `chapter_attention-mechanisms-and-transformers/self-attention-and-positional-encoding.ipynb` |  | stale .executed stamp; notebook was regenerated after passing; stamp exists but current notebook has no executed cells |
| unverified | tensorflow | `chapter_attention-mechanisms-and-transformers/transformer.ipynb` |  | stale .executed stamp; notebook was regenerated after passing; stamp exists but current notebook has no executed cells |
| unverified | tensorflow | `chapter_attention-mechanisms-and-transformers/vision-transformer.ipynb` |  | stale .executed stamp; notebook was regenerated after passing; stamp exists but current notebook has no executed cells |
| unverified | tensorflow | `chapter_builders-guide/custom-layer.ipynb` |  | stale .executed stamp; notebook was regenerated after passing; stamp exists but current notebook has no executed cells |
| unverified | tensorflow | `chapter_builders-guide/init-param.ipynb` |  | stale .executed stamp; notebook was regenerated after passing; stamp exists but current notebook has no executed cells |
| unverified | tensorflow | `chapter_builders-guide/lazy-init.ipynb` |  | stale .executed stamp; notebook was regenerated after passing; stamp exists but current notebook has no executed cells |
| unverified | tensorflow | `chapter_builders-guide/model-construction.ipynb` |  | stale .executed stamp; notebook was regenerated after passing; stamp exists but current notebook has no executed cells |
| unverified | tensorflow | `chapter_builders-guide/parameters.ipynb` |  | stale .executed stamp; notebook was regenerated after passing; stamp exists but current notebook has no executed cells |
| unverified | tensorflow | `chapter_builders-guide/read-write.ipynb` |  | stale .executed stamp; notebook was regenerated after passing; stamp exists but current notebook has no executed cells |
| unverified | tensorflow | `chapter_builders-guide/use-gpu.ipynb` |  | stale .executed stamp; notebook was regenerated after passing; stamp exists but current notebook has no executed cells |
| unverified | tensorflow | `chapter_computational-performance/async-computation.ipynb` |  | stale .executed stamp; notebook was regenerated after passing; stamp exists but current notebook has no executed cells |
| unverified | tensorflow | `chapter_computational-performance/auto-parallelism.ipynb` |  | stale .executed stamp; notebook was regenerated after passing; stamp exists but current notebook has no executed cells |
| unverified | tensorflow | `chapter_computational-performance/hybridize.ipynb` |  | stale .executed stamp; notebook was regenerated after passing; stamp exists but current notebook has no executed cells |
| unverified | tensorflow | `chapter_computational-performance/multiple-gpus-concise.ipynb` |  | stale .executed stamp; notebook was regenerated after passing; stamp exists but current notebook has no executed cells |
| unverified | tensorflow | `chapter_computational-performance/multiple-gpus.ipynb` |  | stale .executed stamp; notebook was regenerated after passing; stamp exists but current notebook has no executed cells |
| unverified | tensorflow | `chapter_computer-vision/anchor.ipynb` |  | stale .executed stamp; notebook was regenerated after passing; stamp exists but current notebook has no executed cells |
| unverified | tensorflow | `chapter_computer-vision/bounding-box.ipynb` |  | stale .executed stamp; notebook was regenerated after passing; stamp exists but current notebook has no executed cells |
| unverified | tensorflow | `chapter_computer-vision/fcn.ipynb` |  | stale .executed stamp; notebook was regenerated after passing; stamp exists but current notebook has no executed cells |
| unverified | tensorflow | `chapter_computer-vision/fine-tuning.ipynb` |  | stale .executed stamp; notebook was regenerated after passing; stamp exists but current notebook has no executed cells |
| unverified | tensorflow | `chapter_computer-vision/image-augmentation.ipynb` |  | stale .executed stamp; notebook was regenerated after passing; stamp exists but current notebook has no executed cells |
| warn | tensorflow | `chapter_computer-vision/kaggle-cifar10.ipynb` | loss=0.78, train_acc=0.722, valid_acc=0.328 | validation accuracy is low relative to training accuracy |
| unverified | tensorflow | `chapter_computer-vision/kaggle-dog.ipynb` |  | stale .executed stamp; notebook was regenerated after passing; stamp exists but current notebook has no executed cells |
| unverified | tensorflow | `chapter_computer-vision/multiscale-object-detection.ipynb` |  | stale .executed stamp; notebook was regenerated after passing; stamp exists but current notebook has no executed cells |
| unverified | tensorflow | `chapter_computer-vision/neural-style.ipynb` |  | stale .executed stamp; notebook was regenerated after passing; stamp exists but current notebook has no executed cells |
| unverified | tensorflow | `chapter_computer-vision/object-detection-dataset.ipynb` |  | stale .executed stamp; notebook was regenerated after passing; stamp exists but current notebook has no executed cells |
| unverified | tensorflow | `chapter_computer-vision/rcnn.ipynb` |  | stale .executed stamp; notebook was regenerated after passing; stamp exists but current notebook has no executed cells |
| unverified | tensorflow | `chapter_computer-vision/semantic-segmentation-and-dataset.ipynb` |  | stale .executed stamp; notebook was regenerated after passing; stamp exists but current notebook has no executed cells |
| unverified | tensorflow | `chapter_computer-vision/transposed-conv.ipynb` |  | stale .executed stamp; notebook was regenerated after passing; stamp exists but current notebook has no executed cells |
| unverified | tensorflow | `chapter_convolutional-modern/alexnet.ipynb` |  | stale .executed stamp; notebook was regenerated after passing; stamp exists but current notebook has no executed cells |
| unverified | tensorflow | `chapter_convolutional-modern/batch-norm.ipynb` |  | stale .executed stamp; notebook was regenerated after passing; stamp exists but current notebook has no executed cells |
| unverified | tensorflow | `chapter_convolutional-modern/cnn-design.ipynb` |  | stale .executed stamp; notebook was regenerated after passing; stamp exists but current notebook has no executed cells |
| unverified | tensorflow | `chapter_convolutional-modern/densenet.ipynb` |  | stale .executed stamp; notebook was regenerated after passing; stamp exists but current notebook has no executed cells |
| unverified | tensorflow | `chapter_convolutional-modern/googlenet.ipynb` |  | stale .executed stamp; notebook was regenerated after passing; stamp exists but current notebook has no executed cells |
| unverified | tensorflow | `chapter_convolutional-modern/nin.ipynb` |  | stale .executed stamp; notebook was regenerated after passing; stamp exists but current notebook has no executed cells |
| unverified | tensorflow | `chapter_convolutional-modern/resnet.ipynb` |  | stale .executed stamp; notebook was regenerated after passing; stamp exists but current notebook has no executed cells |
| unverified | tensorflow | `chapter_convolutional-modern/vgg.ipynb` |  | stale .executed stamp; notebook was regenerated after passing; stamp exists but current notebook has no executed cells |
| unverified | tensorflow | `chapter_convolutional-neural-networks/channels.ipynb` |  | stale .executed stamp; notebook was regenerated after passing; stamp exists but current notebook has no executed cells |
| unverified | tensorflow | `chapter_convolutional-neural-networks/conv-layer.ipynb` |  | stale .executed stamp; notebook was regenerated after passing; stamp exists but current notebook has no executed cells |
| unverified | tensorflow | `chapter_convolutional-neural-networks/lenet.ipynb` |  | stale .executed stamp; notebook was regenerated after passing; stamp exists but current notebook has no executed cells |
| unverified | tensorflow | `chapter_convolutional-neural-networks/padding-and-strides.ipynb` |  | stale .executed stamp; notebook was regenerated after passing; stamp exists but current notebook has no executed cells |
| unverified | tensorflow | `chapter_convolutional-neural-networks/pooling.ipynb` |  | stale .executed stamp; notebook was regenerated after passing; stamp exists but current notebook has no executed cells |
| unverified | tensorflow | `chapter_generative-adversarial-networks/dcgan.ipynb` |  | stale .executed stamp; notebook was regenerated after passing; stamp exists but current notebook has no executed cells |
| unverified | tensorflow | `chapter_generative-adversarial-networks/gan.ipynb` |  | stale .executed stamp; notebook was regenerated after passing; stamp exists but current notebook has no executed cells |
| unverified | tensorflow | `chapter_hyperparameter-optimization/hyperopt-api.ipynb` |  | stale .executed stamp; notebook was regenerated after passing; stamp exists but current notebook has no executed cells |
| unverified | tensorflow | `chapter_hyperparameter-optimization/hyperopt-intro.ipynb` |  | stale .executed stamp; notebook was regenerated after passing; stamp exists but current notebook has no executed cells |
| unverified | tensorflow | `chapter_hyperparameter-optimization/sh-intro.ipynb` |  | stale .executed stamp; notebook was regenerated after passing; stamp exists but current notebook has no executed cells |
| unverified | tensorflow | `chapter_linear-classification/classification.ipynb` |  | stale .executed stamp; notebook was regenerated after passing; stamp exists but current notebook has no executed cells |
| unverified | tensorflow | `chapter_linear-classification/image-classification-dataset.ipynb` |  | stale .executed stamp; notebook was regenerated after passing; stamp exists but current notebook has no executed cells |
| unverified | tensorflow | `chapter_linear-classification/softmax-regression-concise.ipynb` |  | stale .executed stamp; notebook was regenerated after passing; stamp exists but current notebook has no executed cells |
| unverified | tensorflow | `chapter_linear-classification/softmax-regression-scratch.ipynb` |  | stale .executed stamp; notebook was regenerated after passing; stamp exists but current notebook has no executed cells |
| unverified | tensorflow | `chapter_linear-regression/linear-regression-concise.ipynb` |  | stale .executed stamp; notebook was regenerated after passing; stamp exists but current notebook has no executed cells |
| unverified | tensorflow | `chapter_linear-regression/linear-regression-scratch.ipynb` |  | stale .executed stamp; notebook was regenerated after passing; stamp exists but current notebook has no executed cells |
| unverified | tensorflow | `chapter_linear-regression/linear-regression.ipynb` |  | stale .executed stamp; notebook was regenerated after passing; stamp exists but current notebook has no executed cells |
| unverified | tensorflow | `chapter_linear-regression/oo-design.ipynb` |  | stale .executed stamp; notebook was regenerated after passing; stamp exists but current notebook has no executed cells |
| unverified | tensorflow | `chapter_linear-regression/synthetic-regression-data.ipynb` |  | stale .executed stamp; notebook was regenerated after passing; stamp exists but current notebook has no executed cells |
| unverified | tensorflow | `chapter_linear-regression/weight-decay.ipynb` |  | stale .executed stamp; notebook was regenerated after passing; stamp exists but current notebook has no executed cells |
| unverified | tensorflow | `chapter_multilayer-perceptrons/dropout.ipynb` |  | stale .executed stamp; notebook was regenerated after passing; stamp exists but current notebook has no executed cells |
| unverified | tensorflow | `chapter_multilayer-perceptrons/kaggle-house-price.ipynb` |  | stale .executed stamp; notebook was regenerated after passing; stamp exists but current notebook has no executed cells |
| unverified | tensorflow | `chapter_multilayer-perceptrons/mlp-implementation.ipynb` |  | stale .executed stamp; notebook was regenerated after passing; stamp exists but current notebook has no executed cells |
| unverified | tensorflow | `chapter_multilayer-perceptrons/mlp.ipynb` |  | stale .executed stamp; notebook was regenerated after passing; stamp exists but current notebook has no executed cells |
| unverified | tensorflow | `chapter_multilayer-perceptrons/numerical-stability-and-init.ipynb` |  | stale .executed stamp; notebook was regenerated after passing; stamp exists but current notebook has no executed cells |
| unverified | tensorflow | `chapter_natural-language-processing-applications/natural-language-inference-and-dataset.ipynb` |  | stale .executed stamp; notebook was regenerated after passing; stamp exists but current notebook has no executed cells |
| unverified | tensorflow | `chapter_natural-language-processing-applications/natural-language-inference-attention.ipynb` |  | stale .executed stamp; notebook was regenerated after passing; stamp exists but current notebook has no executed cells |
| unverified | tensorflow | `chapter_natural-language-processing-applications/natural-language-inference-bert.ipynb` |  | stale .executed stamp; notebook was regenerated after passing; stamp exists but current notebook has no executed cells |
| unverified | tensorflow | `chapter_natural-language-processing-applications/sentiment-analysis-and-dataset.ipynb` |  | stale .executed stamp; notebook was regenerated after passing; stamp exists but current notebook has no executed cells |
| unverified | tensorflow | `chapter_natural-language-processing-applications/sentiment-analysis-cnn.ipynb` |  | stale .executed stamp; notebook was regenerated after passing; stamp exists but current notebook has no executed cells |
| unverified | tensorflow | `chapter_natural-language-processing-applications/sentiment-analysis-rnn.ipynb` |  | stale .executed stamp; notebook was regenerated after passing; stamp exists but current notebook has no executed cells |
| unverified | tensorflow | `chapter_natural-language-processing-pretraining/bert-dataset.ipynb` |  | stale .executed stamp; notebook was regenerated after passing; stamp exists but current notebook has no executed cells |
| unverified | tensorflow | `chapter_natural-language-processing-pretraining/bert-pretraining.ipynb` |  | stale .executed stamp; notebook was regenerated after passing; stamp exists but current notebook has no executed cells |
| unverified | tensorflow | `chapter_natural-language-processing-pretraining/bert.ipynb` |  | stale .executed stamp; notebook was regenerated after passing; stamp exists but current notebook has no executed cells |
| unverified | tensorflow | `chapter_natural-language-processing-pretraining/similarity-analogy.ipynb` |  | stale .executed stamp; notebook was regenerated after passing; stamp exists but current notebook has no executed cells |
| unverified | tensorflow | `chapter_natural-language-processing-pretraining/word-embedding-dataset.ipynb` |  | stale .executed stamp; notebook was regenerated after passing; stamp exists but current notebook has no executed cells |
| unverified | tensorflow | `chapter_natural-language-processing-pretraining/word2vec-pretraining.ipynb` |  | stale .executed stamp; notebook was regenerated after passing; stamp exists but current notebook has no executed cells |
| unverified | tensorflow | `chapter_optimization/adadelta.ipynb` |  | stale .executed stamp; notebook was regenerated after passing; stamp exists but current notebook has no executed cells |
| unverified | tensorflow | `chapter_optimization/adagrad.ipynb` |  | stale .executed stamp; notebook was regenerated after passing; stamp exists but current notebook has no executed cells |
| unverified | tensorflow | `chapter_optimization/adam.ipynb` |  | stale .executed stamp; notebook was regenerated after passing; stamp exists but current notebook has no executed cells |
| unverified | tensorflow | `chapter_optimization/convexity.ipynb` |  | stale .executed stamp; notebook was regenerated after passing; stamp exists but current notebook has no executed cells |
| unverified | tensorflow | `chapter_optimization/gd.ipynb` |  | stale .executed stamp; notebook was regenerated after passing; stamp exists but current notebook has no executed cells |
| unverified | tensorflow | `chapter_optimization/lr-scheduler.ipynb` |  | stale .executed stamp; notebook was regenerated after passing; stamp exists but current notebook has no executed cells |
| unverified | tensorflow | `chapter_optimization/minibatch-sgd.ipynb` |  | stale .executed stamp; notebook was regenerated after passing; stamp exists but current notebook has no executed cells |
| unverified | tensorflow | `chapter_optimization/momentum.ipynb` |  | stale .executed stamp; notebook was regenerated after passing; stamp exists but current notebook has no executed cells |
| unverified | tensorflow | `chapter_optimization/optimization-intro.ipynb` |  | stale .executed stamp; notebook was regenerated after passing; stamp exists but current notebook has no executed cells |
| unverified | tensorflow | `chapter_optimization/rmsprop.ipynb` |  | stale .executed stamp; notebook was regenerated after passing; stamp exists but current notebook has no executed cells |
| unverified | tensorflow | `chapter_optimization/sgd.ipynb` |  | stale .executed stamp; notebook was regenerated after passing; stamp exists but current notebook has no executed cells |
| unverified | tensorflow | `chapter_preface/index.ipynb` |  | stale .executed stamp; notebook was regenerated after passing; stamp exists but current notebook has no executed cells |
| unverified | tensorflow | `chapter_preliminaries/autograd.ipynb` |  | stale .executed stamp; notebook was regenerated after passing; stamp exists but current notebook has no executed cells |
| unverified | tensorflow | `chapter_preliminaries/calculus.ipynb` |  | stale .executed stamp; notebook was regenerated after passing; stamp exists but current notebook has no executed cells |
| unverified | tensorflow | `chapter_preliminaries/linear-algebra.ipynb` |  | stale .executed stamp; notebook was regenerated after passing; stamp exists but current notebook has no executed cells |
| unverified | tensorflow | `chapter_preliminaries/lookup-api.ipynb` |  | stale .executed stamp; notebook was regenerated after passing; stamp exists but current notebook has no executed cells |
| unverified | tensorflow | `chapter_preliminaries/ndarray.ipynb` |  | stale .executed stamp; notebook was regenerated after passing; stamp exists but current notebook has no executed cells |
| unverified | tensorflow | `chapter_preliminaries/pandas.ipynb` |  | stale .executed stamp; notebook was regenerated after passing; stamp exists but current notebook has no executed cells |
| unverified | tensorflow | `chapter_preliminaries/probability.ipynb` |  | stale .executed stamp; notebook was regenerated after passing; stamp exists but current notebook has no executed cells |
| unverified | tensorflow | `chapter_recurrent-modern/bi-rnn.ipynb` |  | stale .executed stamp; notebook was regenerated after passing; stamp exists but current notebook has no executed cells |
| unverified | tensorflow | `chapter_recurrent-modern/deep-rnn.ipynb` |  | stale .executed stamp; notebook was regenerated after passing; stamp exists but current notebook has no executed cells |
| unverified | tensorflow | `chapter_recurrent-modern/encoder-decoder.ipynb` |  | stale .executed stamp; notebook was regenerated after passing; stamp exists but current notebook has no executed cells |
| unverified | tensorflow | `chapter_recurrent-modern/gru.ipynb` |  | stale .executed stamp; notebook was regenerated after passing; stamp exists but current notebook has no executed cells |
| unverified | tensorflow | `chapter_recurrent-modern/lstm.ipynb` |  | stale .executed stamp; notebook was regenerated after passing; stamp exists but current notebook has no executed cells |
| unverified | tensorflow | `chapter_recurrent-modern/machine-translation-and-dataset.ipynb` |  | stale .executed stamp; notebook was regenerated after passing; stamp exists but current notebook has no executed cells |
| unverified | tensorflow | `chapter_recurrent-neural-networks/language-model.ipynb` |  | stale .executed stamp; notebook was regenerated after passing; stamp exists but current notebook has no executed cells |
| unverified | tensorflow | `chapter_recurrent-neural-networks/rnn-concise.ipynb` |  | stale .executed stamp; notebook was regenerated after passing; stamp exists but current notebook has no executed cells |
| unverified | tensorflow | `chapter_recurrent-neural-networks/rnn-scratch.ipynb` |  | stale .executed stamp; notebook was regenerated after passing; stamp exists but current notebook has no executed cells |
| unverified | tensorflow | `chapter_recurrent-neural-networks/rnn.ipynb` |  | stale .executed stamp; notebook was regenerated after passing; stamp exists but current notebook has no executed cells |
| unverified | tensorflow | `chapter_recurrent-neural-networks/sequence.ipynb` |  | stale .executed stamp; notebook was regenerated after passing; stamp exists but current notebook has no executed cells |
| unverified | tensorflow | `chapter_recurrent-neural-networks/text-sequence.ipynb` |  | stale .executed stamp; notebook was regenerated after passing; stamp exists but current notebook has no executed cells |
| unverified | jax | `chapter_appendix-mathematics-for-deep-learning/distributions.ipynb` |  | stale .executed stamp; notebook was regenerated after passing; stamp exists but current notebook has no executed cells |
| unverified | jax | `chapter_appendix-mathematics-for-deep-learning/eigendecomposition.ipynb` |  | stale .executed stamp; notebook was regenerated after passing; stamp exists but current notebook has no executed cells |
| unverified | jax | `chapter_appendix-mathematics-for-deep-learning/geometry-linear-algebraic-ops.ipynb` |  | stale .executed stamp; notebook was regenerated after passing; stamp exists but current notebook has no executed cells |
| unverified | jax | `chapter_appendix-mathematics-for-deep-learning/information-theory.ipynb` |  | stale .executed stamp; notebook was regenerated after passing; stamp exists but current notebook has no executed cells |
| unverified | jax | `chapter_appendix-mathematics-for-deep-learning/integral-calculus.ipynb` |  | stale .executed stamp; notebook was regenerated after passing; stamp exists but current notebook has no executed cells |
| unverified | jax | `chapter_appendix-mathematics-for-deep-learning/maximum-likelihood.ipynb` |  | stale .executed stamp; notebook was regenerated after passing; stamp exists but current notebook has no executed cells |
| unverified | jax | `chapter_appendix-mathematics-for-deep-learning/multivariable-calculus.ipynb` |  | stale .executed stamp; notebook was regenerated after passing; stamp exists but current notebook has no executed cells |
| unverified | jax | `chapter_appendix-mathematics-for-deep-learning/naive-bayes.ipynb` |  | stale .executed stamp; notebook was regenerated after passing; stamp exists but current notebook has no executed cells |
| unverified | jax | `chapter_appendix-mathematics-for-deep-learning/random-variables.ipynb` |  | stale .executed stamp; notebook was regenerated after passing; stamp exists but current notebook has no executed cells |
| unverified | jax | `chapter_appendix-mathematics-for-deep-learning/single-variable-calculus.ipynb` |  | stale .executed stamp; notebook was regenerated after passing; stamp exists but current notebook has no executed cells |
| unverified | jax | `chapter_appendix-mathematics-for-deep-learning/statistics.ipynb` |  | stale .executed stamp; notebook was regenerated after passing; stamp exists but current notebook has no executed cells |
| unverified | jax | `chapter_appendix-tools-for-deep-learning/utils.ipynb` |  | stale .executed stamp; notebook was regenerated after passing; stamp exists but current notebook has no executed cells |
| unverified | jax | `chapter_attention-mechanisms-and-transformers/attention-pooling.ipynb` |  | stale .executed stamp; notebook was regenerated after passing; stamp exists but current notebook has no executed cells |
| unverified | jax | `chapter_attention-mechanisms-and-transformers/attention-scoring-functions.ipynb` |  | stale .executed stamp; notebook was regenerated after passing; stamp exists but current notebook has no executed cells |
| unverified | jax | `chapter_attention-mechanisms-and-transformers/bahdanau-attention.ipynb` |  | stale .executed stamp; notebook was regenerated after passing; stamp exists but current notebook has no executed cells |
| unverified | jax | `chapter_attention-mechanisms-and-transformers/multihead-attention.ipynb` |  | stale .executed stamp; notebook was regenerated after passing; stamp exists but current notebook has no executed cells |
| unverified | jax | `chapter_attention-mechanisms-and-transformers/queries-keys-values.ipynb` |  | stale .executed stamp; notebook was regenerated after passing; stamp exists but current notebook has no executed cells |
| unverified | jax | `chapter_attention-mechanisms-and-transformers/self-attention-and-positional-encoding.ipynb` |  | stale .executed stamp; notebook was regenerated after passing; stamp exists but current notebook has no executed cells |
| unverified | jax | `chapter_attention-mechanisms-and-transformers/transformer.ipynb` |  | stale .executed stamp; notebook was regenerated after passing; stamp exists but current notebook has no executed cells |
| unverified | jax | `chapter_attention-mechanisms-and-transformers/vision-transformer.ipynb` |  | stale .executed stamp; notebook was regenerated after passing; stamp exists but current notebook has no executed cells |
| unverified | jax | `chapter_builders-guide/custom-layer.ipynb` |  | stale .executed stamp; notebook was regenerated after passing; stamp exists but current notebook has no executed cells |
| unverified | jax | `chapter_builders-guide/init-param.ipynb` |  | stale .executed stamp; notebook was regenerated after passing; stamp exists but current notebook has no executed cells |
| unverified | jax | `chapter_builders-guide/lazy-init.ipynb` |  | stale .executed stamp; notebook was regenerated after passing; stamp exists but current notebook has no executed cells |
| unverified | jax | `chapter_builders-guide/model-construction.ipynb` |  | stale .executed stamp; notebook was regenerated after passing; stamp exists but current notebook has no executed cells |
| unverified | jax | `chapter_builders-guide/parameters.ipynb` |  | stale .executed stamp; notebook was regenerated after passing; stamp exists but current notebook has no executed cells |
| unverified | jax | `chapter_builders-guide/read-write.ipynb` |  | stale .executed stamp; notebook was regenerated after passing; stamp exists but current notebook has no executed cells |
| unverified | jax | `chapter_builders-guide/use-gpu.ipynb` |  | stale .executed stamp; notebook was regenerated after passing; stamp exists but current notebook has no executed cells |
| unverified | jax | `chapter_computational-performance/async-computation.ipynb` |  | stale .executed stamp; notebook was regenerated after passing; stamp exists but current notebook has no executed cells |
| unverified | jax | `chapter_computational-performance/auto-parallelism.ipynb` |  | stale .executed stamp; notebook was regenerated after passing; stamp exists but current notebook has no executed cells |
| unverified | jax | `chapter_computational-performance/hybridize.ipynb` |  | stale .executed stamp; notebook was regenerated after passing; stamp exists but current notebook has no executed cells |
| unverified | jax | `chapter_computational-performance/multiple-gpus-concise.ipynb` |  | stale .executed stamp; notebook was regenerated after passing; stamp exists but current notebook has no executed cells |
| unverified | jax | `chapter_computational-performance/multiple-gpus.ipynb` |  | stale .executed stamp; notebook was regenerated after passing; stamp exists but current notebook has no executed cells |
| unverified | jax | `chapter_computer-vision/anchor.ipynb` |  | stale .executed stamp; notebook was regenerated after passing; stamp exists but current notebook has no executed cells |
| unverified | jax | `chapter_computer-vision/bounding-box.ipynb` |  | stale .executed stamp; notebook was regenerated after passing; stamp exists but current notebook has no executed cells |
| unverified | jax | `chapter_computer-vision/fcn.ipynb` |  | stale .executed stamp; notebook was regenerated after passing; stamp exists but current notebook has no executed cells |
| unverified | jax | `chapter_computer-vision/fine-tuning.ipynb` |  | stale .executed stamp; notebook was regenerated after passing; stamp exists but current notebook has no executed cells |
| unverified | jax | `chapter_computer-vision/kaggle-cifar10.ipynb` |  | stale .executed stamp; notebook was regenerated after passing; stamp exists but current notebook has no executed cells |
| unverified | jax | `chapter_computer-vision/kaggle-dog.ipynb` |  | stale .executed stamp; notebook was regenerated after passing; stamp exists but current notebook has no executed cells |
| unverified | jax | `chapter_computer-vision/multiscale-object-detection.ipynb` |  | stale .executed stamp; notebook was regenerated after passing; stamp exists but current notebook has no executed cells |
| unverified | jax | `chapter_computer-vision/neural-style.ipynb` |  | stale .executed stamp; notebook was regenerated after passing; stamp exists but current notebook has no executed cells |
| unverified | jax | `chapter_computer-vision/object-detection-dataset.ipynb` |  | stale .executed stamp; notebook was regenerated after passing; stamp exists but current notebook has no executed cells |
| unverified | jax | `chapter_computer-vision/rcnn.ipynb` |  | stale .executed stamp; notebook was regenerated after passing; stamp exists but current notebook has no executed cells |
| unverified | jax | `chapter_computer-vision/semantic-segmentation-and-dataset.ipynb` |  | stale .executed stamp; notebook was regenerated after passing; stamp exists but current notebook has no executed cells |
| unverified | jax | `chapter_computer-vision/ssd.ipynb` |  | stale .executed stamp; notebook was regenerated after passing; stamp exists but current notebook has no executed cells |
| unverified | jax | `chapter_computer-vision/transposed-conv.ipynb` |  | stale .executed stamp; notebook was regenerated after passing; stamp exists but current notebook has no executed cells |
| unverified | jax | `chapter_convolutional-modern/alexnet.ipynb` |  | stale .executed stamp; notebook was regenerated after passing; stamp exists but current notebook has no executed cells |
| unverified | jax | `chapter_convolutional-modern/batch-norm.ipynb` |  | stale .executed stamp; notebook was regenerated after passing; stamp exists but current notebook has no executed cells |
| unverified | jax | `chapter_convolutional-modern/cnn-design.ipynb` |  | stale .executed stamp; notebook was regenerated after passing; stamp exists but current notebook has no executed cells |
| unverified | jax | `chapter_convolutional-modern/densenet.ipynb` |  | stale .executed stamp; notebook was regenerated after passing; stamp exists but current notebook has no executed cells |
| unverified | jax | `chapter_convolutional-modern/googlenet.ipynb` |  | stale .executed stamp; notebook was regenerated after passing; stamp exists but current notebook has no executed cells |
| unverified | jax | `chapter_convolutional-modern/nin.ipynb` |  | stale .executed stamp; notebook was regenerated after passing; stamp exists but current notebook has no executed cells |
| unverified | jax | `chapter_convolutional-modern/resnet.ipynb` |  | stale .executed stamp; notebook was regenerated after passing; stamp exists but current notebook has no executed cells |
| unverified | jax | `chapter_convolutional-modern/vgg.ipynb` |  | stale .executed stamp; notebook was regenerated after passing; stamp exists but current notebook has no executed cells |
| unverified | jax | `chapter_convolutional-neural-networks/channels.ipynb` |  | stale .executed stamp; notebook was regenerated after passing; stamp exists but current notebook has no executed cells |
| unverified | jax | `chapter_convolutional-neural-networks/conv-layer.ipynb` |  | stale .executed stamp; notebook was regenerated after passing; stamp exists but current notebook has no executed cells |
| unverified | jax | `chapter_convolutional-neural-networks/lenet.ipynb` |  | stale .executed stamp; notebook was regenerated after passing; stamp exists but current notebook has no executed cells |
| unverified | jax | `chapter_convolutional-neural-networks/padding-and-strides.ipynb` |  | stale .executed stamp; notebook was regenerated after passing; stamp exists but current notebook has no executed cells |
| unverified | jax | `chapter_convolutional-neural-networks/pooling.ipynb` |  | stale .executed stamp; notebook was regenerated after passing; stamp exists but current notebook has no executed cells |
| unverified | jax | `chapter_generative-adversarial-networks/dcgan.ipynb` |  | stale .executed stamp; notebook was regenerated after passing; stamp exists but current notebook has no executed cells |
| unverified | jax | `chapter_generative-adversarial-networks/gan.ipynb` |  | stale .executed stamp; notebook was regenerated after passing; stamp exists but current notebook has no executed cells |
| unverified | jax | `chapter_hyperparameter-optimization/hyperopt-api.ipynb` |  | stale .executed stamp; notebook was regenerated after passing; stamp exists but current notebook has no executed cells |
| unverified | jax | `chapter_hyperparameter-optimization/hyperopt-intro.ipynb` |  | stale .executed stamp; notebook was regenerated after passing; stamp exists but current notebook has no executed cells |
| unverified | jax | `chapter_hyperparameter-optimization/sh-intro.ipynb` |  | stale .executed stamp; notebook was regenerated after passing; stamp exists but current notebook has no executed cells |
| unverified | jax | `chapter_linear-classification/classification.ipynb` |  | stale .executed stamp; notebook was regenerated after passing; stamp exists but current notebook has no executed cells |
| unverified | jax | `chapter_linear-classification/image-classification-dataset.ipynb` |  | stale .executed stamp; notebook was regenerated after passing; stamp exists but current notebook has no executed cells |
| unverified | jax | `chapter_linear-classification/softmax-regression-concise.ipynb` |  | stale .executed stamp; notebook was regenerated after passing; stamp exists but current notebook has no executed cells |
| unverified | jax | `chapter_linear-classification/softmax-regression-scratch.ipynb` |  | stale .executed stamp; notebook was regenerated after passing; stamp exists but current notebook has no executed cells |
| unverified | jax | `chapter_linear-regression/linear-regression-concise.ipynb` |  | stale .executed stamp; notebook was regenerated after passing; stamp exists but current notebook has no executed cells |
| unverified | jax | `chapter_linear-regression/linear-regression-scratch.ipynb` |  | stale .executed stamp; notebook was regenerated after passing; stamp exists but current notebook has no executed cells |
| unverified | jax | `chapter_linear-regression/linear-regression.ipynb` |  | stale .executed stamp; notebook was regenerated after passing; stamp exists but current notebook has no executed cells |
| unverified | jax | `chapter_linear-regression/oo-design.ipynb` |  | stale .executed stamp; notebook was regenerated after passing; stamp exists but current notebook has no executed cells |
| unverified | jax | `chapter_linear-regression/synthetic-regression-data.ipynb` |  | stale .executed stamp; notebook was regenerated after passing; stamp exists but current notebook has no executed cells |
| unverified | jax | `chapter_linear-regression/weight-decay.ipynb` |  | stale .executed stamp; notebook was regenerated after passing; stamp exists but current notebook has no executed cells |
| unverified | jax | `chapter_multilayer-perceptrons/dropout.ipynb` |  | stale .executed stamp; notebook was regenerated after passing; stamp exists but current notebook has no executed cells |
| unverified | jax | `chapter_multilayer-perceptrons/kaggle-house-price.ipynb` |  | stale .executed stamp; notebook was regenerated after passing; stamp exists but current notebook has no executed cells |
| unverified | jax | `chapter_multilayer-perceptrons/mlp-implementation.ipynb` |  | stale .executed stamp; notebook was regenerated after passing; stamp exists but current notebook has no executed cells |
| unverified | jax | `chapter_multilayer-perceptrons/mlp.ipynb` |  | stale .executed stamp; notebook was regenerated after passing; stamp exists but current notebook has no executed cells |
| unverified | jax | `chapter_multilayer-perceptrons/numerical-stability-and-init.ipynb` |  | stale .executed stamp; notebook was regenerated after passing; stamp exists but current notebook has no executed cells |
| unverified | jax | `chapter_natural-language-processing-applications/natural-language-inference-and-dataset.ipynb` |  | stale .executed stamp; notebook was regenerated after passing; stamp exists but current notebook has no executed cells |
| unverified | jax | `chapter_natural-language-processing-applications/natural-language-inference-attention.ipynb` |  | stale .executed stamp; notebook was regenerated after passing; stamp exists but current notebook has no executed cells |
| unverified | jax | `chapter_natural-language-processing-applications/natural-language-inference-bert.ipynb` |  | stale .executed stamp; notebook was regenerated after passing; stamp exists but current notebook has no executed cells |
| unverified | jax | `chapter_natural-language-processing-applications/sentiment-analysis-and-dataset.ipynb` |  | stale .executed stamp; notebook was regenerated after passing; stamp exists but current notebook has no executed cells |
| unverified | jax | `chapter_natural-language-processing-applications/sentiment-analysis-cnn.ipynb` |  | stale .executed stamp; notebook was regenerated after passing; stamp exists but current notebook has no executed cells |
| unverified | jax | `chapter_natural-language-processing-applications/sentiment-analysis-rnn.ipynb` |  | stale .executed stamp; notebook was regenerated after passing; stamp exists but current notebook has no executed cells |
| unverified | jax | `chapter_natural-language-processing-pretraining/bert-dataset.ipynb` |  | stale .executed stamp; notebook was regenerated after passing; stamp exists but current notebook has no executed cells |
| unverified | jax | `chapter_natural-language-processing-pretraining/bert-pretraining.ipynb` |  | stale .executed stamp; notebook was regenerated after passing; stamp exists but current notebook has no executed cells |
| unverified | jax | `chapter_natural-language-processing-pretraining/bert.ipynb` |  | stale .executed stamp; notebook was regenerated after passing; stamp exists but current notebook has no executed cells |
| unverified | jax | `chapter_natural-language-processing-pretraining/similarity-analogy.ipynb` |  | stale .executed stamp; notebook was regenerated after passing; stamp exists but current notebook has no executed cells |
| unverified | jax | `chapter_natural-language-processing-pretraining/word-embedding-dataset.ipynb` |  | stale .executed stamp; notebook was regenerated after passing; stamp exists but current notebook has no executed cells |
| unverified | jax | `chapter_natural-language-processing-pretraining/word2vec-pretraining.ipynb` |  | stale .executed stamp; notebook was regenerated after passing; stamp exists but current notebook has no executed cells |
| unverified | jax | `chapter_optimization/adadelta.ipynb` |  | stale .executed stamp; notebook was regenerated after passing; stamp exists but current notebook has no executed cells |
| unverified | jax | `chapter_optimization/adagrad.ipynb` |  | stale .executed stamp; notebook was regenerated after passing; stamp exists but current notebook has no executed cells |
| unverified | jax | `chapter_optimization/adam.ipynb` |  | stale .executed stamp; notebook was regenerated after passing; stamp exists but current notebook has no executed cells |
| unverified | jax | `chapter_optimization/convexity.ipynb` |  | stale .executed stamp; notebook was regenerated after passing; stamp exists but current notebook has no executed cells |
| unverified | jax | `chapter_optimization/gd.ipynb` |  | stale .executed stamp; notebook was regenerated after passing; stamp exists but current notebook has no executed cells |
| unverified | jax | `chapter_optimization/lr-scheduler.ipynb` |  | stale .executed stamp; notebook was regenerated after passing; stamp exists but current notebook has no executed cells |
| unverified | jax | `chapter_optimization/minibatch-sgd.ipynb` |  | stale .executed stamp; notebook was regenerated after passing; stamp exists but current notebook has no executed cells |
| unverified | jax | `chapter_optimization/momentum.ipynb` |  | stale .executed stamp; notebook was regenerated after passing; stamp exists but current notebook has no executed cells |
| unverified | jax | `chapter_optimization/optimization-intro.ipynb` |  | stale .executed stamp; notebook was regenerated after passing; stamp exists but current notebook has no executed cells |
| unverified | jax | `chapter_optimization/rmsprop.ipynb` |  | stale .executed stamp; notebook was regenerated after passing; stamp exists but current notebook has no executed cells |
| unverified | jax | `chapter_optimization/sgd.ipynb` |  | stale .executed stamp; notebook was regenerated after passing; stamp exists but current notebook has no executed cells |
| unverified | jax | `chapter_preface/index.ipynb` |  | stale .executed stamp; notebook was regenerated after passing; stamp exists but current notebook has no executed cells |
| unverified | jax | `chapter_preliminaries/autograd.ipynb` |  | stale .executed stamp; notebook was regenerated after passing; stamp exists but current notebook has no executed cells |
| unverified | jax | `chapter_preliminaries/calculus.ipynb` |  | stale .executed stamp; notebook was regenerated after passing; stamp exists but current notebook has no executed cells |
| unverified | jax | `chapter_preliminaries/linear-algebra.ipynb` |  | stale .executed stamp; notebook was regenerated after passing; stamp exists but current notebook has no executed cells |
| unverified | jax | `chapter_preliminaries/lookup-api.ipynb` |  | stale .executed stamp; notebook was regenerated after passing; stamp exists but current notebook has no executed cells |
| unverified | jax | `chapter_preliminaries/ndarray.ipynb` |  | stale .executed stamp; notebook was regenerated after passing; stamp exists but current notebook has no executed cells |
| unverified | jax | `chapter_preliminaries/pandas.ipynb` |  | stale .executed stamp; notebook was regenerated after passing; stamp exists but current notebook has no executed cells |
| unverified | jax | `chapter_preliminaries/probability.ipynb` |  | stale .executed stamp; notebook was regenerated after passing; stamp exists but current notebook has no executed cells |
| unverified | jax | `chapter_recurrent-modern/bi-rnn.ipynb` |  | stale .executed stamp; notebook was regenerated after passing; stamp exists but current notebook has no executed cells |
| unverified | jax | `chapter_recurrent-modern/deep-rnn.ipynb` |  | stale .executed stamp; notebook was regenerated after passing; stamp exists but current notebook has no executed cells |
| unverified | jax | `chapter_recurrent-modern/encoder-decoder.ipynb` |  | stale .executed stamp; notebook was regenerated after passing; stamp exists but current notebook has no executed cells |
| unverified | jax | `chapter_recurrent-modern/gru.ipynb` |  | stale .executed stamp; notebook was regenerated after passing; stamp exists but current notebook has no executed cells |
| unverified | jax | `chapter_recurrent-modern/lstm.ipynb` |  | stale .executed stamp; notebook was regenerated after passing; stamp exists but current notebook has no executed cells |
| unverified | jax | `chapter_recurrent-modern/machine-translation-and-dataset.ipynb` |  | stale .executed stamp; notebook was regenerated after passing; stamp exists but current notebook has no executed cells |
| unverified | jax | `chapter_recurrent-neural-networks/language-model.ipynb` |  | stale .executed stamp; notebook was regenerated after passing; stamp exists but current notebook has no executed cells |
| unverified | jax | `chapter_recurrent-neural-networks/rnn-concise.ipynb` |  | stale .executed stamp; notebook was regenerated after passing; stamp exists but current notebook has no executed cells |
| unverified | jax | `chapter_recurrent-neural-networks/rnn-scratch.ipynb` |  | stale .executed stamp; notebook was regenerated after passing; stamp exists but current notebook has no executed cells |
| unverified | jax | `chapter_recurrent-neural-networks/rnn.ipynb` |  | stale .executed stamp; notebook was regenerated after passing; stamp exists but current notebook has no executed cells |
| unverified | jax | `chapter_recurrent-neural-networks/sequence.ipynb` |  | stale .executed stamp; notebook was regenerated after passing; stamp exists but current notebook has no executed cells |
| unverified | jax | `chapter_recurrent-neural-networks/text-sequence.ipynb` |  | stale .executed stamp; notebook was regenerated after passing; stamp exists but current notebook has no executed cells |
| unverified | mxnet | `chapter_attention-mechanisms-and-transformers/bahdanau-attention.ipynb` |  | missing .executed stamp; notebook has no passing execution record |
| unverified | mxnet | `chapter_attention-mechanisms-and-transformers/transformer.ipynb` |  | missing .executed stamp; notebook has no passing execution record |
| fail | mxnet | `chapter_builders-guide/use-gpu.ipynb` |  | 2 error output(s) |
| unverified | mxnet | `chapter_computational-performance/auto-parallelism.ipynb` |  | missing .executed stamp; notebook has no passing execution record |
| unverified | mxnet | `chapter_computational-performance/multiple-gpus-concise.ipynb` |  | missing .executed stamp; notebook has no passing execution record |
| unverified | mxnet | `chapter_computational-performance/multiple-gpus.ipynb` |  | missing .executed stamp; notebook has no passing execution record |
| unverified | mxnet | `chapter_computer-vision/fcn.ipynb` |  | missing .executed stamp; notebook has no passing execution record |
| unverified | mxnet | `chapter_computer-vision/fine-tuning.ipynb` |  | missing .executed stamp; notebook has no passing execution record |
| unverified | mxnet | `chapter_computer-vision/image-augmentation.ipynb` |  | missing .executed stamp; notebook has no passing execution record |
| unverified | mxnet | `chapter_computer-vision/kaggle-cifar10.ipynb` |  | missing .executed stamp; notebook has no passing execution record |
| unverified | mxnet | `chapter_computer-vision/kaggle-dog.ipynb` |  | missing .executed stamp; notebook has no passing execution record |
| unverified | mxnet | `chapter_computer-vision/neural-style.ipynb` |  | missing .executed stamp; notebook has no passing execution record |
| unverified | mxnet | `chapter_computer-vision/ssd.ipynb` |  | missing .executed stamp; notebook has no passing execution record |
| unverified | mxnet | `chapter_convolutional-modern/alexnet.ipynb` |  | missing .executed stamp; notebook has no passing execution record |
| unverified | mxnet | `chapter_convolutional-modern/batch-norm.ipynb` |  | missing .executed stamp; notebook has no passing execution record |
| unverified | mxnet | `chapter_convolutional-modern/cnn-design.ipynb` |  | missing .executed stamp; notebook has no passing execution record |
| unverified | mxnet | `chapter_convolutional-modern/densenet.ipynb` |  | missing .executed stamp; notebook has no passing execution record |
| unverified | mxnet | `chapter_convolutional-modern/googlenet.ipynb` |  | missing .executed stamp; notebook has no passing execution record |
| unverified | mxnet | `chapter_convolutional-modern/nin.ipynb` |  | missing .executed stamp; notebook has no passing execution record |
| unverified | mxnet | `chapter_convolutional-modern/resnet.ipynb` |  | missing .executed stamp; notebook has no passing execution record |
| unverified | mxnet | `chapter_convolutional-modern/vgg.ipynb` |  | missing .executed stamp; notebook has no passing execution record |
| unverified | mxnet | `chapter_convolutional-neural-networks/lenet.ipynb` |  | missing .executed stamp; notebook has no passing execution record |
| unverified | mxnet | `chapter_generative-adversarial-networks/dcgan.ipynb` |  | missing .executed stamp; notebook has no passing execution record |
| unverified | mxnet | `chapter_natural-language-processing-applications/natural-language-inference-attention.ipynb` |  | missing .executed stamp; notebook has no passing execution record |
| unverified | mxnet | `chapter_natural-language-processing-applications/natural-language-inference-bert.ipynb` |  | missing .executed stamp; notebook has no passing execution record |
| unverified | mxnet | `chapter_natural-language-processing-applications/sentiment-analysis-cnn.ipynb` |  | missing .executed stamp; notebook has no passing execution record |
| unverified | mxnet | `chapter_natural-language-processing-applications/sentiment-analysis-rnn.ipynb` |  | missing .executed stamp; notebook has no passing execution record |
| unverified | mxnet | `chapter_natural-language-processing-pretraining/bert-pretraining.ipynb` |  | missing .executed stamp; notebook has no passing execution record |
| unverified | mxnet | `chapter_natural-language-processing-pretraining/word2vec-pretraining.ipynb` |  | missing .executed stamp; notebook has no passing execution record |
| unverified | mxnet | `chapter_optimization/lr-scheduler.ipynb` |  | missing .executed stamp; notebook has no passing execution record |
| unverified | mxnet | `chapter_optimization/minibatch-sgd.ipynb` |  | missing .executed stamp; notebook has no passing execution record |
| unverified | mxnet | `chapter_recommender-systems/autorec.ipynb` |  | missing .executed stamp; notebook has no passing execution record |
| unverified | mxnet | `chapter_recommender-systems/deepfm.ipynb` |  | missing .executed stamp; notebook has no passing execution record |
| unverified | mxnet | `chapter_recommender-systems/fm.ipynb` |  | missing .executed stamp; notebook has no passing execution record |
| unverified | mxnet | `chapter_recommender-systems/mf.ipynb` |  | missing .executed stamp; notebook has no passing execution record |
| unverified | mxnet | `chapter_recommender-systems/neumf.ipynb` |  | missing .executed stamp; notebook has no passing execution record |
| unverified | mxnet | `chapter_recommender-systems/seqrec.ipynb` |  | missing .executed stamp; notebook has no passing execution record |
| unverified | mxnet | `chapter_recurrent-modern/deep-rnn.ipynb` |  | missing .executed stamp; notebook has no passing execution record |
| unverified | mxnet | `chapter_recurrent-modern/gru.ipynb` |  | missing .executed stamp; notebook has no passing execution record |
| unverified | mxnet | `chapter_recurrent-modern/lstm.ipynb` |  | missing .executed stamp; notebook has no passing execution record |
| unverified | mxnet | `chapter_recurrent-modern/seq2seq.ipynb` |  | missing .executed stamp; notebook has no passing execution record |
| unverified | mxnet | `chapter_recurrent-neural-networks/rnn-concise.ipynb` |  | missing .executed stamp; notebook has no passing execution record |
| unverified | mxnet | `chapter_recurrent-neural-networks/rnn-scratch.ipynb` |  | missing .executed stamp; notebook has no passing execution record |

## Passing Metrics Sample

| Framework | Notebook | Metrics |
|-----------|----------|---------|
| pytorch | `chapter_computational-performance/multiple-gpus-concise.ipynb` | test_acc=0.86 |
| pytorch | `chapter_computational-performance/multiple-gpus.ipynb` | test_acc=0.84 |
| tensorflow | `chapter_recurrent-modern/seq2seq.ipynb` | bleu=0.512 |
| jax | `chapter_computer-vision/image-augmentation.ipynb` | loss=0.159, train_acc=0.945, test_acc=0.818 |
| jax | `chapter_recurrent-modern/seq2seq.ipynb` | bleu=0.752 |
| mxnet | `chapter_generative-adversarial-networks/gan.ipynb` | loss_d=0.693, loss_g=0.694 |
