## ./chapter_reinforcement-learning/mdp.md
### pytorch
- [ ] Equation 17.2.2 is missing an expectation value over $a_0$ for the first term. (Link to comment: https://discuss.d2l.ai/t/12084/4)

## ./chapter_reinforcement-learning/qlearning.md
### pytorch
- [ ] Sign error in equation 17.3.3: the update should use a plus sign before the $\alpha$ term, i.e., $Q = (1-\alpha)Q + \alpha(\ldots)$, not a minus sign. (Link to comment: https://discuss.d2l.ai/t/12103/10)

## ./chapter_linear-classification/softmax-regression-scratch.md
### pytorch
- [ ] `cross_entropy` function uses `d2l.log(y_hat[...])` without clipping, so `log(0)` can occur if softmax outputs a zero probability, causing NaN. (Link to comment: https://discuss.d2l.ai/t/51/4)

## ./chapter_linear-classification/classification.md
### pytorch
- [ ] `configure_optimizers` should be added to the `Classifier` class instead of the base `d2l.Module` for better consistency. (Link to comment: https://discuss.d2l.ai/t/6809/6)

## ./chapter_attention-mechanisms-and-transformers/self-attention-and-positional-encoding.md
### pytorch
- [ ] `PositionalEncoding` implementation breaks if `num_hiddens` (encoding_dim) is an odd number due to shape mismatch between `torch.cos(X)` and `self.P[:, :, 1::2]`. (Link to comment: https://discuss.d2l.ai/t/1652/2)

## ./chapter_appendix-tools-for-deep-learning/jupyter.md
### general
- [ ] Missing instructions on how to compile labels so that equation and image numbers/links are displayed correctly within Jupyter notebooks. (Link to comment: https://discuss.d2l.ai/t/421/4)

## ./chapter_appendix-tools-for-deep-learning/colab.md
### general
- [ ] Missing suggestion/workaround to git clone the entire repository to avoid "No such file" errors when reading images in Colab. (Link to comment: https://discuss.d2l.ai/t/424/3)

## ./chapter_appendix-tools-for-deep-learning/sagemaker.md
### general
- [ ] Clarify the requirement for a credit card for non-students and mention the 24-hour waiting period for account activation (and note AWS Educate as an alternative for students). (Link to comment: https://discuss.d2l.ai/t/422/2)

## ./chapter_appendix-tools-for-deep-learning/contributing.md
### general
- [ ] Add a warning that `mxnet.py`, `torch.py`, and `tensorflow.py` are auto-generated and should not be edited directly. (Link to comment: https://discuss.d2l.ai/t/426/2)
- [ ] Provide guidance for Windows 10 users regarding `d2lbook` build library requirements (e.g., reference the GitHub issue or suggest seeking help). (Link to comment: https://discuss.d2l.ai/t/426/2)

## ./chapter_linear-classification/image-classification-dataset.md
### pytorch
- [ ] Default `num_workers` to 0 on Windows to avoid `RuntimeError`. (Link to comment: https://discuss.d2l.ai/t/49/2)
### tensorflow
- [ ] Shuffle before batching (`shuffle().batch()`) rather than after to ensure better randomization across label boundaries. (Link to comment: https://discuss.d2l.ai/t/224/5)

## ./chapter_linear-classification/environment-and-distribution-shift.md
### general
- [ ] Correct Equation 4.7.10 and the confusion matrix $C$ definition; specifically, columns of $C$ should sum to 1. (Link to comment: https://discuss.d2l.ai/t/105/16)
- [ ] Move section 4.7.3.1 before 4.6 as it defines "risk" and "empirical risk" used earlier. (Link to comment: https://discuss.d2l.ai/t/105/18)

## ./chapter_attention-mechanisms-and-transformers/vision-transformer.md
### pytorch
- [ ] Resolve discrepancy between Figure 11.8.1 and code regarding `<cls>` token placement and positional embedding addition order. (Link to comment: https://discuss.d2l.ai/t/8943/9)

## ./chapter_attention-mechanisms-and-transformers/transformer.md
### mxnet
- [ ] During inference, positional encoding is applied to each decoder token individually (num_steps=1), so every generated token receives the same position-0 encoding, causing positional encoding to fail at prediction time. (Link to comment: https://discuss.d2l.ai/t/348/15)

## ./chapter_computer-vision/transposed-conv.md
### pytorch
- [ ] The text and summary claim that a transposed convolutional layer with the same hyperparameters as the convolutional layer will always produce an output shape identical to the original input. However, for strides > 1, multiple input shapes can map to the same output shape in convolution, meaning the transposed convolution shape is not guaranteed to match without additional parameters (like `output_padding`). (Link to comment: https://discuss.d2l.ai/t/1450/2)

### jax
- [ ] The text and summary claim that a transposed convolutional layer with the same hyperparameters as the convolutional layer will always produce an output shape identical to the original input. However, for strides > 1, multiple input shapes can map to the same output shape in convolution, meaning the transposed convolution shape is not guaranteed to match. (Link to comment: https://discuss.d2l.ai/t/1450/2)

## ./chapter_computer-vision/semantic-segmentation-and-dataset.md
### pytorch
- [ ] Usage of `torchvision.io.image.ImageReadMode` causes `AttributeError` on torchvision versions prior to 0.9.0; consider noting the minimum required version. (Link to comment: https://discuss.d2l.ai/t/1480/2)

## ./chapter_convolutional-neural-networks/padding-and-strides.md
### tensorflow
- [ ] The code comment in the "slightly more complicated example" for TensorFlow uses `padding='valid'` but the corresponding comments in other frameworks mention specific padding numbers; a clarifying note that `'valid'` means no padding would prevent confusion. (Link to comment: https://discuss.d2l.ai/t/272/2)
### pytorch
- [ ] Improve the explanation of even kernel size padding distribution to avoid the "padding half a row" misinterpretation. (Link to comment: https://discuss.d2l.ai/t/68/6)

## ./chapter_builders-guide/custom-layer.md
### pytorch
- [ ] Avoid using `.data` (e.g., `self.weight.data`) in the `MyLinear` forward pass as it can interfere with gradient tracking. (Link to comment: https://discuss.d2l.ai/t/59/3)
- [ ] Rename `MyLinear` class to `MyDense` to maintain consistency with the surrounding text and other frameworks. (Link to comment: https://discuss.d2l.ai/t/59/2)

## ./chapter_builders-guide/parameters.md
### pytorch
- [ ] Use `assert` statements to programmatically verify tied parameters instead of relying on visual print inspections. (Link to comment: https://discuss.d2l.ai/t/57/13)

## ./chapter_appendix-mathematics-for-deep-learning/single-variable-calculus.md
### pytorch
- [ ] Add intermediate steps to the derivation of Equation 18.3.10 to clarify the use of the chain rule. (Link to comment: https://discuss.d2l.ai/t/1088)

## ./chapter_recurrent-modern/seq2seq.md
### pytorch
- [ ] The `predict_step` function does not call `self.eval()`, which is necessary to correctly handle layers like dropout during inference. (Link to comment: https://discuss.d2l.ai/t/1062/6)

## ./chapter_recurrent-modern/machine-translation-and-dataset.md
### general
- [ ] The `pad_or_trim` lambda truncates sequences with `seq[:t]`, which can silently drop the `<eos>` token; truncation should preserve `<eos>` (e.g., `seq[:t-1] + ['<eos>']`). (Link to comment: https://discuss.d2l.ai/t/1060/7)

## ./chapter_attention-mechanisms-and-transformers/multihead-attention.md
### mxnet
- [ ] The comment in `forward` assumes input feature dimension is `num_hiddens`, but it could be `query_size`, `key_size`, or `value_size`. (Link to comment: https://discuss.d2l.ai/t/1634/10)
### pytorch
- [ ] The comment in `forward` assumes input feature dimension is `num_hiddens`, but it could be `query_size`, `key_size`, or `value_size`. (Link to comment: https://discuss.d2l.ai/t/1635/10)
### tensorflow
- [ ] The comment in `call` assumes input feature dimension is `num_hiddens`, but it could be `query_size`, `key_size`, or `value_size`. (Link to comment: https://discuss.d2l.ai/t/3869/10)
### jax
- [ ] The comment in `__call__` assumes input feature dimension is `num_hiddens`, but it could be `query_size`, `key_size`, or `value_size`. (Link to comment: https://discuss.d2l.ai/t/18029/10)

## ./chapter_recommender-systems/autorec.md
### mxnet
- [ ] `test_iter` incorrectly uses `train_inter_mat` instead of `test_inter_mat`. (Link to comment: https://discuss.d2l.ai/t/401/2)
### pytorch
- [ ] `test_iter` incorrectly uses `train_inter_mat_t` instead of a test-specific tensor. (Link to comment: https://discuss.d2l.ai/t/401/2)

## ./chapter_recommender-systems/fm.md
### general
- [ ] `d2l.accuracy` is unsuitable for binary classification in FM models because it compares probabilities/logits directly to binary labels after casting, leading to incorrect accuracy metrics. (Link to comment: https://discuss.d2l.ai/t/406/5)

## ./chapter_recommender-systems/deepfm.md
### general
- [ ] Potential bug in embedding layer initialization and forward pass if field-specific indices are not properly offset before being passed to a single large embedding matrix. (Link to comment: https://discuss.d2l.ai/t/407/10)

## ./chapter_generative-adversarial-networks/dcgan.md
### pytorch
- [ ] In section 17.2.2 (The Generator), the text "With a input shape of $n_h^{'} \times n_w^{'} = 16 \times 16$" might be more consistent using $n_h \times n_w$ for input dimensions, reserving the prime notation for output. (Link to comment: https://discuss.d2l.ai/t/1083/3)

## ./chapter_generative-adversarial-networks/gan.md
### pytorch
- [ ] The text says "If the generator does a perfect job, then $D(\mathbf x')\approx 1$, so the above loss is near 0, which results in the gradients that are too small to make good progress for the discriminator." This is factually wrong: if the discriminator does a perfect job at detecting fakes, $D(G(\mathbf{z})) \approx 0$ (not 1), making $-\log(1-D(G(\mathbf{z})))$ near 0 with small gradients. The text confuses generator success with discriminator success. (Link to comment: https://discuss.d2l.ai/t/1082/13)
- [ ] The explanation of vanishing gradients for the generator is slightly confusing: "If the generator does a perfect job, then $D(\mathbf x')\approx 1$, so the above loss is near 0, which results in the gradients that are too small to make good progress for the discriminator." Usually, the problem is the generator not receiving enough gradient when the discriminator is too strong early in training. (Link to comment: https://discuss.d2l.ai/t/1082/9)

## ./chapter_natural-language-processing-pretraining/similarity-analogy.md
### pytorch
- [ ] In the `knn` function, the return line `return topk, [cos[int(i)] for i in topk]` is redundant. Since `torch.topk` returns both values and indices, it can be simplified to `vals, topk = torch.topk(cos, k=k)` and `return topk, vals`. (Link to comment: https://discuss.d2l.ai/t/1336/2)

## ./chapter_natural-language-processing-pretraining/bert.md
### pytorch
- [ ] The standalone demo code for Next Sentence Prediction (NSP) uses `torch.flatten(encoded_X, start_dim=1)` which flattens the entire sequence. Standard BERT NSP only uses the `<cls>` token. While the `BERTModel` implementation later in the page correctly uses `encoded_X[:, 0, :]`, the middle demo code is inconsistent. (Link to comment: https://discuss.d2l.ai/t/1490/2)

## ./chapter_appendix-mathematics-for-deep-learning/information-theory.md
### pytorch
- [ ] Redundant expectation operator in mutual information formula :eqlabel:`eq_mut_ent_def`. The term $p_{X, Y}(x, y)$ is already inside the expectations $E_x E_y$, which is notationally redundant or incorrect. It should be $E_{(x, y) \sim P} [ \log \dots ]$ or a double sum. (Link to comment: https://discuss.d2l.ai/t/1104/6)

## ./chapter_appendix-mathematics-for-deep-learning/random-variables.md
### pytorch
- [ ] The second property of correlation is missing the `sign(a)` term. It should be $\rho(aX+b, Y) = \text{sign}(a)\rho(X, Y)$ to account for negative $a$. (Link to comment: https://discuss.d2l.ai/t/1094/4)
- [ ] The Cauchy distribution probability density function is missing the normalization constant $1/\pi$. It should be $p(x) = \frac{1}{\pi(1+x^2)}$. (Link to comment: https://discuss.d2l.ai/t/1094/2)

## ./chapter_appendix-mathematics-for-deep-learning/multivariable-calculus.md
### mxnet
- [ ] Typo in the first line of the multi-variate differentiation derivation block. The partial derivative term contains $w_N+\epsilon_N$ while the left-hand side of the approximation only goes up to $w_N$. (Link to comment: https://discuss.d2l.ai/t/413/6)

## ./chapter_optimization/sgd.md
### pytorch
- [ ] Typo in the convergence bound formula: the left-hand side is written as `\left[E[\bar{\mathbf{x}}]\right] - R^*` but should be `E[R(\bar{\mathbf{x}})] - R^*` to represent the expected risk gap. (Link to comment: https://discuss.d2l.ai/t/497/5)

## ./chapter_convolutional-modern/resnet.md
### pytorch
- [ ] In the `ResNeXtBlock` class, `groups=bot_channels//groups` should be `groups=groups` to correctly implement grouped convolution. (Link to comment: https://discuss.d2l.ai/t/86/16)
- [ ] The `Residual` block silently produces wrong output shapes when `strides>1` but `use_1x1conv=False`; consider auto-enabling `use_1x1conv` when `strides>1`. (Link to comment: https://discuss.d2l.ai/t/86/18)

## ./chapter_multilayer-perceptrons/kaggle-house-price.md
### mxnet
- [ ] Standardization is applied jointly to train and test features, leaking test-set statistics into training; mean and std should be computed on training data only and then applied to the test set. (Link to comment: https://discuss.d2l.ai/t/106/2)

## ./chapter_attention-mechanisms-and-transformers/attention-scoring-functions.md
### pytorch
- [ ] In `masked_softmax`, the logic for `valid_lens` reshaping and `_sequence_mask` could be improved to handle 1D and 2D `valid_lens` more consistently across different input shapes of `X`. (Link to comment: https://discuss.d2l.ai/t/1064/2)

## ./chapter_recommender-systems/neumf.md
### mxnet
- [ ] Bug in `evaluate_ranking`: `all_items = set([i for i in range(num_users)])` should be `all_items = set([i for i in range(num_items)])`. (Link to comment: https://discuss.d2l.ai/t/403/2)
- [ ] Text says "candidate items" refers to unobserved items, but the `candidates` variable in code holds the rated (observed) items — contradictory naming/description. (Link to comment: https://discuss.d2l.ai/t/403/3)
### pytorch
- [ ] Bug in `evaluate_ranking`: `all_items = set([i for i in range(num_users)])` should be `all_items = set([i for i in range(num_items)])`. (Link to comment: https://discuss.d2l.ai/t/403/2)
- [ ] Text says "candidate items" refers to unobserved items, but the `candidates` variable in code holds the rated (observed) items — contradictory naming/description. (Link to comment: https://discuss.d2l.ai/t/403/3)

## ./chapter_computer-vision/anchor.md
### pytorch
- [ ] Typo in `multibox_prior`: The `* in_height / in_width` in the anchor width calculation may cause values outside [0, 1] when height exceeds width. (Link to comment: https://discuss.d2l.ai/t/1603/16)
### jax
- [ ] Typo in `multibox_prior`: The `* in_height / in_width` in the anchor width calculation may cause values outside [0, 1] when height exceeds width. (Link to comment: https://discuss.d2l.ai/t/1603/16)

## ./chapter_recurrent-neural-networks/rnn-scratch.md
### tensorflow
- [ ] The training result might be abnormal (high perplexity) unless `Y` is transposed to match the transposed `X`. (Link to comment: https://discuss.d2l.ai/t/1052/5)

## ./chapter_gaussian-processes/gp-priors.md
### pytorch
- [ ] The visualization for the neural network kernel appears to be missing from the page. (Link to comment: https://discuss.d2l.ai/t/12116/2)
- [ ] The text states "Since Gaussians are closed under addition, f(x) is also a Gaussian" but should note that w0 and w1 must be assumed independent (or jointly Gaussian) for their linear combination to be Gaussian. (Link to comment: https://discuss.d2l.ai/t/12116/3)

## ./chapter_natural-language-processing-pretraining/bert-pretraining.md
### pytorch
- [ ] The loss calculation in `_get_batch_loss_bert()` uses `loss = nn.CrossEntropyLoss()` with default `reduction='mean'`, then multiplies the scalar result by `mlm_weights_X.reshape(-1, 1)`. This makes the masking ineffective — the result is identical to the unmasked mean loss. It should use `reduction='none'` so per-token losses can be properly masked by `mlm_weights_X.reshape(-1)` before summing. (Link to comment: https://discuss.d2l.ai/t/1497/2)

## ./chapter_hyperparameter-optimization/hyperopt-intro.md
### pytorch
- [ ] Mismatch between code `stats.loguniform(1e-4, 1)` and text "uniform distribution between -4 and -1 in the logarithmic space". The upper bound $10^0=1$ corresponds to $0$ in log space, not $-1$; fix either the code to `stats.loguniform(1e-4, 0.1)` or the text to "between -4 and 0". (Link to comment: https://discuss.d2l.ai/t/12090/4)

## ./chapter_convolutional-neural-networks/lenet.md
### pytorch
- [ ] The implementation uses `nn.Sigmoid` whereas the original LeNet-5 architecture used `Tanh`; the text should clarify this deviation or switch to `nn.Tanh`. (Link to comment: https://discuss.d2l.ai/t/74/15)

## ./chapter_recurrent-neural-networks/rnn.md
### mxnet
- [ ] The text says perplexity is the "harmonic mean" of the number of real choices, but mathematically it is the geometric mean. (Link to comment: https://discuss.d2l.ai/t/337/12)

## ./chapter_computational-performance/auto-parallelism.md
### pytorch
- [ ] Reported typo/error in Fig. 12.3.1 (twogpu): the 3rd and 6th blue boxes show weight updates using unaggregated gradients from GPU0 only, but they should use aggregated gradients. (Link to comment: https://discuss.d2l.ai/t/1681/3)

## ./chapter_appendix-mathematics-for-deep-learning/eigendecomposition.md
### pytorch
- [ ] The stretching factor of 1.97 mentioned in the text is specific to the MXNet seed; for the PyTorch seed (42), the value is 2.45, which may confuse users. (Link to comment: https://discuss.d2l.ai/t/1086/2)
### tensorflow
- [ ] The TensorFlow eigenvalue computation uses `tf.linalg.eigh` instead of `tf.linalg.eig` for the random (non-symmetric) matrix in the "Relating Back to Eigenvectors" section, yielding incorrect results for general matrices. (Link to comment: https://discuss.d2l.ai/t/1087/3)

## ./chapter_computer-vision/image-augmentation.md
### mxnet
- [ ] In `train_ch13`, the fourth metric label in the comment says "no. of predictions" but it tracks `labels.size` which equals the number of examples (same as `labels.shape[0]` for a 1D label vector); the comment and metric legend should say "no. of examples" consistently. (Link to comment: https://discuss.d2l.ai/t/367/7)
- [ ] `RandomRotation` in Gluon may throw a `TypeError` even when called after `ToTensor`. (Link to comment: https://discuss.d2l.ai/t/367/10)

## ./chapter_recurrent-neural-networks/text-sequence.md
### pytorch
- [ ] In `seq_data_iter_sequential`, `random.randint(0, num_steps)` should be `random.randint(0, num_steps - 1)` since Python's `random.randint` is inclusive on both ends. (Link to comment: https://discuss.d2l.ai/t/118/2)

## ./chapter_natural-language-processing-pretraining/glove.md
### general
- [ ] In the GloVe mathematical derivation, the ratio $p_{ij}/p_{ik}$ should be corrected to $p_{ji}/p_{ki}$ where $w_j$ and $w_k$ are center words. (Link to comment: https://discuss.d2l.ai/t/385/6)

## ./chapter_convolutional-neural-networks/why-conv.md
### general
- [ ] Clarify the bias term `u` in formulas 6.1.1 and 6.1.2 to avoid confusion with indexed terms like `u[i,j]`. (Link to comment: https://discuss.d2l.ai/t/64/7)

## ./chapter_computational-performance/parameterserver.md
### general
- [ ] Clarify the synchronization time calculation for 8 V100 GPUs, specifically the factor of 3 in the denominator representing the aggregate bandwidth of the two rings. (Link to comment: https://discuss.d2l.ai/t/366/5)
- [ ] Clarify that the ring synchronization algorithm consists of two phases (reduce-scatter and all-gather) to ensure each GPU has the full aggregated gradient. (Link to comment: https://discuss.d2l.ai/t/366/4)

## ./chapter_appendix-mathematics-for-deep-learning/distributions.md
### pytorch
- [ ] Clarify the definition of $k$ in the Discrete Uniform and Binomial/Poisson sections to avoid confusion. (Link to comment: https://discuss.d2l.ai/t/1098/2)

## ./chapter_appendix-mathematics-for-deep-learning/geometry-linear-algebraic-ops.md
### pytorch
- [ ] The figure `fig_vector-project` (proj-vec.svg) is reported to show the projected line intersecting the y-axis at (0, 2) instead of the correct (0, 1) for the equation y = 1 − 2x. (Link to comment: https://discuss.d2l.ai/t/1084/17)
- [ ] The text says "any of the four collections of three columns are dependent" for matrix C which has 5 columns, but there are 5C3=10 possible collections of 3 columns, not 4; the claim should be clarified (e.g., state that *any* collection of three or more columns is dependent, or provide the correct count). (Link to comment: https://discuss.d2l.ai/t/1084/11)

## ./chapter_computational-performance/hardware.md
### general
- [ ] Consider adding a brief introduction to RDMA (Remote Direct Memory Access) in the Networks and Buses section. (Link to comment: https://discuss.d2l.ai/t/363/2)

## ./chapter_computational-performance/hybridize.md
### pytorch
- [ ] The text asserts that torchscript "computing performance is improved" unconditionally, but users report no speedup on CPU for this small MLP example; a caveat should be added that speedup is most visible on GPU or with more compute-intensive models. (Link to comment: https://discuss.d2l.ai/t/2490/2)

## ./chapter_convolutional-modern/alexnet.md
### pytorch
- [ ] `self.net.apply(d2l.init_cnn)` in the `AlexNet.__init__` has no effect because lazy layers are not yet initialized at construction time; a note or fix should clarify that weight initialization must happen after a forward pass (e.g., via `apply_init`). (Link to comment: https://discuss.d2l.ai/t/76/16)

## ./chapter_convolutional-modern/densenet.md
### mxnet
- [ ] The Taylor expansion notation is ambiguous: the text says "For the point $x = 0$ it can be written as $f(x) = f(0) + \ldots$" but $x$ serves as both the expansion point and the free variable, which is confusing; consider using a separate symbol (e.g., $a=0$) for the expansion point. (Link to comment: https://discuss.d2l.ai/t/87/4)

## ./chapter_convolutional-neural-networks/conv-layer.md
### tensorflow
- [ ] The TensorFlow "Learning a Kernel" code block has a redundant `Y_hat = conv2d(X)` call on the line before the for-loop (line 401 in the .md source); it is immediately recomputed inside the loop and can be removed. (Link to comment: https://discuss.d2l.ai/t/271/3)

## ./chapter_convolutional-neural-networks/pooling.md
### pytorch
- [ ] The text says "shift the whole image by one pixel to the right, i.e., `Z[i, j] = X[i, j + 1]`", but `Z[i,j]=X[i,j+1]` shifts the content leftward (not rightward); the equation should be `Z[i, j] = X[i, j - 1]` for a rightward shift. (Link to comment: https://discuss.d2l.ai/t/72/5)

## ./chapter_installation/index.md
### tensorflow
- [ ] The download/unzip commands use `rm d2l-en.zip` but Windows users need `del d2l-en.zip`; add a Windows-specific note or variant for each framework tab. (Link to comment: https://discuss.d2l.ai/t/436/13)

## ./chapter_linear-regression/generalization.md
### pytorch
- [ ] The paragraph "While ideally we would only touch the test data once, to assess the very best model or to compare a small number of models with each other" uses "test data" where it should say "validation data", since assessing/comparing models is the role of the validation set, not the test set. (Link to comment: https://discuss.d2l.ai/t/97/2)

## ./chapter_linear-regression/linear-regression-concise.md
### pytorch
- [ ] Exercise 1 asks how to change the learning rate when replacing "aggregate loss" with "average over the loss", but `nn.MSELoss()` already uses mean reduction by default, making the question misleading or ill-posed; the exercise needs to be reworded or corrected. (Link to comment: https://discuss.d2l.ai/t/45/2)

## ./chapter_linear-regression/linear-regression.md
### pytorch
- [ ] The text says "maximize the *likelihood* of the entire dataset" but likelihood is technically a function of the parameters, not the data; the phrasing should be "maximize the likelihood function" or "maximize the likelihood of the parameters". (Link to comment: https://discuss.d2l.ai/t/258/11)

## ./chapter_optimization/momentum.md
### mxnet
- [ ] The gradient formula `∂_x h(x) = Q(x − Q^{-1}c)` is incorrect: expanding gives `Qx − c`, but the true derivative of `h` is `Qx + c`; it should read `Q(x − x*)` where `x* = −Q^{-1}c`, or equivalently be written as `Qx + c`. (Link to comment: https://discuss.d2l.ai/t/354/2)

## ./chapter_preliminaries/calculus.md
### mxnet
- [ ] The use of `∇_x` notation for `Ax` and `x^T A` (which are R^n→R^m functions, not R^n→R scalars) confuses readers expecting gradient (scalar-output) notation; multiple users requested clarification or a note about the denominator-layout convention being used. (Link to comment: https://discuss.d2l.ai/t/32/18)

## ./chapter_preliminaries/pandas.md
### mxnet
- [ ] `inputs.fillna(inputs.mean())` raises a FutureWarning in newer pandas versions because `numeric_only` defaults to `None`; fix by using `inputs.fillna(inputs.mean(numeric_only=True))`. (Link to comment: https://discuss.d2l.ai/t/28/5)

## ./chapter_recommender-systems/ranking.md
### general
- [ ] The hinge loss link pointing to the MXNet incubator API is broken; replace with an updated or generic reference. (Link to comment: https://discuss.d2l.ai/t/402/4)

## ./chapter_recurrent-modern/beam-search.md
### general
- [ ] The text says "$L^\alpha$ in the denominator penalizes long sequences," but since log-probabilities are negative, dividing by $L^\alpha$ actually makes the score less negative (compensates for longer sequences); the wording should be clarified to avoid confusion. (Link to comment: https://discuss.d2l.ai/t/338/5)

## ./chapter_recurrent-modern/gru.md
### pytorch
- [ ] The GRU section declares biases as $\mathbb{R}^{1 \times h}$ but the sum produces $\mathbb{R}^{n \times h}$ results; the text should note that broadcasting is applied (as the LSTM section already does). (Link to comment: https://discuss.d2l.ai/t/1056/2)
