# Writing pass: part and chapter introductions

Files: `chapter_basics/index.md` and the `index.md` of chapters 1, 2, 4, 5, 6, 7, 8. Chapter 3 (`chapter_linear-classification/index.md`) is the approved model and was not edited.

Only the text between the label and the `toc` block was changed. Each file passes `check_prose_only.py` ("OK: only prose changed") and `tools/lint_source.py`; every `:numref:` label used exists. Word counts are of the body before the `toc` block, with each reference counted as one word and the figure caption excluded.

## Decisions across the introductions

- **Legendre and Gauss** are told in the chapter 2 introduction only (it has the dates, Ceres and the citation). The part introduction keeps one clause and a pointer.
- **The garment** is introduced in the part introduction. Chapters 1, 4, 6 and 8 recall it in one clause.
- **History paragraphs** now open nine different ways: a recipe (part), a date (ch. 1), the model then astronomy (ch. 2), statisticians and Fisher (ch. 3, unchanged), a failure (ch. 4), early programs (ch. 5), the visual cortex (ch. 6), two numbers (ch. 7), a game (ch. 8). None uses "The question is older than".
- **Routes** group sections into movements and no longer use "asks" as the repeated verb. Every section label that was referenced is still referenced.
- After reading all nine in order, two routes that ended on "The last N sections ..." (ch. 2 and 7, against ch. 3 and 5) were reworded, and ch. 8's "Three sections ..." was changed because ch. 1 and ch. 6 use the same opening.

## Part introduction (Basics)

`chapter_basics/index.md`, 360 words.

**History paragraph, old first sentence:** The core of the answer is older than computers.

**New first sentence:** Our answer is a recipe with three ingredients: a model with parameters, a loss that measures the model's errors, and an algorithm that reduces the loss by changing the parameters.

**New route:**

> The first three chapters teach the recipe on models small enough to
> understand completely. :numref:`chap_preliminaries` collects the tools:
> tensors, derivatives, and enough probability to reason about uncertainty.
> :numref:`chap_regression` fits a linear model to numbers and asks whether
> the fit will hold for new data, and :numref:`chap_classification` does the
> same for categories, which is where the garments return. The next two
> chapters make the model deep. :numref:`chap_perceptrons` adds hidden layers
> and deals with the troubles they bring, and :numref:`chap_computation`
> turns to the software that assembles such a model, saves it, and runs it on
> a GPU. The last three chapters build what we know about the data into the
> model. Pixels have neighbors, which leads to convolutional networks
> (:numref:`chap_cnn`) and to the ideas that let them grow to more than a
> hundred layers (:numref:`chap_modern_cnn`). Words come in order, which
> leads to models of text and other sequences (:numref:`chap_rnn`).

**Claims added or changed:**

- Legendre and Gauss removed; replaced by one clause, "The recipe itself is two centuries old, and astronomers used it first", with a pointer to the chapter 2 introduction, which keeps the story.
- Added: "so that nobody has to derive them by hand" (of automatic differentiation).
- Added judgement: the first three chapters use "models small enough to understand completely".
- First appearance of the garment example; kept here in full.

## Chapter 1, Preliminaries

`chapter_preliminaries/index.md`, 420 words.

**History paragraph, old first sentence:** The mathematics behind these steps was not invented for deep learning. (mid-paragraph; the paragraph opened "It would store the photographs ...")

**New first sentence:** Probability began in 1654, in letters between Pascal and Fermat about a game of chance, and calculus is more than three centuries old.

**New route:**

> The chapter takes the program's needs in order. Two sections are about data:
> how to store it in tensors and compute with it (:numref:`sec_ndarray`), and
> how to get there from a table of real records, with its missing entries and
> its text categories (:numref:`sec_pandas`). Three sections follow a
> prediction from a weighted sum to a gradient: linear algebra for the
> prediction and for the size of its error (:numref:`sec_linear-algebra`),
> calculus for the direction in which the loss falls (:numref:`sec_calculus`),
> and automatic differentiation, which lets the framework compute the
> derivatives for us (:numref:`sec_autograd`). :numref:`sec_prob` is about
> doubt: how evidence updates a probability, and when a measured difference is
> larger than sampling noise. Finally, :numref:`sec_lookup_api` shows how to
> find out what a function does when the book does not say.

**Claims added or changed:**

- Garment example reduced to one clause ("the program that sorts photographs of garments"); the list of categories is gone.
- The history is now its own paragraph; the "It would store ..." paragraph that the review asked to keep is unchanged.
- Added reading advice (review, structural finding 2): the chapter is long; readers who know NumPy and calculus can skim, but should read sec_autograd and the standard error in sec_prob. Checked: probability.md has "The Standard Error of a Frequency".
- Dropped from the route: norms and "repeated multiplication by a matrix" for sec_linear-algebra (now "the prediction and the size of its error"), and the minibatch example for sec_prob (review IDX-2).
- Dropped: "together with the intuition behind each result".

## Chapter 2, Linear Regression

`chapter_linear-regression/index.md`, 417 words.

**History paragraph, old first sentence:** Legendre published the method in 1805 to compute the orbits of comets. (mid-paragraph; the paragraph opens "Linear regression answers with the simplest rule", unchanged)

**New first sentence:** The method comes from astronomy. Legendre published it in 1805 to compute the orbits of comets.

**New route:**

> The mathematics comes first and takes one section:
> :numref:`sec_linear_regression` derives the model, its loss, and minibatch
> stochastic gradient descent, the algorithm that trains it. The next four
> sections are about code. We set up a small design of classes that later
> chapters reuse, so that each of them changes only the part under study
> (:numref:`sec_oo-design`), generate data whose true parameters we know
> (:numref:`sec_synthetic-regression-data`), and train the model twice: once
> with every part written by hand and checked (:numref:`sec_linear_scratch`),
> and once with the layers, losses, and optimizers of a framework
> (:numref:`sec_linear_concise`). With an implementation that works, we return
> to the question of trust. :numref:`sec_generalization_basics` asks when a
> model that fits its training data predicts new data well, and how to choose
> among models. :numref:`sec_weight_decay` shows how to limit a model's
> flexibility by degrees, rather than by dropping features.

**Claims added or changed:**

- Legendre, Gauss, Ceres and the citation are kept here (the one home of the story among the introductions); wording of the facts unchanged.
- "Linear regression contains every part of learning in its simplest form: ..." became "Simple as it is, linear regression is a complete learning method, with ...", to avoid repeating the part introduction's recipe sentence.
- Added: "Neither failure announces itself" (of bad generalization and silent implementation errors).

## Chapter 4, Multilayer Perceptron

`chapter_multilayer-perceptrons/index.md`, 434 words.

**History paragraph, old first sentence:** The question is older than deep learning.

**New first sentence:** The trouble starts with a much smaller example than a sleeve: no threshold on a weighted sum computes the exclusive or of two bits.

**New route:**

> We begin by building the model: :numref:`sec_mlp` shows which functions a
> network with hidden layers can represent and why it needs a nonlinearity, and
> :numref:`sec_mlp-implementation` trains one on the clothing images to see
> what it gains over softmax regression and what it costs. The three problems
> then get a section each, in that order (:numref:`sec_backprop`,
> :numref:`sec_numerical_stability`, :numref:`sec_generalization_deep`).
> Dropout (:numref:`sec_dropout`) is a regularizer made for hidden units. To
> finish, :numref:`sec_kaggle_house` takes the whole toolkit away from images,
> to a Kaggle competition on house prices.

**Claims added or changed:**

- Garment example reduced: "we taught a computer to recognize clothing" and "Fashion-MNIST image" became "the 784 pixels of a clothing image".
- Question reworded per review IDX-2: "a model in which the evidence from one input depends on the others".
- History reordered to start from the failure (Minsky and Papert 1969, then Rosenblatt 1958). Same facts. Sharpened: "no threshold on a weighted sum computes the exclusive or" (was "what such classifiers cannot compute; the simplest example is the exclusive or").
- Added per review IDX-4: "such as the product of the two bits" (XOR = a + b - 2ab is linear in a, b, ab), and "somebody has to supply the features".
- Added judgement: "The first two problems have satisfactory answers. The third is the deepest, and what we can say about it is still incomplete." (generalization-deep.md lists a predictive theory of test error as open.)
- Added prerequisites sentence (review, structural finding 2): sec_generalization_basics and sec_weight_decay.
- Dropout placed in one clause ("a regularizer made for hidden units").

## Chapter 5, Computation

`chapter_builders-guide/index.md`, 427 words.

**History paragraph, old first sentence:** What happens to a model between its definition and a trained result is a matter of software.

**New first sentence:** Early neural network programs derived and coded the gradients of each model by hand.

**New route:**

> The chapter builds a model, then chooses the numbers it computes in, then
> runs it. Building takes four sections: how a model is assembled from modules
> (:numref:`sec_model_construction`), what state it holds and how much memory
> training it needs (:numref:`sec_parameters`), how to set that state before
> training (:numref:`sec_init_param`, which continues the analysis of
> :numref:`sec_numerical_stability`), and how to add a layer that the library
> lacks (:numref:`sec_custom_layer`). :numref:`sec_numerics` is about number
> formats: what a model saves by computing in 16 bits instead of 32, and what
> it can lose without any error message. The last three sections run the model.
> We save it and load weights trained by others (:numref:`sec_read_write`),
> place it on a GPU and account for the memory (:numref:`sec_use_gpu`), and ask
> when two runs of the same code give the same result (:numref:`sec_repro`).

**Claims added or changed:**

- Added the running example: "Take a language model with 8 billion parameters, to which most sections of this chapter return. Its weights alone take 32 GB in the 32-bit format we have used so far." git grep finds the 8-billion model in model-construction, parameters-state-memory, custom-layers, numerics, saving-loading and gpus-devices-memory (6 of 8 sections); 32 GB in fp32 matches parameters-state-memory.md.
- Removed the sentence "What happens to a model ... is a matter of software" (review: not true of the initialization section); the route now says that sec_init_param continues the analysis of sec_numerical_stability.
- Failure list cut from four to two (review IDX-2); number format and device moved to the route.
- Added judgement: "This is the one idea to take from the chapter" (the module tree) and "which makes it tempting to skip. We advise against that".
- Dropped from the route: "a differentiable function" (custom layers) and "how to look inside a running model" (sec_repro); both remain in the sections and the toc.

## Chapter 6, Convolutional Neural Networks

`chapter_convolutional-neural-networks/index.md`, 494 words.

**History paragraph, old first sentence:** The question is older than deep learning.

**New first sentence:** The visual cortex offers a hint.

**New route:**

> We derive the layer before we use it. :numref:`sec_why-conv` starts from the
> two facts and finds that they leave little choice: a fully connected layer
> made to respect both is a convolution. :numref:`sec_conv_layer` computes with
> the layer and lets it learn an edge detector from data. Three sections then
> supply the controls of a practical layer: padding, stride, and dilation,
> which set the size of the output and how far each output sees
> (:numref:`sec_padding`); channels, several in and several out
> (:numref:`sec_channels`); and pooling, which summarizes a neighborhood so
> that small shifts do not matter (:numref:`sec_pooling`). :numref:`sec_lenet`
> assembles a complete convolutional classifier and compares it with the MLP on
> the clothing images.

**Claims added or changed:**

- First paragraph, the figure field_visual and its paragraph are kept (the figure paragraph only gained the new first sentence).
- Claim removed (review IDX-3): "and far less data". The chapter shows the parameter saving only.
- The terms *locality* and *translation equivariance* are no longer named in the introduction (review IDX-2); their content is stated ("Each output then depends only on nearby pixels, and a shifted input gives a shifted output") and why-conv.md defines them.
- Removed "They remain the default where data are scarce, latency matters, or every pixel needs a prediction" (review IDX-4: chapter 7 says it).
- Added: "We should not make too much of this." before the existing caveat about the brain analogy.
- Word count is above 400 because the history and figure paragraph were moved here on purpose.

## Chapter 7, Modern Convnets

`chapter_convolutional-modern/index.md`, 433 words.

**History paragraph, old first sentence:** Much of the answer was found on ImageNet, a benchmark of more than a million photographs in a thousand categories.

**New first sentence:** The network that won the ImageNet competition in 2012 had 8 layers, and the winner three years later had more than 150.

**New route:**

> The first four sections follow the growth in depth: the network of 2012
> (:numref:`sec_alexnet`), networks built from repeated blocks instead of layer
> by layer (:numref:`sec_blocks`), the normalization of activations
> (:numref:`sec_batch_norm`), and the residual connection
> (:numref:`sec_resnet`), where we also see why deeper plain networks train
> worse. :numref:`sec_efficient_cnns` goes the other way and makes a network
> cheap enough for a phone, at some cost in accuracy. From there on the chapter
> is about evidence as much as about layers. We ask how much of a reported
> accuracy comes from the training procedure (:numref:`sec_training_recipes`),
> how far a ResNet gets when it is modernized one change at a time
> (:numref:`sec_convnext`), and what rules appear when we study populations of
> networks instead of single designs (:numref:`sec_cnn-design`).

**Claims added or changed:**

- Claim changed (review IDX-1): "connections that let a deep network start out as a shallow one" became "connections that let each block add a correction to its input instead of replacing it", since the ResNet section does not zero-initialize the branch.
- "Normalization and residual connections reappear in every transformer" moved next to the list ("The last two reappear in every transformer"); the training-recipe point is its own paragraph (review IDX-2).
- Added: "A benchmark this small shows that a network trains and what it costs. The evidence that an idea matters at scale comes from the ImageNet results" (review, structural finding 1).
- Reworded: "Telling a training recipe apart from an architecture matters ..." became "This is awkward for anyone who compares two published accuracies, because the better number may belong to the better training recipe and not to the better network."
- Not done: the review suggested saying that all networks share one training procedure defined in sec_alexnet. I did not verify that every section uses it, so I left it out.

## Chapter 8, Sequence Models

`chapter_recurrent-neural-networks/index.md`, 411 words.

**History paragraph, old first sentence:** The question is older than neural networks.

**New first sentence:** Claude Shannon made a game of this.

**New route:**

> The chapter sets up the problem before any recurrent network appears. We
> predict a sequence from its own past and see why forecasting many steps ahead
> is harder than forecasting one (:numref:`sec_sequence`), cut text into tokens
> (:numref:`sec_text-sequence`), and build language models that only count,
> which also gives us the measure by which language models are compared
> (:numref:`sec_language-model`). The counts run out, and the recurrent network
> is our answer: :numref:`sec_rnn` replaces the window by a hidden state,
> :numref:`sec_rnn-scratch` implements and trains the model and compares it
> with the counts, and :numref:`sec_bptt` explains why gradients through the
> recurrence vanish or explode. The last section, :numref:`sec_decoding`, turns
> next-token distributions into text.

**Claims added or changed:**

- Second paragraph split in two (review IDX-1). The chain rule of probability and the table of counts are left to their sections; "prefix" became "the text already read".
- Added gloss for token: "which may be a character or a word" (text-sequence.md compares both, then bytes and BPE).
- "tokenization, perplexity and decoding" became "cutting text into tokens, scoring a model, and turning its predictions into text" (review IDX-2).
- Route keeps the toc order (implementation before backpropagation through time); review IDX-3 concerns the hand-off inside rnn.md.

## Unsure

- Chapter 4: "The first two problems have satisfactory answers" is a judgement the sections support but do not state in these words.
- Chapter 5: sections quote the 8-billion model at 32 GB (fp32) and at 16 GB (bf16); the introduction uses the fp32 figure.
- Chapter 7: "The network that won the ImageNet competition in 2012 had 8 layers" restates the old text ("It had 8 layers") with the competition named.
- Several introductions run 410 to 435 words, slightly above the 400 of guide §12.1; chapter 6 is at 494 because of the moved figure paragraph.
