# Writing pass: `chapter_convolutional-neural-networks/lenet.md`

Prose words (checker, includes exercises and tab notes): 2534 before, 2534
after. The body lost about 150 words of repetition and gained about 150 in
the reading of the result. Checker: "OK: only prose changed". Lint: clean.

## Before and after

1. Opening.
   - Before: "We now have every part of a convolutional network: the convolutional layer of :numref:`sec_conv_layer`, padding and stride to control its output (...). This section puts them together into a complete classifier and asks whether it is better than the MLP we trained on the same data."
   - After: "The preceding sections supplied the parts: the convolutional layer (...), padding and stride (...), channels (...), and pooling (...). None of them has classified anything yet. We now assemble them into a complete classifier and ask the obvious question: is it any better than the MLP?"
2. The MLP.
   - Before: "It flattens each $28\times28$ image into a vector of 784 numbers before its first layer. Shuffling the pixels in the same way in every image would therefore not change what it can learn: the MLP does not know which pixels are neighbors, and it must learn a pattern separately at every position where the pattern can occur."
   - After: "Recall that the MLP reads an image as 784 numbers in no particular order (:numref:`sec_why-conv`), whereas convolutional layers keep the spatial layout and use the same kernel at every position."
3. The 1998 network.
   - Before: "... but it used a scaled tanh as activation, pooling with a trainable weight and bias, partial connections between the two convolutional layers, and radial-basis output units. Its 120-unit layer was formally a $5\times5$ convolution, ... It also took $32\times32$ images. The MNIST digits fit in a $20\times20$ box ..."
   - After: "It differed in details that need not concern us: a scaled tanh as activation, pooling with a trainable weight and bias, partial connections between the two convolutional layers, and radial-basis output units. One difference does leave a trace in our code. The original took $32\times32$ images, so our first convolutional layer pads its $28\times28$ input by two pixels on each side, which gives it the same extent and the same $28\times28$ output."
4. Shape check.
   - Before: "To check that the code builds this network, we pass a single-channel $28 \times 28$ image through it ..." / "The shapes follow the rules of :numref:`sec_padding`."
   - After: "Does the code build this network? To find out, we pass a single-channel $28 \times 28$ image through it ..." / "It does, and the shapes follow the rules of :numref:`sec_padding`."
5. Learning rate.
   - Before: "The learning rate needs more care than for the MLP. The derivative of the sigmoid, $\sigma'(x) = \sigma(x)(1 - \sigma(x))$, is at most $1/4$. ... At the MLP's learning rate of 0.1, LeNet therefore learns slowly ... We use a learning rate of 2.0 instead, twenty times larger."
   - After: "Unfortunately, the learning rate cannot stay where it was. The trouble is the sigmoid, which passes back at most $1/4$ of the gradient (:numref:`subsec_activation-functions`). ... We use a learning rate of 2.0 instead. The factor of twenty is empirical: the argument says only that the rate must grow."
6. Reading the result.
   - Before: "Even at this learning rate the validation accuracy can stay at chance, 10%, before it climbs; how long it stays there varies from run to run, from no epoch at all to several. After 30 epochs LeNet reaches a validation accuracy of about 88--90%, against about 87% for the MLP, with fewer than a third of the MLP's parameters."
   - After: "After 30 epochs LeNet reaches a validation accuracy of about 88--90%, against about 87% for the MLP. It does not always start at once: the accuracy can stay at chance, 10%, for anything from no epoch at all to several, depending on the run."
7. What the result means (new).
   - After: "How much should we read into a gain of one to three points? Not too much. The MLP's own accuracy moves by about a point from one epoch to the next (:numref:`sec_mlp-implementation`), and LeNet differs from the MLP in its depth, its activation function, and its learning rate as well as in its convolutions."
8. Chapter ending (new; replaces the bare restatement of the premise).
   - After: "What the run does show is that the accuracy of the MLP can be had, and slightly improved on, with fewer than a third of its parameters. ... On $28\times28$ images little is at stake. On the megapixel photographs of :numref:`sec_why-conv`, where a single fully connected layer needs a billion weights or more, the same idea decides whether the network can be built at all. Whether such a network can also be trained is another matter, and the four sigmoids above give a first hint of the difficulty."

## Repetition removed

- Pixel shuffling (owner: `why-conv.md`, also the chapter introduction): the
  opening's three-line retelling is now one clause with a reference.
- The four-section recap in the opening: reduced to a list of references in
  one sentence.
- "About 204,000 parameters" in the opening: dropped; the count is given
  once, exactly, with the cost table.
- "Fewer than a third of the parameters": was in the cost section, the result
  paragraph, and the Summary; the result paragraph now uses it once as its
  conclusion, the cost section derives it, the Summary restates it.
- "Channels grow while pooling halves the resolution": was in the
  architecture, the shape reading, and the Summary; removed from the
  architecture paragraph.
- Layer widths 120, 84, 10: no longer repeated in the shape reading.
- Sigmoid derivative at most $1/4$ (owner: chapter 4,
  `subsec_activation-functions`): formula removed, one clause with reference.
- Convolution parameter and multiply-add counts (owner: `channels.md`,
  "Parameters and Cost"): introduced with "Recall from
  :numref:`subsec_multi-output-channels`"; the formula stays because the
  table is read with it.
- Same loss, optimizer, epochs: stated in the opening; the training
  paragraph and the Summary now say "as for the MLP" / "trained like the MLP".

## Review findings

Applied (prose): LEN-3 (1998 paragraph cut to what the code needs); LEN-4 (b)
(the factor of twenty is called empirical, "therefore" removed); LEN-5
(result first, plateau in one sentence, conclusion limited to what the run
shows); LEN-7 (one phrasing, "passes back at most $1/4$", in body and
Summary; the "because" is gone); LEN-8 in part (PyTorch tab note shortened;
not moved).

Deferred:
- LEN-1 (title "LeNet"): the full title is quoted in the "Where this fits"
  slide tables of all seven files of the chapter; needs one change for the
  chapter.
- LEN-2 (permutation or shift experiment): new cells and outputs.
- LEN-4 (a), (c) (show the run at 0.1, or train the ReLU variant): new cells.
  The "two to four epochs" and "about 73% after 10" stay as run-to-run notes;
  they cannot be checked from the store.
- LEN-6 (replace `tab_lenet_modern` by two sentences): the label is
  referenced from `chapter_computational-performance/multiple-gpus.md`, and
  labels may not change in this pass. The table stays in the Summary.
- Chapter finding 4 (say which parts of LeNet break equivariance and
  invariance): that is the answer to exercise 5, part 1; adding it needs the
  exercise changed.
- Learned feature maps: new cell and figure.

## Claims added or changed

- "LeNet differs from the MLP in its depth, its activation function, and its
  learning rate as well as in its convolutions." Checked against
  `mlp-implementation.md` (one hidden layer, ReLU, learning rate 0.1). The
  minibatch size also differs (256 there, 128 here); the text does not say
  so, and "as for the MLP ... with minibatches of 128 images" could be read
  as if it were the same.
- "The MLP's own accuracy moves by about a point from one epoch to the next":
  cited from `mlp-implementation.md`.
- "A gain of one to three points": the arithmetic of the existing "about
  88--90%" against "about 87%". Stored final validation accuracies, read
  from the four plots: LeNet about 0.89 to 0.90 (PyTorch about 0.89,
  TensorFlow, JAX, MXNet about 0.90); MLP about 0.87 to 0.875. Epochs at
  chance in the store: JAX none, PyTorch and TensorFlow two, MXNet three,
  which fits "from no epoch at all to several".
- "A sharper test would shuffle the pixels ...: the MLP could not notice,
  while LeNet's kernels would find pixels side by side that were never
  neighbors." A prediction, not a reported result; no outcome for LeNet's
  accuracy is claimed. If LEN-2 is run later, this sentence becomes the
  experiment.
- "The factor of twenty is empirical: the argument says only that the rate
  must grow." No claim about how 2.0 was found.
- "On $28\times28$ images little is at stake"; "a single fully connected
  layer needs a billion weights or more" (the $10^9$ and $10^{12}$ of
  `why-conv.md`).
- "Whether such a network can also be trained is another matter": points to
  the next chapter's opening (deeper sigmoid stacks trained worse).
- Removed: the receptive-field explanation of the $32\times32$ input, and the
  remark that the 120-unit layer was formally a $5\times5$ convolution.

## Slides that now differ from the body

- "This Is LeNet-5's Layout, Not LeNet-5": still lists the 120-unit layer as
  a formal $5\times5$ convolution and the "digits fit in 20×20" note; the
  body no longer has either.
- "The MLP Reaches 87% with 204k Parameters ...": retells the shuffling
  argument that the body now only cites.
- "Four Sigmoids Shrink the Gradient ...": "Use 2.0, twenty times the MLP's
  rate" without the body's remark that the factor is empirical; the
  derivative formula is on the slide but no longer in the body.
- "LeNet Beats the MLP with a Third of the Parameters": the body is more
  guarded ("Not too much") and names the confounders; the slide does not.
- No number on the slides disagrees with the body.
