# Saving, Loading, and Pretrained Weights

Source: `chapter_builders-guide/saving-loading.md` (read to the slides marker, line 1394). Prose a PyTorch reader sees: about 2,600 words plus 690 words of exercises; across all four tabs about 3,900 words in 44 tab blocks, and 17 code cells per framework.

## Verdict

The strongest section of the second half, and the one that reads most like a textbook. Three of its four parts have a sharp example that the reader will remember: a pickle file that prints a message when it is loaded, a safetensors header read with eight lines of standard-library Python, and an Adam optimizer that, restarted without its state, moves every weight by the full learning rate and raises the loss a thousandfold. Each is followed by an explanation that accounts for the number on the screen. The section falls short of the bar in two places. The checkpoint experiment never tests the claim the opening makes, that training "resumes exactly", and two of the five compartments of the checkpoint (random-number state and step) are never exercised. And the last part, on pretrained weights, loses the thread: it ends in a subsection on freezing that repeats `sec_parameters` and belongs to fine-tuning. The most important fix is to make the resume experiment a real test: an interrupted run compared, bit for bit, with an uninterrupted one.

## Scores

| Dimension | Score |
|---|---|
| Clarity of explanation | 4 |
| Clarity of thought | 4 |
| Pedagogy | 4 |
| Language | 4 |
| Ease and enjoyment | 4 |

## What works

- The opening. It starts from the reader ("Each model we have trained so far has lived only in the memory of the process that trained it"), scales the problem with one well-chosen fact (419 interruptions in 54 days of Llama 3 pre-training, "about one every three hours"), and closes the paragraph with a consequence: "A run that cannot continue from where it stopped would never finish."
- "A model is code plus state." The distinction organizes the whole section and is paid off immediately after the first state listing: "It does not record the `MLP` class, its `forward` computation, or the activation function."
- "Code Execution on Load". "Python's standard serialization, `pickle`, does not store data. It stores a program that reconstructs the data, and loading the file runs that program." Then a five-line `Tripwire` class proves it. The harmless `print` standing in for `os.system` is exactly the right size of example.
- Reading the header by hand. "Because the header is JSON at a known offset, we can read it with nothing but the Python standard library", followed by a check of the byte ranges against $256 \times 20 \times 4$. The reader verifies the figure against the file.
- The fresh-Adam paragraph: "Every parameter with a nonzero gradient moves by the full learning rate of 0.05, however small that gradient is." The stored output (`0.002, 4.3103, 0.1922, 1.0051, 2.0529`) supports it, and the equation is used, not just displayed. The four-word close "Weights alone do not resume a run" is earned.
- The atomic-write paragraph (temporary name, then rename) is three sentences and complete.
- The damaged state dictionary, and the working rule that follows: "we state in advance which names may be missing, the head here, and treat every other entry of either set as an error."
- The closing paragraph names what the reader can now do and hands off to the next section in one sentence.

## Structural findings

1. **"Resumes exactly" is promised and not tested.** The opening says the section "writes a checkpoint from which training resumes exactly". The experiment shows that the loss "keeps decreasing slowly" with the saved optimizer and jumps without it. It never compares the resumed run with a run that was not interrupted, which is what "exactly" means and what an engineer would check. Train 105 steps without stopping, train 100 plus checkpoint plus 5, and compare the final weights for equality. That one comparison also motivates `sec_repro` two sections later.
2. **Two compartments of the checkpoint are decoration.** `fig_bg_checkpoint_contents` lists five things. The experiment is full-batch regression with no shuffling, no dropout, and no learning-rate schedule, so the random-number state and the step counter do nothing. Either give the experiment minibatches (then dropping the RNG state changes the order and the comparison of finding 1 fails without it), or say in one sentence that this experiment exercises only three of the five.
3. **Adam arrives here in full.** The section introduces Adam's complete update, with bias correction, three chapters before the optimization chapter. The equation is used well afterwards, so the cost is justified in part, but the reader needs only one fact from it: with zero state, the first step is $\eta\, g_i/(|g_i| + \epsilon)$. See SAV-5 for a lighter form.
4. **The pretrained-weights part drifts.** It has one idea that belongs here (load by name, read the report of what did not fit) and one that does not (freezing, with batch-normalization caveats per framework, before the reader has met batch normalization or ResNet). `sec_parameters` already has a subsection "Frozen Parameters". Cut the freezing subsection to two sentences and a pointer to `sec_fine_tuning`.
5. **Order is right.** State, then safe formats, then checkpoints, then weights from elsewhere: each part needs the one before. Do not reorder.

## Passage-level findings

**SAV-1** (minor, thought). Under "Training Checkpoints": "with 8 billion parameters takes 96 GB, six times the 16 GB of its weights".
The factor of six mixes two effects: three fp32 tensors per parameter instead of one, and fp32 instead of bf16. A reader will attribute all of it to the optimizer. Write "96 GB: three times the 32 GB of the fp32 weights, and six times the 16 GB file that is enough for inference in bf16." The same comparison in the opening ("the weights alone take 16 GB in bf16, while a resumable checkpoint with fp32 weights and the optimizer's state takes 96 GB") needs the same care.

**SAV-2** (minor, pedagogy). Under "Model State": "A model holds many tensors, so a file format for its state must store a".
The first cell saves and reloads `{'x': ..., 'y': ...}`. It teaches the name of a function and nothing about models. Start at the `MLP` state listing, which is the first cell that carries an idea.

**SAV-3** (minor, thought). Under "Code Execution on Load": "such checks have had holes: PyTorch fixed a way around `weights_only=True`".
Six lines earlier, version 2.6 is the release that made `weights_only=True` the default; here it is the release that fixed a hole in it. Both are true and the reader cannot tell what happened. Version numbers will also date. Write "such checks have had holes, in PyTorch's `weights_only` loader among others", and keep one version number at most.

**SAV-4** (moderate, language). Under "The safetensors Format": "The *safetensors* format is one: a file holds tensors and a description of".
"Is one" points back across a heading and a TensorFlow tab block to "A format that contains no program at all". A subsection cannot open with a pronoun whose antecedent is in another subsection. Write "The *safetensors* format contains no program: a file holds tensors and a description of them, and nothing else."

**SAV-5** (moderate, explanation). Under "Training Checkpoints": "Many of the later chapters train with *Adam*, an optimizer that keeps two".
A four-equation aligned display with $\mathbf{m}$, $\mathbf{v}$, $\hat{\mathbf{m}}$, $\hat{\mathbf{v}}$, $\beta_1$, $\beta_2$, $\epsilon$, $t$ is a lot of new notation for a section about files, and `sec_parameters` has already said that Adam keeps two numbers per parameter. Keep the words ("each parameter moves against its averaged gradient, by an amount scaled down where its gradients have been large") and the two running averages, and move the bias-corrected step to the point of use, where the jump is explained: "At the first step of a fresh Adam the averages are zero, and after bias correction the step of each parameter is $\eta\, g_i / (|g_i| + \epsilon)$." Then the display is three symbols long and arrives with its purpose.

**SAV-6** (minor, language; pattern). Same passage: "running averages for every parameter, of its gradient $\mathbf{g}$ and of".
The same two tensors are "running averages" here, "moment estimates" in `numerics.md` (exercise 2) and `gpus-devices-memory.md` ("the two moment estimates of Adam"), "first moment" and "second moment" in the table of `parameters-state-memory.md`, and "second moment" again in exercise 4 of this section. Pick one name for the chapter.

**SAV-7** (major, pedagogy). Opening: "asks which file formats are safe to load from a stranger, writes a".
See structural findings 1 and 2. The phrase "resumes exactly" on the next source line sets a standard the experiment does not meet.

**SAV-8** (minor, pedagogy). Under "Training Checkpoints": "function of 20 inputs. The weights start from Xavier".
The variance formula of Xavier initialization and the remark on biases are irrelevant to checkpointing and cost three lines. "An MLP with one hidden layer of 64 units, fit by Adam with learning rate 0.05 to 256 noisy examples" is enough.

**SAV-9** (minor, pedagogy). Under "Training Checkpoints": "To check that the checkpoint holds everything the model needs, we corrupt".
`'perturbed 48271.99 -> restored 0.0020'` shows that loading overwrites weights, which the safetensors round trip already showed. It also does not check "everything the model needs" (the optimizer is not tested until the next cell). Fold it into the resume comparison or drop it.

**SAV-10** (moderate, language). Under "Training Checkpoints": "few thousandths or less to between about 0.5 and 6 after a single step, and after four steps" and the next line "it is still hundreds of times higher than at the checkpoint, or more."
The sentence is hedged to cover four frameworks' outputs ("a few thousandths or less", "between about 0.5 and 6", "hundreds of times higher ... or more") and loses its force. The claim that holds in every tab is simpler: "With a fresh Adam, a single step raises the loss by a factor of several hundred or more, and four steps later it has not recovered."

**SAV-11** (minor, language). Under "Pretrained Weights": "In practice, many models do not start from random weights. A network trained on a large dataset such as ImageNet, 1.2 million images".
This repeats the second paragraph of the opening ("in practice, many models start from weights that someone else trained") almost word for word, and the line is unwrapped in the source, which suggests a late edit. Open with the task: "To reuse an ImageNet classifier for a task with 10 classes, we keep all layers except the last ..."

**SAV-12** (minor, correctness). Under "Pretrained Weights": "example is ResNet-50, a convolutional network with about 25 million" against, later, "Of the more than 23 million parameters of the network, 20,490 remain".
Both are right (the 1,000-class head holds 2 million of the 25), but the reader sees two totals for one network. Add "(23.5 million without its 1,000-class head)" at the first mention.

**SAV-13** (moderate, language). Under "Pretrained Weights": "what a careful load reports, we build a damaged state dictionary, standing".
The sentence runs to about 55 words with three appositives ("standing in for a file with missing entries", "which a new head replaces anyway", "a loss nobody intended") and uses "stages" of a network the reader has not seen. Draft: "To see what a careful load reports, we remove two groups of entries from the pretrained state. The first is the head, which we mean to replace. The second is the last quarter of the network (`layer4`), which stands in for an accident: a file that lost part of its contents. We load what remains into a new, randomly initialized ResNet-50 with a 10-class head."

**SAV-14** (moderate, pedagogy). Under "Pretrained Weights": "Finally, we *freeze* the pretrained layers of the complete network `net`, so".
See structural finding 4. The tab blocks that follow discuss running statistics of batch normalization in training mode, `grad_req`, `nnx.DiffState`, and `use_running_average`, none of which the reader can place yet. The count "20,490 trainable of 23,528,522" is a nice number but it is the same demonstration as the "Frozen Parameters" subsection of `sec_parameters`.

**SAV-15** (minor, thought). Under "Pretrained Weights", MXNet and TensorFlow tabs: "The total, 23,547,402, includes 18,880 biases: Gluon's `resnet50_v1` gives".
An audit of why two frameworks' totals differ from PyTorch's. This is written for the authors (writing guide section 4.5). Cut both tab blocks.

**SAV-16** (minor, pedagogy). End of "The safetensors Format": "format that contains no program and publish a checksum with the file so that".
The checksum is recommended in a subordinate clause and never shown. One line (`hashlib.sha256`) in the header-reading cell would make the advice concrete, or leave it to exercise 6, which already asks for it.

## Exercises

Six per tab, the best set in the second half. Exercise 2 (predict how the header length changes, then read one tensor with file seeks) and exercise 6 (crash-safe manifest with `fsync`) extend the section's own examples and have checkable answers. Exercise 4 (float16 round trip) ties this section to `sec_numerics` and ends with a real question about the second moment. Exercise 1 is a fair warm-up. Exercise 5 (checkpoint averaging) is a good problem about a different idea (stochastic weight averaging) and is the one to drop if the count must fall; `sec_parameters` has "Weight Averages", where it would fit better. Add, if structural finding 1 is not adopted in the body: "Show that 100 steps, a checkpoint, and 5 more steps give the same weights as 105 uninterrupted steps. Then train on shuffled minibatches and find which compartment of the checkpoint the equality now needs."

## Still open from October

"mmap and meta-device loading" (October: cut) is reduced to one sentence ("a reader can map the file into memory and read one tensor without reading the others"), which is the right size. Nothing else from October applies to this section.
