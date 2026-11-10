# Modern Convnets (chapter introduction)

Source: `chapter_convolutional-modern/index.md`. About 330 words of introduction, then a 570-word "Resources and Further Reading" list.

## Verdict

A good chapter introduction, close to the house template and better than most: it starts from the LeNet the reader has just trained, names a real obstacle ("For a long time this did not work"), and ends its first paragraph on a question the reader can hold for eight sections. The roadmap is one question per section. Two things need fixing. The second paragraph loses its thread in its last three sentences, and one of its promises ("connections that let a deep network start out as a shallow one") is not kept by the ResNet section as written. The resources list is useful but its own wording ("the same tour", "this chapter retells") concedes the history-tour reading that the introduction works to avoid.

## Scores

| Dimension | Score |
|---|:-:|
| Clarity of explanation | 4 |
| Clarity of thought | 4 |
| Pedagogy | 4 |
| Language | 4 |
| Ease and enjoyment | 3 |

## What works

- The opening sentence starts from the reader's own result: "it classifies Fashion-MNIST slightly better than the MLP with fewer than a third of its parameters. A natural next step is to make it larger". The question that closes the paragraph, "What does it take to make a convolutional network much deeper, and better?", is the organizing question the October review said the chapter lacked.
- "It had 8 layers, and three years later the winning network had more than 150." One comparison carries the whole motivation.
- The roadmap phrases every section as a question ("asks why deeper plain networks train worse and how a residual connection fixes it"). Keep this form.
- The scope paragraph is honest about method: "Every network in the chapter is trained on Fashion-MNIST; ImageNet results are quoted from the papers."

## Structural findings

- The introduction never tells the reader what the Fashion-MNIST experiments can and cannot show. Sections 7.1 to 7.4 all land between 91% and 93% on upsampled 28 by 28 images, and each section then has to explain why its idea matters "on ImageNet". One sentence here would prepare the reader: the small benchmark checks that each network trains and lets us count its cost; the evidence that an idea pays off at scale comes from the papers. See the chapter-level note on experiments in the report.
- The chapter's shared training procedure (10 epochs, batch 128, warmup then cosine decay) is the fixed background of every experiment, and "Telling a training recipe apart from an architecture" is announced here as a theme. The introduction should say in one clause that all networks of the chapter are trained with one procedure, defined once in :numref:`sec_alexnet`.

## Passage-level findings

**IDX-1** (moderate, thought). Under the chapter title, second paragraph. Quote: "connections that let a deep network start out as a shallow one".
This is the zero-initialized residual branch idea, and it is a fine one-line statement of why residual networks train. But the ResNet section does not deliver it: its `Residual` block uses He initialization for both convolutions and a batch-norm scale of 1, so at initialization the branch output has the same magnitude as the input and the network does not start out shallow. Either rephrase here ("connections that let each block add a correction to its input instead of replacing it"), or make the ResNet section keep the promise (zero-initialize the last batch-norm scale of each block and say so). See RES-3.

**IDX-2** (minor, language). Same paragraph. Quote: "Normalization and residual connections reappear in every transformer."
The last three sentences of this paragraph are three separate remarks (later gains came from training; transformers reuse two ideas; recipes confound comparisons). The middle one interrupts the other two, which belong together. Order: the three obstacles and their fixes; "Two of these, normalization and residual connections, reappear in every transformer."; then, as its own short paragraph, the point about training recipes and published comparisons.

**IDX-3** (minor, explanation). First paragraph. Quote: "Deeper stacks of sigmoid layers trained worse than shallow ones".
True, and the reader met the reason in LeNet (each sigmoid scales the gradient by at most 1/4). Half a clause would connect it: "Deeper stacks of sigmoid layers trained worse than shallow ones, since every sigmoid shrinks the gradient".

**IDX-4** (minor, language). Resources list. Quotes: "the same tour as this chapter's first half" and "these are the primary sources this chapter retells".
Both phrases describe the chapter as a tour and a retelling, which is the weakness the rewrite set out to remove. Write "covers the same networks as the first four sections, with the same parameter and FLOP accounting" and "the primary sources for this chapter".

**IDX-5** (minor, language). Resources list, throughout. The list carries all 33 em dashes of the file, often two per entry ("free PDF; Chapter 11 (Residual networks) analyzes *why* residual connections ease optimization — loss-surface and gradient-propagation arguments complementing"). If the house rule against dashes applies to this list, each entry can be "Title, authors. One sentence on what it adds." Also "the 'what you actually do in 2026' companion to this chapter" will date within a year and reads as a quip.

**IDX-6** (minor, pedagogy). Resources list, "Foundational papers". The list names AlexNet, batch norm, ResNet, two recipe papers, MobileOne, ConvNeXt, and the Smith et al. comparison, but none of VGG, NiN, GoogLeNet or RegNet, each of which has a section or a third of one. Either add them or say the list is selective.

## Exercises

None, correctly.

## Still open from October

The October finding "a catalogue of about 25 architectures with no organizing question" is answered at the level of this introduction. Residue of the catalogue survives inside sections (see the notebook files).
