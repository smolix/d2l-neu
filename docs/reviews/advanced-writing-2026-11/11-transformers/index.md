# Transformers (chapter introduction)

Source: `chapter_transformers/index.md` (read to the `<!-- slides -->` marker at line 94). About 330 words of introduction, a table of contents, and an unnumbered "Resources and Further Reading" list of 17 annotated links (about 600 words). No code, no figures.

## Verdict

The first three paragraphs are good and should be kept nearly as they are: they start from something the reader has used (an assistant writing a reply), name the gap in attention in one clause, and turn the chapter's topics into five problems of depth and scale, each stated in a sentence. The fourth paragraph then drops the voice and lists the seven sections as seven sentences of the form "X asks how", which is the template the author has asked to remove. The introduction does not say which sections are the core and which can be skipped, does not connect to the question the part introduction poses (how to get from the recurrent network of the Basics to a current language model), and promises in two places more than the sections show. The resource list is useful but unranked, sits before the reader has read anything, and three of its annotations repeat claims that the section reviews found unsupported. The most important fix: replace the roadmap paragraph with a route that judges (what is the core, what is optional, in what order to read the rest).

## Scores

| dimension | score | note |
|:--|:-:|:--|
| Clarity of explanation | 4 | Nothing undefined; the block is described in two sentences a reader can hold. |
| Clarity of thought and focus | 3 | A real argument in paragraph three, then a list; two overclaims; no statement of what matters most. |
| Pedagogy | 3 | Motivates the chapter; gives no route, no prerequisites, no sense of cost (which sections need a GPU, how long the runs take). |
| Language | 3 | Paragraphs one to three are plain and have a voice. Paragraph four is one sentence pattern seven times; the resource list uses an em dash in every entry. |
| Ease and enjoyment | 3 | The opening questions pull. The roadmap and the link list do not. |

## What works

- **The opening questions**: "What else does a unit need? How can a stack that deep be trained? And how can it write fast enough for us to read along?" Three questions the reader can feel, each answered by a section. Keep.
- **The description of the block** in two sentences ("It pairs attention with a small feed-forward network that transforms each position on its own, and both add their outputs to a residual stream that carries the signal upward").
- **Paragraph three** ("Each setting answers a problem of depth or scale"). It turns the cache, the mixture of experts and scaling laws into problems before they are named: "More parameters cost more computation unless most of them sit idle for any one token. And a run that costs millions has to be sized before it starts." This is how the roadmap should have been written too.
- **The scope paragraph** says what the chapter leaves out and where it goes. Right idea; see IDX-6 for the references.
- **The annotations in the resource list** say what each resource corresponds to in the chapter, which most reading lists do not.

## Structural findings

1. **The roadmap is a list, and it ranks nothing.** Seven sections, seven sentences, each with "asks". The chapter is the centre of the part and about 25,000 words of body text; a reader needs to be told that the block, the GPT and the first half of the cache section are the core on which the later chapters build, that the wirings and the vision transformer can be read in either order, and that the mixture of experts and scaling laws are for the reader who wants to understand how large models are sized and why their parameter counts and their cost per token differ. See IDX-1 for a draft.
2. **No link to the part's question.** `chapter_advanced/index.md` opens with the recurrent network of the Basics and asks what it takes to get from there to current language models. This introduction never mentions that network. One sentence would do it, and it would commit `sec_gpt` to the comparison its reviewer found missing (`gpt.md`, structural finding 1): "By the end of :numref:`sec_gpt` we have trained a transformer on the same kind of text as the recurrent network of :numref:`chap_rnn` and can compare the two."
3. **The resource list is in the wrong place and has no order of importance.** Seventeen links under an unnumbered heading inside the chapter introduction, next to a numbered section called "Further Reading". A reader meets "the first-token sink and the windowed cache" and "fine-grained experts balanced by a bias" before any of these terms mean anything. Move the list to the top of `further-reading.md` (as its first heading, "Companions to the chapter"), and leave in the introduction at most three resources named in a sentence: one build-along (Karpathy's video), the Annotated Transformer, and one maintained comparison of current models.

## Passage-level findings

**IDX-1** (moderate, language and pedagogy: template). Roadmap paragraph. Quote: ":numref:`sec_transformer-block` asks what a block needs besides attention,". Seven consecutive sentences of the form "section asks how or what". The part introduction uses the same pattern for its chapters, so the reader meets it twice in a row. Direction: tell the route as an argument and judge. Draft: "We start with one block and measure what its normalization does to a deep stack (:numref:`sec_transformer-block`), then stack blocks into a language model, train it, and load GPT-2 into the same class (:numref:`sec_gpt`). These two sections are the chapter; everything after them changes one thing. :numref:`sec_kv-cache` makes generation affordable and shows that its speed is then set by memory traffic. :numref:`sec_transformer` removes the causal mask or adds a second stack, and :numref:`sec_vision-transformer` feeds the block image patches; the two can be read in either order, or skipped by a reader who wants only language models. The last two sections are about size: what a run costs and how to split its budget (:numref:`sec_scaling-laws`), and how to store more parameters than each token uses (:numref:`sec_moe`)."

**IDX-2** (moderate, pedagogy). Same paragraph; nothing to quote because it is absent. The introduction says neither what the reader must bring (the attention layer, masks and RoPE of :numref:`chap_attention`; AdamW, the schedule and `train_steps` of :numref:`chap_optimization`) nor what the runs cost. The section reviews found that no section states a running time (`gpt.md`, structural finding 5). One sentence here ("every experiment runs in a few minutes on one GPU; the two that take longer say so") would set the reader's expectations for the whole chapter, once the sections can back it.

**IDX-3** (moderate, correctness; confidence high). Quote: "and where its normalization must sit for a deep stack to train." Post-norm stacks train: GPT-1 and BERT are post-norm, the chapter's own `further-reading.md` says that with a rescaled stream "post-norm stacks of a thousand layers train", and the stored output of `gpt.md` shows the post-norm model with a warmup ending at the same loss as the pre-norm one (1.55 against 1.54). What the chapter shows is that placement decides how sensitive a stack is to initialization and to the learning-rate schedule (see TB-5 and GPT-13). Write "and what the placement of its normalization changes in a deep stack".

**IDX-4** (minor, correctness). Quote: "A deep stack can lose its", continuing "signal at initialization or diverge in training." The chapter shows no divergence. `gpt.md` says of its failing run: "does not diverge, but its loss stops at about 3.2 nats". Write "or stall in training", which is what the reader will see.

**IDX-5** (minor, thought). Quote: "reading that memory, not the arithmetic, sets the speed." This is the thesis of `sec_kv-cache`, stated here as fact. In the default (PyTorch) tab the measured decode step is eleven times slower than the memory bound allows, because of kernel launches (KV-6, KV-8). The sentence is right for a compiled or large model; it should stay, and the cache section must be made to show it.

**IDX-6** (minor, dependency). Quote: "Models part, kernels and parallelism to". The scope paragraph names "the Language Models part" without a label, while the two neighbouring pointers carry one. Labels exist for the skeleton chapters (`chap_lm_pretraining`, `chap_lm_encoders`, `chap_lm_serving`). Either cite them or keep the unlabelled form everywhere; the sections currently mix both, and two of them point to the retired `chap_nlp_pretrain` (ED-7, SL-14). Also: `:numref:`sec_transformer`` is the label of the encoders and decoders section, a leftover from when it was the chapter's main section; rename it in the editing pass (grep first).

**IDX-7** (moderate, correctness: annotations that repeat unsupported claims). Three entries of the resource list. Quotes: "and the reason post-norm models need warmup, which :numref:`sec_gpt` tests."; "whose matched-parameter comparison :numref:`sec_transformer-block` repeats in miniature."; "whose law :numref:`sec_scaling-laws` fits and whose bend its miniature study shows." The first: `sec_gpt` shows that a long warmup avoids a stall; it measures no gradient and does not test the paper's reason (GPT-13). The second: the miniature comparison is 600 steps of a model still improving by 0.25 nats per 200 steps (TB-19). The third: the section does not fit the law (it compares the published constants with a published refit), and the bend rests on one point (SL-7). Reword each to what the section does, after the sections are revised.

**IDX-8** (minor, language). Resource list. Quote: "All are freely accessible online." Two lines later: "(the book itself is paid)". And every one of the 17 entries joins title and annotation with an em dash, which the author has asked to avoid; use a colon or a period. The list also omits the resource the author names as the bar for the part: the Illustrated Transformer appears only in the attention chapter's list, although its subject is the encoder-decoder of `sec_transformer`.

## Code and experiments

None.

## Exercises

None.

## Still open from October

Nothing specific to this file.

## For the chapter review

- The introduction's best paragraph (five problems of depth and scale) is the argument the chapter should follow, and the sections mostly do. Its roadmap is the template the author asked to remove.
- Two overclaims here ("must sit", "diverge") come from the same place as the chapter's most serious finding: the argument for pre-norm is stated more strongly than the experiments support.
- The resource list duplicates the function of `further-reading.md`; merge them.
