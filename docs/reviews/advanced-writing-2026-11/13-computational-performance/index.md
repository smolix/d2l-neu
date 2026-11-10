# Computational Performance (chapter introduction)

Source: `chapter_computational-performance/index.md` (lines 1 to 93, up to the slides marker). About 330 words of introduction and about 650 words of annotated resources. No code.

## Verdict

The first two paragraphs are good and have a voice: a question the reader can feel ("Where does the rest of the time go?"), a piece of history that earns its place (processors outran their memory, and in 2009 one picture drew the consequence), and a statement of the three limits in three sentences. The third paragraph then drops into the route template, seven sentences of the form "X asks how", and the introduction ends on a scope note. It does not say what the reader needs to run the chapter (which sections need a GPU, which need two), what can be skipped, or which machine every number comes from. And the method it announces is "measure a program, find the resource that limits it, relieve that resource, and measure again": there is no "predict" in it, although prediction before measurement is what the chapter does at its best and what the section reviewers found missing most often. The most important fix is to state the method as the chapter should practise it (count, predict, measure, explain the difference) and to replace the route list by a paragraph that judges.

## Scores

| dimension | score | reason |
|---|---|---|
| Clarity of explanation | 4 | The three limits are clear before any section starts. |
| Clarity of thought and focus | 3 | The method is stated without its predictive step; two promises are ones the sections do not keep. |
| Pedagogy | 3 | Motivates well; gives no prerequisites, no hardware requirements, no guidance on what to skip. |
| Language | 4 | Plain and exact; the route is a formula. |
| Ease and enjoyment | 4 | The opening and the history make a reader want to start. |

## What works

- **The opening.** "We took the time of a step and its memory as given. That stops working when a run takes days, a model does not fit on the card, or a second GPU makes training slower." Three failures, each of which a later section explains, in one sentence.
- **The history.** "Computer architects met this question long before deep learning did." Four sentences place the roofline in its origin and say why a deep learning reader inherits it. This is the only history in the chapter and it is the right amount.
- **"Which of the three holds a step back decides which change speeds it up."** The chapter's thesis in one line.
- **"This view explains choices that earlier chapters made, such as large minibatches and 16-bit numbers."** It tells the reader that the chapter pays off debts from the Basics.
- **The scope paragraph** says what is left out and where it goes.
- **The resources** are well chosen (the scaling book, the Ultra-Scale Playbook, Horace He's essay, CS336) and each entry says which section it accompanies.

## Structural findings

1. **The method lacks its first step.** See IDX-2. This sentence is the chapter's contract, and the chapter review measures the sections against it.
2. **The route is a list.** See IDX-1.
3. **Nothing on what the reader needs.** The chapter assumes `sec_use_gpu` (asynchronous dispatch, `d2l.Benchmark`, peak memory, checkpointing), `sec_numerics` (bf16, tf32, loss scaling), `sec_gpt` and `sec_kv-cache`. It never says so in one place. Nor does it say that sections one to four and the first half of the case study need one GPU, that the two multi-GPU sections assert two, or that a reader with no GPU can follow the counting and the stored outputs but can run almost nothing. One short paragraph would do both.
4. **The capture machine is introduced in a subordinate clause** ("On the GPU that runs this book's notebooks"). Every number in the chapter comes from four RTX 4090 cards in one host without peer-to-peer transfers. The introduction is the place to say so once, and to say that the reader's own machine will give other numbers and the same relations.
5. **No chapter ending.** The case study's last paragraph is the chapter's last paragraph of teaching. The question this introduction asks is answered there only halfway (see the review of `fast-transformer.md`, structural finding S2).

## Passage-level findings

**IDX-1** (moderate, pedagogy and language). Third paragraph.
> :numref:`sec_perf_model` asks which of the three resources limits an

Six of the seven sentences have the verb "asks" (the fifth has "builds ... and asks"). The paragraph gives each section the same weight and none a judgment. A reader cannot tell that the first section is the one everything rests on, that the hardware section can be read for its two tables, or that the two multi-GPU sections can be skipped by a reader with one card without losing the case study's single-GPU half. Draft: "Everything rests on :numref:`sec_perf_model`: one quantity, the arithmetic intensity, and one habit, to predict a time before measuring it. :numref:`sec_hardware` explains where the two constants of that model come from; a reader in a hurry can take its two tables and move on. :numref:`sec_compilation` and :numref:`sec_memory_precision` are the tools for one GPU, and most readers will use them every day. The two sections on several GPUs need two cards to run but not to read. :numref:`sec_fast_transformer` applies everything to one GPT and reports what each change was worth."

**IDX-2** (moderate, thought). Second paragraph.
> (:numref:`sec_numerics`). It also gives a method: measure a

The method as stated starts with a measurement. The roofline's value is that it tells the reader what to expect before the program runs; a measurement without an expectation cannot surprise. The performance-model section says of the third limit "Only measurement reveals the third" (PM-16) and the hardware section, in its last table, states the procedure this sentence should state: "count its FLOPs and bytes, divide by $P$ and $\beta$, and compare the result with the host cost of its operations" (HW-17). Draft: "It also gives a method: count the arithmetic and the bytes of a step, predict its time, measure it, and explain the difference. The difference is where the time went."

**IDX-3** (minor, correctness). First paragraph.
> :numref:`sec_gpt`. Dividing the arithmetic of one training step by the rate

Three loose points in two sentences. (i) "the rate in the GPU's specification": NVIDIA publishes no tensor-core rate for GeForce cards; the 165 TFLOP/s is derived (PM-1, HW-9). Write "the peak rate of the GPU". (ii) "several times longer": the case study's 25 to 30 ms against 6 ms compares a tf32 step with the bf16 peak, so half of the gap is the number format (FT-1). (iii) "the GPT of :numref:`sec_gpt`": the case study doubles its width to 512, for a stated reason. None is wrong; each will make an attentive reader feel a seam when the case study arrives. One clause each.

**IDX-4** (moderate, thought: promises the sections do not keep). Third paragraph.
> production libraries hide communication behind computation and divide a

Two promises. The first is the one the section's own measurement disconfirms: on this machine the exposed communication is 15 to 19 ms against 18 ms for no overlap at all (MGP-12, structural finding S6 of that review). The second, to "divide a model that no single GPU holds", is described there and never run, in that section or anywhere in the chapter (S2 of that review). After the revision of that section, write what it found: one process per GPU fixes the launcher, overlap helps only on fast links, and sharding trades memory for communication at a price the section computes.

**IDX-5** (minor, dependencies). Scope paragraph.
> kernels that the frameworks provide instead of writing new ones

The reference that follows is `sec_custom_layer`. That section writes a custom layer (RMSNorm as a module) from existing operations and lists "writing GPU kernels (not in this book)" among its skips. It does not write kernels, so the reference points at the wrong thing. The place that discusses hand-written kernels is "Writing Fused Kernels" in `sec_performance-further-reading`; cite that.

**IDX-6** (minor, pedagogy). "Resources and Further Reading".
> The resources below accompany the sections of this chapter; each names the

Sixteen entries in four groups, none ranked. Say which to read first: Horace He's essay is fifteen minutes and gives the three regimes of the first section in another voice; the scaling book is the natural second text for the whole chapter. Two details. The entry for the memory blog names a private function (`torch.cuda.memory._record_memory_history`), which will date with MEM-7. "by one of the authors of `torch.compile`" is a loose attribution that I cannot confirm (I believe the author worked on the PyTorch compiler stack, in particular functorch and AOTAutograd); "by a PyTorch compiler developer" is safe. The entries use a dash as separator between title and annotation; this is the resources format of every chapter introduction, so I record it only as a book-wide convention that conflicts with the author's preference against dashes in prose.

## Code and experiments

None.

## Still open from October

Nothing specific to this file.

## For the chapter review

- The introduction's method sentence omits prediction (IDX-2); the chapter review should say whether the sections deliver the method the reviewers expected or the weaker one stated here.
- Two of the seven route sentences promise more than their sections deliver (IDX-4).
- The opening question is the chapter's frame and needs its answer in the case study, in milliseconds.
- No appendix reference. One reference to the Language Models part, as a scope note, not load-bearing.
- One reference points at the wrong section (IDX-5).
