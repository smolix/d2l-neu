# State Space Models (chapter introduction)

Source: `chapter_recurrent-modern/index.md` (lines 1 to 89; the file has no slides section). About 330 words of introduction and about 650 words of resources (15 entries).

## Verdict

A short, correct introduction with a good first paragraph: the chat assistant that holds a key and a value for every token of a long report, and the question whether a memory of fixed size can do as well. It then names the two problems of a fixed state (sequential training, forgetting) and the three sources of the answers (gates, linear dynamical systems, attention without the softmax), which is the chapter's real plan and is well put. After that it stops leading. The route is six sentences of the form "X asks how", it does not say which sections carry the chapter and which a reader may skip, and it does not say where the story ends: that no recurrent model in the chapter replaces attention and the last section puts attention back. The opening question ("still model language well?") is one the chapter's experiments cannot answer, since every model is tested on 128-character windows where memory hardly matters. The most important fix is to replace the route by a paragraph with a judgment in it, and to restate the question as one the chapter does answer.

## Scores

| Dimension | Score |
|---|---|
| Clarity of explanation | 4 |
| Clarity of thought and focus | 3 |
| Pedagogy | 3 |
| Language | 4 |
| Ease and enjoyment | 3 |

## What works

- The opening case: "When a chat assistant answers a question about a long report, it holds a key and a value for every token of the report". Concrete, tied to `sec_kv-cache`, and it is the case `hybrids.md` closes on with its cache arithmetic. Keep.
- "generating the millionth token costs as much as generating the tenth." The benefit in one clause.
- "The answers in this chapter come from three directions: gates from the 1990s, linear dynamical systems from physics and control theory, and attention without the softmax. The three meet in recurrent layers that train in parallel and compute a form of attention." This is the chapter's thesis, and it is a claim with content.
- The scope statement: "the chapter teaches the mechanisms (state, update, readout, and cost) rather than a ranking of current models". The right policy for a field that changes yearly.
- Prerequisites are stated in one sentence, and the one section with a specific prerequisite (`sec_attention-at-scale` for `sec_matrix-state`) is named.

## Structural findings

1. **The introduction does not say how the story ends.** A reader arriving from the transformer chapter wants to know whether these models replace attention. The honest answer, which the chapter reaches in its last section, is no: they replace most attention layers and a few remain. The introduction mentions hybrids in one closing clause of the second paragraph. Say it as the chapter's conclusion, up front: a fixed state cannot look up an arbitrary earlier token, the chapter shows how far a fixed state goes, and its last section shows what a little attention adds.
2. **No guidance on what matters and what can be skipped.** A reader who wants the main line needs the gated sum and the half-life from `sec_lstm`, the linear recurrence and the scan from the first half of `sec_ssm`, the outer-product memory and its capacity from `sec_matrix-state`, the delta rule, and the recall experiment. The continuous-time half of `sec_ssm`, the chunked algorithms and the layer details can be skipped at first reading. The introduction should say so in two sentences.
3. **The title and the content.** The chapter is called "State Space Models"; two of its six sections (`sec_ssm`, `sec_mamba`) are about state space models in the technical sense, and the last three are about linear attention with a matrix state. The introduction never says in what sense the whole chapter is about state space models (a state of fixed size, a linear update, a readout). One sentence would justify the title and give the reader the common frame: the scope statement's "state, update, readout" is that frame and should be promoted from a parenthesis.
4. **The resources list is long and unranked.** Fifteen entries in four groups, each with a one-line annotation of similar weight. The annotations are informative and mostly sober. What is missing is a recommendation: which one or two to read first. See IDX-6.

## Passage-level findings

**IDX-1** (moderate, thought). Heading: introduction. Quote: "of the report. Can a model read a long text with a memory of fixed size and". The question ends "still model language well?" The chapter cannot answer it: its only language experiment is character-level Penn Treebank on windows of 128 characters, where the section reviews found that all models lie between about 2.96 and 3.36 and that `lstm.md` itself says the task "rewards long memory little". What the chapter does answer is narrower and better: what can a state of fixed size keep, how is it trained in parallel, and what can it not do. Restate the question in those terms, and say that the evidence on language at scale is in the cited literature.

**IDX-2** (moderate, language). Heading: introduction. Quote: ":numref:`sec_lstm` asks how a network can learn what to keep.". Six consecutive sentences with the verb "asks" (seven question clauses). This is the route template the Basics review flagged (seven "X asks" sentences in chapter 4). The questions themselves are well chosen; the form hides the argument that links them. Draft: "Gates let a network learn what to keep, but a gate that reads the previous state forces training to run step by step (:numref:`sec_lstm`). Make the update linear in the state and all positions can be computed at once (:numref:`sec_ssm`); let each token set its own decay and the model can keep a name and drop a comma (:numref:`sec_mamba`). The state of such a model turns out to be a table of key and value pairs with a capacity we can calculate (:numref:`sec_matrix-state`) and a write rule we can improve (:numref:`sec_deltanet`). What no table of fixed size can do, and how little attention it takes to repair it, closes the chapter (:numref:`sec_hybrids`)."

**IDX-3** (minor, correctness). Heading: introduction. Quote: "it once per token, so generating the millionth token costs as much as". True for the recurrent models. The paragraph then says of classical networks that training "leaves most of a GPU idle", which is fair. But the chapter introduction promises "cost" as one of four things taught ("state, update, readout, and cost"), and the section reviews found that no section gives the cost of a recurrent layer against an attention layer in work, sequential depth and memory (see the chapter review). Either the promise is kept by adding that comparison, or "cost" should not be in the list.

**IDX-4** (minor, language). Heading: introduction. Quote: "sequence one step at a time and leaves most of a GPU idle. And a state of fixed size must forget, which raises the question of what it should". An unreflowed source line (163 characters), the trace of a patch. The section reviews found the same in `lstm.md`, `ssm.md` and `mamba.md`. Reflow.

**IDX-5** (minor, thought). Heading: introduction. Quote: "ranking of current models. It teaches algorithms rather than GPU kernels,". A good promise that the chapter does not fully keep: the reviewer of `matrix-state.md` found that from "Chunked Computation" on that section "reads as kernel documentation" (MS-11, MS-13), and `mamba.md` has a subsection on the memory of the scan. Keep the sentence and hold the sections to it.

**IDX-6** (minor, pedagogy). Heading: Resources and Further Reading. Quote: "of the ideas, surveys, and course counterparts. All are freely accessible". Fifteen entries, none marked as the one to start with. Suggest opening each group with a recommendation ("If you read one thing: Gu's essay on the tradeoffs of SSMs and transformers, which argues the point of :numref:`sec_hybrids`"), and cutting to about ten. Candidates to cut: the ASAP seminar (a venue, not a resource), one of the two Mamba implementations, and the second course. The entries use " — " as a separator between title and annotation, 25 times; the house rule against em dashes applies to prose, and a list separator may be exempt, but a colon would serve.

**IDX-7** (minor, correctness, unverified). Heading: Resources and Further Reading. Quote: "a bonus chapter of *Build a Large Language Model (From Scratch)* that implements the Gated DeltaNet layer of Qwen3-Next and Kimi Linear". Three claims about external material that I cannot check from here: that the linked directory `ch04/08_deltanet` exists under that name, that Stanford CS336 "Lecture 4" covers attention alternatives and mixtures of experts (lecture numbering changes by year, and the link goes to the course root), and that CMU 10-423 lecture 22 is the SSM lecture (the link pins the spring 2025 offering). Have someone open the three links; cite a year for the Stanford lecture.

## Code and experiments

None.

## Exercises

None, as for other chapter introductions.

## Still open from October

Not checked against the October review beyond the section reviews' notes. The private vocabulary October objected to in this chapter ("ladder", "rung") does not occur here.

## For the chapter review

- The introduction states the thesis ("the three meet") and the scope (mechanisms, not rankings) correctly and briefly. It lacks a conclusion (hybrids), a reading guide, and a question that the chapter's experiments can answer.
- "Cost" is promised and not delivered in the sections.
- The route template of the Basics chapters recurs here unchanged.
- The division of labour between this file's resources list (material that explains what the chapter teaches) and `further-reading.md` (topics the chapter does not teach) is stated in both files and is sound.
