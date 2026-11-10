# Further Reading

Source: `chapter_optimization/further-reading.md` (lines 1 to 399; the file has no slides). About 3,300 words in five groups and 22 topics, citing 57 works. No code, no figures, no exercises. Read after the ten section reviews.

## Verdict

As an annotated bibliography this is well above the usual standard. Almost every topic starts from something the chapter did, says what problem the other work addresses, and says what it changes, in four to eight sentences; several entries carry a judgement ("so it is a record of engineering rather than a benchmark of optimizers"; "The theory is clean, but on deep networks these methods have not given consistent improvements"). The descriptions of the papers I know agree with the papers. Three things keep it from doing its job fully. It does not rank: 22 topics stand at equal weight, with no word on which three a reader should follow first and which are for specialists. A quarter of it is about Muon and its variants, the same disproportion the section review found in `muon.md`, and five of those paragraphs lean on the variant tour in `muon.md` that the section review recommends moving here, so the two files have to be revised together. And about ten statements report results of papers from 2025 and 2026 with numbers ("up to six times less time", "one to two percent", "a tuned Muon ties NorMuon") that I could not confirm and that will date first. The single most important fix: an opening paragraph that ranks, and a rule for the frontier entries (say what the method does; quote a number only when the chapter's own argument needs it).

## Scores

| Dimension | Score |
|---|---|
| Clarity of explanation | 4 |
| Clarity of thought and focus | 3 |
| Pedagogy | 3 |
| Language | 4 |
| Ease and enjoyment | 3 |

## What works

- **Each topic is anchored in the chapter.** ":numref:`sec_scheduler` explained warmup through the sharpness of the loss at initialization. Gotmare et al. asked instead what warmup changes inside the network." The reader knows at once what is new. Keep this device in every entry.
- **Sharpness is given both sides in one paragraph**: what sharpness-aware minimization does, what it costs ("at twice the gradient cost per step"), and the objection of Dinh et al. that sharpness "depends on how the function is parametrized". This is how a contested idea should be reported.
- **Connected minima** ties the permutation argument of the first section to mode connectivity and to weight averaging in four sentences, and ends on why it matters ("the basis of methods that merge separately trained models").
- **"Hyperparameter Transfer in Production"** continues :numref:`sec_scaling` with three cases (muP used as designed, standard parametrization with fitted power laws, a method reported without being published) and closes with the honest limit: "Transfer in production now extends beyond width, the one direction the chapter covers."
- **"Layerwise Adaptive Learning Rates"** uses LARS and LAMB as the worked example of the chapter's rule about baselines. A reader remembers the method because of the story.
- **Candour about a benchmark**: the speedrun's headline time "is a record of engineering rather than a benchmark of optimizers".
- The scope statement is exact: "Nothing else in the book depends on this section." I checked: `sec_optimization-further-reading` is cited only from the chapter introduction and from the further reading of the appendix chapter.
- Every appendix pointer I followed lands on a passage that holds the cited content (`subsec_mdl-quasi-newton`, `subsec_mdl-gd-smoothness` for the line search, `subsec_mdl-trust-region`, `subsec_mdl-fixed-budget-rates`, `subsec_mdl-nonconvex-sgd`, `sec_mdl-variance-reduction`). All 57 citation keys exist in `d2l.bib`.

## Structural findings

1. **No ranking** (FR-1). The opening lists the five groups and promises that each topic "states the problem it addresses and what it changes relative to the chapter". It does not say what to read first.
2. **Muon takes a quarter of the section.** "Steepest Descent Under a Chosen Norm", "Faster Orthogonalization", "Distributed Orthogonalization", "Muon from the Speedrun to Production", half of "Attention-Logit Stability" and half of "Benchmarks and Their Protocols" come to about 850 of 3,300 words, against about 120 words for all of second-order optimization without the Hessian and 110 for variance reduction. The chapter body has the same tilt. If the variant material of `muon.md` moves here (see the section review, structural finding 1), the Muon part of this section would exceed 2,000 words. Then it needs to be a single subsection with its own lead paragraph that ranks ("what changed Muon's use in practice: weight decay and the update scale; what makes it cheaper; what is still open"), and the older classical topics need to keep their share.
3. **Five entries assume the variant tour of `muon.md`.** "Dion, in :numref:`sec_muon`", "NorMuon's per-neuron scaling", "The QK-clip of :numref:`sec_muon`", "the RMS-matching factor of :numref:`sec_muon`" and "a tuned Muon ties NorMuon" are written for a reader who has just read "Per-Neuron Normalization" and "Muon at Scale". If those subsections are cut from the section, Dion, NorMuon and QK-clip must be introduced here in one sentence each. This is the one place where two files of the chapter must be edited in the same pass.
4. **The section holds two facts that the body needs and lacks.** (a) Moonlight "found that without decay, Muon falls behind AdamW late in long runs as the weight norms grow", and Kovalev's reading "explains why it needs weight decay". The body of `muon.md` displays and implements Muon without weight decay and runs its comparison with decay off (MUON-7). (b) "rankings can flip late in a run as the learning rate decays, so optimizers should be compared at the end of training rather than along the way". The body compares optimizers at a constant rate and leads with the early gap (MUON-4, ADAM-12). Both belong in the teaching sections, in a sentence each; here they are too far from the experiments they qualify.
5. **Looked for and not found.** A pointer to the book's own chapter on hyperparameter optimization (`chapter_hyperparameter-optimization`, in the Attic), which is where a reader who wants automated tuning should go from :numref:`sec_practice`. A pointer to the further reading of the appendix chapter for the theory texts. Sign-based methods beyond one sentence in `muon.md` (Lion). None of these needs more than a line.

## Passage-level findings

**FR-1** (moderate, pedagogy). Opening paragraph.
Anchor: "topic states the problem it addresses and what it changes relative to the"
Add a ranking in the authors' voice. Draft: "If you follow three threads, take these. For what the loss surface of a network looks like, 'Connected Minima' and the sharpness debate. For why deep learning does not use the classical second-order methods, 'Newton Steps Without the Hessian'. For how to judge a claimed improvement, 'Benchmarks and Their Protocols'. The Muon topics are a snapshot of a field that moved every month while this was written; read them for the ideas, and expect the numbers to have changed."

**FR-2** (moderate, correctness; medium confidence). Under "Rates for Stochastic Gradient Descent".
Anchor: "$1/\sqrt{T}$ after $T$ steps if its iterates are averaged"
The $1/\sqrt{T}$ bound for convex objectives with bounded gradients and averaged iterates is the classical result of Nemirovski and Yudin (and, for the online setting, Zinkevich). Polyak and Juditsky (1992) showed something different: averaging the iterates of a run with slowly decaying steps attains the optimal asymptotic rate and constant on strongly convex problems. Cite Nemirovski and Yudin for the bound and for the lower bound, and keep Polyak and Juditsky for what averaging buys, which is also the root of :numref:`sec_practice`'s weight averaging.

**FR-3** (minor, thought). Same subsection.
Anchor: "ball, but quoted no rates. For convex objectives with bounded gradients, SGD"
The section says of :numref:`sec_sgd` that it "quoted no rates". The review of `sgd.md` lists exactly this as looked for and not found (its structural finding 7): one sentence there, that noise lowers the rate on the valley from geometric to $1/t$, would answer "what is lost" where the question is asked. This paragraph can then begin from that sentence.

**FR-4** (minor, correctness). Under "Connected Minima".
Anchor: "found such permutations by matching weights or activations and confirmed"
"confirmed the conjecture for wide networks" is stronger than a set of experiments can be. Ainsworth et al. found permutations that leave almost no barrier for sufficiently wide networks on the tasks they tried, and report failures for narrow ones. "and found almost no barrier for wide networks" says that. In the sentence before it, the "polygonal chain with one bend or a quadratic Bézier curve" is Garipov et al.; Draxler et al. found their paths by a different method. Attribute the curve shapes to the first paper only.

**FR-5** (minor, correctness; unverified, low confidence). Under "Steepest Descent Under a Chosen Norm".
Anchor: "norm, which explains why it needs weight decay. On where Muon's gain over"
I know Kovalev (2025) as the reading of orthogonalized updates as a trust-region method under the spectral norm. I cannot confirm that the paper derives the need for weight decay; the empirical statement is Moonlight's, which the section cites 130 lines later. Check the paper, or attribute the weight-decay point to Moonlight only.

**FR-6** (moderate, focus; unverified). Under "Faster Orthogonalization" and "Distributed Orthogonalization".
Anchors: "improved the speedrun times of the next" and "to six times less time spent in the optimizer"
Nine methods in two paragraphs, each with one sentence and most with a number: one to two percent, up to twice as fast, one of five steps, a quarter or an eighth of the rows, up to six times less time. Four of the cited works are dated 2026 (`Zhang.Amsel.Chen.ea.2026`, `Amsel.Zhang.Ahn.ea.2026`, `Kim.Oh.2026`, `Jordan.2026`) and I could not check any of their numbers; the 2025 ones (CANS, Turbo-Muon, MuonBP, the coefficient search) match what I know in kind but not to the digit. This is a variant list at equal weight. Group by idea and drop most numbers: (a) better polynomials (the heuristic search, CANS, Polar Express in the body); (b) cheaper arithmetic for the same polynomial (Gram form, preconditioning the input); (c) orthogonalizing less (low rank, a subset of rows, per shard). End on the one theoretical statement, which is the useful one: a few steps lose almost nothing against exact orthogonalization. Note that the body's own measurement on a real gradient qualifies that statement for five steps of the quintic (MUON-1). The first anchor also holds a stale pointer: "the speedrun times of the next paragraph" refers to the speedrun, which is described five subsections later under "Muon from the Speedrun to Production", not in the next paragraph.

**FR-7** (minor, correctness; unverified). Under "Kronecker-Factored Preconditioners".
Anchor: "SOAP is among the fastest methods in the matched-tuning studies cited in"
Plausible from what I recall of the Wen et al. study (Muon and SOAP as the matrix methods that keep a small gain). The plural "studies" needs a second one: before its variants part, `muon.md` cites one matched-tuning study (`Wen.Hall.Ma.ea.2025`); the second (Semenov et al.) is cited only in this file. Write "study", or name both here.

**FR-8** (minor, thought). Under "Steepest Descent Under a Chosen Norm".
Anchor: "which :numref:`sec_adam` located Adam's own advantage over SGD."
:numref:`sec_adam` reports that Kunstner et al. located it there; its own experiment shows the same modest gap on both test models and does not separate frequent from rare tokens (ADAM-12). If the experiment of ADAM-12 is added, this sentence becomes true as written. Until then: "the regime in which the papers cited in :numref:`sec_adam` locate".

**FR-9** (minor, correctness). Under "Warm Restarts".
Anchor: "grow in length. Each cycle ends with a decayed model, so one run yields"
The use of the end-of-cycle models as an ensemble is the snapshot-ensembles idea of Huang et al. (2017), which built on warm restarts; Loshchilov and Hutter mention the possibility but the method and its results are in the later paper. One clause or one citation. This subsection also answers what the review of `lr-scheduler.md` looked for (one sentence on warm restarts, since the cosine citation is the warm-restarts paper); a pointer from there to here closes that.

**FR-10** (minor, thought). Under "Smaller and Sharded Optimizer State".
Anchor: "Adam keeps two numbers per parameter, which for a large model costs more"
"costs more memory than the weights" is twice the weights at equal precision; say "twice". The paragraph is the third place in the book with the ledger of optimizer memory, after "Optimizer-State Memory" in `adam.md` and :numref:`sec_training_systems`, and the first two disagree with chapter 13 about where bf16 copies live (ADAM-21). Whatever home the ledger gets, this paragraph should cite it and keep only the three ways to shrink the state.

**FR-11** (minor, language). Under "A Training Logbook".
Anchor: "is rarely published. The OPT team :cite:`zhang2022opt` released its"
The citation key is in a different style from the other 56 (`Author.Author.Year`). Cosmetic, but a linter will catch it later than a reviser can fix it now. "two months" is as I remember the logbook; unverified.

**FR-12** (minor, language). Whole file.
Anchor: "### A Method Without a Learning Rate"
Headings name topics, which is right for a reference section, except this one, which is a riddle for Adadelta. Call it "Adadelta". The entry itself is good: it says in three sentences why the method has no learning rate "in name" and where the scale comes from in practice.

## Statements about papers that I could not confirm

For the editing pass, in one place. None is implausible; each should be checked against its source or lose its number.

- Cesista, You and Jordan: "six quintics", "one to two percent".
- Gram Newton–Schulz: "only two multiplications by the rectangular matrix", "up to about twice as fast".
- Turbo-Muon: "saves one of the five steps".
- Kim and Oh: "doubly exponentially fast in $q$".
- Dion3: "a quarter or an eighth", "largest $\ell_1$ norms", "up to six times less time".
- MuonBP: "a separate learning rate for each kind of step".
- Semenov, Pagliardini and Jaggi: the three rankings (AdEMAMix, Muon with decay, Muon without).
- Modded-NanoGPT optimization track: "a tuned Muon ties NorMuon".
- GLM-4.5 "pretrained with Muon" (I believe so); Llama 4's transferable per-layer rates (matches Meta's announcement as I recall it).
- Wang et al.: gain concentrated in value and output projections and MLP weights (matches my memory of the abstract).

Confirmed from memory with confidence: Li et al. on filter normalization; Foret et al.; Dinh et al.; Liu and Nocedal; Martens (2010) and the autoencoders; Dauphin et al.; Ghadimi and Lan; Adadelta; K-FAC and the natural gradient; SOAP's construction; Scion's per-layer norms; Adafactor, 8-bit optimizers and ZeRO; schedule-free methods and the AlgoPerf results; LAWA, model soups and the post-hoc EMA of Karras et al.; the speedrun record of about a fifth (31.4 to 24.9 minutes); Moonlight's weight decay finding; QK-norm; Nado et al.; Cerebras-GPT and DeepSeek's power laws; the OPT logbook. I also checked one internal claim by hand: Newton's method in :numref:`sec_gd` does end at a local maximum ($f''(26.83) \approx -6$ for $f(x) = x\cos(0.15\pi x)$), as "climbed to a local maximum" says.

## Code and experiments

None.

## Exercises

None, correctly.

## Still open from October

Nothing specific to this file.

## For the chapter review

- The section is the right home for the variant material that the reviews of `muon.md`, `batch-size.md`, `adam.md` and `practice.md` want out of the teaching sections (NorMuon and AdaMuon, communication accounting and Dion, the Polar Express construction if the appendix does not take it, the large-batch history, the z-loss). Taking all of it would add 1,500 to 2,000 words. Without a ranking and a firmer rule about numbers, the section would then become the survey the teaching sections were relieved of. Take the ideas, not the paragraphs.
- Two facts stated only here qualify experiments in the body: Muon needs weight decay in long runs, and optimizer rankings should be read at the end of a decayed run (structural finding 4).
- The split of labour with the chapter introduction's "Resources" (material for what the chapter teaches there, topics it does not teach here) is stated in both places and is kept.
- Pointers into the appendix are all accurate. This is the only file of the chapter that sends the reader to the appendix for SGD rates, variance reduction, quasi-Newton methods and trust regions; `sgd.md` itself has no pointer to the appendix at all.
