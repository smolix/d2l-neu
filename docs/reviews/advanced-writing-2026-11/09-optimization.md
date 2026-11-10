# Chapter 9, Optimization Algorithms: writing review

Sources: the twelve rendered `.md` files of `chapter_optimization/` in `_quarto.yml` order (introduction, ten sections, further reading), each read in full up to the slides marker by a section reviewer or, for the introduction and the further reading, by me. About 41,000 words of body prose, 10,000 words of exercises, 3,300 words of further reading. Stored PyTorch and JAX outputs were checked against the prose by the section reviewers; I rechecked the claims marked "verified" below in the sources and in `outputs/`. No source text was changed.

Section reviews:
[index](09-optimization/index.md) ·
[optimization-intro](09-optimization/optimization-intro.md) ·
[gd](09-optimization/gd.md) ·
[sgd](09-optimization/sgd.md) ·
[momentum](09-optimization/momentum.md) ·
[adam](09-optimization/adam.md) ·
[lr-scheduler](09-optimization/lr-scheduler.md) ·
[muon](09-optimization/muon.md) ·
[batch-size](09-optimization/batch-size.md) ·
[scaling](09-optimization/scaling.md) ·
[practice](09-optimization/practice.md) ·
[further-reading](09-optimization/further-reading.md)

**Author decisions (Alex Smola, 2026-11-10).** (a) `adam.md` is split into Adam, the two test models and AdamW ("Split adam is OK"); the harness section goes after `sgd.md` as recommended below unless the author says otherwise. (b) Results sentences that rest on a gap inside the run-to-run noise are replaced by the firm lesson the run does show, not by a hedge: "OK with weaker sentences IF the lessons don't come out as vague." For this chapter the section reviews name those lessons (early-training lead, width of the usable range, position and drift of an optimum). See `overview.md`, "Author decisions".

## Verdict

This is a strong chapter with a method, and in a dozen places it is at the bar. The method is to predict a number, run, and read the output: the 47 steps of gradient descent on the valley and the condition number that explains them, the noise ball whose two radii the formula gives before the cloud is measured, the table of four momentum settings with a deliberately bad one, the first Adam step that grows with width, the critical batch size fitted on two models, clipping that touched 13 of 2,000 updates. The mathematics is nearly clean: the reviewers found one wrong explanation (MOM-5), one misleading experiment reading that reverses the long-run facts (ADAM-3), one statement of a published rule that does not match the rule (SCALE-6), and one stored output that contradicts the text above it (SCALE-9). The material is current for a book of 2026 to 2027, and the prose is plain and almost free of mannerisms.

It is not yet an excellent chapter, for four reasons, in order of weight. (1) **The experiments on networks do not carry the claims made for them.** Momentum is never shown to help on data. Adam's advantage, the Muon comparison, the AdamW grid, the Adam batch-size rule and the muP transfer cost each rest on a single-seed difference of the size of the run-to-run noise, and in each case the prose then explains what the experiment did not show or retells the papers that did. What these experiments do show robustly (early-training gaps, the width of the usable range, the position of an optimum) is usually on the page and not led with. (2) **The chapter is built as valley, then network, and the network arrives inside the Adam section**: a 370-line test harness between the definition of Adam and the first evidence about it, in a file that is three sections. (3) **Every section from Adam on ends with a tour of recent papers**, results quoted and not reproduced, and one optimizer of 2024 has as many words as SGD and gradient descent together. (4) **The chapter does not end with the reader able to do what the part introduction promises**, "how to choose a learning rate, schedule and batch size": every ingredient is taught and no page assembles them. Fix the second and fourth by restructuring, the third by cutting to Further Reading, and the first on the development machine, with prose that claims less in the meantime.

## Scores

1 to 5 as the section reviewers gave them; 5 means a page one would be happy to find in MacKay.

| Section | Explanation | Thought | Pedagogy | Language | Enjoyment |
|---|---|---|---|---|---|
| [index](09-optimization/index.md) | 4 | 3 | 3 | 3 | 3 |
| [optimization-intro](09-optimization/optimization-intro.md) | 4 | 3 | 4 | 4 | 4 |
| [gd](09-optimization/gd.md) | 4 | 3 | 3 | 4 | 3 |
| [sgd](09-optimization/sgd.md) | 4 | 3 | 3 | 4 | 3 |
| [momentum](09-optimization/momentum.md) | 4 | 4 | 3 | 4 | 4 |
| [adam](09-optimization/adam.md) | 4 | 3 | 3 | 4 | 3 |
| [lr-scheduler](09-optimization/lr-scheduler.md) | 4 | 3 | 4 | 4 | 3 |
| [muon](09-optimization/muon.md) | 4 | 3 | 3 | 4 | 3 |
| [batch-size](09-optimization/batch-size.md) | 4 | 4 | 4 | 4 | 3 |
| [scaling](09-optimization/scaling.md) | 4 | 3 | 4 | 4 | 4 |
| [practice](09-optimization/practice.md) | 4 | 3 | 3 | 4 | 3 |
| [further-reading](09-optimization/further-reading.md) | 4 | 3 | 3 | 4 | 3 |

The scores do not discriminate. Explanation is 4 in all twelve rows and language is 4 in eleven; only thought and pedagogy move, between 3 and 4. I see no difference in calibration between the reviewers: they describe the same shape in section after section (a first part at 4 to 5, a later part at 2 to 3, averaged to 3 or 4), and say so in words ("The part up to 'Sampling without Replacement' would score 4 or 5 on every line; the part after it, 2 or 3"). Read by their verdicts instead:

- **Strongest:** `batch-size.md` (one question, a measurable quantity, a prediction that could fail, ranges that hold in both tabs); `momentum.md` on the valley; the first half of `scaling.md`; the valley and edge-of-stability parts of `optimization-intro.md`; the first third of `sgd.md`, of `adam.md` and of `muon.md`.
- **Weakest:** `gd.md` ("competent and correct, but ordinary", and it re-answers a question the previous section answered); the second half of `sgd.md` (hardware and helpers); `adam.md` as a file (three sections, and its central comparison does not show its subject); `practice.md` as a whole (four topics and no procedure); the last third of `muon.md` (variants).

## Technical correctness and currency

No false theorem and no failing derivation. Findings with dimension `correctness`, by severity; "verified" means I rechecked it.

**Stored output against prose, or a reading that the output does not support**

1. SCALE-9 (major; verified in `outputs/`). The transfer cost of the width-128 learning rate at width 1,024 prints 20% (standard) and 5% (muP) in PyTorch and 2% and 6% in JAX. The JAX reader sees the opposite of the paragraph above the output, and "Under muP the best loss also falls" fails there too (0.286 then 0.293).
2. ADAM-3 (moderate; reviewer confident, recomputed for 200 steps). "RMSProp gets close to the minimum where AdaGrad stalled" is true at step 20 and reversed later: RMSProp ends oscillating at about $\eta/2$, AdaGrad converges. The cause is a property the chapter needs and never states: a normalized step does not shrink with the gradient.
3. ADAM-18 (major, filed under thought). The conclusion of the decay grid ("a column for AdamW, a whole grid for the coupled form") rests on minima that move by one grid step; the two tabs differ by 0.4 nats in one entry; the fixed-point paragraph cannot explain a difference between the two forms.
4. SGD-11 (moderate). The shared prose describes a "level part" of the time-per-example curve that only the PyTorch output shows.
5. SGD-13 (major, filed under pedagogy). The wall-clock race is won by full-batch gradient descent, the method the section argues against, and the prose says "only because the dataset is tiny".
6. ADAM-17 (moderate; medium confidence, not run). The explanation of the collapse under coupled decay (learned, then forgot) does not fit the stored curve, which never falls much below the unigram entropy.
7. LAND-11 (moderate). "though not on every step": no output shows the loss rising on single steps, the defining property of the regime.
8. PRAC-12 (moderate). "Gradient noise in language models is heavy-tailed" is asserted directly after a cell that recorded 2,000 gradient norms with none above four times the median after step 12.
9. SCHED-4, SCHED-9, MUON-4 (moderate, filed under thought and pedagogy): readings that go beyond the stored results; see Experiments.
10. Smaller: LAND-8 (46 against the 47 measured), GD-17 ("seven steps" not printed), MUON-14 (three iterations already suffice on the Gaussian matrix), MUON-13 (the round-off floor), SCHED-14 ("two to three points" is one to two), PRAC-10, PRAC-19, BATCH-10.

**Errors and statements that need a hypothesis**

11. MOM-5 (moderate; reviewer confident; verified: $0.635^{31} \approx 10^{-6}$ and $40 \cdot 0.635^{40} \approx 5 \times 10^{-7}$). At the tuned setting the error falls as $t\rho^t$ at every $t$, not only "after the first few steps"; the appendix has it right.
12. SCALE-6 (moderate; reviewer sure of the published rules). "The initialization column is the one that does *not* change" is not what muP prescribes for the output matrix; `MuMLP` implements a harmless variant and presents it as the rule.
13. MUON-3 (moderate; confident). The output head is a dense matrix, so "do not transform activations through a dense matrix product either" is false for it; keeping it on AdamW is an empirical choice.
14. MUON-6 (moderate). An entrywise RMS of $0.2\,\eta$ holds for exact orthogonalization; the section's own measurement shows about two thirds of that on a real gradient.
15. ADAM-21 (moderate; verified against `fast-transformer.md`). "The forward and backward passes use bf16 weights and gradients, 2 bytes each" contradicts the recipe of chapter 13, where "The weights, their gradients, and Adam's moment estimates stay in fp32". The same 112 GB ledger is worked again in `training-systems.md`.
16. SGD-4 (moderate, by omission). The noise ball assumes noise of constant variance. For a model that fits its training data the noise vanishes at the minimum and constant-step SGD converges; the reader of this book trains such models.
17. LAND-10 (moderate). The condition number as defined is not finite for a network whose Hessian has zero and negative eigenvalues; the text changes the definition without saying so. LAND-17: the Summary's "Real networks ... train at the edge of stability" generalizes from full-batch gradient descent.
18. GD-6 (moderate). The run at $\eta = 2$ is explained as an accident of the first jump; the deep minimum is unstable at that rate from any start ($f'' \approx 1.81 > 2/\eta$). GD-12: the storage of the Hessian is the weakest argument against Newton's method, since Hessian-vector products avoid it.
19. SGD-9 (moderate). "on the order of a hundred floating-point operations" per byte is the bf16 figure; the experiment that follows runs in float32.
20. SCHED-5 (moderate; medium confidence, not run). Batch normalization without weight decay makes the effective rate fall by itself; the schedule comparison and the warmup result at a nominal rate of 7.5 are confounded by it. SCHED-2 (minor; confident): a superfluous condition on the quadratic.
21. Smaller, all with a one-clause fix: GD-5 (smoothness assumption for the $O(\epsilon^2)$ remainder), GD-16, GD-14, MOM-7, ADAM-4 (RMSProp's default), ADAM-9 (verified: Basics gives 170,000 characters, not 100,000), ADAM-15 (geometric mean), SGD-14 and BATCH-5 and ADAM-11 (numbers no cell prints: $L \approx 2.1$, "less than a third", four byte-frequency statistics), BATCH-14, PRAC-1, PRAC-13.
22. Defective exercises: Exercise 1 of `gd.md` (all runs on $x^4$ overflow from the given start; the reviewer ran them) and Exercise 5 of `muon.md` (part (a) proves a property of wide matrices and part (b) applies it to `token_emb`, which is tall).

**Claims about papers and systems that the reviewers could not confirm**

LAND-12 (the mechanism of warmup in Kalra and Barkeshli; the reviewer's reading follows more directly from the section's own experiment), LAND-6 (what Dauphin et al. measured), SCHED-11 (which paper compared linear decay to zero), MUON-10 (NorMuon's 6% to 15%), MUON-12 (byte counts, Dion's rank), MUON-17 (the Sophia detail), ADAM-14 and ADAM-19 (attribution of the block-heterogeneity result; the averaging-horizon view is due to Wang and Aitchison, who are not cited), ADAM-20 (the OLMo 2 anecdote), BATCH-17 (who introduced warmup), PRAC-2 (two rows of the recipe table), PRAC-14 (the bfloat16 motive attributed to PaLM), PRAC-22, SCALE-3, and in my own two files IDX-5 (Boyd and Vandenberghe do not treat SGD; confident), FR-2 (the $1/\sqrt{T}$ bound credited to Polyak and Juditsky), FR-5 and the list at the end of the further-reading review. One of the reviewers' open items is settled by the chapter itself: MUON-7 could not confirm that Muon needs decoupled weight decay in long runs, and the Further Reading section states it with its source (Moonlight). The teaching section and the chapter's own further reading therefore disagree about what Muon's update is.

**Currency.** The chapter is up to date, and that is a strength: the edge of stability, warmup–stable–decay, the critical batch size, muP, Muon, the 2024 and 2025 pretraining reports. Three places describe practice a 2026 reader will not recognize without a sentence more: schedules are taught on plain SGD only, while the section's motivation is AdamW on language models (SCHED-10); Muon is shown without weight decay (MUON-7); the optimizer-memory ledger describes a mixed-precision layout the book does not use (ADAM-21). The frontier numbers in the closing subsections will date first; see Focus.

## Arc and order

The chapter has one line of argument and states it in its second paragraph: curvature differs between directions, the gradient is an estimate, and each method answers one of the two. Sections 1 to 4 (landscapes, gradient descent, SGD, momentum) follow it on one quadratic. Sections 5 to 7 are the optimizers and schedules in use. Sections 8 to 10 are about scale and practice. The line holds, and the hand-offs are mostly good ("how can SGD move faster along the flat directions ... where gradient descent is slow and the noise ball is widest?"). What needs to change:

**Sections that are really two or three.**

- `adam.md` is three: Adam (lines 1 to 382 and 943 to 1253), the test harness (383 to 942), AdamW and weight decay (1254 on). ADAM-8 and the structural findings of that review give the seams. `sec_adamw` already sits on a `##` heading and is cited as a section from 18 other files (verified), nine of them skeleton files of the Language Models part; it was a file of its own until the book-pass revision. Split it.
- `sgd.md` is two: noise (to the end of "Sampling without Replacement") and hardware plus helpers (SGD-1). The reviewer recommends shrinking the second part instead of splitting; I agree, and the harness section below absorbs what is left of it.
- `practice.md` is four short topics (guard, finish, choose, compare). Reorder so the chapter ends on choosing, and replace its closing (PRAC-24) by the missing procedure.
- `muon.md` has a seam before "Relatives and Variants"; the recommendation is to cut, not to split.

**Where the test harness should live (recommendation).** A short section of its own in this chapter, placed directly **after `sgd.md` and before Momentum**, titled for what it is ("Two Test Models"), carrying the labels `subsec_tinylm` and `subsec_fashioncnn` unchanged plus a new `sec_` label. Reasons:

1. *It is cited as a place.* `subsec_tinylm` is cited from chapters 10 and 11, and `d2l.TinyLM`, `d2l.train_steps` or `d2l.lr_sweep` are used in 17 files across chapters 9 to 13 and the Language Models skeletons (verified). A reader sent there from the attention chapter should land on a page about the model, not in the middle of Adam.
2. *The chapter turns from valley to network at one point, and that point should be visible.* Sections 1 to 3 need nothing but the valley. Everything after can then use a network. Placed before Momentum, the harness gives `momentum.md` the experiment it lacks (MOM-9: four runs on data and no gain) and removes the airfoil regression from the chapter's argument, where it serves two sections and shows the effect in neither (SGD-13, MOM-9).
3. *It is the natural home for three things that are now scattered.* The helper code that `sgd.md` narrates for later sections (`Timer`, `train_with_rule`, `train_with_optimizer`; SGD-16); the comparison protocol, which is stated for one comparison in `adam.md` and as a general classification only in the last 650 words of the chapter (PRAC-21); and the noise discipline ("A difference between two schedules means something only if it is larger than the difference between two runs of the same schedule", now in `lr-scheduler.md`), together with a measured run-to-run spread for both models that every later section can cite instead of quoting ranges from unprinted repeats.
4. *Its first run can be the baseline.* Trained with the minibatch SGD of the section before, the harness section produces the loss curve that Momentum, Adam and Muon then have to beat.

The section should say in its first paragraph that the transformer is used as an instrument two chapters before it is explained, with a pointer to chapter 11, and list the helpers in a small table (name, what it returns, first use) instead of narrating them (ADAM-8, ADAM-10). I would not move the harness to the tools appendix: chapters 10 to 13 cite it as part of the argument, and the main text should not depend on an appendix for its running example. If the author prefers a change that needs no new experiment, the fallback is the same section placed directly before Adam; that fixes points 1 and 3 and leaves Momentum without a network.

**Order.** Move Muon to follow AdamW, before Schedules (the reviewer of both sections proposes it from each side). Muon answers the same question as Adam, which direction, and the norm table (SGD, Adam and Muon as three balls) then closes the "direction" group. I checked the dependencies: `muon.md` does not mention schedules, warmup or cosine decay, and `lr-scheduler.md` does not cite `sec_muon`, so nothing breaks. The resulting order is: landscapes, gradient descent, SGD, test models, momentum, Adam, AdamW, Muon, schedules, batch size, scaling, practice. Twelve sections of 3,000 to 4,500 words in place of ten of 2,300 to 6,700.

**Boundaries that leak.**

- `optimization-intro.md` and `gd.md` both own gradient descent on a quadratic (LAND-1, GD-1). The first section derives the ceiling, the best rate and the condition number; the second opens by posing them as open and restarts from $x^2$. Assign the valley analysis to the first section and rebuild `gd.md` on what only it has: the descent guarantee with its three-line proof (GD-4), the general quadratic (GD-7, LAND-9; only one of the two files), the learning rate as a filter on minima (GD-6), Newton, preconditioning.
- `optimization-intro.md` previews `sgd.md` in three paragraphs without an example (LAND-14).
- The arithmetic-intensity argument, the cache remark and gradient accumulation in `sgd.md` belong to chapter 13 (SGD-10, SGD-12).
- The learning-rate rules in `batch-size.md` are used 250 lines before they are stated (structural finding 1 of that review, BATCH-9).
- The tuning protocol in `practice.md` governs every comparison of the chapter and arrives after all of them (PRAC-21).

**The chapter has no ending.** PRAC-24: the last body paragraph is a classification and a pointer to the next chapter.

## The most important findings

1. **Give the test models their own section and split Adam from AdamW.** ADAM-8, structural findings of `adam.md`, SGD-16, PRAC-21; placement as recommended above.
2. **Make the comparisons show a mechanism, or claim only what they show.** ADAM-12 (Adam against SGD: same modest gap on both models, an untuned edge of SGD's grid, three papers retold in place of the result; promote Exercise 3, frequent against rare tokens), MUON-4 (final gap of the size of the noise; lead with the early gap and the width of the usable range), ADAM-18 (the decay grid), BATCH-15 (the square-root rule cannot be separated from its neighbours at a batch ratio of 8), SCALE-9 and SCALE-10 (lead with the position of the optimum, which both tabs show).
3. **Show momentum helping on data, and remove the experiments that are explained away.** MOM-9 (four null results), SGD-13 (the race won by gradient descent), SGD-10 (a memory argument followed by a measurement of call overhead), SCHED-6 (warmup shown only at a rate nobody would use; the combined run never read).
4. **End the chapter with a worked procedure for configuring a new model.** PRAC-24, PRAC-5, structural finding 2 of `practice.md`, structural finding 3 of `batch-size.md`. The part introduction promises it, the chapter introduction's resource list describes it (IDX-6), and the ingredients are all taught.
5. **Assign gradient descent on the quadratic to one section and raise the level of `gd.md`.** LAND-1, GD-1, GD-2, GD-4, GD-6, GD-7.
6. **Cut Muon to about 4,000 words and test Newton–Schulz on a real gradient where it is introduced.** MUON-1 (the section demonstrates on a Gaussian matrix and admits 700 lines later that real gradients behave differently), MUON-2, MUON-7 (weight decay), MUON-9 to MUON-12 (variants to Further Reading).
7. **State three small results that the chapter keeps using and never states.** The scale invariance of Adam's update and what it says about the learning rate (ADAM-5); that a normalized step does not shrink with the gradient, so Adam at a constant rate does not settle (ADAM-3); the limit of the constant-variance noise model (SGD-4). Add from the exercises into the body: why the stable range widens with $\beta$ (MOM-3), why momentum at a matched step does not reduce noise (MOM-10), the rotational equilibrium under weight decay (ADAM-19), the three-line origin of the steps-to-target formula (BATCH-7).
8. **Fix the correctness items.** SCALE-9, MOM-5, ADAM-3, SCALE-6, MUON-3, ADAM-21, LAND-10, GD-6, and the two defective exercises.
9. **Read the network experiments of Schedules for what they show.** SCHED-4 (the final rate sorts the test accuracies; the "budget" does not show), SCHED-5 (batch-normalization confound), SCHED-9, SCHED-10 (say what carries over to Adam).
10. **Replace the paper tours by judgement.** ADAM-19, BATCH-18, BATCH-19, MUON-10, PRAC-3, PRAC-15, SCHED-8, SCALE-17; in each case two or three sentences in the authors' voice and a pointer to Further Reading, which needs a ranking to receive them (FR-1, FR-6).

## Focus

The first four sections have one question each and answer it. From Adam on, each section does its own work for about two thirds of its length and then closes with what the field does: five to eight citations at a sentence apiece, results quoted and not reproduced. Three reviewers describe this independently. The instances:

- `adam.md`: "Sources of the Gap" (three papers in place of the experiment, ADAM-12), "Weight Decay in Large-Scale Training" (500 words, three papers, no cell, ADAM-19), "Optimizer-State Memory" (ADAM-21).
- `muon.md`: about 3,000 of 6,100 words are relatives and variants: optimized polynomials, library versions, Shampoo, NorMuon with two cells, communication accounting, Dion, QK-clip (structural finding 1 of that review; MUON-10, MUON-11, MUON-12, MUON-17).
- `batch-size.md`: "Limits of the Rules" and "Growing the Batch", eight papers in 450 words (BATCH-18, BATCH-19).
- `scaling.md`: "The Spectral View", two sections' worth of claims in 400 words (structural finding 2 of that review, SCALE-7, SCALE-17).
- `lr-scheduler.md`: four worst-case results in eleven lines (SCHED-8).
- `practice.md`: the reading of the recipe table as counts (PRAC-3), the z-loss (PRAC-15), two benchmark papers.
- `optimization-intro.md` has the older form of the same fault: a catalogue of hazards (local minimum, cubic, saddle, plateau) at 700 words and four figures, after which the text says that what limits training "is usually something else" (LAND-4, LAND-5).
- `further-reading.md` is where this material belongs, and it has the same tilt: a quarter of it is Muon, and its frontier paragraphs are variant lists with numbers (FR-6).

Proportion follows from this. In words of body prose: Muon 6,100; Adam with AdamW 6,700, of which 1,080 is harness; Schedules 4,500; Batch Size 4,400; Practice 3,950; SGD 3,900; Momentum 3,000; Landscapes 3,000; Scaling 2,850; Gradient Descent 2,300. One optimizer of 2024, whose advantage the chapter itself reports as "tens of percent, not ... multiples" and shrinking with scale (PRAC-22), has as many words as SGD and gradient descent together.

The reader should finish knowing what matters and what can be skipped. The chapter says this well inside sections ("these conditions are a guide rather than a recipe"; "A model of 0.47 million parameters demonstrates the mechanism but cannot rank optimizers") and nowhere for the chapter as a whole (IDX-1).

## Experiments

The author's decision 4 and the guide's rule ("fix the experiment or remove it. Do not keep it and explain what it fails to show") apply more often here than anywhere in the appendix. Grouped by kind, with cell IDs in the section reviews.

**Experiments that do not show the phenomenon, kept and explained.**

- Momentum on the airfoil regression: final losses 0.244, 0.245, 0.263, 0.245, 0.244 across five runs (MOM-9).
- The wall-clock race in `sgd.md`, won by full-batch gradient descent (SGD-13); the batched matrix product, which measures a fixed cost per call and not the memory argument that precedes it (SGD-10).
- Adam against SGD on two models "chosen to contrast" that give the same gap, 0.06 and 0.05 nats, crossing at the same step (ADAM-12).
- Decay exemptions: a 2,000-step run to show that nothing happens (ADAM-20).
- Warmup at a nominal rate of 7.5, and a warmed-up cosine run whose result is never read (SCHED-6).

**Conclusions on differences within the run-to-run noise** (the author's decision 5).

- Muon hybrid against AdamW at step 2,000: 0.026 nats (PyTorch) and 0.042 (JAX), against a stated spread of about 0.02 (MUON-4).
- The coupled-decay minimum that "moves" by one grid step, with the two tabs 0.4 nats apart in one entry (ADAM-18).
- The Adam learning-rate shift for an eightfold batch: predicted 2.8, measured 1.9 and 3.2, stated range 1.5 to 5, from a parabola through three points of a flat basin (BATCH-15).
- muP transfer cost in percent (SCALE-9), from a last-iterate score after 400 steps at a constant rate (SCALE-2).
- SGD's "tuned" rate in the Adam comparison is the last grid point before divergence on a factor-of-three grid (ADAM-12), and `practice.md` then shows that clipping moves that edge (PRAC-11).

**Experiments read for more than they show.** The schedule comparison on the network, read as the valley formula "at work on a network" (SCHED-4); warmup–stable–decay, whose extra progress on this network is memorization (SCHED-9); the coupled-decay collapse (ADAM-17); the EMA, which is below the live weights for more than half of the run (PRAC-18).

**Unfair or confounded comparisons.** Both arms of the Muon comparison at a constant rate, one section after the reader learned what that does to the final loss (MUON-4; moot if Muon moves before Schedules, but then say that a decayed run lowers both); selection by training loss in one comparison and by validation loss in the other (MUON-18); batch normalization without weight decay under every schedule (SCHED-5); the CNN's learning rate not retuned per batch size (code and experiments of `batch-size.md`); a 15-epoch EMA run compared with 30-epoch decayed runs (PRAC-19).

**Checks that cannot fail.** Three parabola runs that multiply by 0.6 ten times (GD-2); the $1/b$ law measured by resampling the gradients whose covariance was just computed (SGD-7); per-neuron normalization printing that normalized rows have equal norm (MUON-9); `torch.optim.SGD` confirmed to be SGD (SGD-16); the optimizer-memory cell, which multiplies three constants (ADAM-21); two more momentum runs that reproduce a curve on the page (MOM-9). Confirming cells that earn their place, because they show where an approximation stops: the noise-ball measurement in `sgd.md`, the first-step checks of Adam and AdamW, the four schedules on the valley, the coordinate check.

**The two tabs.** Shared prose must be true of both. It is not in SCALE-9 and SGD-11, and it is only loosely true in the testbed cell of `lr-scheduler.md` (standard deviation 1.4 points in PyTorch, 0.3 in JAX). The reviewer of the last three sections found the contradiction only by reading the JAX outputs; this check should be repeated for every chapter of the part.

**What the experiments do show, and should be led with.** The early-training gap of Muon (0.15 to 0.25 nats in the first 500 steps) and the decade of learning rates over which its loss stays within 0.05 of its best; the collapse of coupled decay at $\lambda = 0.1$ to the unigram entropy; the final rate sorting test accuracy and removing its jitter; the optimum moving 4 to 8 times under standard parametrization and at most one grid step under muP; clipping firing on a dozen steps of 2,000.

A list for the development machine, each from a section review: a regime where momentum pays (MOM-9); frequent against rare tokens under Adam and SGD, with a sign-descent arm (ADAM-12); the Newton–Schulz spectrum of a real `TinyLM` gradient (MUON-1); a `TinyLM` run with AdamW that diverges cold and trains with a ramp (SCHED-6, SCHED-10); the batch-size rule at a ratio of 64 or by the largest stable rate (BATCH-15); the width sweep to 4,096 or three seeds (structural finding 1 of `scaling.md`); the noise scale over training as the closing experiment of `batch-size.md`; the per-step loss fraction at the edge of stability (LAND-11); a large enough regression for the wall-clock panel (SGD-13); the EMA started at epoch 5 (PRAC-18).

## Repetition

**Inside the chapter.**

| What | Where | Single home |
|---|---|---|
| The valley, its curvatures, the 30-step run, the 47 steps | Introduced in `optimization-intro.md`, re-introduced with its numbers in `gd.md` (GD-1, GD-9), `sgd.md` (SGD-3), `momentum.md` (MOM-1, MOM-2, structural finding 3), `adam.md` (ADAM-1); "47" appears three times each in the first two sections (verified) | `optimization-intro.md`; elsewhere one clause and a reference |
| Gradient descent on a quadratic (ceiling, best rate, $\kappa$) | `optimization-intro.md` and `gd.md` | `optimization-intro.md` |
| The $1/b$ law | Preview in `optimization-intro.md` (LAND-14), derived and measured in `sgd.md`, displayed again with its reason in `batch-size.md` (BATCH-2), and in Basics | `sgd.md` (`subsec_sgd-variance`) |
| The $\eta/b$ invariance | `sgd.md`, `lr-scheduler.md`, `batch-size.md` twice | A named fact in `sgd.md` |
| Robbins–Monro conditions with their glosses | `sgd.md` and `lr-scheduler.md` "Progress and Noise" | `sgd.md`; `lr-scheduler.md` keeps what is new (they say little for a fixed $T$) |
| The three decisions | `optimization-intro.md` ("Three Decisions", LAND-16), cited at the top of `lr-scheduler.md`, and "The Chapter's Three Decisions" in `practice.md` (PRAC-24) | The chapter introduction, once |
| Description of `TinyLM` ("two blocks, about 0.47 million parameters") | `adam.md`, `muon.md`, `batch-size.md`, `practice.md` (PRAC-9) | The harness section |
| Adam's results as ranges | "1.3 to 2.5 times as fast" three times, "0.03 to 0.2 nats" four times (ADAM-13) | Once, with the comparison |
| Muon's saving in optimizer state | Three times in `muon.md` (MUON-5) | "Parameter Groups" |
| "Noise is not only a nuisance" | Twice in `optimization-intro.md` (LAND-15) | With the saddle-escape result |
| The tuning protocol | `adam.md` (its own), `practice.md` (general) | The harness section, cited from both |
| The optimizer-memory ledger (112 GB) | `adam.md`, `training-systems.md`, a third telling in `further-reading.md` (ADAM-21, FR-10) | Chapter 13 or the tools appendix |
| Warmup | Mechanism in `optimization-intro.md` (LAND-12), taught in `lr-scheduler.md`, history in `batch-size.md` (BATCH-17), "adaptive warmup" in `practice.md` | `lr-scheduler.md`; the three accounts should agree on the mechanism and on who introduced it |

**With Basics.** Minibatch SGD, unbiasedness and the noise ball with its constant are in `subsec_linear-regression-minibatch-sgd`; `sgd.md` derives them afresh and cites Basics 240 lines in (SGD-2, structural finding 2 of that review). Cite at the opening and add what is new here, the curvature. Global-norm clipping is defined again in `practice.md` (PRAC-7; small). The risk against empirical risk figure of `optimization-intro.md` repeats `sec_generalization_basics` (LAND-3). The EMA of `sec_training_recipes` is cited and not re-derived, which is right. Optimizer objects are introduced in `sgd.md` as if new (SGD-16).

**With other Advanced chapters.** Chapter 13 owns arithmetic intensity, the cache, `d2l.Benchmark` and gradient accumulation; `sgd.md` teaches all four in short form (SGD-10, SGD-12, SGD-16) and its Exercises 6 to 8 belong there. The memory ledger, as above. Chapter 11 owns the transformer; `muon.md` explains gradient rank through LayerNorm and the residual stream (MUON-11) and `adam.md` prints attention code the reader cannot yet read (ADAM-8).

**With the appendix (chapter 40).** The appendix review listed 14 items taught in both, and the author's rule sorted them: small results in both, long derivations in the appendix, network experiments here. That sorting has been applied on the appendix side. Verified state of the sources, and what remains to do on this side:

- *Closed.* The opening sentence shared by the two introductions (the appendix introduction is rewritten). The warmup pointer in `lr-scheduler.md` now cites `subsec_mdl-schedules-warmup` for what that subsection shows ("until the momentum average has filled"). The "ladder" that `muon.md` cites exists again in `subsec_mdl-preconditioning-ladder`. All 18 distinct section and equation labels of the appendix cited from the chapter resolve, and the section reviewers found each pointer to hold the cited content.
- *Edge of stability.* The appendix now cites `sec_optimization-intro` and has a label, but it still repeats the experiment in two NumPy cells (verified, "The two cells below repeat that experiment in NumPy"). The only thing those cells have that this chapter lacks is the fraction of late steps on which the loss rose. Print it here (LAND-11); the appendix cells can then go.
- *Noise ball.* Small, allowed in both. But `sgd.md` contains no reference to the appendix at all (verified: the only section of the chapter with none besides `batch-size.md`, `scaling.md` and `practice.md`). Add one pointer to `subsec_mdl-stochastic-gradients` for the transient and the loss floor, and one to `subsec_mdl-step-size-decay` together with the missing sentence on the rate (structural finding 7 of the `sgd.md` review, FR-3). The two valleys differ (curvatures 0.2 and 4 here, 1 and 1,000 there); a clause at the pointer should say so.
- *The finite-population factor.* Derived over 25 lines here and cited from the appendix (":numref:`sec_sgd` derives it", verified), which is the rule inverted. State the result with its one-line reason here; move the derivation, or an exercise for it, to the appendix; change the appendix sentence in the same pass (SGD-8).
- *Robbins–Monro.* State once here (in `sgd.md`), cite from `lr-scheduler.md`, point to "The Robbins--Monro Conditions" in the appendix; fix SCHED-2.
- *Schedules on the valley.* The appendix reviewer proposed the appendix as the home, because its recursion is exact. The reviewer of `lr-scheduler.md` finds the simulated valley experiment the best part of the section. Keep it here; it is the section's foundation and it shows the case the formula misses. Add one sentence that the appendix computes the same ranking without sampling ("Schedule Shapes under the Exact Recursion"); `lr-scheduler.md` does not cite it now.
- *Descent lemma.* In the appendix only, three lines, and the answer to the question `gd.md` asks. Duplicate it (GD-4) and replace the pointer to the whole chapter by `subsec_mdl-gd-smoothness`. GD-10: do not write "the appendix proves both" of a theorem and a cost statement.
- *Momentum.* Already the model. Fix MOM-5 from the appendix's correct statement, and add a clause that the appendix writes the heavy ball with the learning rate inside the velocity.
- *Adam.* Division of labour is right. Two small results that the appendix has and the main text needs: scale invariance (ADAM-5) and the fixed points of the two decay forms (ADAM-18, where the main text currently uses the fixed point for an argument it cannot make). Exercises 5 and 7 of `adam.md` cannot be done without appendix code; make them self-contained or move them.
- *Warmup.* The appendix says ":numref:`sec_scheduler` measures this in a real model" (verified); the section measures no curvature. Add the measurement here, which is the better book, or change the appendix sentence (SCHED-7).
- *Muon.* Add a pointer to `subsec_mdl-orthogonalized-updates` for the polar factor and the cubic's convergence (MUON-15); note that this label is in the linear algebra chapter (`mdl-svd-low-rank.md`), not in chapter 40. Decide whether the Polar Express construction goes there or to Further Reading.
- *Notation.* Still open from the appendix's list: the placement of $\epsilon$ in Adam, and $\mathbf{w}$ there against $\mathbf{x}$ here (ADAM-22).

The main text reads without the appendix, with the three exceptions named: the $2/L$ guarantee (GD-4), Exercises 5 and 7 of `adam.md`, and the clause in Exercise 3 of `optimization-intro.md` that asks the reader to compare with the appendix's derivation.

## Recurring patterns

**One object under many names.**

- The largest curvature: $h_2$, $\lambda_{\max}$, "the sharpness", $L$ (LAND-18, SGD-14). A mode: "direction", "coordinate", "mode", "eigendirection" (MOM-6).
- Adam's $\hat{\mathbf{v}}$ and its root: eight names, one of them wrong ("variance estimate" for a second moment, in a heading and a label; ADAM-7).
- "Budget": coined for $\sum_s \eta_s$, used about 30 times, and from the middle of `lr-scheduler.md` also for the length of training (SCHED-1).
- "Critical batch size" and noise scale (BATCH-11).

**Notation drift between adjacent sections.** $\lambda$ is curvature through `lr-scheduler.md` and the decay coefficient in `adam.md`, in adjacent subsections (open since October). $\mathbf{v}$ is the velocity in `momentum.md` and the second-moment average in `adam.md`; $\epsilon$ is a tolerance and then Adam's constant (MOM-13, ADAM-22). The update is $\mathbf{x}_t \leftarrow \mathbf{x}_{t-1}$ in `momentum.md` and $\mathbf{x}_{t+1} = \mathbf{x}_t$ in `adam.md`. $\mathbf{x}$ is the parameter vector while `X` is the data and an exercise writes features as $\tilde{\mathbf{x}}$ (SGD-18). $\log$ is natural and not said (LAND-8).

**Results the body uses, parked in exercises.** MOM-3 and MOM-10 (cited from the body with "shows why"), ADAM-12 (Exercise 3 is the experiment the section needs), ADAM-19 (Exercise 9), BATCH-19 (Exercise 1), SCALE-8 (Exercise 3), SCALE-17 (Exercise 7), PRAC-5 (Exercise 5), MUON-1 (Exercise 10 tests the section's premise). The exercises of this chapter are very good; in eight places they are better than the text they follow.

**Numbers no cell prints.** SGD-14, ADAM-11, BATCH-5, BATCH-9, BATCH-15 ("about 2.5 on average"), SCALE-8, GD-17, and the steps read off a plot in `muon.md`.

**Double endings and roadmaps.** "We can now ..." before the Summary (SGD-15); a hand-off written twice (MOM-12); three endings in `optimization-intro.md` (LAND-16); "One question remains open" for a question that is postponed (BATCH-20); Summaries of eight or nine sentences that list every topic (SCHED-14, SCALE-18, PRAC-25) and in two cases state one run as a general fact (LAND-17, PRAC-25).

**Cell introductions by formula.** Nine "We draw ... / We train ..." in `lr-scheduler.md` (SCHED-13); six question-then-cell openings in `muon.md` (MUON-19); "Four ...", "Four ...", "Five ..." as headings (SCHED-15). In the chapter introduction, ten "asks" (IDX-1).

**Framework plumbing in prose.** Tab notes on loaders, warm-up passes, default initializers, `approximate=False`, Optax state and batch-norm momentum conventions: SGD-17 (three pairs), ADAM-8 (twelve notes), SCHED-12, PRAC-8, the TF32 note in `muon.md`. `sgd.md` and `practice.md` are the two sections where the tabs fragment the reading. Helper code for later sections is narrated where it happens to be defined (GD-8, SGD-3, SGD-16). Unused imports and second imports cells in the first three sections (LAND-2, SGD-19, and the code notes of `gd.md`). No TensorFlow or MXNet material anywhere in the rendered sections; `legacy-timer-lib.md` holds the two legacy `Timer` variants and is build-only.

**Legacy cells at their old length.** `show_trace` and the parabola plots in `gd.md` are drawing code (GD-3) kept beside rebuilt material; the hazards figures in `optimization-intro.md` (LAND-4, LAND-5).

**Small tics.** "exact" and "exactly" six times for the bias correction (ADAM-6); ornament in `muon.md` ("speaks for the optimizer", "a matrix in disguise"; MUON-20); "quietly" (BATCH-16); "race" surviving in cell identifiers and one exercise.

**Voice.** Present, and the best thing about the language: "One clock cannot serve both populations", "which makes $\epsilon$ a tuning knob rather than a numerical detail", "The problem is not posed until we say which norm defines the ball", "although nobody chose its horizon when training started", "it is the one that ships", "A directory of loss curves without sentences becomes unreadable within weeks". No reviewer found clickbait. What is missing is the same voice at the level of the chapter: which of all this matters.

## Dependencies

**From Basics.** Empirical risk and generalization (`sec_generalization_basics`), minibatch SGD and the noise ball (`subsec_linear-regression-minibatch-sgd`), gradients and the Hessian (`subsec_calculus-grad`), weight decay (`sec_weight_decay`), initialization (`sec_numerical_stability`), clipping (`sec_rnn-scratch`), LeNet and batch normalization, the training recipe with AdamW, cosine decay and an EMA (`sec_training_recipes`, cited from five sections as the thing to be explained). All references resolve and each use is appropriate. The chapter answers questions Basics raised, which is the right relation.

**From later chapters, used early.** `TinyLM` is a pre-norm transformer with multi-head attention, taught in chapters 10 and 11. `adam.md` handles this honestly ("an optimizer does not need to know what attention does"), and `muon.md` then breaks the promise: parameter names (`qkv`, `proj`), the residual stream, LayerNorm, attention logits and heads carry arguments there (MUON-11, MUON-12, structural finding 4 of that review). `scaling.md` says transformers need "one further rule" and sends the reader to a blog post, although the chapter's main model is one (SCALE-7). Sharding vocabulary from chapter 13 is used undefined (MUON-12). With the variants cut and the harness in its own section, what remains is one deliberate forward dependency, stated once.

**On the appendix.** The chapter reads without it, with the three small exceptions listed under Repetition. No reference into the appendix points to the wrong place. Three sections (`batch-size.md`, `scaling.md`, `practice.md`) make no reference to it and need none; `sgd.md` makes none and should make two.

**Forward references to the Language Models skeletons.** None from this chapter. The dependency runs the other way: nine `chapter_lm-*` files cite `sec_adamw`, which today resolves to a subsection.

**What depends on this chapter.** `sec_adam` is cited from seven other chapters; the test models from chapters 10 to 13; `sec_batch_size` from chapter 13; `sec_scaling` and `sec_practice` from chapter 11; `sec_muon` from chapter 13 and the appendix. `sec_momentum`, `sec_gd` and `sec_sgd` are hardly cited from outside. So the parts of the chapter that the rest of the book stands on are Adam, AdamW and the harness, which are the parts now folded into one file.

## Connection to the main text

- `sec_adamw` and `subsec_tinylm` are used as section-level targets and are not sections (above).
- SCALE-16: `scaling.md` corrects an impression left by `muon.md` (the constant 0.2 was chosen to match AdamW's update size, so Muon as taught does not have the width independence its derivation suggests). One clause with a forward reference belongs in `muon.md`.
- PRAC-22: `practice.md` reports that Muon's advantage shrinks with scale, and `muon.md` compares on a model of 0.47 million parameters. Tell the reader in one sentence what to conclude about the chapter's own result.
- PRAC-11: the divergent SGD grid point of `adam.md` trains once clipped, so that grid was limited by its first steps. `adam.md` should point forward to this; it changes how its comparison is read.
- GD-6 and the edge of stability in `optimization-intro.md` are the same idea (a learning rate rules out minima sharper than $2/\eta$) and do not refer to each other.
- `lenet_bn` is saved in `lr-scheduler.md` for `practice.md`; the confound of SCHED-5 travels with it.
- The norm table of `muon.md` (three balls, three optimizers) is the chapter's best unifying picture and arrives two sections after Adam. With Muon moved up it closes the direction group; a one-line preview belongs in the introduction (IDX-3).
- Uses named without a label: the chapter introduction does not name the test models or where they are defined; `batch-size.md` refers to "a subsection the reader has not read" for the 512 cap (BATCH-9).
- Missing pointers that reviewers looked for: from `lr-scheduler.md` to warm restarts in Further Reading (FR-9); from `practice.md` to the book's hyperparameter-optimization chapter; from `sgd.md` to the appendix.

## What to cut

About 6,500 words of body prose, most of it to Further Reading or chapter 13, none of it argument:

- `optimization-intro.md`: the local-minimum figure and the cubic (LAND-4, LAND-5), the risk-gap figure (LAND-3), "Noisy Gradients" down to a paragraph (LAND-14), the second roadmap (LAND-16). About 700 words.
- `gd.md`: the three parabola runs and their plots (GD-2, GD-3), the damped Newton run (GD-13), the `train_2d` narration (GD-8). About 600 words, partly replaced by the descent lemma and the eigenbasis paragraph.
- `sgd.md`: the arithmetic-intensity argument and the micro-batch paragraph to chapter 13 (SGD-10, SGD-12), the variance cell or its 60 lines of set-up (SGD-7), the derivation of the finite-population factor (SGD-8), "Concise Implementation" (SGD-16), the tab notes (SGD-17). About 1,300 words and 150 lines of code.
- `momentum.md`: the recap subsection (MOM-2), two of the data runs (MOM-9). About 300 words.
- `adam.md`: the CNN comparison to a table row and a paragraph (ADAM-13), the decay grid (ADAM-18), "Weight Decay in Large-Scale Training" to the two-line result and a judgement (ADAM-19), the exemption run (ADAM-20), the memory ledger (ADAM-21). About 1,200 words.
- `muon.md`: optimized polynomials, library versions as a subsection, NorMuon, the rank argument, communication accounting and Dion. About 2,200 words.
- `batch-size.md`: the Malladi conditions, the sharp-minima history, most of "Growing the Batch" (BATCH-13, BATCH-18, BATCH-19). About 350 words.
- `lr-scheduler.md`: the worst-case paragraph (SCHED-8), the tab blocks (SCHED-12), half of the Summary. About 350 words.
- `practice.md`: the counts paragraph (PRAC-3), the z-loss (PRAC-15), "The Chapter's Three Decisions" (PRAC-24). About 400 words.
- Exercises to cut, fix or move: `gd.md` 1 (fix), 4 (cut or reword); `sgd.md` 6 to 8 (to chapter 13); `muon.md` 5 (fix), 9 (to Adam); `scaling.md` 7 (cut or reduce); `practice.md` 7 (to Batch Size); `lr-scheduler.md` 1(b) (reword), 7 (to SGD); `adam.md` 5 and 7 (self-contained or to the appendix).

## What is missing

- A page that takes a new model from nothing to a learning rate, a schedule, a batch size and a guard, citing the section that justifies each step (PRAC-24).
- Momentum, and warmup, shown to matter on a network.
- The mechanism behind Adam's advantage, measured (frequent against rare tokens), or a plain statement that it appears at scale.
- Adam on the valley: the running example is used for AdaGrad and RMSProp and dropped when the title method arrives.
- The general quadratic: gradient descent decouples along the eigenvectors of the Hessian, which is why a two-coordinate valley is general (LAND-9, GD-7).
- What noise costs in rate (one sentence in `sgd.md`).
- How schedules carry over to momentum and Adam (SCHED-10), how to choose the peak rate and the warmup length.
- Weight decay in Muon's update (MUON-7), and the objection that small singular directions of a minibatch gradient may be noise (MUON-2).
- The muP rules for a transformer, since the chapter's model is one, and the authors' judgement on whether to use muP (structural finding 3 of `scaling.md`).
- Symptoms: how a loss curve looks when the rate is too high, too low, or the noise dominates; what to log in every run (structural finding 5 of `practice.md`).
- A measured run-to-run spread for the two test models, stated once.
- In the introduction: the running examples, the prerequisites, what can be skipped (IDX-1), and a ranking in Further Reading (FR-1).
- A closing paragraph for the chapter.

## What is best and should set the model

- **`optimization-intro.md`, the valley and the edge of stability.** "The steep coordinate sets a ceiling ... The flat coordinate sets the pace"; a sweep that poses the question the formula then answers; a 2021 result reproduced on 25 parameters with the ceilings drawn in the plot.
- **`sgd.md`, "The Noise Ball".** Six lines from one recursion, a prediction of both radii, a measurement that also shows where the approximation stops, and the sentence that joins the chapter's two difficulties: the flat directions "are also where the noise carries the iterate farthest".
- **`momentum.md`, the table of four settings.** Prediction before measurement, with a deliberately bad row that teaches what the theorem alone does not; concrete critical values a reader can check by hand; candour about where the theory stops.
- **`adam.md`, AdaGrad to Adam.** Each method introduced by the failure of the one before, with a number ("7.6 $\eta$ = 3.0 ... is not enough to cover the distance of 5"); the $\epsilon$ paragraph that ends in a judgement; the three-curve experiment in which one $\lambda$ means two things, read against a computed baseline.
- **`lr-scheduler.md`, two numbers for a schedule and branching off the plateau.** A device the reader carries through the section; every printed number explained by a formula; a current idea demonstrated at the reader's scale.
- **`muon.md`, "Norms and Steepest Descent".** One question, three balls, three optimizers the reader knows, and a cost count applied to the reader's own model with the unflattering 12% reported.
- **`batch-size.md`, from the opening to the collapse onto one curve.** The whole section in three sentences with numbers before any symbol; an estimator derived in four lines and coded in four; "a prediction ... that could fail, so we test it"; an honest account of what a two-parameter fit does and does not test.
- **`scaling.md`, the first Adam step.** "All $n$ terms of the sum have the same sign, because the update is built from the very activations that it multiplies", from two facts the reader has, with its limit stated.
- **`practice.md`, clipping.** A run that diverges in twenty steps, saved by clipping that touched 13 updates, read as an adaptive warmup; and "What clipping protects is Adam's state."
- **The exercises**, throughout: graded, each with a question, often with a surprise in the answer.

The common factor is the one the earlier reviews named: a number predicted, then printed, then explained. Where the chapter leaves that method (a paper quoted in place of a run, a small gap read as a win), it leaves the bar.

## Where the section reviews disagree or need reconciling

- **The three decisions.** PRAC-24 says the classification "appears here for the first time" in `practice.md`. It does not: `optimization-intro.md` has a subsection "Three Decisions" (LAND-16) and `lr-scheduler.md` cites it in its opening (verified). The recommendation of PRAC-24 stands and is strengthened: the frame is stated at the end of section 1, recalled in section 6 and restated at the end of section 10, and never in the introduction.
- **The noise ball against the appendix.** The appendix review proposed the appendix as its home; the overview's sorting and the reviewer of `sgd.md` treat it as a small result allowed in both. The author's rule supports the second reading, and the derivation here is one of the best passages of the chapter. Keep it; add the pointer.
- **Schedules on the valley.** Same disagreement, same resolution (keep here, cite the exact recursion).
- **Protocol.** The reviewer of `adam.md` praises "Protocol stated before results"; the reviewer of `practice.md` says the protocol arrives after every comparison it governs (PRAC-21). Both are right about different things: `adam.md` states the protocol of its own comparison, `practice.md` the general classification. One statement in the harness section serves both.
- **Where the harness goes.** The reviewer of `adam.md` leaves the placement (before Adam or before Momentum) to the author; the reviewer of `momentum.md`, the same person, notes that before Momentum would give that section a network. My recommendation is above.
- **Heavy tails.** `adam.md` reports from Kunstner et al. that heavy-tailed token frequencies explain Adam's advantage; `practice.md` asserts heavy-tailed gradient noise against its own recorded norms (PRAC-12). These are different tails (of class frequencies, of gradient norms). A reviser should keep them apart, and the byte-level `TinyLM` with 256 classes may show neither strongly, which is the argument for the 1,024-token vocabulary proposed in ADAM-12.

## Could not check

- Rendered HTML and slides. The reviewers read sources, stored PyTorch and JAX outputs and rasterized figures.
- No PyTorch or JAX on the review machine: no network run was reproduced. Valley computations, the noise ball, momentum roots, schedule formulas, Newton–Schulz maps and exercise answers were recomputed in NumPy by the section reviewers.
- The airfoil and WikiText data are not on the machine: $L \approx 2.1$ (SGD-14) and the four byte-frequency statistics (ADAM-11) are unchecked.
- Statements attributed to papers were checked from memory only; the unverified ones are listed under Technical correctness and in the further-reading review. Works dated 2026 could not be checked at all.
- Run-to-run ranges quoted in the prose rest on repeats that are not in the store (one stored run per framework).
- The proposed experiments are untested; each section review says what to verify.
