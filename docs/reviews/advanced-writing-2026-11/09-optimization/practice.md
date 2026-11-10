# Practice

Source: `chapter_optimization/practice.md`, lines 1 to 750 (slides ignored). About 3,950 words of body prose plus 1,000 words of exercises; 6 code cells in PyTorch, 8 in JAX.

## Verdict

The parts of this section are good and the whole is not yet a section. The clipping experiment is a small gem: a run that diverges in twenty steps, saved by clipping that touched 13 of 2,000 updates, and read correctly as an adaptive warmup. The paragraph on what clipping protects in Adam (its second-moment state, not the step) is original and exact. The weight-averaging experiment shows what it claims. But the section has four questions (how to configure, how to guard, how to finish, how to compare), frames them all around "a run that can be done only once" at the scale of PaLM, and closes the chapter with a taxonomy of three decisions and a pointer to the next chapter. The part introduction promises that this chapter teaches "how to choose a learning rate, schedule and batch size". A reader who has a new model and one GPU does not get that here: "Tuning Budgets" is 220 words, says nothing about the schedule or the batch size, and its concrete advice is for a team with tens of full-scale runs. The single most important fix: make the closing of the chapter a worked procedure for configuring a new model with the tools of the ten sections, in place of "The Chapter's Three Decisions", and frame the section for the reader's own runs, with the frontier reports as evidence and not as the subject.

## Scores

| Dimension | Score |
|---|---|
| Clarity of explanation | 4 |
| Clarity of thought and focus | 3 |
| Pedagogy | 3 |
| Language | 4 |
| Ease and enjoyment | 3 |

## What works

- **The opening's three facts** each carry a point: a sweep "costs four times the run itself", PaLM "saw its loss spike about twenty times along the way, although gradient clipping was enabled", and "the checkpoint at the end is one draw from a noisy trajectory, yet it is the one that ships." The last is a sentence with voice.
- **The clipping demonstration** (`subsec_practice-nan` and "Clipping Frequency"). It reuses the divergent grid point the reader already met in :numref:`sec_adam`, changes one thing, and prints the number that makes the point: 13 (PyTorch) or 11 (JAX) clipped steps of 2,000, median gradient norm a quarter of the threshold. "Here clipping did the job of a warmup, and an adaptive one. It shortened exactly those first steps that were too long ... and it needed no schedule to tell it when to stop." The experiment teaches something the formula does not. Keep every sentence.
- **Guard against brake** ("If clipping fires on most steps, it rescales nearly every update and thereby changes the update rule itself"). A useful distinction with a diagnostic the reader can log.
- **"Clipping and Adam"**: the bound $\eta(1-\beta_1)/\sqrt{1-\beta_2} \approx 3.2\,\eta$ on a spike's update, and then "What clipping protects is Adam's state." I checked the numbers (3.16 and 0.447). This is a result I have not seen stated this plainly elsewhere, and it explains why the reports pair clipping with $\beta_2 = 0.95$.
- **Lag is not bias** (lines 511 to 514): "The average starts at the initial weights, so its coefficients already sum to one, and there is nothing to correct." Heads off the confusion with Adam's bias correction that every reader will have. And $0.999^{235} \approx 0.79$ is a worked number at the right moment.
- **"Experiment Records"** ends with "A directory of loss curves without sentences becomes unreadable within weeks." True, and the kind of thing only someone who has done it writes.
- The October complaint about hedges contradicting advice here is fixed: "a defensible starting point ... not proven optima" is one consistent position.

## Structural findings

1. **Four sections under one title.** Configuration (550 words), clipping and spikes (1,300), weight averaging (900), comparing optimizers (650), plus a closing. Each could be read alone, and the only thread is the conceit of following one large run "through its life". By the guide's "one idea per section" this is a split candidate, but the pieces are too short to stand as sections, and three of them have natural homes. The cleaner solution is to recognize the two real topics. **Seam:** after "Loss Spikes and Recovery" plus "Weight Averaging" (protecting and finishing a run: clipping, spikes, checkpoints, EMA; two experiments; about 2,200 words), and a second part made of "Choosing a Configuration" and "Comparing Optimizers" merged (how to choose and how to know you chose well: budgets, protocol, records; about 1,200 words now, with room for the worked procedure of finding 2). If the file is not split, reorder it this way: guard, finish, then choose and compare, so that the chapter ends on choosing.
2. **The chapter does not end with the reader able to configure a new model.** What the chapter has taught, section by section, is enough: the stable range of $\eta$ and how to find its edge (`gd.md`, `adam.md`), warmup and decay shapes (`lr-scheduler.md`), the noise scale and the $\sqrt{b}$ rule (`batch-size.md`), transfer from a proxy (`scaling.md`), clipping and EMA (here). What is missing is the page that puts them in order for a concrete case. I would replace "The Chapter's Three Decisions" by a worked configuration of one model the reader has not tuned in this chapter (or `TinyLM` with AdamW from a cold start), in about 400 words and ideally one cell:
   - batch size: the largest that the device runs at full throughput and that memory allows; check with `noise_scale` that it is not far above the noise scale after a few hundred steps;
   - optimizer and fixed settings: AdamW, $(0.9, 0.95)$ or the defaults, weight decay 0.1 on matrices, clipping at 1 with the clip rate logged;
   - learning rate: short runs on a factor-of-three grid, take the largest rate that is clearly stable, one grid point below the edge;
   - schedule: warmup over the first one to few percent of steps, then cosine or warmup–stable–decay, with the reason from :numref:`sec_scheduler` in a clause;
   - at the end: EMA or a decayed endpoint, and a rerun with a second seed before believing a difference.
   Each line cites the section that justified it. This is the page a reader will photocopy, and at present it does not exist anywhere in the chapter. (A new cell is an experiment change and can wait; the prose can be written now.)
3. **Framing.** "A large training run allows none of this" sets the section in a world of 6,144 TPUs. Most of the content applies to any run the reader cannot afford to repeat fifty times, which for a reader with one GPU begins at a few hours. Say so in the opening, and the clipping, averaging and record-keeping advice stops reading as a report from elsewhere.
4. **Paper retelling.** "Loss Spikes and Recovery" (330 words) narrates the PaLM report twice over (z-loss, then rewind and skip), and "Tuning Effort and the Verdict" (130 words) retells two benchmark papers. The PaLM rewind story is worth its space: it is concrete, surprising ("the same minibatches, replayed from a different, earlier checkpoint, caused no spike") and leads to actionable advice on checkpoints. The z-loss paragraph is not: it describes a remedy and then reports "the spikes came anyway". Move the z-loss to exercise 3, which already teaches it better than the prose.
5. **What a reader would look for and not find:** how to tell from a loss curve that the learning rate is too high or too low (the chapter has many such curves and never collects the symptoms); what to log in every run (gradient norm, clip rate, update-to-weight ratio, the learning rate); how long warmup should be. The first two would fit "Experiment Records" in three sentences each.

## Passage-level findings

**PRAC-1** (minor, correctness). Under "Reported Configurations". Anchor: "four prominent pretraining reports
from 2024 to
2026".
The four cited reports are dated 2024 (Llama 3, DeepSeek-V3) and 2025 (OLMo 2, Kimi K2) in `d2l.bib`. Write "from 2024 and 2025". The line break after "to" is also a leftover of an edit.

**PRAC-2** (minor, correctness; unverified details). Under "Reported Configurations", the table. Anchor: "| OLMo 2 7B :cite:`OLMo.2025` | AdamW | 0.9, 0.95 |".
I can confirm from memory the Llama 3 row (peak $8 \times 10^{-5}$, 8,000 warmup steps, cosine to $8 \times 10^{-7}$, batch ramp 4M, 8M, 16M tokens) and the DeepSeek-V3 row (all entries, including the batch ramp over 469B tokens; the report also has a final constant phase at $7.3 \times 10^{-6}$ that the table omits, which is fine). The OLMo 2 row (cosine to 10% of peak, embeddings exempt from decay, 1,024 sequences) and the Kimi K2 row (500 warmup steps, 67M tokens) agree with what I recall but I could not check them: mark for verification against the reports.

**PRAC-3** (moderate, language and focus). Under "Reported Configurations". Anchor: "Three of the four runs use AdamW. The fourth, Kimi K2, applies Muon to its".
The paragraph that reads the table is 220 words of counts: "Three of the four ...", "The two reports that disclose ...", "Three of the four decay ...", "the two that report clipping ...", "All four warm up ...", "Two ramp ... and two keep it fixed". The reader can count rows. What the reader cannot do is rank: which entries are consensus worth copying, which are choices that vary, and why. Suggested shape: one sentence for the consensus ("AdamW or a Muon hybrid, $\beta_2 = 0.95$, decay 0.1, clipping at 1, warmup then decay"), the $\beta_2$ explanation (keep: it is the one place a row is explained), one sentence for what varies (peak rate, batch ramp), and the candid closing sentence (keep: "The dashes record missing evidence, not implied defaults"). About 110 words.

**PRAC-4** (minor, explanation). Under "Reported Configurations". Anchor: "from
$8 \times 10^{-5}$ for the dense 405-billion-parameter model to".
"dense" is unexplained (as opposed to the mixture-of-experts DeepSeek-V3 and Kimi K2, which the reader has not been told are such). More useful than the span of the four values is the pattern the reader can use: larger models take smaller peak rates, which is the drift of :numref:`sec_scaling` seen in practice. Say that, or drop "dense".

**PRAC-5** (moderate, pedagogy). Under "Tuning Budgets". Anchor: "With a handful of runs, keep the recurring settings and use every run to".
This is the only passage in the chapter that tells the reader how to tune, and it is two paragraphs. It names the learning rate, then weight decay and schedule, and stops. It does not say what grid, how long the tuning runs should be relative to the final run, how to pick from the sweep (the best point, or one below the edge?), or anything about batch size. See structural finding 2. The observation about $\eta\lambda$ is good and specific; the last sentence ("the tuning becomes a study of its own, and the protocol of :numref:`subsec_practice-tuning` applies") points 430 lines ahead to a subsection that gives a classification, not a protocol.

**PRAC-6** (minor, thought). Under "Tuning Budgets". Anchor: "A joint grid should therefore vary the
product $\eta\lambda$ and also move along lines on which the product is
constant.".
Any two-dimensional grid does both. The intended advice is to parametrize the grid by $(\eta, \eta\lambda)$ instead of $(\eta, \lambda)$, so that the axes are the two quantities with separate effects. Say that.

**PRAC-7** (minor, thought). Under "Gradient Clipping". Anchor: "The clipping column of :numref:`tab_practice_recipes` guards against a
failure that this chapter has already met.".
Good link. The paragraph then defines global-norm clipping again in full, although :numref:`sec_rnn-scratch` introduced it and the reader has used it. The formula and the worked number ("a gradient of norm 5 is multiplied by 0.2") can stay as a three-line reminder; the sentence on preserving relative layer sizes is the new content and should lead.

**PRAC-8** (minor, language; tab prose). Under "Gradient Clipping" and `subsec_practice-nan`. Anchor: "An Optax transformation is a pair of pure functions, `init` and `update`,".
Three tab blocks in this file explain framework mechanics at 40 to 60 words each (the PyTorch wrapper and device synchronization; Optax chaining; Optax state). The guide allows a sentence or two per tab. The JAX reader also gets an extra 12-line cell (`record_norms`) whose only job is to log a norm. This is the one place in my three sections where the two tabs noticeably fragment the reading. Shorten each block to one sentence and move `record_norms` into `d2l` if a second section needs it.

**PRAC-9** (minor, explanation). Under `subsec_practice-nan`. Anchor: "The model is `TinyLM`, the decoder-only transformer
of :numref:`subsec_tinylm`: two blocks, about 0.47 million parameters,".
Fourth full description of `TinyLM` in the chapter, and `d2l.train_steps` is introduced again as "The library function". One clause each.

**PRAC-10** (minor, correctness). Under `subsec_practice-nan`. Anchor: "The first step lowers the loss, and from the second on the unclipped run".
The rendered plots show the unclipped loss falling for two to three steps before it turns (PyTorch: slight rise at step 3, a dip, then the climb from about step 6; JAX similar). "After the first few steps" is what both tabs show. The mechanism described next is right.

**PRAC-11** (minor, correctness). Under `subsec_practice-nan`. Anchor: "to a final loss comparable with that of the
best SGD run in :numref:`sec_adam`".
Stored values: 1.621 (PyTorch) and 1.615 (JAX) for the clipped run at $\eta = 1$ against 1.654 for the best unclipped SGD run at $\eta = 0.3$ in `adam.md`. So the clipped run at the higher rate is a little better, by an amount that may or may not exceed seed noise. "comparable" is the safe word and correct. But note what this implies and the text does not say: clipping moved the stability edge, so the SGD grid of :numref:`sec_adam` was limited by its first steps and not by the bulk of training. That is a real lesson about reading sweeps (the "diverging grid point" may mark an initialization problem) and deserves a sentence.

**PRAC-12** (moderate, correctness). Under "Clipping Frequency". Anchor: "Gradient noise in language models is heavy-tailed, so long
after the start an occasional minibatch produces a gradient many times".
The claim is from the cited paper and fair as a report. But the section has just recorded 2,000 gradient norms of a language model and found that none after step 12 exceeded four times the median (clipping "idle for the rest of the 2,000 steps"). The reader's own evidence says the opposite of the sentence. Either say that this small model with full-window batches does not show the tail, or use the recorded norms (print the maximum of `norms[100:]` over the median) so the claim is measured. As written, the prose asks the reader to believe a citation against the cell output.

**PRAC-13** (minor, thought). Under "Clipping and Adam". Anchor: "A spike a hundred times the typical gradient keeps $v$ above its
usual level for more than two thousand steps".
$v$ is above its usual level for ever, by a decaying amount. The computation behind the sentence: the spike adds $(1-\beta_2) \cdot 10^4 = 10$ times the usual $v$, and $10 \cdot 0.999^t$ falls below 1 at $t \approx 2{,}300$. So: "keeps $v$ at more than twice its usual level for about 2,300 steps". Also note for the reader that global-norm clipping acts on the whole gradient vector: it protects a coordinate's $v$ only when the spike is large enough to raise the global norm above $\theta$. One clause.

**PRAC-14** (moderate, correctness; unverified). Under "Loss Spikes and Recovery". Anchor: "where they lose precision in
bfloat16".
The PaLM paper motivates its z-loss by saying that keeping $\log Z$ near zero increases training stability; I do not recall it giving the bfloat16 argument, which I associate with later reports. The sentence reads as PaLM's reasoning. Either attribute it or drop the clause. The rest of the PaLM account (about 20 spikes, restart from about 100 steps before, skip 200 to 500 batches, same batches harmless from another checkpoint) matches the paper as I remember it.

**PRAC-15** (minor, focus). Under "Loss Spikes and Recovery". Anchor: "PaLM used it throughout training, and the spikes came
anyway".
A remedy introduced over six sentences and then reported not to have worked. See structural finding 4: cut the z-loss paragraph to one sentence with a pointer to exercise 3.

**PRAC-16** (minor, thought). Under "Weight Averaging". Anchor: "When the fluctuations
around the center of the region are only weakly correlated, they partly
cancel in the mean.".
The condition matters and is passed over. Successive SGD iterates are strongly correlated; what decorrelates them is time, about $1/(\eta\lambda)$ steps in a direction of curvature $\lambda$. This is why the window must be long (1,000 steps here) and why $\alpha = 0.9$ does little, which exercise 4 then asks about. One sentence would connect the claim, the choice of $\alpha$ and the exercise.

**PRAC-17** (minor, focus). Under "The Exponential Moving Average". Anchor: "Stochastic weight averaging takes an equal-weight average of checkpoints".
The subsection opens with a variant the section does not use. Open with the EMA (which the reader met in :numref:`sec_training_recipes`) and mention the equal-weight tail average in a clause where "Averaging and Decay" discusses averaging checkpoints.

**PRAC-18** (moderate, pedagogy). Under "Averaging at a Constant Learning Rate". Anchor: "For the first few epochs it trails the live
weights by a wide margin".
In both stored plots the EMA is below the live weights until epoch 8 or 9 of 15, so more than half of the experiment shows the lag and under half shows the benefit. The text then says "Practical EMAs therefore start averaging late in training or let $\alpha$ grow from a small value" and does neither. A reader copying this cell gets a model that is worse than the live one for 2,000 steps. For the writing pass: say "for the first eight epochs or so". For the experiment pass: start the average at epoch 5 (one `if`), which shortens the lag discussion to a sentence and makes the plot show the point.

**PRAC-19** (minor, correctness). Under "Averaging at a Constant Learning Rate". Anchor: "while the EMA's rises
steadily, by about one to two points, and ends at its highest value".
Stored: EMA last five epochs 0.897 to 0.910 (PyTorch), 0.900 to 0.908 (JAX); live 0.839 to 0.891 and 0.816 to 0.899. Consistent. "roughly 91%" for the decayed schedules agrees with `lr-scheduler.md` outputs (0.910 to 0.911). Note the EMA is still rising at epoch 15, so "reaches about the final accuracy of the decayed schedules" compares a 15-epoch run with 30-epoch runs; say so, since it favours the EMA.

**PRAC-20** (minor, thought). Under "Averaging and Decay". Anchor: "Averaging is
therefore most useful when no decay is planned, or when a decayed endpoint
is expensive to reach".
Then: "Large runs average checkpoints even after decay, where the gain is smaller", with Llama 3 as the example. The second sentence half-retracts the first. The honest summary is short: averaging and decay are substitutes; with a full decay the gain is small and cheap, so teams take it anyway. The final two sentences on recomputing batch-normalization statistics address the reader as "you" and are advice for a case the section did not run; move them into the earlier paragraph on batch-normalization statistics.

**PRAC-21** (moderate, thought). Under "Comparing Optimizers". Anchor: "Every comparison in this chapter rests on a tuning protocol, and the
protocol is part of the result.".
A good sentence, and the subsection that follows is the chapter's statement of method. It arrives in the last 650 words of the last section, after the reader has read every comparison it governs. The October review noted "optimizers raced ... with protocol caveats". The scientific/nuisance/fixed classification belongs where the first race is run (:numref:`sec_adam`), in the five sentences it takes; this section then looks back in one paragraph and spends its space on what is new here (budgets, records). Chapter-level item.

**PRAC-22** (minor, correctness). Under "Tuning Effort and the Verdict". Anchor: "found that published speedups of about $2\times$ over AdamW shrink to
about $1.4\times$ at 100 million parameters and toward $1.1\times$ at 1.2
billion".
Matches my recollection of the paper (claimed 1.4 to 2 times; measured about 1.4 at 0.1B falling to about 1.1 at 1.2B for the matrix-based optimizers). "ten optimizers" I believe is right but did not check. Note the tension the chapter should resolve in one sentence: :numref:`sec_muon` races Muon against AdamW on a 0.47-million-parameter model, and this paragraph says such advantages shrink with scale. The reader should be told directly what to conclude about the chapter's own Muon result.

**PRAC-23** (minor, language). Under "Experiment Records". Anchor: "Change one factor
at a time to estimate its effect, and use a factorial design when the".
"factorial design" is unexplained jargon for this reader. "and vary two together, on a grid, when the question is how they interact" says it.

**PRAC-24** (major, pedagogy). Under "The Chapter's Three Decisions". Anchor: "The methods of this chapter settle three decisions: the direction of an".
This is the last body paragraph of a ten-section chapter. It offers a classification (direction, step size over time, noise handling) that appears here for the first time, asserts that "When a run misbehaves, the first question is which of the three decisions failed" without a single example of how one would tell, and ends on a pointer to the next chapter. It is a recap paragraph before the Summary, which the author has ruled out, and it is not the ending the part introduction promises. Replace it with the worked configuration of structural finding 2. If a diagnostic classification is wanted, make it one: three symptoms with their likely cause (loss rises within the first steps: step too large for the initial curvature, add warmup or clipping; loss falls then plateaus high with a jittery curve: noise ball, decay or enlarge the batch; loss falls smoothly but slowly: step too small or a poorly conditioned direction, raise $\eta$ or change the optimizer).

**PRAC-25** (minor, thought). Under "Summary". Anchor: "Global-norm clipping with a threshold above the typical gradient norm
fires only on the first few steps".
Stated as a general fact; it is what happened in one run of one model. "In our run it fired only on the first dozen steps". The Summary is nine sentences and lists every topic; with the section refocused it should fall to five.

## Code and experiments

- **`practice-a-nan-averted-2` and `-3`** (clipping): shows the phenomenon, at the right scale, with a printed count that the prose reads accurately in both tabs (13 of 2,000, last at step 12, median 0.25; JAX 11, 10, 0.26). The prose range "a few dozen at most ..., all of them within the first forty" is loose against these values but is a deliberate range. Best experiment of the section.
- **`practice-weight-averaging-2` and `-3`** (EMA): shows the effect; half the run is lag (PRAC-18). The comparison with decayed schedules uses a number from another section and a different epoch count (PRAC-19).
- **Claims no cell supports:** heavy-tailed gradient noise, against the recorded norms (PRAC-12); "in the sweeps of this chapter a factor of three in the learning rate separated the best run from a diverging one" is supported by `adam.md` (0.3 against 1) and is fine.
- **No experiment for** configuration, spike recovery, or comparison protocol: half of the section's four topics are prose only. A section called "Practice" that ends the chapter should have the reader do the thing once (structural finding 2; exercise 5 is that experiment and could be promoted).
- **PyTorch `Clipped` wrapper**: `d2l.train_steps` evidently needs only `step` and `zero_grad`. Fine and compact. The tab text about keeping norms on the device is plumbing (PRAC-8).
- No TensorFlow or MXNet material. No figure-drawing code. One "exactly" ("exactly those first steps"), which is earned.

## Exercises

Seven problems, 1,000 words (25% of the body). Mostly strong.

- Exercise 1 (decay timescale) is answerable from the table plus the stated sequence length and dataset sizes; I get about 0.19 epochs for DeepSeek-V3 and about 0.035 for OLMo 2. Good use of the table.
- Exercise 2 (clipping regimes) has four parts and is the best in the section; part 4 (warmup in place of clipping) tests the body's main claim. Part 1 (reimplement and compare with the library to $10^{-6}$) is a drill; cut it or keep it short. The attribution line to CS336 sits oddly after part 4 when only part 1 is adapted.
- Exercise 3 (logit drift) is a clean pencil problem and should carry the z-loss material (PRAC-15).
- Exercise 5 (ten-run budget) is the exercise form of the missing closing. Keep, and consider doing a three-run version in the text.
- Exercise 6 (defaults against a budget) uses runs "that :numref:`sec_adam` and :numref:`sec_muon` already made"; the reader must find final losses in two other notebooks, some of which (SGD at 0.1, Adam at $10^{-3}$) are grid points and one of which (Muon hybrid at $10^{-3}$) should be checked to exist in the grid. It does in the stored `muon.md` output.
- Exercise 7 (batch size and the tuned rate) belongs to :numref:`sec_batch_size`, where it would replace the weak Adam sweep discussed there; with a batch ratio of 4 it has the same resolution problem.
- Add: one exercise on diagnosing a run from its curves (three curves, name the failure).

## Still open from October

- "Hedges contradict advice (`practice.md:54`, `:510`)": fixed.
- "Cut: ... 'Configurations Reported'": the subsection survives as "Reported Configurations". I would keep the table (it is evidence the reader cannot easily assemble, and exercise 1 uses it) and halve the paragraph that reads it (PRAC-3).
- The chapter-level note that `TinyLM` is described repeatedly still holds (PRAC-9).

## For the chapter review

- This section closes the chapter and is where the chapter's promise is tested. Verdict on that promise: after these three sections the reader can scale a learning rate with batch size and with width, knows that the learning rate is the first thing to tune, and has a credible starting configuration for a language model. The reader cannot yet walk from "new model" to "learning rate, schedule, batch size" by a stated procedure, because no section owns that procedure. Schedules are chosen in `lr-scheduler.md`, batch size in `batch-size.md`, the learning rate by sweeps everywhere, and this section adds budgets without assembling them. One page here would close the gap (structural finding 2, PRAC-24).
- Order of the three closing sections is right (batch, width, then practice). Lengths are proportionate except that the configuration question, the most important for the reader, gets the least space of anything in the three.
- Overlaps: clipping with `sec_rnn-scratch` (small, acceptable); EMA with `sec_training_recipes` (cited, not re-derived, good); the noise ball restated from `sgd.md` and `lr-scheduler.md` (one clause, good); the tuning protocol belongs earlier, in `adam.md` (PRAC-21).
- Assumes nothing from the appendix; there are no `sec_mdl-*` references. Forward reference to `chap_performance` is a pointer only.
- Pattern across my three sections: experiments are single-seed with ranges quoted from unprinted repeats. In `batch-size.md` this is handled well, in `scaling.md` it fails once, here it is fine. A second pattern: each section closes by narrating recent papers at one sentence apiece (here: two benchmark papers and the PaLM report). The frontier material is current for a 2026 to 2027 book, which is a strength, but it ages fastest and is least taught.
