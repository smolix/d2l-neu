# Scaling Laws

Source: `chapter_transformers/scaling-laws.md` (read to the `<!-- slides -->` marker at line 771). About 3,300 words of body prose, 170 words of tab prose and 1,750 words of exercises; about 5,200 in all. Stored outputs read: `outputs/pytorch/chapter_transformers/scaling-laws.json`, the JAX file, and the stored plot. Figure `img/mdl-transformers-chinchilla.svg` rendered and inspected. The allocation formulas, both fits, the deployed-model table of the cell, and the three-point power-law fit were recomputed with NumPy.

## Verdict

Two thirds of this section meet the bar and one third does not belong to it. The cost model ($6ND$, with the reason for the factor 4, the size of what it omits, and a framework count that could have disagreed) and the allocation part (the law, a three-line derivation of the optimum, the published constants against the refit, and the discovery that only the refit reproduces Chinchilla's own 20 tokens per parameter) are first-rate: derived, checked, candid about what the fits do not determine, and correct in every number I recomputed. The named results and constants are right and current to 2025. Two things hold the section back. First, the miniature study is read for a bend that it barely shows and cannot attribute: the "power law" is a line through three points, the largest model misses it by 0.04 nats, and a constant term $E$ alone bends the curve whether or not data is short. The same fifteen runs contain a much stronger demonstration of the section's actual question (at equal compute, the small model trained long beats the large model trained briefly by more than 0.1 nats) which the prose never reads. Second, the last part, "Configurations of Deployed Models", is the chapter's closing survey (a seven-row table of released models and a constructor demo) attached to a section on scaling; it is about 600 words that answer a different question. The single most important fix: read the grid along lines of equal compute, and drop or demote the three-point extrapolation.

## Scores

| dimension | score | note |
|:--|:-:|:--|
| Clarity of explanation | 4 | The $6ND$ derivation and the optimum are as clear as they can be made. The miniature study's variables ($N$ non-embedding here, all parameters in the law) need care the text mostly gives. |
| Clarity of thought and focus | 3 | The bend is overread; the section ends with a survey of deployed architectures that is not about scaling. |
| Pedagogy | 3 | Excellent on cost and allocation; the central experiment does not show the central trade-off; the last cell prints numbers the prose compares with a baseline it does not print. |
| Language | 4 | Plain, exact, with judgment ("The constants come from fits, and the fits disagree."). |
| Ease and enjoyment | 4 | The opening and the two-fits cell are a pleasure. The table section reads as an appendix. |

## What works

- **The opening** (lines 4 to 11). "A large model offers no second try." Then two numbers eleven orders of magnitude apart ($2 \times 10^{14}$ and $3.8 \times 10^{25}$; recomputed: $6 \cdot 405\text{B} \cdot 15.6\text{T} = 3.79 \times 10^{25}$) and the question. This is the voice the guide asks for.
- **Why the backward pass costs twice the forward** (lines 119 to 131): two products per layer, $\mathbf{W}^\top \boldsymbol{\delta}$ and $\boldsymbol{\delta}\mathbf{x}^\top$, each the cost of the forward product. Most texts assert the factor; this one derives it in four sentences.
- **What the count omits, with its size** (lines 142 to 152): $4nd$ against $24d^2$, ratio $n/(6d)$, 8% here and 100% for a wide model at 24,576 tokens. Then the check: the PyTorch counter gives 1.082 times $2ND$, and $1 + 128/(6 \cdot 256) = 1.083$. A measurement that could have failed and that lands on the correction to three digits. The prose should say so more loudly than "about the size of".
- **The embedding census** and the rule $V \gtrsim 12Ld$, with GPT-2 as "an 85-million-parameter transformer attached to lookup tables of 39 million entries".
- **Matched schedules** (lines 276 to 282): each run's cosine ends at that run's budget, with the reason and the historical consequence. This is the detail that separates a scaling study from a loss curve.
- **The optimum** (`eq_chinchilla_optimum`, `eq_chinchilla_allocation`). Derivation verified. "At the optimum the two terms that the budget can shrink fall at equal marginal rates" is the right sentence, and the figure (a budget line tangent to a loss contour) is the right picture.
- **Two fits, one cell** (`scaling-laws-the-optimal-split`). Recomputed: published constants give 32B and 2.98T (93 tokens per parameter); the refit gives 72B and 1.33T (18). The reader sees a famous number fail to follow from its own published constants. This is the most memorable result of the three closing sections, and the sentence after it ("This ratio belongs to that study's data, tokenizer, and model family") is the right caution.
- **Training past the optimum**: applying the law to Llama 3 8B (1,875 tokens per parameter, a tenth of the optimal size) explains a design decision the reader has seen in every model card.
- **Exercises 2, 8 and 9**: variations on the factor six; the drift of the ratio under the two fits (recomputed: 50 to 122 under the published fit between $10^{21}$ and $10^{25}$ FLOPs, 21.6 to 17.1 under the refit); the lifetime-cost optimum.

## Structural findings

1. **The grid is read along the wrong axis.** The section's question is how to divide a budget between $N$ and $D$. The grid varies both, so it contains iso-compute comparisons, and they are decisive. From the stored PyTorch output, with compute proportional to $ND$: the 1.77M model at 16.4M tokens ($ND = 2.9 \times 10^{13}$) reaches 1.29; the 4.72M model at 8.2M tokens ($3.9 \times 10^{13}$) reaches 1.34; the 14.2M model at 4.1M tokens ($5.8 \times 10^{13}$) reaches 1.45. More compute, worse loss, by 0.05 and 0.16 nats: five to fifteen times the run-to-run spread. That is the Chinchilla trade-off in miniature, it needs no new runs, and it leads straight into "The Optimal Split". The current reading (curves against $N$ at fixed $D$, a fitted line, a bend) is the Kaplan-style view, the weaker result at this scale, and the one that needs hedging. Recommended: keep the plot, add a second reading of the printed table along its anti-diagonals (or a three-line cell that prints loss against $6ND$), and make exercise 10 (IsoFLOP profiles) the natural continuation.
2. **"Configurations of Deployed Models" is a different section.** It asks "what do released models look like?", answers with a table, and cites five other sections in its caption. It is the chapter's synthesis, and a good one in content, but the guide (12.2) says sections do not end with surveys, and here it makes the scaling section end twice (once at "For a model that will serve trillions of tokens...", once at the table). The first cell's ID, `scaling-laws-scaling-laws-and-the-modern-recipe`, suggests the two were once announced together in the title. Seam: the heading `## Configurations of Deployed Models`. Move it to a short closing section of the chapter ("The Modern Recipe", after MoE), or into `index.md` or `further-reading.md`. The section's own Summary gives it one clause, which is about its weight here.
3. **$6ND$ is derived twice in the chapter.** `gpt.md` ("Training Compute", `eq_gpt-six-nd`) states the forward $2N$, "the backward pass takes about twice as many", and $C \approx 6ND$; this section derives it again under a second label (`eq_six_nd`) and then says it is "the estimate that :numref:`sec_gpt` used to place its run". One owner (guide 7.7): this section owns the derivation; `gpt.md` cites the formula in a clause and keeps its table of runs. One equation label.
4. **Order.** This section uses only the dense GPT. Its last body paragraph must say that MoE "does not fit this accounting", and the MoE section leans forward on it. Placing scaling laws directly after `kv-cache.md` or before `moe.md` would let both read in dependency order (see "For the chapter review").
5. **Missing, briefly.** (a) Kaplan's own answer: the reader is told the earlier fit "favored larger models" but not what it said (model size growing much faster than data). One clause gives the contrast its content. (b) What a scaling law is not: it predicts pretraining loss, not downstream ability; one sentence. (c) How to read "0.05 nats" (see SL-12).
6. **Length.** Without the configurations part the section is about 2,700 words, well proportioned.

## Passage-level findings

**SL-1** (minor, language). Opening, second paragraph. Anchor: "The answer rests on two regularities that can be measured on runs a million
times smaller."
"A million times smaller" than what is not said (Chinchilla's fitted runs were at most a few hundred times smaller than Chinchilla; ours are $10^{11}$ times smaller than Llama 3). Either drop the multiplier or tie it to a fact.

**SL-2** (minor, explanation). Under "Blocks and Embeddings". Anchor: "It is 1.4% of our model with its 256 bytes, but
nearly a third of GPT-2"
The cell prints "1%". Print one decimal or write "about 1%".

**SL-3** (minor, explanation). Same paragraph. Anchor: "The miniature study below does the same, so there $N$ means
the number of non-embedding parameters. The law of
:citet:`hoffmann2022training` that follows it counts all parameters."
Correct and necessary, and a third convention follows in "Checking the Arithmetic" ("Here $N$ counts all two-dimensional parameters"). Three meanings of $N$ in one section, each announced where used, is manageable only because the embedding is 1% here; say that once, here, so the reader stops tracking it: "for our byte-level models the three counts agree to about 1%".

**SL-4** (minor, correctness, sure). Under "Checking the Arithmetic". Anchor: "The counted cost lies 8 to 12% above the estimate, about the size of the
$n/(6d)$ correction for the attention scores"
Right for both tabs (PyTorch 1.082 and 1.082; JAX 1.094 and 1.116). The stronger statement is available and should be made: in PyTorch, where the counter sees only matrix products, the excess is the attention correction to three digits ($1.083$ predicted); in JAX the counter also includes elementwise work, which accounts for the remaining 1 to 3%. As written the sentence gives one range for two different measurements, and the tab prose has already told the reader that the two counters count different things.

**SL-5** (minor, focus). Same subsection, end. Anchor: "A count of FLOPs also
converts into time."
Four sentences on utilization that the section does not use again (exercise 4 does). They read as a note for the performance chapter. Cut to one sentence with the pointer, or move into exercise 4.

**SL-6** (moderate, thought). Under "A Miniature Scaling Study". Anchor: "so a line fitted to small models predicts the loss of larger
ones"
This sets up the power law in $L$ itself, $L = aN^{-\alpha}$, as the expectation. The law taught forty lines later has a constant, $L = E + A/N^{\alpha} + \ldots$, and is never straight on these axes. Kaplan et al. could fit pure power laws because their losses were far above the entropy floor; byte-level losses of 1.2 to 1.4 nats with a floor that may be half of that are not. The reader is given an expectation that the section's own law says is wrong, so that its failure can be shown. Introduce the constant here: "the reducible part of the loss falls as a power law".

**SL-7** (major, correctness of inference). Under "Reading the Curves". Anchor: "The three smallest models at the largest budget
lie close to a straight line, but the two larger models fall short of
its extension: the curve flattens."
Recomputed from the stored PyTorch losses (1.38, 1.35, 1.29, 1.25, 1.22): the line through the first three has slope $-0.040$; the residuals of the five points are $-0.001$, $+0.001$, $0.000$, $+0.010$, $+0.034$ (the cell prints 0.041 from unrounded losses; JAX prints 0.035). Three points and two fitted constants will always "lie close to a straight line". The fourth point misses by 0.01, inside the stated run-to-run spread of "a hundredth or two". The claim therefore rests on one point, 0.035 to 0.04 nats off a line extrapolated over a factor of eight in $N$ from a fit that spans a factor of five. That is suggestive, not a measurement of a bend. The October review flagged hedges here; the hedges are gone, and the claim is now stated more firmly than five single runs allow.

**SL-8** (major, thought). Same paragraph. Anchor: "With the data fixed, the loss approaches a floor that more
parameters do not lower, and more data lowers the floor."
Two problems. (a) No floor is observed: at every budget the largest model is still 0.02 to 0.04 nats better than the next. (b) Flattening on log-log axes does not show that data is the limit. With $D$ infinite, $E + A/N^{\alpha}$ still flattens toward $E$. What would show a data limit is an interaction: more data helping large models more than small ones. The grid shows the opposite or nothing: going from 4.1M to 16.4M tokens lowers the loss by 0.32, 0.26, 0.24, 0.24, 0.23 nats from the smallest to the largest model. (The law of `eq_chinchilla_law` predicts a gap that does not depend on $N$; exercise 7 part 3 asks the reader to notice this, so the authors know.) Replace the paragraph by what the fifteen numbers do show: loss falls with $N$ at every $D$ and with $D$ at every $N$; each tripling of $N$ buys less than the last; and at equal compute the trade favours data at this scale (Structural 1). Then the law arrives as the simplest form with those three properties, not as the explanation of a bend.

**SL-9** (minor, language). Same subsection. Anchor: "Two checks support this reading."
Three follow (run-to-run spread, no repeated text, training loss against validation loss), and the third sentence about spread is the one the reader needs before the reading, not after. Move "rerunning moves most points by a hundredth or two of a nat" up to where the plot is introduced.

**SL-10** (minor, focus). Same paragraph. Anchor: "Repetition would change these results only slowly:"
A paper result (Muennighoff et al., correct as far as I can check: up to 9B parameters, about four epochs nearly as good as fresh data) attached to an experiment that has no repetition. The law's treatment of repeated tokens is raised again at the end of "The Chinchilla Law" ("The law counts $D$ repeated tokens as $D$ fresh ones"). Say it once, there.

**SL-11** (minor, correctness, fairly sure). Under "The Chinchilla Law". Anchor: ":citet:`Pearce.Song.2024` trace to it most of the reason why the
earlier fit of :citet:`kaplan2020scaling` favored larger models"
This is that paper's claim, as I recall it. A second analysis (Porian et al., 2024, "Resolving discrepancies in compute-optimal scaling of language models") attributes the same disagreement to three other factors: the cost of the output layer, warmup length, and learning-rate tuning by scale. The section itself has just named schedule matching as a cause (line 280). "One analysis traces most of it to..." would be accurate; the present wording presents a contested attribution as settled. Unverified detail: I am confident both papers exist and of their main claims, not of the proportion each assigns.

**SL-12** (moderate, explanation). Under "Training Past the Optimum". Anchor: "The law puts it at a tenth of the compute-optimal
size and predicts about 0.05 nats of extra loss."
The reader cannot tell whether 0.05 nats is a little or a lot; the scale of the loss here (nats per token of Chinchilla's tokenizer, $E = 1.82$) differs from the 1.2 to 1.4 nats per byte of the grid above. Recomputed under the refit: the reducible loss is 0.204 for Llama 3 8B as trained against 0.150 at the optimum for its budget, a third more; equivalently, the compute-optimal model with the same predicted loss has 34B parameters, reads 0.64T tokens, and needs 18% of the training compute. So the 8B model costs about five and a half times the training compute of the model that would match it, and each generated token costs about a quarter (8B against 34B). Those two numbers are the trade the paragraph is about. The current comparison ("a tenth of the computation that the optimal model would need") is with the 80B model of equal budget, which has a lower loss, so "the smaller model is the better choice" compares models of different quality. Exercise 9 part 2 computes exactly the right numbers; quote its answer in the body and let the exercise derive it.

**SL-13** (minor, correctness). Same paragraph. Anchor: "The largest Llama 3 model sits close to the optimum; Meta's own fit on its
data put the optimum for that budget at 402 billion parameters and 16.55
trillion tokens"
Both halves are right (the paper's figures are as quoted), but the cell just printed "0.65 x optimal size" for this model under the refit, and the prose says "close" without comment. Say that it is 0.65 of the size the Chinchilla refit would choose, costing 0.001 nats, and almost exactly what Meta's own fit chose: a small, concrete instance of "a different corpus moves the ratio".

**SL-14** (moderate, correctness, sure). Same paragraph. Anchor: "the Language Models part (:numref:`chap_nlp_pretrain`)"
`chap_nlp_pretrain` is the label of `chapter_natural-language-processing-pretraining/index.md`, the retired NLP Pretraining chapter that now sits in the Attic, not a chapter of the Language Models part (`chapter_lm-*`). The link sends the reader to word2vec and BERT. Remove the `:numref:` (the other sections of this chapter refer to "the Language Models part" without one) or point to the right `chapter_lm-*` label once it exists. The same wrong target is in `encoders-decoders.md` line 374.

**SL-15** (minor, correctness, unverified). Cell `scaling-laws-training-past-the-optimum`. Anchor: "'OLMo 2 7B': (7e9, 4.05e12)"
Llama 3 (15T and 15.6T), Chinchilla and Gemma 3 27B (14T) match my memory of the reports. For OLMo 2 7B I recall about 4T pretraining tokens but cannot confirm 4.05T. Check, and cite the sources of the token counts in the prose (only Llama's is cited).

**SL-16** (moderate, focus). Under "Configurations of Deployed Models". Anchor: "The models trained on such budgets are built from the components of this
chapter."
See Structural 2. The table's entries are accurate to the best of my knowledge (GQA ratios for Llama 3 and Qwen3, RoPE bases, Gemma's 5:1 local to global layers with window 1,024, DeepSeek-V3's 512-dimensional latent, GPT-OSS's window of 128 on alternate layers with learned sinks, 128 or 32 experts with 4 active). Entries I believe but could not confirm: OLMo 3's "window 4096 on 3 of 4 layers", YaRN in GPT-OSS, and the two "newest designs" sentences (the hybrid model cited as `Kimi.Team.2025b`, and Llama 4 removing positional encoding from some attention layers). Wherever it lives, the table needs a date in its caption text ("as of late 2025") since it will age faster than anything else in the chapter, and the paragraph after it should say which column matters: the authors list where rows agree and differ but do not judge.

**SL-17** (minor, correctness of wording). Same subsection. Anchor: "a model that reads trillions of tokens about once has nothing to
memorize"
It has a great deal to memorize and does memorize some of it; what it lacks is repeated exposure. "Sees almost nothing twice, so there is little overfitting for dropout to prevent."

**SL-18** (minor, experiment). After the last cell. Anchor: "Sharing key--value heads removes part of the key and value projections, a
tenth of the parameters or so at this width"
The cell prints GPT-2 (5.26M), Llama 3 8B (4.40M) and Qwen3-32B (4.30M). The baseline for "a tenth" is the default model without sharing (4.98M by my count: 4.72M in blocks plus 0.26M of embedding), which is not printed; the printed neighbour, GPT-2, differs also by biases and a position table. Add a row for the default configuration, so that 4.98 against 4.40 and 4.30 (12% and 14%) and 12.0 against 3.0 and 1.5 KiB can be read off.

**SL-19** (minor, language). Last body sentence. Anchor: "Deployed models are thus
configurations of the chapter's one block, trained at budgets that the law
of this section helps to plan."
A recap that joins two unrelated halves in one sentence, directly before the Summary (decision 6). It goes when the configurations part moves.

## Code and experiments

- `scaling-laws-a-parameter-census`: confirms $12Ld^2$ to 0.1 to 0.2% and shows the embedding share. Teaches something beyond the formula (where GPT-2's parameters are). Keep.
- `scaling-laws-checking-the-arithmetic`: a check that can fail and nearly did in an informative way (the 8% excess). Keep; read it more sharply (SL-4). The PyTorch tab's second imports cell (`scaling-laws-1`) breaks the one-imports-cell rule; fold it into the first.
- `scaling-laws-five-sizes-one-diet-1` and `-2`: fifteen runs, both tabs agree to 0.01 to 0.02 nats, which itself supports the stated noise level. Read for more than they show on the bend (SL-7, SL-8) and for less than they show on the trade-off (Structural 1). The three-point fit line should go or be relabelled as a guide for the eye.
- `scaling-laws-the-optimal-split`: correct in every digit. The best cell of the section.
- `scaling-laws-training-past-the-optimum`: correct arithmetic; inputs partly unverified (SL-15); interpretation needs a unit (SL-12).
- `scaling-laws-recipe-rows-as-constructor-calls`: a shape check plus two printed numbers per row; baseline missing (SL-18). It demonstrates the API more than an idea, and leaves with the configurations part.
- Tab prose: two notes of about 85 words each explain what each framework's counter counts. They are needed to read the output, but they are long, and the JAX note's caution about GPU kernels is plumbing for a comment.
- Figure `mdl-transformers-chinchilla.svg`: the picture is right and useful. Defects against the house checklist: the label "$C = 5.76 \times 10^{23}$" crosses a contour and a budget line; "Llama 3 8B" sits on the dashed budget line; the caption says Llama 3 8B is "on a budget about a quarter larger" but no line for that budget is drawn, so the point appears to lie on Chinchilla's line. Either draw its budget line or move the remark out of the caption.

## Exercises

Ten problems, about half the body. A very good set; exercises 1 and 2 extend the cost model, 3 makes the embedding-count effect computable, 7 and 8 probe how little the fits determine, 9 does the lifetime optimum properly. Changes: exercise 10 (IsoFLOP profiles) becomes the direct continuation of the body if Structural 1 is adopted, and could shrink to its parts 2 and 3. Exercise 6 ("A sixth size") is a good prediction exercise and is the honest test of the bend; if the authors want the bend in the body, this run belongs there instead of the extrapolated line. Exercise 4 (DeepSeek-V3 utilization) depends on an H800 peak figure that I could not check ("about 990 TFLOP/s in BF16"); verify or phrase as "take". The attribution to Stanford CS336 on exercise 10 is good practice.

## Still open from October

"Hedges that contradict the conclusion (`scaling-laws.md:475` vs `:499`)": the hedge stack is gone, but the underlying problem (the grid does not identify a bend or its cause) is now stated as fact instead of hedged (SL-7, SL-8). "Cut: configuration tables ... (about 1,200 words)": partly done; one table of seven rows and about 600 words remain (Structural 2).

## For the chapter review

- This section is two things: the chapter's quantitative close (cost, loss, allocation) and the chapter's synthesis (deployed configurations). Split at `## Configurations of Deployed Models`; the second part is a chapter close, not scaling.
- Order: scaling laws depend only on `sec_gpt` (and on `sec_scaling` in chapter 9 for the $1/d$ learning-rate rule, which that section states for width $n$ of an MLP; the reference holds). MoE depends lightly on scaling laws. A dependency-ordered close would be: vision transformer, scaling laws, mixture of experts, the modern recipe.
- Repetition across the chapter: $6ND$ derived in `gpt.md` and here under two labels; the WikiText-103 byte corpus described in full in `gpt.md`, `moe.md` (lines 614 to 620) and here (lines 250 to 254); the `d2l.LMOptimizer` recipe described three times. One home (`sec_gpt`) and a clause elsewhere.
- Wrong cross-reference: `chap_nlp_pretrain` (SL-14), also in `encoders-decoders.md`. Worth a chapter-wide grep for labels of retired chapters.
- No appendix dependence. The optimum is a two-line calculus exercise and is rightly kept in the main text.
- Currency: the law, the refit, the inference-aware variant and the table are current to late 2025. For a 2026 to 2027 book the one thing to watch is that the "20 tokens per parameter" rule is by now mainly cited to be departed from; the section already says so and shows it, which is the right stance.
