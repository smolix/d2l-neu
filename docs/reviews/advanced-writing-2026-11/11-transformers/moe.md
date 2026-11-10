# Mixture of Experts

Source: `chapter_transformers/moe.md` (read to the `<!-- slides -->` marker at line 1076). About 4,250 words of body prose, 140 words of tab prose and 1,700 words of exercises; about 6,100 in all. Stored outputs read: `outputs/pytorch/chapter_transformers/moe.json`, the JAX file, and both stored plots of each. Figures `img/mdl-transformers-moe.svg` and `img/mdl-transformers-moe-fine.svg` rendered and inspected.

## Verdict

The best of the three closing sections and close to the bar. It has a felt problem ("Can a model hold more parameters without spending more computation on each token?"), an accounting the reader can redo on a napkin and that reproduces published model sizes, a from-scratch layer that does real dispatch instead of masking, a failure mode with a mechanism, two remedies compared in one run, and a headline experiment whose outcome (the mixture overtakes the dense model after about 1,500 steps) the October review said was missing. Every model fact I could check is right. What keeps it from the bar is that both experiments are entangled with an artifact of the implementation, the fixed-capacity buffer that drops tokens. The imbalance the reader sees is present before training and comes from byte frequencies, not from the feedback loop the text has just explained; the "no balancing" run loses mostly because up to 44% of its assignments are dropped; and the final comparison needs a 180-word paragraph about capacity factors before it can be run, then wins by 0.02 nats in the default tab. The single most important fix: separate the three effects (frequency skew at initialization, self-reinforcement during training, dropping) so that each claim in the prose has a number behind it, and say how large the final gain is against run-to-run noise.

Does it belong in a chapter on the transformer? Yes. It changes one sublayer of the block and is taught by building it; it is not a survey. The only survey material is the last subsection (fine-grained and shared experts, with a table of four released models), about 450 words that the body neither implements nor tests.

## Scores

| dimension | score | note |
|:--|:-:|:--|
| Clarity of explanation | 4 | Routing, the gradient through $p_i$, and both balancing rules are clear. The dispatch paragraphs are dense and have no worked example. |
| Clarity of thought and focus | 3 | One line of argument, but three mechanisms (skew, feedback, dropping) are mixed in the experiments, and one configuration in the accounting is not the one trained. |
| Pedagogy | 4 | Count, build, break, repair, compare: the right sequence. The exercises carry the missing ablations. Loses a point for a headline result near the noise. |
| Language | 4 | Plain, exact, and in places has a voice ("The name '8x7B' thus describes what the model stores, not the work per token"). |
| Ease and enjoyment | 4 | A reader wants to keep going until the capacity-factor paragraph, where the section starts explaining its apparatus. |

## What works

- **The opening** (lines 4 to 19). "Continuing 'the integral of' draws on different knowledge from continuing 'Act I, Scene'." A concrete image in one line, then the question, then DeepSeek-V3's 671 and 37 billion as the proof that the question has an answer. October asked to keep lines 4 to 6; still true.
- **The accounting cell** (`moe-conditional-computation`). Twelve lines of arithmetic reproduce 656B of DeepSeek-V3's 671B and Mixtral's 46.7B stored and 12.9B active. I recomputed all of them, and the Qwen3-235B figures of exercise 2 (235.0B stored, 22.2B active). This is "code teaches" at its best: the reader learns where the parameters of a real model sit.
- **The two assumptions paragraph** (lines 112 to 120). It says what the count leaves out (memory, communication) and what it presumes (even load), and the second assumption motivates the rest of the section. A model of how to state a caveat once.
- **Raw probability against renormalized weight** (lines 154 to 167). The observation that a renormalized top-1 router "learns nothing from the loss at all" is the kind of detail that decides whether an implementation trains, and exercise 1 makes the reader derive it.
- **"Follow the feedback loop."** (lines 427 to 438). A mechanism told as a sequence of events, with a named end state. October asked to keep it; it survived.
- **The two balancing rules side by side** with their formulas, and the sentence that states the difference: "The controller steers where tokens go; the gradients serve prediction alone."
- **Honesty about costs** (lines 809 to 816): memory, optimizer state, buffer rows, data movement, drops.
- **Exercises 3, 5 and 6** (balancing-loss gradient, controller in isolation, capacity factor with the binomial estimate). Each isolates one mechanism and has a computable answer.

## Structural findings

1. **Three causes of imbalance are run together.** The text teaches self-reinforcement (`subsec_moe-feedback`), then the experiment shows a model that is already at 2.2 to 2.3 times the even share with 17 to 19% dropped before any training, explains that by byte frequency (a fourth of the paragraph after the plot), and concludes "Training without balancing does not undo the imbalance." In the JAX tab the untrained and the unbalanced trained model are nearly the same (2.2 and 2.6 times the share); only in the PyTorch tab does training make it clearly worse (5.2). So the stored outputs do not show the feedback loop; they show a frequency skew that persists. The Summary then lists both causes as findings ("both because similar tokens go to the same expert and because routing reinforces itself"). The ablation that separates them is exercise 4 part 1 (freeze the router). Either bring that one run into the body, or teach the two causes in the order the experiment shows them: first "identical tokens go to the same expert, and bytes are not equally frequent", then "training reinforces whatever the router starts with".
2. **Dropping is a property of this layer, and it drives both experiments.** With `capacity_factor=1.25` and byte tokens, a dropped assignment means the token skips its feed-forward network. The unbalanced run's worse loss (1.85 against 1.56 to 1.60 in PyTorch; 1.69 against 1.54 to 1.59 in JAX) is attributed to "experts that receive almost no tokens add parameters that the model hardly uses, and many tokens skip their expert altogether". The second cause is likely the larger one, and one evaluation would say: report the unbalanced model's validation loss with a capacity factor large enough to drop nothing. Without it the reader cannot tell wasted capacity from skipped computation.
3. **The capacity-factor paragraph before the final experiment** (lines 736 to 749) is the guide's "paragraph of confounders" (section 9): 180 words on why $c = 2$, what happened in the authors' runs at $c = 1.25$, and a range of outcomes no cell shows. It also quietly doubles the feed-forward computation of the mixture ($EC = 2n$ rows), which the claim "at the same computation per token" then has to take back three paragraphs later ("the buffers hold twice as many rows as there are tokens"). Options, in order of preference: (a) use a tokenizer-free fix that removes the problem, namely balance the final model with the bias controller, which reaches 0 to 1% drops at $c = 1.25$ in the section's own comparison; (b) keep $c = 2$ and cut the paragraph to two sentences; (c) evaluate both models without dropping. Under any option, state the computation honestly once.
4. **"Fine-Grained and Shared Experts"** is the survey part: a paper's two design steps, one reported result, a table of four released models. It is well told and the figure is good, but the body tests none of it (exercises 8 and 9 do). It also duplicates `further-reading.md` ("Sparsity and Expert Width", lines 288 to 305), which cites the same two scaling studies as lines 795 to 807 here. Keep the two-step idea and the figure (about 200 words), move the table and the released-model comparison to the configuration table of `scaling-laws.md` or to further reading, where three of the four models already appear.
5. **Order within the chapter.** The argument "the gain needs data" (lines 799 to 807) and exercise 7 lean on `sec_scaling-laws`, which comes next. The dependence is light, but scaling laws need only the dense GPT, and MoE reads better with tokens-per-parameter already in hand. See "For the chapter review".
6. **Length.** About 4,250 words is proportionate. Cuts: the capacity paragraph (150 words), the table and its commentary (200), the second scaling-study citation (60).

## Passage-level findings

**MOE-1** (moderate, correctness, sure). Under "Conditional Computation". Anchor: "We apply it to two deployed models and to our own configuration, the one
trained at the end of this section."
The cell computes `moe_accounting('ours', 256, 683, 8, 1, 6, unit=1e6)`: width 256, six layers, 25.2M expert parameters. The model trained at the end of the section is `small_gpt`: width 128, two blocks, 2.26M parameters in all. The "ours" row is the six-block GPT of `sec_gpt`, which appears only in exercise 7. Change the call to `(128, 341, 8, 1, 2)` (expect about 2.1M stored and 0.26M active) or change the sentence. The ratio of 8 quoted afterwards holds either way.

**MOE-2** (minor, language). Same subsection. Anchor: "a token uses less than two sevenths of the
parameters"
The play on "8x7B" (two experts of seven billion) is opaque: 12.9 of 46.7 is 28%, and the reader has to work out why sevenths. "A token uses 12.9B of the 46.7B, a little over a quarter" says it.

**MOE-3** (minor, explanation). Under "Conditional Computation". Anchor: "(The
37B on its model card add"
A parenthesis that resolves the mismatch between the 23B just computed and the 37B of the opening. Good that it is there; it would be stronger as a printed line, as the cell already does for Mixtral, so that the reader sees the 37B assembled. As prose it is an assertion.

**MOE-4** (moderate, explanation). Under "Dispatching Tokens to Experts". Anchor: "We number the assignments of each expert in the
order of the tokens and copy the token of an expert's $r^\textrm{th}$
assignment into slot $r$ of that expert's buffer."
This paragraph and the one after the code ("The cumulative sum over the one-hot encoding numbers the assignments...") are the hardest in the section, and they describe an index manipulation in words. A six-token example would carry it: tokens routed to experts (2, 0, 2, 2, 1, 2) with $E = 3$, $C = 3$ give ranks (0, 0, 1, 2, 0, 3), so the last token is dropped; show the one-hot cumsum as a small array. The Illustrated Transformer standard is exactly this: one tiny case drawn out. Also say here, not only in exercise 6, what the rule implies: tokens late in the flattened minibatch are dropped first, so the last sequences of a batch lose their experts.

**MOE-5** (minor, thought). Same subsection. Anchor: "Fixed shapes come with dropping: a balanced router loses few
assignments, an unbalanced one many."
True of this implementation, and the section never says that dropping is a choice of the early papers (GShard, Switch) that current releases avoid. Readers will take token dropping as intrinsic to mixtures of experts. One sentence: implementations that group tokens by expert with variable-length segments drop nothing; we use fixed buffers because they compile and batch simply.

**MOE-6** (minor, pedagogy). Under "A Mixture-of-Experts Layer". Anchor: "Besides its output, the layer keeps four things for the experiments
below, which its method `record` updates for every batch:"
An announcement of bookkeeping that the reader cannot yet use ("which stays zero until we need it", "that we derive when we use it"). The four attributes are plumbing; a comment per attribute in the code does the same work and the prose can go.

**MOE-7** (minor, pedagogy). Tab prose after the layer. Anchor: "No group size leaves the device, so the layer never waits for the host."
A sentence only someone who has written the naive version (a Python loop with `.item()`) can parse. Either say what the alternative would be, or delete. The JAX note is clear and needed.

**MOE-8** (minor, experiment). Under "Dispatching Tokens to Experts", after the shape check. Anchor: "A fresh router spreads random tokens roughly evenly: no expert's
share is far from the even share of $1/8$"
A check on Gaussian vectors that the next experiment immediately overturns for real data ("unlike the layer that we fed random vectors above"). It teaches a false baseline for 25 lines before it is corrected. Either drop the shares from this cell (keep the parameter count, which is used) or make the contrast the point: random vectors spread evenly, bytes do not, and say why in one place.

**MOE-9** (moderate, thought). Under "Positive Feedback in Routing". Anchor: "This loop can end in *routing
collapse*: a few experts receive most of the assignments, and much of the
stored capacity goes unused."
See Structural 1. The mechanism is correct and well told, but the section's runs do not isolate it, and in the JAX tab the trained unbalanced model is barely more concentrated than the untrained one. Add the frozen-router run (exercise 4 part 1) as a fourth line of the comparison, or soften the Summary to what was shown: the skew is there at initialization and training without balancing keeps or worsens it.

**MOE-10** (minor, correctness, fairly sure). Under "Balancing Without a Loss". Anchor: "the model's loss contains no
balancing term, and no gradient competes with the language-modeling one"
True of the method of Wang et al. and of this layer. DeepSeek-V3 itself, cited two sentences earlier as training "with their method", keeps a small sequence-wise balance loss beside the bias controller (its report gives a weight of $10^{-4}$, as I recall). The text does not claim otherwise, but a reader will conclude that V3 trains without any balancing loss. Add a clause.

**MOE-11** (minor, explanation). Same subsection. Anchor: "DeepSeek-V3
used $u = 0.001$ for the first 14.3 trillion of its training tokens; our
runs take 800 steps, and we use $u = 0.01$"
The number is right. What the reader needs to choose $u$ is its scale against the probabilities it competes with: a bias of 0.01 is added to probabilities near $1/8$, and after 100 steps in one direction it has moved by 1, more than any probability. Say that, and that the controller therefore always wins if imbalance persists (exercise 5 explores it).

**MOE-12** (moderate, thought). Under "Balancing Methods Compared", after the plot. Anchor: "The fresh model is unbalanced before it has seen any data, unlike the
layer that we fed random vectors above"
This paragraph is the most interesting observation in the section (the first block sees little but the byte identity; one byte in five is a space) and it arrives as an after-the-fact explanation of a surprise. It deserves to be the prediction. Before the cell: "A byte-level model gives the router very little to work with in the first block. What should the load look like before training?" Then the cell confirms. As it stands, eleven lines explain the initialization row and three sentences cover the result the subsection is named for.

**MOE-13** (moderate, correctness of inference). Same passage. Anchor: "Both runs end at or below the validation loss of the unbalanced one, and
the further that run has collapsed, the wider the gap."
"The further ... the wider" is a dose-response claim drawn from two runs, one per tab (PyTorch: 5.2 times the share, gap about 0.25; JAX: 2.6, gap about 0.10 to 0.15), each with its own initialization. It may be true, and the authors may have run more; say so as a range over repeated runs (decision 5) or drop the comparative. And attribute the gap to its cause (Structural 2).

**MOE-14** (moderate, pedagogy). Under "A Mixture-of-Experts GPT". Anchor: "In our runs at
$c = 1.25$, the first block of some models dropped a tenth of its tokens
for a thousand steps or more."
Results of experiments the reader cannot see, used to justify a setting, in a paragraph that postpones the section's main experiment. See Structural 3.

**MOE-15** (moderate, correctness). Same subsection. Anchor: "Its validation loss falls below the dense model's within the
first two thousand steps and stays below it to the end, by a few hundredths
of a nat."
Stored outputs: dense 1.32 against 1.30 (PyTorch) and 1.27 (JAX). "A few hundredths" covers 0.02 to 0.05, and the plot confirms the crossover near 1,500 steps in both tabs, so the sentence is accurate. But the reader is not told how large the run-to-run spread of this model is (the scaling section gives "a hundredth or two" for similar models), so in the default tab the headline gain equals the noise. The next sentence, "At the same computation per token, the stored experts make a better model", is then stronger than the evidence, and the computation is not the same (Structural 3). Give the range over repeated runs in the prose, as elsewhere in the book, and compare the gain with something the reader can weigh: by the dense scaling curve of `sec_scaling-laws`, 0.03 nats is what roughly doubling a dense model of this size buys. That turns a small number into a result.

**MOE-16** (minor, explanation). Same passage. Anchor: "each fresh
expert's output is scaled by a probability near $1/8$"
Correct, and it is a cost of the raw-probability weight praised earlier (with $k = 1$ the whole feed-forward sublayer starts at an eighth of its dense strength). One clause linking back ("the price of letting the router learn from $p_i$") would connect the two passages.

**MOE-17** (minor, focus). Same subsection. Anchor: "Large studies report the same effect."
Three studies in two paragraphs (Switch, Clark et al., Krajewski et al.), two of which are retold again in `further-reading.md`. Keep Switch here (it is the design the layer follows), move the other two to further reading.

**MOE-18** (minor, correctness, unverified). Under "Fine-Grained and Shared Experts". Anchor: "than a GShard model with the same stored and active parameters (1.81
against 1.87)"
I believe these are the Pile losses of the 2B models in the DeepSeekMoE paper (1.808 and 1.867) and that the "1.5 times" comparison is the paper's own, but I cannot confirm the digits. Check against the paper's table. The rest of the subsection (16 of 2 becoming 64 of 8; one shared plus 7 of 63; $\binom{64}{8} \approx 4.4 \times 10^9$) is right.

**MOE-19** (minor, language). Same subsection. Anchor: "say one for chemical vocabulary and
another for nested clauses"
A vivid guess presented as how experts specialize. Published analyses (Mixtral's among them) found routing follows surface features such as token identity and position more than topic, and this section's own first-block observation agrees. Keep the image but mark it as hoped-for ("in the ideal case"), or replace it with what is observed.

**MOE-20** (minor, language). Under the table. Anchor: "The three later models use many more experts than Mixtral and between 4
and 8 active per token, and most of them make each expert narrower than
the model itself."
"Most of them" is two of three. The table values check out (width ratios 3.5, 0.29, 0.375, 1.0; DeepSeek-V2 with two shared experts). If the table stays, add the stored-to-active ratio per model, which is the quantity this section taught the reader to compute.

## Code and experiments

- `moe-conditional-computation`: correct; "ours" row is the wrong configuration (MOE-1).
- `moe-a-mixture-of-experts-layer-2`: shape checks plus a share printout on Gaussian tokens that misleads (MOE-8). The parameter counts (4,198,400 and 1,051,136) match the prose.
- `moe-three-runs-one-budget-3`: stored lines agree with the prose at its stated precision in both tabs. The two remedies swap order between tabs (1.60 and 1.56 in PyTorch, 1.54 and 1.59 in JAX), and the prose correctly says neither is consistently ahead. What the experiment cannot show: the feedback loop (Structural 1) and the cause of the unbalanced model's loss (Structural 2). Each run draws its own initialization; a shared seed across the three runs would remove one source of spread at no cost.
- `moe-three-runs-one-budget-4`: plot read correctly. In PyTorch the unbalanced run has three experts near zero share; in JAX four.
- `moe-a-mixture-of-experts-gpt-1`: 0.43M against 2.26M, ratio 5.3, as stated.
- `moe-a-mixture-of-experts-gpt`: see MOE-15. The dense model is also trained through `train_moe_lm` with `aux_weight=0.01`, which is harmless (no expert layers) but looks odd; a comment would help.
- Numbers no cell prints: the outcomes at $c = 1.25$ (MOE-14); "about 50 million bytes, more than a hundred per parameter" (correct: $6000 \cdot 64 \cdot 128 = 49$M, 114 per dense parameter).
- No cell merely confirms a derivation. No TensorFlow or MXNet material. The two tab notes are short and specific to their framework.
- Figures: `mdl-transformers-moe.svg` is clear, but the label "$\times\, p_3 = 0.24$" sits on the green arrow from $\mathrm{FFN}_3$ (house rule 3); move it left. `mdl-transformers-moe-fine.svg` is clean and carries its argument.

## Exercises

Nine problems, about 40% of the body: a strong set, weighted toward mechanism. Keep all. Notes: exercise 4 part 1 should move to the body (Structural 1). Exercise 7 ("When the mixture loses") depends on `sec_scaling-laws`, the next section; fine if the order stays, natural if the order changes. Exercise 9 is honest about the body ("this section does not test it"), which is another sign that the fine-grained subsection is survey. Exercise 2 part 2 (expected distinct experts read per decode step) is the best serving problem in the chapter. One to add: evaluate the unbalanced model of the comparison with dropping disabled (Structural 2), if that does not go into the body.

## Still open from October

"The MoE experiment shows no gain and is left uninterpreted (`moe.md:740`)": addressed. The mixture now ends below the dense model and the result is interpreted; what remains is that the gain is 0.02 nats in the default tab and reached only after a change of capacity factor (MOE-14, MOE-15). The "keep" items (opening, feedback loop) are intact.

## For the chapter review

- Fit: a real teaching section that modifies one sublayer of the chapter's block through `ffn_factory`; it uses `d2l.GPT`, `d2l.FeedForward`, `d2l.LMOptimizer`, `d2l.train_steps` and `d2l.lm_perplexity` as given (all labels resolve: `sec_gpt`, `sec_adam`, `sec_kv-cache`, `sec_transformer-block`). No appendix dependence.
- Order: consider scaling laws before this section. Scaling laws use only the dense GPT; this section's regime argument ("more than a hundred [bytes] per parameter", "the gain needs data") and exercise 7 would then rest on something taught, and the closing configuration table (now in `scaling-laws.md`) could follow MoE as a short chapter close of its own.
- Forward reference to the Language Models part ("a systems problem that the Language Models part takes up"): not load-bearing.
- Overlap: Clark et al. and Krajewski et al. are told here and in `further-reading.md`; the released-model table overlaps `tab_modern-recipe` in `scaling-laws.md` (DeepSeek-V3, Qwen3, GPT-OSS appear in both).
- Chapter-wide pattern: the small experiments are run once per tab and the prose is careful about precision, but the reader is never given the run-to-run spread of the small GPT in this section, so cannot weigh 0.02 nats. One sentence in `sec_gpt` giving that spread, cited here and in scaling laws, would serve the whole chapter.
