# Chapter 12, State Space Models: writing review

Sources: the eight `.md` files of `chapter_recurrent-modern/`, in book order. The six teaching sections were read in full by section reviewers; I read all six review files, reviewed `index.md` and `further-reading.md` myself, and checked the most serious cross-section claims against the sources and the stored outputs in `outputs/pytorch/chapter_recurrent-modern/` and `outputs/jax/chapter_recurrent-modern/`. About 27,000 words of prose and exercises in the six sections, plus about 1,000 in the introduction and 1,500 in the further reading.

Per-section reviews:
[index](12-state-space-models/index.md) ·
[lstm](12-state-space-models/lstm.md) ·
[ssm](12-state-space-models/ssm.md) ·
[mamba](12-state-space-models/mamba.md) ·
[matrix-state](12-state-space-models/matrix-state.md) ·
[deltanet](12-state-space-models/deltanet.md) ·
[hybrids](12-state-space-models/hybrids.md) ·
[further-reading](12-state-space-models/further-reading.md)

**Author decision (Alex Smola, 2026-11-10).** "Redesign the experiments, otherwise it's pointless." The alternative offered in this file, to say once that the cheaper models do about as well at this scale, is rejected. Each step of the chapter's ladder gets an experiment in which that step alone changes, on a task hard enough to show it, with parameter counts, a transformer baseline and a cost table against attention. The proposals under "Experiments" and "What is missing" below are the starting point; the work is for the development machine. Until then the prose pass corrects only what the stored outputs contradict (LSTM-11 first). See `overview.md`, "Author decisions".

## Verdict

This chapter has the best line of argument a textbook treatment of this material could have, and in places the best teaching: the gated sum built in three steps, the scalar recurrence and the composition rule for the scan, the two-dimensional memory that spoils when a third key arrives, "read before writing", the cache arithmetic. The prose is plain, candid about its own experiments, and free of the mannerisms October objected to. Each section answers the limitation on which the previous one ended. On explanation the chapter is close to the bar, and no other text I know carries a reader from the LSTM to Gated DeltaNet in one notation.

It is not at the bar, and the reason is one thing: the experiments do not support the story the prose tells, and the prose mostly knows it and steps around it. The first experiment of the chapter shows the plain recurrent network recalling a symbol over longer spans than the LSTM and the GRU with default gates, and the text says the opposite in its introduction and its summary (confirmed below). A bias initialization matches S4D; the minGRU nearly matches Mamba; the delta rule's benefit is never shown on a trained model for the problem that motivates it; the common language benchmark puts the LSTM first and a transformer near the bottom and separates nothing; and the last section draws its conclusions from about a dozen runs that no cell prints. There is no cost comparison with attention anywhere, though the introduction promises cost and the reader has just come from two chapters on attention. What matters most is to decide what each experiment shows, say that in the chapter's own candid voice, and replace the three or four experiments that show nothing by ones that show the mechanism. The explanations can stay almost as they are.

**Do the experiments support the thesis?** In part. They support: gates start with short memories and initialization decides (the keep-gate rows); a linear recurrence can be computed for all positions at once; a model without input-dependent dynamics cannot copy selectively (S4D at 0.2); a matrix memory holds a number of pairs set by its size (the capacity law); a fixed state loses pairs as the load grows and one attention layer restores them (the recall sweep). They do not support: that each step of the chapter's ladder (minGRU, S4D, Mamba, matrix memory, delta rule) improves on the one before; that the continuous-time view adds something a gate bias does not; that Mamba needs both of its ingredients; that the delta rule is why Gated DeltaNet recalls more; that any of these models "models language well" with a fixed memory on long text, which is the introduction's question. In three sections the older, cheaper idea does about as well at this scale. That is a finding, and the chapter should state it once instead of leaving each section to explain it away.

**Are the last three sections settled enough for a textbook?** The principles are: the outer-product memory and its capacity, decay, the duality with masked linear attention, the delta rule as a gradient step, the counting bound and the cache arithmetic are exact, old or both, and will not date. The particular layers are not: the rank-16 decay gate with exponent 1/16, the width-4 convolution, the head sizes and the hybrid ratios are designs of 2024 and 2025. And the empirical account in `hybrids.md` (the "slots" explanation, the attribution of the recall gap to the write rule, the four ablations) is research in progress written up as a result. The remedy is not to cut the sections but to shift their weight: more of the settled principle, a smaller and simpler layer, and only those experimental claims that a cell in the book prints.

## Scores

1 to 5 as the reviewers gave them; 5 means a page one would be happy to find in MacKay.

| Section | Explanation | Thought | Pedagogy | Language | Enjoyment |
|---|---|---|---|---|---|
| [index](12-state-space-models/index.md) | 4 | 3 | 3 | 4 | 3 |
| [lstm](12-state-space-models/lstm.md) | 4 | 3 | 3 | 4 | 3 |
| [ssm](12-state-space-models/ssm.md) | 4 | 3 | 4 | 4 | 4 |
| [mamba](12-state-space-models/mamba.md) | 3 | 3 | 3 | 4 | 3 |
| [matrix-state](12-state-space-models/matrix-state.md) | 4 | 3 | 3 | 4 | 3 |
| [deltanet](12-state-space-models/deltanet.md) | 4 | 4 | 3 | 4 | 4 |
| [hybrids](12-state-space-models/hybrids.md) | 3 | 3 | 3 | 3 | 3 |
| [further-reading](12-state-space-models/further-reading.md) | 4 | 3 | 3 | 4 | 3 |

Two reviewers covered the six sections (one the first three, one the last three); their calibration looks the same to me. The scores discriminate little: 32 of 40 are a 3 or a 4 in the same pattern. Read with the reviews, the order is clear. Strongest: `deltanet.md` (one question, one answer, derivation, picture, table; its only weakness is its evidence) and the first half of `ssm.md`. Each of `lstm.md`, `ssm.md` and `matrix-state.md` has a first third or half that would score 5 and a second part that would score 3, which the single row hides. Weakest: `hybrids.md` (right question, clean headline result, then a heuristic explanation and a laboratory notebook) and `mamba.md` (the hardest step of the chapter made in one dense paragraph, and experiments that do not separate its subject from the minGRU). Pedagogy is 3 in seven of eight rows for the same reason everywhere: the experiments.

## Technical correctness and currency

The mathematics of the chapter is sound: reviewers re-derived the scan, the zero-order hold, the adjoint recurrence, the capacity proposition, the delta-rule transition, the chunked delta algebra, the implicit step, the Welch bound and the memory arithmetic and found no error in a derivation. The errors are in the reading of experiments and in a few overstatements. Ordered by severity.

**Stored outputs that contradict the prose.**

- **LSTM-11** (major; confirmed by me in both stores). The delayed recall table. Stored PyTorch, delays 5 to 200: RNN 1.00 1.00 1.00 1.00 1.00 0.13; LSTM 1.00 1.00 0.37 0.12 0.13 0.12; GRU 1.00 1.00 1.00 0.12 0.13 0.12; both keep-gate rows 1.00 throughout. JAX: RNN 1.00 1.00 1.00 1.00 0.12 0.75; LSTM 1.00 1.00 0.87 0.12 0.13 0.12; GRU 1.00 1.00 1.00 0.12 0.13 0.13. The plain network recalls at 50 steps in both frameworks and at 100 in PyTorch, where both default-gated networks are at chance. The prose calls it "erratic" and concludes "Gates make a long memory possible and learnable" without saying that the ungated network had one. With it: **LSTM-1** (the introduction's "a few dozen steps at best"), **LSTM-19** (the Summary's "Both remember over spans where a plain network fails", `lstm.md` line 915, and the same "few dozen steps" at line 907), and **SSM-20** (the sentence in `ssm.md` that steps around the RNN row). Certain. The explanation the reviewer proposes (LeCun normal recurrent matrix with spectral radius near one, hidden state small, so the Jacobian product neither vanishes nor explodes at the start) is plausible and untested; the reviser should verify it before printing it, for instance by running the RNN row with PyTorch's default initialization.
- **MS-3** (major; certain, re-run by the reviewer). "doubling $d_k$ doubles the number of pairs held" contradicts the table above it, in which doubling $d_k$ quadruples the pairs held at a given recall, because the codebook width grows with $d_k$. Recall depends on $d_k d_v$, the size of the state.
- **MS-4** (moderate; certain). "it grows faster than in proportion to $n$" reads an artifact of the codebook's nonzero mean as a property of correlated keys. The error with correlated keys is still proportional to $n - 1$, about four times larger.
- The timing run in `matrix-state.md` (cell `matrix-state-chunked-computation-3`): the prose says the dual and chunked forms take "about a millisecond or less"; the stored PyTorch output has the dual form at 2.45 ms. Minor.

**Conclusions the experiments do not carry.**

- **HY-10** (major). The "slots" account rests on "80 to 90%" against "90% or more", two ranges that touch, neither printed.
- **HY-11** (moderate). "The delta rule pays off" is drawn from two layers that differ in write rule, form of decay, gate parametrization and chunk size, one of which has an optimization plateau. See also DN-13 and structural finding S1 of `matrix-state.md`.
- **SSM-5** (moderate) and **MAMBA-17** (moderate). Parameter counts. "a third of the GRU's weights per layer" frames a perplexity gap between models that differ by 28% in size; Mamba's 3.0 against the minGRU's 3.2 comes with about 50% more parameters, and the text compares its count with the LSTM's only. Related: LSTM-15 (the LSTM model has 58% more parameters than the RNN model, printed and not read).
- **MAMBA-15** and **MAMBA-5** (moderate, dimension thought). The claim that Mamba needs both ingredients rests on 1.00 against 0.97.

**Statements that are wrong or too strong.**

- **DN-4** (moderate; certain). "so the memory cannot" grow: the transition never amplifies, but the writes still add. Growth is additive, not excluded.
- **DN-9** (moderate; certain). The regression table calls a one-pass gradient method "least squares"; the closed form needs the pseudoinverse when the keys do not span the space. The reviewer's measurement (structural finding S2 of `deltanet.md`) shows how far the delta rule can be from the least-squares fit.
- **LSTM-13** (moderate). Stacked recurrent networks "are trained with the same techniques as stacked transformer blocks" is anachronistic for the LSTM systems of the 2010s; it is this book's design choice.
- **LSTM-8** (minor). The library GRU differs from the equations; disclosed only in an exercise. **LSTM-2** (minor, filed under explanation) overstates the singular-value condition.
- **SSM-8** (minor). The two tabs run different scan schedules and the shared prose does not say so.
- **HY-2** (minor; certain). The bit count omits the bits that say which keys were seen (about 400 bits, not 128). **HY-8** (minor; certain, confirmed by me). `d2l.cosine_warmup` and `eq_cosine_warmup` are defined in `chapter_convolutional-modern/alexnet.md`, not in `sec_batch_norm`.
- **DN-5** (minor; confirmed numerically). The parity construction needs the same nonzero value written each time.
- **MAMBA-10** (minor, medium confidence). Backpropagation through the doubling scan stores about twenty tensors of the size of the state, not ten.

**Claims about papers and systems, unverified.**

- **MS-12** (moderate; fairly sure). The layer called "gated linear attention" has a short convolution and unit keys that are not in the cited paper; `hybrids.md` then compares it with "the reference implementation of the layer".
- **MAMBA-12** (minor; the reviewer is confident from the released code). The rank of the step-size projection is $\lceil d/16 \rceil$ in the reference, not $\lceil H/16 \rceil$, while the text says "As in" the paper.
- **MAMBA-1** (minor). "matched or exceeded transformers of the same size" needs attribution and a scale.
- **SSM-15** (minor, medium) and **FR-2** (moderate, unverified): whether S4D's real rates "approximate" the HiPPO design and whether they work "nearly as well as the full matrix". Two reviewers doubt it independently. `ssm.md` builds on these rates, so check the paper.
- **FR-6** (moderate; fairly sure). The expressivity result is conditional on a complexity conjecture, and "non-diagonal transitions can" should be "non-diagonal and input-dependent".
- **DN-6**, **DN-11**, **MS-9**, **HY-5**, **FR-3**, **FR-4**, **FR-5**, **FR-7**, **FR-8**, **IDX-7**, **SSM-2** (all minor): finite precision for the parity result; "the update of Longhorn"; Mamba-2 and RetNet decays; Arora et al. and Jelassi et al.; the linear recurrent unit; S5; H3; Griffin's block pattern; the 2025 hybrids; three resource links; why transformers replaced recurrent networks.

**Numbers no cell prints.** `hybrids.md`: about a dozen (plateau lengths, the reference implementation, doubled training, the 128-key run, four ablations; structural finding S2 and HY-9, HY-10, HY-12). `matrix-state.md`: the overflow statistics (MS-11). `mamba.md`: "about 520,000", "about 115,000", "a few gigabytes" (MAMBA-9, MAMBA-10, MAMBA-17). `ssm.md`: the counts behind "a third of the GRU's weights" (SSM-5). `lstm.md`: "is still falling at the last step" (LSTM-16). `deltanet.md`: none.

**Currency.** The teaching sections stop in 2024. `mamba.md` does not say that the scan-based layer it builds was followed by Mamba-2, nor that pure state space language models did not displace transformers (structural finding 4 there). `hybrids.md` names Jamba only (HY-13), although the chapter has just built the layer that the 2025 hybrids use. `lstm.md` does not say where gated networks are still used or that a forget bias of 1 is the standard default (structural finding 6 there). All of this is in `further-reading.md`, correctly; three or four sentences need to move forward.

## Focus

The chapter is focused where October found it unfocused. Surveys and variant lists are out of the sections and in `further-reading.md`; no section retells a paper. What remains:

- **Text written for the authors.** The initialization paragraph in `lstm.md` (LSTM-9), the overflow report in `matrix-state.md` (MS-11), the model paragraph and the three paragraphs of unseen runs in `hybrids.md` (structural findings S2 and S5, HY-6, HY-9). These are the chapter's worst pages and they are all the same kind: a record of what the authors tried.
- **Implementation detail at kernel level** in a chapter whose introduction says "It teaches algorithms rather than GPU kernels": the per-coordinate masks, fold and sub-chunks in `matrix-state.md` (S1, MS-13), the memory accounting of the scan in `mamba.md` before any selective model has been trained (structural finding 5 there).
- **Small strays**: bidirectional networks (LSTM-14), induction heads (HY-6), network-valued memories (DN-12), the complex-rate panels of the discretization figure (SSM-14).
- **`further-reading.md`** is a survey by design and a good one; it needs ranking (FR-1) and a clause per topic on what has lasted.

## Experiments

This is the chapter's main problem, so in full.

**The common benchmark separates nothing.** Stored PyTorch perplexities on character-level Penn Treebank, as collected in `matrix-state.md` S3: RNN 3.12, LSTM 2.96, GRU 3.01, minGRU 3.20, S4D 3.36, Mamba 3.00, gated linear attention 3.09 (3.20 without convolution, SiLU and gate), Gated DeltaNet 3.10, hybrid 3.16, transformer 3.27. The 1997 model is first; the plain RNN of Basics is level with both matrix-memory layers; the transformer is next to last. `lstm.md` says the task "serves mainly to check that a new model learns" (LSTM-17), and then five sections train on it and four of them end by explaining why their mechanism does not show (LSTM-17, SSM-21, MAMBA-18, MAMBA-20, MS-14, DN S1, HY S4). Model sizes range from about 286,000 to 542,000 parameters and are printed in one section only. Proposal: one table at the end of the chapter with perplexity and parameter count for every model, introduced by one sentence that says what it is (a check that each layer learns text, not a ranking), and no perplexity-based conclusions in the sections. The Penn Treebank runs in `deltanet.md` and `matrix-state.md` can then go or shrink to a line.

**Experiments read for more than they show.** The recall table in `lstm.md` (LSTM-11). Stability as an advantage of the continuous-time view, and the recall controls in `ssm.md` that show a gate bias does as well (SSM-12, SSM-19). Selective copying, where the minGRU scores 0.97 (MAMBA-5, MAMBA-15). The interference cell (MS-3, MS-4). The recall sweep as evidence for the write rule and for "slots" (HY-10, HY-11, HY-4). The transformer's perplexity as evidence that "perplexity alone can miss what a model cannot recall" (HY S4).

**Checks that cannot fail, or only confirm.** `deltanet-the-delta-rule-1` prints `1.00 1.00 1.00 1.00` for orthonormal keys at $\beta = 1$, which is the identity derived twenty lines above (DN-2). The reset-state check in `ssm.md` cannot answer no as posed (SSM-11). The interference cell confirms a proposition, which is allowed once, and adds a recall table. The seven or so "max deviation" cells across the chapter (`lstm-implementation-2`, `ssm-implementation-2`, `ssm-the-s4d-layer-4`, `mamba-training-with-a-scan-2`, `matrix-state-unrolling-with-decay-3`, `matrix-state-chunked-computation-2`, `deltanet-chunked-computation-3`) test code, can fail, and are legitimate (SSM-9, MAMBA-11); there are too many, two can merge in `matrix-state.md`, and each is followed by the same sentence about rounding error.

**Unfair or confounded comparisons.** Unequal parameter counts throughout (LSTM-15, SSM-5, MAMBA-17). Gated linear attention against Gated DeltaNet differing in four things (HY-11, DN-13). Three parts ablated at once (MS-14). `TinyLM` against the hybrid, which differ in convolution, position encoding and more (HY S4, HY-7). The learning rate changed from 0.003 to 0.01 between `lstm.md` and `ssm.md` without a reason. Timings against an interpreted Python loop presented as the gain of the algorithm (SSM-10, and the timing cell in `matrix-state.md`).

**Experiments the argument needs and the chapter lacks.**

1. A model trained on rebinding. The delta rule is motivated by a key written twice; no network in the chapter ever learns to overwrite (DN S1). This is the largest gap between what the chapter argues and what it shows.
2. A setting that separates Mamba from the minGRU, or a plain statement that this scale does not (MAMBA structural finding 2).
3. A transformer in the comparisons before the last section, on equal terms, and one table of work, sequential depth and generation memory against attention (SSM structural finding 4, MAMBA structural finding 3, IDX-3).
4. A trained model using a matrix memory in the section that introduces it (MS S2).
5. The recall curve at a larger state, which would test the capacity account directly (HY S1, HY-4).
6. Recall measured in text, which is exercise 6 of `hybrids.md` (HY S4).

These should be planned together. The three reviewers of the last three sections each proposed a new experiment for their section (1, 4 and 5); one recall harness with a rebinding option and a width option, introduced in `matrix-state.md` and reused twice, would serve all three and replace three Penn Treebank runs.

**Scale.** The recall sweep (four models, 16,000 steps at batch 256), the selective copying runs on sequences of 1,028 and the 30 models of the delayed recall table are GPU experiments. None says how long it takes. Not reproducible between captures: the JAX cells seed with `random.getrandbits(32)`.

## Arc and order

The chapter has one line of argument, and it is its great strength: gates as learned decay; remove the state from the gates and the recurrence becomes linear and parallel; choose the decays by a rate and a step size; let the step size depend on the input; see that the state is then a matrix of key and value pairs; correct a pair instead of adding to it; count what a fixed state cannot hold and add attention. Each section's last paragraph poses the next section's question. The order should not change.

- **`lstm.md` is two things**: a lesson on gates and the chapter's test bench (`SeqLM`, `RecallModel`, `train_queries`, `fit_lm`, `PennTreebank`, `init_lecun`). Seven cells and about 290 source lines lie between the recall task and its result (structural finding 1, LSTM-9). Give the bench one marked subsection at the end.
- **`ssm.md` is two sections**, split at "## State Space Models" (structural finding 1 there). The first half is the chapter's best teaching. The second half has lost its reason: after the control experiment the reader does not know what the differential equation is for (SSM-12, SSM-19). A split forces the second part to state its own question.
- **The step from S4D to Mamba is the chapter's hardest and gets one paragraph** (MAMBA-6). The identity that makes it easy, that a step size through a softplus is a GRU gate, is in exercises in both `ssm.md` (exercise 7) and `mamba.md` (exercise 1). It belongs in the main text (MAMBA structural finding 1).
- **The claim that Mamba's state is a matrix memory** is the hinge between the two halves of the chapter and is given in words in both places (MAMBA-19, MS-8). It needs its equation, once, in `matrix-state.md`, and a heading.
- **`matrix-state.md` has four jobs**; its second half shrinks if the layer takes a scalar decay per head (S1). I support this for clarity and for the comparison in `hybrids.md`, with one caution the section reviewer could not see: `further-reading.md` notes that Kimi Linear, one of the two deployed delta-rule hybrids, uses a decay per key coordinate (FR-8). So the scalar decay is the teaching choice, not the current one, and the text should say so.
- **The chapter has no ending.** The recap in the last body paragraph of `hybrids.md` (HY-14) is the only summary. The regression table of `deltanet.md` extended by rows for the scalar and diagonal states and by a column for cost at inference, with the consolidated perplexity table next to it, would close the chapter (the reviewers of `deltanet.md` and `hybrids.md` both suggest this).
- **Length against importance.** `ssm.md` (6,000 words) and `lstm.md` (5,000) are the longest; `mamba.md` (3,600), whose subject is hardest, is among the shortest and can take the 300 words its review asks for.

## Most important findings

1. **Make the first experiment and its prose agree.** State what the delayed recall table shows, including that a plain network with a well-scaled recurrent matrix carries a symbol 50 to 100 steps, and restate what gates buy. LSTM-11, LSTM-1, LSTM-19, LSTM-10, LSTM-12, SSM-20.
2. **Say once, in the chapter's voice, that at this scale the cheaper idea does nearly as well**, and stop drawing conclusions from Penn Treebank. One table with parameter counts at the end of the chapter. LSTM-15, LSTM-17, SSM-5, SSM-21, MAMBA-17, MAMBA-18, MS-14, and structural findings S3 of `matrix-state.md`, S1 of `deltanet.md`, S4 of `hybrids.md`.
3. **A result is in a cell, in an exercise, or not in the book.** Remove or print the dozen unseen runs of `hybrids.md` and the overflow report. HY-9, HY-10, HY-12, HY-6, MS-11, MAMBA-9, LSTM-16.
4. **Give the cost against attention, with a transformer in the comparison.** One table of work, depth and memory at generation, placed at the end of the scan section, and a pointer from `mamba.md`. IDX-3, MAMBA-1, and structural findings 4 of `ssm.md` and 3 of `mamba.md`.
5. **Train a model on rebinding, and show what the delta rule costs.** Replace the cell that cannot fail. DN-2, DN-9, DN-4, structural findings S1 and S2 of `deltanet.md`.
6. **Say what the continuous-time view is for, and stop claiming stability as its gift.** Split `ssm.md` at the seam. SSM-12, SSM-19, SSM-1, SSM-13, SSM-17.
7. **Put the gate identity and the shapes into the main text of `mamba.md`,** and say what the experiments do and do not show about its second ingredient. MAMBA-6, MAMBA-15, MAMBA-5, MAMBA-8.
8. **Correct the reading of the interference experiment and tell the capacity story with one quantity,** the number of entries of the state, in `matrix-state.md` and `hybrids.md`. Drop "slots". MS-3, MS-4, MS-2, HY-3, HY-4, HY-10.
9. **Simplify the layer of `matrix-state.md` so that the two memory layers differ in the write rule alone,** and say where each part of the layer comes from. MS-12, MS-13, DN-13, HY-11, with the caution of FR-8.
10. **Bring the chapter up to its publication date and give it a route and an ending.** MAMBA-1, HY-13, HY-14, IDX-1, IDX-2, FR-1.

## Repetition

**Inside the chapter.**

- The speed problem (recurrences train one step at a time): four times in `lstm.md` (structural finding 4, LSTM-18), again at length at the head of `ssm.md`. Home: the opening of `ssm.md`; one sentence in the Summary of `lstm.md`.
- The motivation for the state space half: three times in a row in `ssm.md` (structural finding 3). Stability: five times (SSM-12). Home: once each, under "Discretization".
- "A step size is a gate": in words three times in `mamba.md`, proved in exercise 1 there and in exercise 7 of `ssm.md`. Home: the main text of `mamba.md`; one exercise keeps the variant.
- The state as a key and value memory: the roadmap and the close of `mamba.md` (MAMBA-2, MAMBA-19), then `matrix-state.md` (MS-8). Home: `matrix-state.md`, with a one-sentence hand-off in `mamba.md`.
- Capacity, in three currencies: $(n-1)/d_k$ in `matrix-state.md`, bits and "slots" in `hybrids.md`. Home: `matrix-state.md`, as $d_k d_v$; `hybrids.md` keeps the counting bound and the Welch bound as the precise form of "training cannot remove this".
- The GRU equations, reprinted in `ssm.md` (SSM-4): small, serves the argument, may stay. The half-life numbers: small, may stay.
- The bound $1/(1 - f_\textrm{max})$ in exercises of `lstm.md` and `ssm.md`; the ablation exercise in `matrix-state.md` (9) and `deltanet.md` (9); the 64 MB figure twice in `matrix-state.md` (MS-10). Keep one of each.
- "on this task X matters little", after every Penn Treebank run. Home: one table.
- Network-valued memories: `deltanet.md` (DN-12) and `further-reading.md`. Home: `further-reading.md`.

**With other Advanced chapters.** `sec_attention-at-scale` in chapter 10 already teaches linear attention as a recurrence with a matrix state, the fast-weight reading, interference as a lossy summary, and the chunked form; `matrix-state.md` teaches the recurrence, the unrolling and the attribution again with the state transposed (I confirmed the sentence "the transpose of the layout in" at `matrix-state.md` line 74 and the associative-memory passage at `attention-at-scale.md` line 837) and $d$ renamed $d_k$ (S4, MS-1). Home: chapter 10 owns the unrolling and the cost argument; this chapter owns the example, the capacity law, decay, the mask and the layers. One layout in both. The KV cache arithmetic in `hybrids.md` builds on `sec_kv-cache` without repeating it. The scan's memory accounting touches `sec_memory_precision` in chapter 13, which `further-reading.md` cites and `mamba.md` could.

**With Basics.** "Vanishing Gradients" in `lstm.md` re-derives the Jacobian product that `chapter_recurrent-neural-networks/bptt.md` owns with an experiment at three spectral radii (structural finding 5). The scalar reading is new and short; open by citing the Basics result and cut a third. The Basics experiment at three spectral radii is also the natural place to anchor the explanation of the RNN row (LSTM-11).

**With the appendix.** One overlap, and it is the allowed kind. `subsec_mdl-ode-driven-linear` in the dynamics chapter derives the zero-order hold for a general matrix (`eq_mdl-ode-zoh`) and points to `subsec_zoh` here; I confirmed both labels exist. `ssm.md` states the scalar result without derivation. By the author's rule the two-line scalar derivation should be in the main text (SSM-13) and the matrix case cited once in the appendix.

## Recurring patterns

**Honest readings that stop short.** The pattern the reviewer of `mamba.md` named: the prose reports the number candidly and then does not draw the conclusion. The plain RNN beats the default LSTM ("erratic"). A bias initialization matches S4D ("the initialization decides", and then the differential equation is defended in one clause). The minGRU nearly matches Mamba ("nearly solves the task"). The memory "does not help much on this task". The delta rule makes no difference on text. The transformer is worst at language and the text does not ask why. Six sections, six instances. The candour is the chapter's best quality of voice; it needs to be carried one sentence further each time.

**Runs no cell performs.** About fifteen results across `matrix-state.md` and `hybrids.md`, plus the parameter counts and memory figures of `mamba.md`. Several are the authors ruling out their own bugs ("so the plateau belongs to the model and not to our code").

**Comparisons at unequal size or with several things changed.** Listed under Experiments.

**The equivalence-check formula.** Seven "max deviation" or "relative deviation" cells, each followed by a sentence of the form "The two agree to the rounding error" (SSM-9 counts four in the first three sections; the last three add three).

**Two endings.** A closing paragraph before the Summary in `lstm.md` (LSTM-18) and `hybrids.md` (HY-14). Less frequent than in the Basics chapters.

**The route template.** Six "asks" sentences in the introduction (IDX-2).

**Second imports cell.** In the JAX tab of every section (`import random`), and in both tabs of `lstm.md` (`import os`). House rule: one.

**Patched, unreflowed paragraphs.** `index.md` (IDX-4), `lstm.md` (LSTM-3, LSTM-11), `ssm.md` (SSM-23, three places), `mamba.md` (MAMBA-2). Double Discussions links in `ssm.md`, `mamba.md` and `further-reading.md` (SSM-22, FR-10).

**Coined terms.** October's are gone ("ladder", "rung", "crowding"). One new one carries an argument: "slots" (HY-3). "pays off", "pays an eighth" (HY-11).

**Notation drift.**

- $\mathbf{b}$: bias in `lstm.md`, the write $\mathbf{b}_t$ in `ssm.md` and `mamba.md`, input weight $\bar b$ and input matrix $\mathbf{B}$ in the state space half, the key direction $\mathbf{B}_t$ in `mamba.md` (SSM-3).
- Channels: $d$ in `ssm.md`, $H$ in `mamba.md` (MAMBA-6). Key width: $d$ in chapter 10, $d_k$ here, with the state transposed (MS-1).
- $\beta_t$ in $(0, 1)$ in the layer and up to 2 in the theory (DN-3).
- Against the appendix: state $h$ and input $x$ here, state $\mathbf{x}$ and step $h$ there, and the input index shifted by one (noted in the review of `ssm.md`). The appendix flags it; the citation here should too.
- Done well: the pair $(\mathbf{a}_t, \mathbf{b}_t)$ and `linear_scan` are used unchanged from `ssm.md` through `deltanet.md`, and the half-life is the chapter's one unit of memory.

## Dependencies

**From Basics.** The simple RNN, backpropagation through time and the character language model (`chap_rnn`); LeCun initialization (`sec_numerical_stability`); the cosine schedule (cited to the wrong section, HY-8). All restated before use. The reader of Basics can follow the chapter to the minGRU and the scan without help.

**From earlier Advanced chapters.** Attention and Nadaraya and Watson pooling (chapter 10), linear attention and chunking (`sec_attention-at-scale`, a real prerequisite of `matrix-state.md`, stated in the introduction), rotary embeddings and windows, the transformer block and the KV cache (chapter 11), `d2l.TinyLM` (cited to `sec_adam` in chapter 9). The chapter reads in book order. It does not read well before chapters 10 and 11, and should not have to.

**From the appendix.** Nothing. No section cites the appendix (I found no `sec_mdl-*`, `subsec_mdl-*` or `eq_mdl-*` reference in the chapter), and with the two-line derivation of SSM-13 added none needs it. One citation should be added, from `eq_zoh` to `subsec_mdl-ode-driven-linear`. The appendix's reference into this chapter (`subsec_zoh`, `sec_ssm`) is correct.

**Forward references to the Language Models part.** None in the chapter (no `chapter_lm-*` label is cited). In the other direction the skeleton chapters lean on this one: `chapter_lm-architectures/hybrids-at-scale.md` cites it six times and six other skeleton files cite it, so the labels `sec_hybrids`, `sec_deltanet`, `sec_matrix-state` and `sec_mamba` must survive a split of `ssm.md` or any renaming.

**Within the chapter.** Every later section depends on the harness defined in `lstm.md`; `mamba.md` on `linear_scan`, `S4D` and `MinGRU`; `deltanet.md` on `decay_mask` and the layer class of `matrix-state.md`. A reader cannot enter the chapter in the middle, which is acceptable if the introduction says so (it does not).

## Connection to the main text

- Inbound references are healthy: Basics (`bptt.md`, the RNN chapter introduction, `sequence.md`), chapters 10, 11 and 13, the part introduction and the preface all point here.
- Missing outbound: a cost table that uses the terms of chapters 10 and 11; a reference from `mamba.md` to chapter 13 for the kernel and the memory of the scan; the appendix citation above; the bptt experiment for the RNN row.
- Stale or wrong: HY-8 (`sec_batch_norm` for `cosine_warmup`); the Discussions links.
- Named without a label: the forget-gate bias of 1 and chrono initialization (in an exercise only); "induction head" (HY-6); "the reference implementation of the layer" (HY-9, MS-12).
- Stacking conventions: `SeqLM` is said to follow `sec_transformer-block`, correctly, but presented as how recurrent networks are trained (LSTM-13).

## What to cut

About 2,500 words, none of it argument:

- `lstm.md`: the initialization paragraph to two sentences (LSTM-9); the bidirectional paragraph (LSTM-14); the closing paragraph before the Summary (LSTM-18); a third of "Vanishing Gradients"; the 5 and 10 columns of the recall table.
- `ssm.md`: four of five stability claims; two of three motivations for the second half; the complex-rate panels of the figure.
- `mamba.md`: the roadmap's preview of the next section (MAMBA-2); the mixed worked number (MAMBA-9).
- `matrix-state.md`: the "Linear Attention" subsection to a clause (S4); the overflow paragraph (MS-11); the per-coordinate path, rank-16 gate and exercise 8 if S1 is adopted; one of the two equivalence cells; the timing run or its prose; the Penn Treebank ablation or its conclusion.
- `deltanet.md`: the cell that cannot fail (replaced, DN-2); the Penn Treebank run (replaced); the two citations on network memories (DN-12); exercise 9.
- `hybrids.md`: the slot construction (HY-3); all but one or two of the unseen results (S2); induction heads (HY-6); the recap paragraph (HY-14); the output-gate and SiLU ablations (HY-12).
- `index.md`: five of the fifteen resources (IDX-6).

## What is missing

- The explanation of the RNN row, and one sentence on where gated networks are used today.
- Work, sequential depth and memory at generation, for a recurrent layer against an attention layer, as a table.
- What the differential equation contributes that a gate bias does not.
- The gate identity and a table of shapes in `mamba.md`; what became of Mamba.
- The equation that shows Mamba's state as a matrix memory.
- A trained model that stores and looks up pairs, in the section that introduces the memory; a trained model that overwrites.
- What the delta rule costs when keys overlap.
- The 2025 hybrids in `hybrids.md`, and the matched comparison at scale (Waleffe et al., now only in `further-reading.md`) as the evidence the small experiments cannot give.
- A chapter summary: one table of the memories and their costs, one table of perplexities with parameter counts.
- A reading guide and a conclusion in the introduction; a ranking in the further reading.

## What is best and should set the model

- **`lstm.md`, "Gates".** Running sum, then "a running sum cannot forget", then the gated sum, with half-lives of 7, 69 and 693 steps as the unit the whole chapter then uses. And the objection answered honestly: the cell "provides one path along which gradients can survive".
- **`ssm.md`, the scalar recurrence and the composition rule.** "$1, 0.9, 0.81, 0.729$", then the pair $(0.25, 2.5)$ computed by hand, then the remark that "a block of a thousand steps costs no more to store or to combine than a single step". "Nothing new happens here."
- **`ssm.md`, the recall experiment with two controls,** and the willingness to print the result that deflates the section's subject.
- **`mamba.md`, the opening and "Contributions to the State".** "A memory that keeps everything for the same time fills up with commas; a memory that forgets quickly loses the name." And an experiment that measures where the state comes from and says what the plot does not determine.
- **`matrix-state.md`, the first third.** Two dimensions, two pairs, a third key at 45 degrees, 7.95; a four-line proof; a measurement on the predicted lines. This is the standard the rest of the chapter should meet.
- **`deltanet.md`, from the opening to the regression table.** `rate = 0.05` then `rate = 0.07`; "The fix is to read before writing."; the failure measured before the cure; "Aligning gives superposition; minimizing the error gives correction."; and the paragraph on learning at two levels.
- **`hybrids.md`, the opening sentence and the inference-memory subsection.** "before it knew which of them would be asked for"; one formula, one worked number, one picture.
- **The exercises.** All six sets were judged strong and graded. Several carry content the main text should have (the gate identity, the forget bias of 1, the width experiment, recall in text); the revision should promote those and keep the rest.

## Disagreements and tensions between the section reviews

No review contradicts another on a fact. Three tensions the reviser should know of:

- The reviewer of `mamba.md` proposes pointing to the `TinyLM` perplexity in `hybrids.md` (about 3.25 to 3.4) to place Mamba's 3.0. The reviewer of `hybrids.md` finds that number unexplained and the comparison confounded (S4). Do not use it as a baseline until it is a fair one.
- The reviewer of `deltanet.md` measured that with random overlapping keys the delta rule at $\beta = 1$ recalls fewer old pairs than plain addition (0.54 against 0.96 at 128 pairs in width 64). The stored recall sweep in `hybrids.md` has Gated DeltaNet above gated linear attention at every load. Both can be true (learned write strengths, different decays, an unfinished plateau), and the reviewer of `hybrids.md` says so, but the chapter's sentence "The delta rule pays off" cannot stand on the present evidence.
- Three reviewers each ask for a new trained experiment in their own section (`matrix-state.md` S2, `deltanet.md` S1, `hybrids.md` S1). Taken separately that adds compute to a chapter that is already heavy; taken together they are one harness used three times.

## Could not check

- No training run could be repeated (no PyTorch or JAX on this machine). Every statement about run-to-run variation rests on one stored run per framework.
- The proposed explanation of the RNN row (spectral radius of the LeCun normal matrix) was checked numerically for the matrix by the section reviewer, not by training.
- Claims about papers were judged from memory by all reviewers, including me; those marked unverified above need someone with the papers.
- The three resource links of IDX-7.
- Rendered HTML pages and the slides sections were not read.
