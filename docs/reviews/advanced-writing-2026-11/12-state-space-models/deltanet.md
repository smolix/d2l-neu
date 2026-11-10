# The Delta Rule

Source: `chapter_recurrent-modern/deltanet.md` (read to the `<!-- slides -->` marker at line 744). About 3,000 words of prose, 800 words of exercises. Stored outputs read: `outputs/pytorch/chapter_recurrent-modern/deltanet.json` and the JAX file. The NumPy overwrite experiment was re-run and extended to overlapping keys; the chunked algebra was re-derived by hand against the code. Figure `img/mdl-modernrnn-delta-geometry.svg` rendered and inspected.

## Verdict

This is the best of the three sections and the most focused section I have read in this chapter: one question (how can a memory replace what it stored?), one answer (read before you write), derived as a gradient step, pictured as a projection, and then placed in a table that orders every memory of the chapter. The derivation, the transition $\mathbf{I} - \beta_t\mathbf{k}_t\mathbf{k}_t^\top$, the chunked algebra and the implicit step are all correct (re-derived). The principle is as settled as anything in the book (Widrow and Hoff, 1960), and the section teaches it as a principle; Gated DeltaNet is one equation on top. What keeps it from the bar is its evidence. The delta rule's one demonstration is a cell that cannot fail (orthonormal keys, $\beta = 1$: the derivation three lines earlier already says the read returns the new value and nothing else is touched). No model in this chapter is ever trained on a task in which a key is rebound, so the reader never sees a network learn to use the rule on the problem that motivates it. And the section does not say what the rule costs: with overlapping keys each write at $\beta = 1$ erases part of every pair whose key overlaps the new one, so a one-pass delta rule is not least squares and can recall old pairs worse than plain addition (measured below). The most important fix: replace the Penn Treebank run, which shows nothing, by a small trained rebinding task, and replace the cannot-fail cell by one with overlapping keys that shows both what the rule repairs and what it gives up.

On the three questions. Settled: yes; the rule is sixty years old, its use as a write is from 2021, and by 2025 it is in released models. Claims about recent work: the attributions (Schlag et al., Yang et al. for the chunked algorithm, Yang, Kautz and Hatamizadeh for the gated form, Grazzi et al. on negative eigenvalues, Wang, Shi and Fox on regression) agree with what I know; the description of Longhorn is slightly generous (DN-11). Principle or survey: principle throughout. The two sentences on network-valued memories are the only survey, and they are short.

## Scores

| dimension | score | note |
|:--|:-:|:--|
| Clarity of explanation | 4 | The rule, the transition and the regression view are clear when they arrive. The second half of the chunked computation (the matrices $\mathbf{A}$ and $\mathbf{W}_\textrm{end}$) is given in words only. |
| Clarity of thought and focus | 4 | One line of argument from first sentence to last. Loses a point for calling a one-pass gradient method "least squares" and for a boundedness claim that is too strong. |
| Pedagogy | 3 | Motivation, a measured failure, a derivation, a picture: all good. Then a check that cannot fail, a training run that shows no effect, and no trained evidence for the section's claim. |
| Language | 4 | Clean and direct, with several sentences worth keeping. |
| Ease and enjoyment | 4 | Reads quickly up to "Chunked Computation" and again from "Memory as Regression". |

## What works

- **The opening** (lines 4 to 12): `rate = 0.05` then `rate = 0.07`, the hero moved from Paris to Rome, and the precise statement of failure, "a key written twice returns the sum of its two values". It continues the closing example of the previous section. Keep.
- **"The fix is to read before writing."** (line 14). The section in six words.
- **Measuring the failure first** (cell `deltanet-a-key-written-twice-1`): 1.00, 0.51, 0.27, 0.21, read as "The memory has room for all eight keys, and it still fails, because it cannot replace a value." The experiment isolates rebinding from interference by using orthonormal keys, and says so.
- **The derivation** (lines 118 to 158): loss, gradient, step, then "Read it as two steps", then the three cases ($\beta = 1$ replaces, orthogonal directions untouched, $0 < \beta < 1$ blends "like the update gate of a GRU"). Every step is there and the link to the GRU ties the section to :numref:`sec_lstm`.
- **"Aligning gives superposition; minimizing the error gives correction."** (line 164). The additive write as a gradient step on $\mathbf{v}_t^\top\mathbf{S}\mathbf{k}_t$ is a small, exact observation that most treatments omit.
- **The transition and its figure** (lines 179 to 202). "The decay of the previous section forgets in every direction at once. The delta rule forgets in one direction, the one it is about to write." The figure shows the four values of $\beta$ in one plane and needs no further words.
- **The regression table** (`tab_dn-regression`) and the two-level paragraph: "The weights learn which pairs are worth storing; the memory stores the pairs of the sequence at hand." This is the idea a reader keeps after the model names are forgotten.
- **Exercises 3, 4, 6 and 8** (below).

## Structural findings

**S1. The section has no trained evidence for its claim.** The delta rule is motivated by rebinding. The only rebinding experiment is NumPy with hand-set $\beta = 1$ and orthonormal keys. The trained experiment is Penn Treebank, where "Gated DeltaNet reaches a perplexity of about 3.1, the same as gated linear attention", and the text then points to "the recall experiments of the next section", in which every key is distinct and none is ever rebound. So nowhere in the chapter does a network learn to overwrite. This is the experiment to add: the recall task of :numref:`sec_hybrids` with a small key set in which half of the keys are written a second time with a new value, queried for the latest value, for gated linear attention against Gated DeltaNet. It is small (the models have width 32), it tests the section's title, and by the hand-set measurement the gap should be large. It replaces the Penn Treebank cell.

**S2. What the rule costs is missing.** The text says of the transition that for directions orthogonal to $\mathbf{k}_t$ "what is stored under other keys is untouched". Keys in a trained model are not orthogonal once there are more of them than dimensions. Then a write at $\beta = 1$ removes the component along $\mathbf{k}_t$ of every stored pair, so each write damages every pair whose key overlaps the new one, and the oldest pairs are damaged most. I measured this with the section's own setup (width 64, random unit keys, each written once, codebook of 256, 20 trials):

| pairs $n$ | squared read error: additive / delta $\beta{=}1$ / least squares | recall: additive / delta / least squares |
|:-:|:-:|:-:|
| 32 | 0.48 / 0.29 / 0.00 | 1.00 / 1.00 / 1.00 |
| 64 | 0.99 / 0.61 / 0.00 | 1.00 / 0.89 / 1.00 |
| 128 | 1.98 / 1.06 / 0.50 | 0.96 / 0.54 / 1.00 |
| 256 | 3.98 / 1.49 / 0.75 | 0.73 / 0.27 / 0.85 |

The delta rule halves the squared error and still recalls far fewer pairs than addition once $n \ge d_k$: it lowers the error by shrinking old reads toward zero, which removes signal along with noise. With $\beta = 0.5$ the loss is smaller (recall 0.81 at $n = 128$) but still below addition. With correlated keys ($\rho = 0.5$) the order reverses and the delta rule wins clearly (recall 0.77 against 0.50 at $n = 64$). Least squares beats both everywhere. This is a better lesson than the present one: correction helps where keys are rebound or share a common component, costs old pairs where keys merely overlap at random, and is one gradient step, not a fit. It is also why the write strength is learned and why the decay and the write strength are separate gates. Exercise 5 points at part of this; the main text should say it, and the table of the regression view should not call the delta rule "least squares" without qualification (DN-9).

**S3. The chunked computation is the right length but stops explaining halfway.** The first half (the corrected values $\mathbf{u}_t$, the triangular system) is derived in the text and is elegant: "If we knew the $\mathbf{u}_t$, the memory would be the decayed additive memory of the previous section." The second half (lines 313 to 324) introduces $\mathbf{A}$ as "the decay of the chunk minus what its corrections remove" and $\mathbf{W}_\textrm{end}$ as what "collects the chunk's own writes", with no formula; the reader must reverse-engineer them from `(Z * to_end).mT @ K`. Two display lines close the gap (DN-7). Not a candidate for cutting: this is the one place in the chapter where a non-diagonal transition is trained in parallel, and the idea transfers.

**S4. No split.** At 3,000 words the section is the right size. If something must go, it is the Penn Treebank run (S1) and the two citations to network-valued memories, which the Further Reading notebook already covers ("Memories That Are Networks").

## Passage-level findings

**DN-1** (minor, explanation). Under "A Key Written Twice". Quote: "the latest value wins only about half of the time, which is what a sum of two". Good. The next clause, "after eight, it wins about one time in five", invites the question why not one in eight. The answer (the latest value was also drawn earlier in about one trial in five, since the codebook has only 32 entries) is exercise 1. A clause would do: "(more than one in eight, because a codebook of 32 values repeats)".

**DN-2** (major, pedagogy). Under "The Delta Rule". Quote: "The delta rule recalls the latest value every time, however often the key". Cell `deltanet-the-delta-rule-1` prints `1.00 1.00 1.00 1.00`. With orthonormal keys and $\beta = 1$ this is the identity $\mathbf{S}_t\mathbf{k}_t = \mathbf{v}_t$ derived twenty lines above, together with "what is stored under other keys is untouched": the cell cannot print anything else. It is a check that cannot fail, of the kind the author has asked to remove. Replace it by the measurement of S2: random unit keys, more keys than the eight orthonormal ones (say 32 and 96 in width 64), additive against delta at $\beta = 1$ and $\beta = 0.5$. The rule then still wins on rebinding (in my run, 32 random keys rewritten 8 times: additive 0.18, delta 0.97) and the reader also sees the cost at 96 keys. A cell whose result the reader cannot predict from the last paragraph teaches; this one does not.

**DN-3** (minor, thought). Under "The Delta Rule". Quote: "The network computes it from the input with a sigmoid, and". The layer's choice of $\beta_t \in (0, 1)$ is stated here in passing, and forty lines later the text explains that $\beta_t$ up to 2 is what makes parity computable. The reader is left to wonder why the book's layer then uses a sigmoid. One sentence where the layer is built: "We keep $\beta_t$ in $(0,1)$, as the original layer does; exercise 7 doubles the range."

**DN-4** (moderate, correctness; certain). Under "The Transition". Quote: "no direction is ever amplified, so the memory cannot". The premise is right (for unit keys and $0 \le \beta_t \le 2$ the transition has norm at most one) and the conclusion is too strong. The write term $\beta_t\mathbf{v}_t\mathbf{k}_t^\top$ still adds: $\|\mathbf{S}_t\|_F^2$ can grow by up to $\beta_t^2\|\mathbf{v}_t\|^2$ per step when each new key is nearly orthogonal to the rows of the memory, and for $\beta = 1$ with nearly parallel keys and opposed values the memory approaches a point at distance of order $1/\theta$, $\theta$ the angle between the keys. What holds is that the transition never amplifies what is stored, so growth is at most additive in the writes, as in any memory of this chapter, and never geometric. Rewrite: "no direction is ever amplified: whatever the memory holds can only shrink or keep its size before the new write is added. Without unit keys the factor $1 - \beta_t\|\mathbf{k}_t\|^2$ can fall below $-1$, and the memory then grows geometrically." The matching sentence under "The Layer" ("and the memory can grow without bound") is correct as it stands.

**DN-5** (minor, correctness; confirmed numerically). Under "The Transition". Quote: "computes the parity of a sequence of any". True, with one detail the reader needs in order to check it: with a constant value $v$ and $\beta_t = 2$ on the marked token, the read alternates between $2v$ and $0$ (I ran it: 2, 0, 0, 2, 2, 2, 0, 2, 0 for the bits 1 1 0 1 0 0 1 1 1). As written, a reader who sets $\mathbf{v} = \mathbf{0}$ gets a memory that stays at zero. Add "with the same value written each time, the read alternates between $2\mathbf{v}$ and $\mathbf{0}$".

**DN-6** (minor, correctness; unverified beyond memory). Under "The Transition". Quote: "cannot compute parity at arbitrary lengths. This includes the decays of the". This matches my recollection of Grazzi et al. (2025): the result is for finite precision and for transitions whose eigenvalues are nonnegative, and their fix for the delta rule is the range $(0, 2)$ for $\beta_t$. "removes the limitation in practice" is fair for parity; the paper is careful that harder state-tracking tasks need more (the Further Reading notebook says so). Add "in finite precision" to the first sentence.

**DN-7** (moderate, explanation). Under "Chunked Computation". Quote: "matrix $\mathbf{A}$ is the decay of the chunk minus what its". Give the two lines the code implements. With $\gamma_m$ the decay of the whole chunk and $\mathbf{D}$ the diagonal matrix of decays from each step to the chunk's end: $\mathbf{S}_m = \gamma_m\mathbf{S}_0 + \mathbf{U}^\top\mathbf{D}\mathbf{K}$, and substituting $\mathbf{U} = \mathbf{W} - \mathbf{Z}\mathbf{S}_0^\top$ gives $\mathbf{S}_m = \mathbf{S}_0\,(\gamma_m\mathbf{I} - \mathbf{Z}^\top\mathbf{D}\mathbf{K}) + \mathbf{W}^\top\mathbf{D}\mathbf{K}$. (Checked against `chunked_delta_rule`: `A` and the second element passed to the scan are these two matrices.) Then the names $\mathbf{A}$ and $\mathbf{W}_\textrm{end}$ mean something.

**DN-8** (minor, thought). Under "Chunked Computation". Quote: "and a scan would compose". The text rejects a scan over tokens because it "would compose dense matrices" and fifty lines later scans over chunks with `A1 @ A2`, dense $d_k \times d_k$ products. Both are right, and the reader deserves the reason: $T/m$ products instead of $T$, each of cost $d_k^3$, against $m\,d_k^2$ per chunk for the rest of the work, so the scan over chunks is affordable when $m$ is of the order of $d_k$. Also "One triangular solve per chunk replaces $m$ sequential steps" slightly oversells: forward substitution is itself $m$ dependent steps. What is gained is that the steps are rows of one batched solve over all chunks and heads, not $m$ updates of a $d_v \times d_k$ state. One sentence each.

**DN-9** (moderate, correctness; certain). Under "Memory as Regression". Quote: "| Delta rule | least squares | one gradient step per pair |". And in the text: "When $\mathbf{K}_t^\top\mathbf{K}_t$ is invertible". Two imprecisions. (a) The closed form needs at least $d_k$ keys that span the space; the next sentences then discuss orthonormal keys, for which $\mathbf{K}_t^\top\mathbf{K}_t$ is a projection and is invertible only when there are exactly $d_k$ of them. Say "with the pseudoinverse when the keys do not span $\mathbb{R}^{d_k}$", or state the result for the minimum-norm solution. (b) One gradient step per pair does not fit least squares; it moves toward it, and S2 shows by how much it can miss. The table's "What it fits" column should read "least squares, approximately: one pass of gradient steps" for the delta rule, and the sentence "The delta rule is a gradient step on the same objective, one pair at a time" should add what follows from that: later pairs are fitted better than earlier ones.

**DN-10** (minor, pedagogy). Under "Memory as Regression". Quote: "In all of these models, learning happens at two levels.". This paragraph is one of the most useful in the chapter and arrives after the table, the implicit step and two citations. A reader meeting "one step of gradient descent" under "Writes as Gradient Steps" will already ask what is being trained and when. Move the paragraph there, right after the delta rule is named, or give a one-sentence version there and keep the full one here.

**DN-11** (minor, correctness; unverified, from memory of the paper). Under "Memory as Regression". Quote: "network sets the write strength to; this is the update of Longhorn". The implicit-step solution is correct (re-derived: step $\beta_t/(1+\beta_t\|\mathbf{k}_t\|^2)$, eigenvalue $1/(1+\beta_t\|\mathbf{k}_t\|^2)$). As I recall, Longhorn derives this closed form and then implements a diagonal approximation of it, so that it fits a Mamba-style elementwise recurrence. "this is the idea behind Longhorn" is safe; "the update of Longhorn" may not be literally true.

**DN-12** (minor, focus). Under "Memory as Regression". Quote: "Other designs replace the matrix by a small". One sentence with two citations on memories that are networks. The Further Reading notebook has a paragraph on exactly this. Cut it here or reduce it to a pointer.

**DN-13** (minor, thought). Under "The Layer". Quote: "because the chunked delta rule above". The reason given for a scalar decay is that the book's function takes one. That is a reason about the code, and it leaves the decay of this layer different in kind from that of the layer it is compared with. See S1 of the review of `matrix-state.md`: with a scalar decay there, this sentence becomes "Its decay is the same".

## Code and experiments

- `deltanet-a-key-written-twice-1`: keeps. Stored `1.00 0.51 0.27 0.21` in both frameworks (seeded NumPy), matching "about half" and "about one time in five".
- `deltanet-the-delta-rule-1`: cannot fail (DN-2). Replace.
- `deltanet-chunked-computation-3`: an implementation test of a nontrivial algorithm against the loop; it can fail and should stay. One line of output would do; "agree to rounding error" is the right reading. There is no timing of the chunked delta rule against the loop, and the reader will want one number after being told that chunking is the point (the previous section timed its three forms).
- `deltanet-language-modeling-2`: stored 3.10 (PyTorch), 3.09 (JAX), against 3.09 and 3.10 for gated linear attention. The prose reads this correctly as no difference and then explains why none was to be expected. An experiment whose null result is predicted in advance is not worth a training run in the text; see S1.
- Numbers no cell prints: none in this section. That is to its credit.
- The JAX tab has a second imports cell (`import random`).
- Figure `fig_dn-geometry`: clear, in house style, and carries the explanation of the transition. No change.

## Exercises

Nine, graded from a two-line argument (1) to a training experiment (7). This is a good set.

- Keep 1 (order does not matter; it explains the 0.21 of DN-1), 2, 3 (the link to the step-size limit $2/L$ of gradient descent is the right connection to chapter 9), 4, 6, 7, 8.
- Exercise 6 (recursive least squares) is the best: "Why does recursive least squares, the best fit of all pairs, recall the latest value poorly?" makes the reader see that the best fit and the latest value are different goals. It answers a question the text should at least raise (S2).
- Exercise 5 asks for the measurement of S2 with correlated keys only. If S2 enters the text, change it to the multi-pass question alone (part 2), which is a good one: several passes of the delta rule are the Kaczmarz iteration.
- Exercise 9 repeats the ablation exercise of the previous section for this layer. Cut one of the two.
- Add: the rebinding task of S1 with a decay-only model, to show that decay cannot do the job (the trained version of exercise 2).

## Still open from October

October listed "RWKV-7, DeltaProduct, and the formal-language survey in `deltanet.md`" as off-focus and wanted the test-time-regression section absorbed. Both done: the surveys are in the Further Reading notebook and the regression view is one subsection here. Nothing from October remains open in this file.

## For the chapter review

- Order is right: it answers the limitation on which :numref:`sec_matrix-state` ends, and its regression table is the natural summary of the chapter's memories. Consider making that table, extended by one row for the scalar and diagonal states of :numref:`sec_ssm` and :numref:`sec_mamba`, the chapter's summary.
- It depends on :numref:`sec_matrix-state` for `decay_mask`, the layer class and the dual form, on :numref:`sec_ssm` for `associative_scan`, and on chapter 10 for Nadaraya–Watson (one clause, `sec_attention-pooling`; the reference is correct). Nothing from the appendix; the least-squares formula is stated in place.
- The chapter never trains a model on rebinding (S1). This is the largest gap between what the chapter argues and what it shows.
- The comparison between this layer and the previous one is confounded by the form of the decay (DN-13), which matters for the conclusion drawn in :numref:`sec_hybrids`.
- The same Penn Treebank finding as elsewhere in the chapter: no effect, explained afterwards.
