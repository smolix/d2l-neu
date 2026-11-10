# Further Reading

Source: `chapter_recurrent-modern/further-reading.md` (lines 1 to 231; the file has no slides section). About 1,500 words, four groups, fourteen topics, about forty citations. No code, no exercises.

## Verdict

As a map of what the chapter left out, this is well made. Each topic opens with the problem it addresses, relates the work to a section of the chapter by label, and says in one or two sentences what changes. The sentences are plain and mostly careful; several are better than the corresponding sentence in the teaching sections (the account of S4D's two initializations is more accurate here than in `ssm.md`, and the paragraph on released hybrids has the 2025 models that `hybrids.md` lacks). It is the right home for the surveys that the October review wanted out of the sections. Two things keep it from being as useful as it could be. First, nothing is ranked: fourteen topics at equal weight, from the theory behind an initialization to genomics, with no word on which a reader should follow first or which results have lasted. Second, a handful of claims about papers are stated more firmly than the papers support, or compress a conditional result into an unconditional one. I could not check any of them against the papers here; each is marked below with what I believe.

## Scores

| Dimension | Score |
|---|---|
| Clarity of explanation | 4 |
| Clarity of thought and focus | 3 |
| Pedagogy | 3 |
| Language | 4 |
| Ease and enjoyment | 3 |

## What works

- The framing sentence: "Each topic states the problem it addresses and what it changes relative to the chapter." The section then does exactly that, fourteen times. "Nothing else in the book depends on this section" tells the reader they may skip it.
- "Memory by Polynomial Projection" is a model of how to summarize a line of papers: one question ("what a fixed-size state should remember about an input ... that keeps growing"), one answer, then what each later paper changed. A reader of `ssm.md` who wondered where $-(n+1)$ came from is answered in 200 words.
- "Such kernels have no recurrent form with a small state, so generation either recomputes the convolution or approximates the kernel by a recurrence afterwards." A consequence stated, not just a method described.
- "whether this recalls more per byte of state than a matrix of the same size is a current research question." Candour about what is not known.
- "Between a Fixed State and a Full Cache" names the axis (how much is stored) and places two designs on it. This is the chapter's own question continued.

## Structural findings

1. **Rank the entries.** A reader finishing the chapter has perhaps an evening for further reading. Tell them where to spend it. A first paragraph of three sentences would do: the two topics that bear most directly on the chapter's argument (released hybrids; what fixed-state models can compute), the one that completes a derivation the chapter cut short (polynomial projection), and the remark that the applications at the end are pointers for readers in those fields. Within groups, order by importance, not by date.
2. **Say what has lasted and what has not.** The section is written in an even present tense, so that S4's low-rank correction, H3, Hyena, Zamba and Qwen3-Next all appear equally current. For a book of 2026 to 2027 a clause per topic suffices: S4's structured matrix was superseded by the diagonal form within a year; long convolutions without a recurrent form did not become language models; the delta-rule hybrids are what was being released when the book went to press.
3. **Move two items toward the sections.** (a) The sentence on Qwen3-Next and Kimi Linear is the connection between the layer the reader built in `deltanet.md` and deployed models; `hybrids.md` names only Jamba (HY-13 in that review). One sentence there should carry it, with the details kept here. (b) The Mamba-2 sentence under "Kernels for Selective Scans" is the fact that `mamba.md` lacks about what became of the layer it teaches (structural finding 4 of that review).
4. **Overlap with the sections is small and mostly right.** `deltanet.md` has two sentences on memories that are networks which duplicate "Memories That Are Networks" here (DN-12); cut them there. The recomputation of states in the Mamba kernel is told in `mamba.md` and again here; here it is one clause, which is fine.
5. **"Beyond Language" is the weakest group.** Three subsections of four to six lines each, each naming two models. They are accurate as far as I know and teach little. Either merge them into one paragraph ("the same layers have been applied where sequences are long and attention is too costly: ...") or give each the one idea that is specific to the domain, as "Genomics" does with reverse-complement equivariance.

## Passage-level findings

**FR-1** (minor, pedagogy). Heading: introduction. Quote: "The sections of this chapter teach one representative of each idea. This". The introduction lists five kinds of topic and gives no order of importance. See structural finding 1.

**FR-2** (moderate, correctness, unverified). Heading: Memory by Polynomial Projection. Quote: "Both work nearly as well as the full matrix.". As I recall :citet:`Gu.Gupta.Goel.Re.2022`, the diagonal variants with complex eigenvalues come close to S4 on the long-range benchmarks, while restricting to real rates costs accuracy on some tasks. "Both" then claims more for the real rates, the choice of `ssm.md`, than the paper shows. The reviewer of `ssm.md` raised the same doubt independently (SSM-15). Safer: "The complex variant works nearly as well as the full matrix; the real one is simpler and, in that paper's ablations, somewhat weaker." Someone with the paper should confirm, since `ssm.md` builds on the real rates.

**FR-3** (minor, correctness, unverified). Heading: Recurrences Without a Differential Equation. Quote: "so that the state does not grow. It matches S4 on long". My recollection of :citet:`Orvieto.Smith.Gu.ea.2023` is that the linear recurrent unit matches the deep state space models on the Long Range Arena. "long sequence benchmarks" is vaguer than it need be; name the benchmark and say "comes close to or matches".

**FR-4** (minor, correctness, fairly sure). Heading: Recurrences Without a Differential Equation. Quote: "parallel scan rather than a convolution, which is the route the selective". S5's two changes are one multi-input, multi-output SSM per layer and the parallel scan. Mamba takes the second and not the first: `mamba.md` gives each channel its own $N$ states. As written, "which is the route" can be read as covering both. Write "computed with a parallel scan rather than a convolution; the selective models of :numref:`sec_mamba` also use the scan, but keep one SSM per channel".

**FR-5** (minor, correctness, unverified). Heading: Long Convolutions. Quote: "token with earlier ones and recall what followed them, and H3 closed much of". "closed much of the gap to attention on language" is a loose claim about a paper. As I recall, H3 came within a fraction of a perplexity point of a transformer on OpenWebText, and its best result used two attention layers, that is, a hybrid. If so, the second fact is the one that matters for this chapter and should be said.

**FR-6** (moderate, correctness, fairly sure). Heading: Expressivity of Fixed-State Models. Quote: "in a class of parallel computations that cannot track the state of a general". Two compressions. (a) The result of :citet:`Merrill.Petty.Sabharwal.2024` places these models in the circuit class $\mathsf{TC}^0$; that such circuits cannot compose permutations of five elements is a standard conjecture ($\mathsf{TC}^0 \neq \mathsf{NC}^1$), not a theorem, so "cannot" needs "under a standard conjecture of complexity theory". (b) The next sentence, "Nonlinear recurrent networks and recurrences with non-diagonal transitions can", is too broad: as I recall the paper, a non-diagonal transition that is the same at every step (as in S4) is also in the class; what escapes it is a transition that is non-diagonal and depends on the input. Write "recurrences whose transition matrices are non-diagonal and depend on the input can". This matters because the delta rule of `deltanet.md` is exactly such a transition, and the reader should see why it is on the right side of the line.

**FR-7** (minor, correctness, fairly sure). Heading: Hybrid Models in Practice. Quote: ":cite:`De.Smith.Fernando.ea.2024` alternates a gated diagonal recurrence". As I recall Griffin, the pattern is two recurrent blocks followed by one block of local attention, not strict alternation. Write "interleaves ... two to one". Unverified against the paper.

**FR-8** (minor, correctness, unverified). Heading: Hybrid Models in Practice. Quote: "layers, three to one, and Kimi Linear :cite:`Kimi.Team.2025b` does the same". The ratios for Qwen3-Next and Kimi Linear agree with my knowledge, as does the description of Kimi's decay as acting on each key coordinate. Note for the chapter review: this means the decay per key coordinate that `matrix-state.md` builds into its layer, and that its reviewer proposes to replace by a scalar per head, is the form one of the two deployed delta-rule models uses. The Jamba ratio here (one in eight layers) agrees with `hybrids.md` (4 of 32).

**FR-9** (minor, thought). Heading: Memories That Are Networks. Quote: "this recalls more per byte of state than a matrix of the same size is a". Good. Add what the chapter's own theory says about the question: by the capacity argument of `matrix-state.md` a linear memory holds a number of pairs proportional to its number of entries, so a network memory must beat that count to be worth its cost. One clause connects the open question to something the reader has proved.

**FR-10** (minor, language). Heading: end of file. Quote: "[Discussions](https://d2l.discourse.group/)". Two Discussions links, the first to the forum root, as in `ssm.md` and `mamba.md` (SSM-22). Keep the second.

## Code and experiments

None. That is right for this section.

## Exercises

None, and none needed.

## Still open from October

October asked that the HiPPO, NPLR and DPLR material, RWKV-7, DeltaProduct, the formal-language survey and the catalogue of shipped hybrids leave the teaching sections. They are all here now, each in a paragraph. Resolved.

## For the chapter review

- This section holds the chapter's currency: Mamba-2, Mamba-3, Qwen3-Next, Kimi Linear, the matched comparison of :citet:`Waleffe.Byeon.Riach.ea.2024`. The teaching sections stop at 2024 (Jamba) and do not say what became of the layers they build. Two sentences should move forward into `mamba.md` and `hybrids.md`.
- It is more careful than `ssm.md` about what S4D's real rates are ("the diagonal of the HiPPO matrix itself" against "approximate that design"); align the section with it, and check FR-2.
- The matched comparison of Waleffe et al. (pure recurrent models behind on recall tasks, hybrids with a small share of attention at or above the transformer) is the evidence at scale for the thesis that the chapter's own small experiments cannot supply. It deserves a sentence in `hybrids.md`, attributed.
- Unranked, like the resources list of the introduction.
