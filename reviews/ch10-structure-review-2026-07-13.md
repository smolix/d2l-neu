# Structural review: Chapter 10 "Gated and Linear Recurrence" (2026-07-13)

> **DECISION (2026-07-13): shelved. Neither Option A nor B will be applied
> in place.** Instead, ch10 will be **relocated later** to sit *after* attention,
> optimization algorithms, and transformers have been introduced, and the SSM
> treatment expanded past 4 notebooks at that point. That move turns every
> forward reference cataloged below into a backward reference and gives the
> G1/G2/G3 completeness gaps their own notebooks. This reverses the earlier
> "SSMs before attention" design decision. **This document is the input to that
> future relocation, not a to-do list for now.** Keep ch10 where it is.



*Review + proposal only; no chapter edits made. Scope: the two concerns raised by
Alex, (1) whether the modern-SSM treatment is complete and coherent, and (2)
whether the chapter leans too heavily on material introduced later in the book
(transformers/attention in ch. 11, advanced optimizers in ch. 12). Grounded in a
full read of `chapter_recurrent-modern/{index,lstm,seq2seq,ssm,mamba}.md`, the
ch. 9 and ch. 11 index files, `chapter_optimization/`, `_quarto.yml` +
`CHAPTER_NUMBERING`, the four modernization design docs, and `d2l.bib`. The
handoff doc's "decisions already made" list (SSMs before attention; hybrids/SSD
deferred to attention/LLM chapters) is respected, not relitigated.*

**Book order that matters here:** ch. 9 sequence/LM/RNN → **ch. 10 this chapter**
→ ch. 11 attention/transformers → ch. 12 optimization → … → ch. 15/16 NLP
pretraining/applications. Two facts that reframe the optimizer concern: gradient
clipping is *taught in ch. 9.5* (`rnn-implementation.md`), so it is a backward
reference; and **AdamW + warmup + cosine schedules already appear, used and
motivated at recipe level, in §8.6** (`chapter_convolutional-modern/
training-recipes.md`, `sec_training_recipes`). Ch. 10 never references §8.6. Also:
warmup and LR schedules do **not** occur anywhere in ch. 10 — the only optimizer
forward-lean is unexplained `Adam`.

---

## 1. Forward-reference catalog

Method: every `:numref:` in the chapter was extracted and classified, plus a
grep sweep for prose-level references (transformer/attention/BERT/ELMo/KV
cache/induction heads/Adam/warmup/schedules). Headline finding first:

> **Every `:numref:` in ch. 10 points backward except two, and both are the
> chapter's explicit handoffs to ch. 11** (`index.md:57`, `mamba.md:1339`,
> both `chap_attention-and-transformers`). All remaining forward references are
> prose-level name-drops or code provisioning, not structural dependencies.
> Nothing in ch. 10 *requires* the reader to understand attention or Adam to
> follow a derivation or run a cell.

| # | Location | Points to | What it is | Verdict | Mitigation (if avoidable) |
|---|----------|-----------|------------|---------|---------------------------|
| 1 | `index.md:44-45`, `:53-57` | ch. 11 | "look back at everything at once (attention, the next part)"; phone-number limit + handoff numref | **Inevitable** | None. A chapter intro may preview its successor; this is the standard device and it is done in self-contained terms (a lookup vs. a bounded state). |
| 2 | `lstm.md:19-23` | ch. 11 | Gates motivated partly by "transformer MLP blocks multiply one branch by another" | **Avoidable** (1 sentence) | Reframe as explicitly forward-looking: "the architectures that displaced the LSTM kept its central trick, as we will see in the next chapter." No transformer anatomy needed to make the point that the gate outlived the cell. |
| 3 | `lstm.md:987-992` | ch. 15/16 | ELMo → "bidirectional attention of BERT, which we study in the pretraining chapters" | **Inevitable** | Historical pointer by name; understanding bi-RNNs does not depend on it. Fine as-is. |
| 4 | `lstm.md:994-1035` | ch. 11+ | "Gates beyond Recurrent Networks" subsection: Highway, GLU/SwiGLU formula, table incl. transformer-MLP row | **Mostly fine; partially avoidable** | Keep the subsection — it is the load-bearing setup for 10.3/10.4 and most rows (Mamba, Griffin, xLSTM) are *this chapter's* material. Mark the SwiGLU/transformer rows explicitly as a look-ahead ("you will meet these blocks in ch. 11"); the gate math itself needs no attention knowledge. |
| 5 | `seq2seq.md:286-287` | ch. 11 | `valid_len` recorded "which the attention models of the next chapter will need" — the field is *unused* by any ch. 10 model | **Avoidable (prose), keep the data** | Keeping the array is cheap and avoids ch. 11 churn; the sentence is already honest. At most soften to "recorded for later use". Low priority. |
| 6 | `seq2seq.md:814-890` + prose `:814-815`, slide `:1262` | ch. 11 | `save_attention_weights=` parameter threaded through all four tabs of the *taught* `predict_step`, dead in this chapter, advertised as "the attention models of the next chapter reuse it" | **Avoidable** — the clearest concrete win | Strip the hook from ch. 10's version; ch. 11 re-attaches an extended `predict_step` via the repo's own `@d2l.add_to_class` idiom (exactly how ch. 10 already extends `MTFraEng` and `BPETokenizer`). Teaching code should not carry dead parameters. Cost: ch. 11 contract change (`bahdanau-attention.md` etc. consume it, per handoff §"attention-chapter contracts"), recapture of touched notebooks. If that cost is unwanted now: delete the two prose/slide advertisements and leave the code; the reader then never notices the parameter. |
| 7 | `seq2seq.md:688-704` (+ recipe at `:780`) | ch. 12 (but §8.6 exists) | "swaps SGD for Adam, which converges faster on this task" — Adam used with no anchor in either direction | **Avoidable** | The reader has in fact already met adaptive optimizers: named + partially explained in §3.7 weight-decay (`weight-decay.md:236-245`, AdamW decoupling, forward ref to `sec_adam`), and *used with a recipe-level explanation* in §8.6 (`sec_training_recipes`) and ConvNeXt. Ch. 10 just fails to point at any of it. Fix locally: a 2–3 sentence aside at first use — "Adam adapts a per-parameter step size from running gradient statistics; we used its AdamW variant in the modern recipe of `:numref:`sec_training_recipes``, and derive it in `:numref:`chap_optimization``. Here it converges in a fraction of SGD's epochs." One aside, then §10.3/10.4 just say "as in `:numref:`sec_seq2seq``". |
| 8 | `seq2seq.md:996-999` | future LLM chapter | "COMET and … LLM to judge, which we return to when we have the models to build them" | **Avoidable (dangling)** | The LLM chapter does not exist yet (overview §"The third piece" plans it). Soften to "beyond this book's current scope" or keep as a deliberate hook — but be aware it is currently a promise with no landing site. |
| 9 | `seq2seq.md:1099-1108` | ch. 11 | The fixed-vector bottleneck → "two ways to defeat it… attention… the rest of this chapter" | **Inevitable — and exemplary** | This is how a forward reference should look: the escape is named, not used, and the chapter takes the other branch. Leave alone. |
| 10 | `ssm.md:15-28` | — | Parallelism motivated **against CNNs** (`chap_cnn`), not transformers | **Already right** | Worth stating: the training-efficiency argument for linear recurrence is made entirely in convolution/recurrence terms the reader has. No change needed; this is the pure-recurrence framing the concern asks for. |
| 11 | `ssm.md:846-848`, `:1215-1219` | ch. 11 | LRA results "ahead of every transformer of its day" / "both RNNs and transformers had failed" | **Inevitable** | Historical benchmark claims; require zero attention machinery. Fine. |
| 12 | `ssm.md:1030-1031`, `mamba.md:319-322`, `:1045-1056` | ch. 12 | Adam again (with genuine reasons given: "the SSM's exponentials make plain SGD's step sizes awkward"; Mamba overfits → dropout + gentler LR) | **Partly avoidable** | Once item 7's aside exists, these become backward references. Do **not** force SGD here: on `log Δ`/`log a` parameterizations SGD is genuinely fragile, and contorting the models to keep an SGD-only diet would damage the material to satisfy a purity rule. The honest caveat paragraph (`mamba.md:1098-1114`, Adam-vs-SGD asymmetry of the capstone) is already best practice. |
| 13 | `mamba.md:27-29` | ch. 11 | "returned recurrence to serious competition with transformers" | **Inevitable** | Historical framing. Fine. |
| 14 | `mamba.md:149-154` | ch. 11 | "Transformer interpretability work calls the circuits that do this *induction heads*" | **Avoidable (trim)** | The associative-recall motivation ("Mrs. Watchett … Mrs. ___") is already self-contained; the induction-heads name-drop teaches nothing yet. Compress to the Arora citation, or keep the name in a parenthetical. Minor. |
| 15 | `mamba.md:745-750` | ch. 11 | "Where a transformer alternates attention blocks with MLP blocks… the full comparison with attention must wait until we have built attention" | **Inevitable** | Explicit, one-sentence, correctly deferred. The right pattern. |
| 16 | `mamba.md:1226-1271` "What a Fixed State Cannot Do" | ch. 11 | Jelassi copying theorem; MQAR; Gu's brains-vs-databases (KV cache) | **Inevitable in substance** | The load-bearing argument (a fixed state holds `k log₂ V` bits, information theory doesn't care about the update rule) is transformer-free. The comparisons *are* the honest scientific content and the chapter's raison-de-handoff. One cheap improvement: gloss the KV cache in self-contained terms on first use ("a model that simply appends every token's representation to a growing store — the transformer of the next chapter"). |
| 17 | `mamba.md:1273-1317` "The Recurrent Frontier" | ch. 11 + future LLM ch. | Hybrids (Jamba/Griffin, "local attention", attention-to-recurrence ratios); siblings (RWKV/xLSTM/GLA); **"Mamba-2's state space duality … needs the attention machinery of the next chapter, and we defer it there"** | **Mostly inevitable; one dangling promise** | This already *is* the "looking ahead" coda the concern hypothesizes — it just isn't labeled as one. The genuine defect: **ch. 11 contains no SSD/Mamba-2/linear-attention material at all** (verified by grep across `chapter_attention-mechanisms-and-transformers/*.md`), so "we defer it there" points at nothing. Either reword to "a later chapter" / "once attention is in hand", or log the SSD revisit as a firm obligation of the planned LLM chapter (overview §2 already assigns it there). |
| 18 | `mamba.md:1338-1339` | ch. 11 | Closing handoff numref | **Inevitable** | The chapter's designed ending. Leave alone. |

**Tally.** 18 catalog entries; **10–11 inevitable/appropriate** (items 1, 3, 9,
10, 11, 13, 15, 16, 18, and most of 4 and 17), **7 avoidable/fixable** (2, 5, 6,
7, 8, 12, 14, plus the SSD wording in 17). None of the avoidable ones is
structural: **every fix is a local edit** — one Adam aside, one code-plumbing
removal (or two-sentence deletion), three softened sentences, one relabeled
coda. No section needs to move to fix the forward-reference problem.

**On the two specific questions posed:**

- *(a) Can the SSM efficiency argument be made in pure recurrence terms?* It
  largely already is. Training-side: `ssm.md:15-28` argues from CNNs
  (`chap_cnn`) and the scan benchmark (`#ssm-implementation-3`) — no
  transformers involved. Inference-side: the claim rests on ch. 9's
  `subsec_rnn-constant-memory` ("constant memory per step"), also
  transformer-free. The only place inference efficiency leans on transformers
  is the brains-vs-databases coda, where the comparison *is* the point. The
  real weakness on the inference side is not a forward reference but a missing
  demonstration — see §2, gap G1.
- *(b) Can training use only tools available by ch. 10?* Clipping: already a
  ch. 9 tool (non-issue). Warmup/schedules: not used in ch. 10 (non-issue).
  Adam: used in 10.2/10.3/10.4 and genuinely warranted for the SSM
  parameterizations; the right fix is the introduce-once aside anchored to
  `sec_training_recipes` (backward) + `chap_optimization` (forward), not an
  SGD-only diet.

---

## 2. Modern-SSM completeness and coherence

### What the chapter gets right (and should not be touched)

The 10.3→10.4 spine is, honestly, the strongest textbook-scale SSM treatment I
know of, and it matches the approved design exactly:

- **The linearization story** (`ssm.md` §"Linearizing the Recurrence",
  `subsec_mingru`): minGRU as the deletion route, with the affine-coefficient
  form `:eqref:`eq_affine_recurrence`` doing double duty (parallelism +
  readable memory), LRU cited. Clean.
- **The parallel-scan story** (`subsec_parallel-scans`): associative combine
  derived, log-depth schedule figured, implemented in three frameworks,
  verified against the sequential loop, *benchmarked* (the "trains like a CNN"
  plot). This is the pedagogical core and it lands.
- **The SSM derivation** (`subsec_ssm`, `subsec_zoh`): the boxed "step size is
  a gate" correspondence is the best bridge in the chapter, and stability by
  parameterization closes the loop with `subsec_bptt-gradient-pathologies`.
- **Three views verified numerically** (`subsec_ssm-conv`,
  `#ssm-recurrence-is-convolution`): loop == scan == convolution to 1e-7, with
  the correct editorial choice (implement the scan, not the FFT, because the
  scan survives selectivity).
- **HiPPO at the right altitude** (`subsec_hippo`): stated + reconstruction
  figure, derivation cited not derived; S4→S4D→S5 compressed to a paragraph;
  the implementation sits at "S4D's parameterization, S5's scan" and says so.
- **The S4D-vs-LSTM initialization narrative** (`ssm.md:1191-1219`): memory by
  design vs. memory by folklore is a genuinely original teaching result.
- **The selectivity progression** (`mamba.md`): selective copying as the
  motivating *experiment* (not just prose), gate derived a third time, "what
  selectivity costs" (conv view dies, scan survives) demonstrated in code,
  hardware-awareness as prose. The capstone (three answers, one task, honest
  caveats) and the copy-task revisit close the arc properly.
- **Honest limits** (`subsec_fixed-state-limits`): the k·log V bit argument,
  Jelassi and MQAR summarized with the explicit admission that they only bite
  beyond textbook scale, and the capacity-cliff exercise as the in-reach proxy.

Sequencing within 10.3–10.4 is correct: minGRU → scan → SSM → HiPPO → S4D →
selectivity → Mamba → limits → frontier. I found no mis-sequencing inside the
SSM material itself; each concept is used only after it is built.

### Gaps (ranked)

**G1 — The recurrent-inference story is claimed, demonstrated at toy level,
and then contradicted by the implementation. This is the one substantive
completeness gap.** The chapter's whole bargain is "parallel training *and*
O(1)-state inference" (`ssm.md:30-32`, `:410-414`: "at inference we still
update … one token at a time, in constant memory, exactly like any RNN";
`index.md:51-53`). But:

- `S4D.forward`, `SelectiveSSM.forward`, and `Mamba.forward` only implement
  the parallel/scan path; `Mamba.forward` returns `(…, None)` for its state
  (`mamba.md:795-797`) — the stack *cannot* carry state across calls.
- The capstone's generation path (`step_fn`, `mamba.md:1124-1137`) re-runs the
  model on the **entire prefix for every generated token** — O(T²) generation
  that quietly contradicts the constant-memory claim the chapter sells.
- The only recurrent stepping shown is on a random toy system
  (`#ssm-recurrence-is-convolution` view (i); `#mamba-what-selectivity-costs…-1`).

A strong modern-SSM chapter should *close the loop*: take the trained S4D (or
Mamba) and step it token-by-token with a carried `(H, N)` state, verify parity
with the scan output, and ideally time stepped generation against
prefix-re-running as the prefix grows. This is also the **transformer-free
inference-efficiency argument** that concern 2(a) asks for: "one state update
per token vs. re-reading the whole prefix" needs no attention chapter to land,
and it sets up the KV-cache contrast in ch. 11 perfectly. Suggested shape: a
short "Inference, one token at a time" subsection in 10.3 after
`subsec_s4d` (LTI stepping is three lines from `:eqref:`eq_ssm_disc``), a
parity assert, one timing plot; plus a 10.4 exercise (or subsection) to add a
`step()` to `SelectiveSSM`/`MambaBlock` (the depthwise conv needs a width-4
rolling buffer — itself a nice teaching point, since it is exactly what
production Mamba inference caches).

**G2 — The SSM↔attention duality is deferred to a chapter that doesn't deliver
it.** `mamba.md:1300-1305` defers Mamba-2's SSD "to the next chapter";
`chapter_attention-mechanisms-and-transformers/` has zero SSM/Mamba/linear-
attention content (grep-verified). The intended home per the overview is the
planned LLM chapter. Until that exists, the reader is promised a payoff the
book never provides. Fix now (wording): "once attention is in hand" / "in a
later chapter". Fix eventually (content): the LLM chapter's SSM-revisit
section (SSD, GLA/DeltaNet lineage, hybrid design points) should be treated as
a recorded debt of this chapter's design, not an optional extra —
`Dao.Gu.2024`, `Yang.Wang.Shen.ea.2024` are already in `d2l.bib`.

**G3 — The "siblings converged" claim floats above anything the reader
built.** `mamba.md:1292-1305` says RWKV/xLSTM/GLA all reached "a matrix-valued
state updated by a learned decay plus an outer-product write." The chapter
only ever builds *vector* states with elementwise decay. The one-paragraph
prose is the right altitude for the frontier section, but a single displayed
equation — `S_t = S_{t-1}·diag(α_t) + v_t k_tᵀ`, "note this is still an affine
recurrence our scan evaluates; read out by `S_t q_t`" — would ground the claim
in the chapter's own machinery at a cost of ~6 lines, without teaching linear
attention. Optional; if it feels like scope creep, leave the prose but expect
readers to take it on faith until the LLM chapter.

**G4 — Redundancy: S4D + S4DBlock restated verbatim in `mamba.md`**
(`mamba.md:197-317`, ~120 lines × 3 tabs) because sections must execute
standalone. The chapter already solved this exact problem for the scan:
`associative_scan` is `#@save`d in 10.3 and picked up as
`d2l.associative_scan` in 10.4 (`mamba.md:89-92`, pytorch tab). Do the same
for `S4D`/`S4DBlock` (pytorch at minimum; tf/jax tabs currently restate the
scan too, so full parity may not be achievable — but the prose "we restate the
layer and its residual block verbatim" would shrink to one line on the tab
that matters). Kills ~360 lines of drift-prone duplication.

**G5 — Minor omissions, all defensible at textbook scale (listed for
completeness, no action urged):** complex/rotational eigenvalues get one
honest paragraph (`ssm.md:666-670`) — adequate; LRA is cited as history rather
than run — correct call; no needle-in-haystack at length — explicitly and
honestly disclosed (`mamba.md:1252-1257`); no long-convolution (Hyena) or
RetNet coverage — not in `d2l.bib`, out of the approved scope; bidirectional
SSMs unmentioned — fine.

### Coherence verdict

**The SSM story is strong and correctly sequenced; it does not need
consolidation or expansion of its spine.** What it needs is (i) the
inference-side demonstration (G1) to make its central bargain real, (ii) the
dangling duality promise resolved (G2), and (iii) the mechanical
de-duplication (G4). What *dilutes the chapter as an SSM chapter* is not the
SSM sections but the fact that ~45% of the chapter's bulk (10.1's four legacy
absorptions + 10.2's MT engineering) sits before the SSM material begins —
which is the structural question, addressed next.

---

## 3. Alternative structures

Two options considered seriously; two rejected up front.

**Rejected: moving SSMs after attention.** Relitigates an approved decision
(handoff §1), and the before-attention placement is the book's differentiating
move (overview §5.1). The chapter's internal evidence supports the decision:
nothing in 10.3/10.4 needs attention, and the handoff writes ch. 11's
motivation.

**Rejected: splitting ch. 10 into two chapters** (gated+seq2seq / linear+
selective). Would give the SSM material "its own arc" nominally, but renumbers
every downstream chapter (`CHAPTER_NUMBERING`, labels, crossrefs from ch. 11,
12, 15, 16, mdl-appendix), contradicts the slots-unchanged constraint the
design committed to, and buys nothing that Option B below doesn't buy cheaper.

### Option A (recommended): keep the 10.1–10.4 skeleton; consolidate in place

| § | Section | Changes |
|---|---------|---------|
| 10.1 | Gated Recurrent Networks | Reframe the one transformer-MLP motivation sentence (cat. #2); tag the SwiGLU rows of "Gates beyond Recurrent Networks" as explicit look-ahead (cat. #4). Nothing else. |
| 10.2 | Encoder-Decoder Models | Add the **introduce-Adam-once aside** (2–3 sentences, backref `sec_training_recipes`, forward ref `chap_optimization`) at `seq2seq.md:688` (cat. #7). Remove the `save_attention_weights` plumbing from the taught `predict_step` and its two advertisements, re-attaching in ch. 11 via `add_to_class` (cat. #6) — or, minimal variant, delete only the advertisements. Soften the COMET/LLM-judge promise (cat. #8). |
| 10.3 | Linear Recurrence and SSMs | **Add "Inference, one token at a time"** after `subsec_s4d` (G1): step the trained S4D recurrently from `:eqref:`eq_ssm_disc``, assert parity with the scan, one timing comparison vs. prefix re-running. `#@save` `S4D`/`S4DBlock` (G4). Point Adam usage back to the 10.2 aside (cat. #12). |
| 10.4 | Selective State Space Models | Drop the verbatim S4D restatement in favor of the `d2l.` classes (G4). Trim induction-heads to a citation (cat. #14). Add a Mamba `step()` exercise or short subsection (G1, selective half; the conv rolling-buffer is the teaching point). Merge "What a Fixed State Cannot Do" + "The Recurrent Frontier" under one explicitly labeled coda — e.g. **"Looking Ahead: Recurrence in the Age of Attention"** — with a one-line contract sentence ("terms named here — attention, KV cache — are defined in the next chapter; read this as a preview") and the KV-cache gloss (cat. #16). Fix the SSD deferral wording (G2/cat. #17). |
| index | — | Unchanged except any coda-name echo. |

*Fixes:* both user concerns, at local-edit cost. All avoidable forward refs
resolved or explicitly fenced into a labeled preview coda; the SSM chapter
gains its missing inference demonstration and loses its one duplication.
*Costs:* recapture of the 3–4 touched notebooks per framework (CPU-feasible
except the timing cells, which want the GPU box for honest numbers); if the
`predict_step` hook moves, a coordinated edit + recapture in ch. 11's
bahdanau/transformer notebooks (the handoff's build-trap history says: verify
genuine re-execution). No renumbering, no label churn, no slide restructuring
beyond the edited sections.

### Option B: reseat seq2seq at the chapter tail (lstm → ssm → mamba → seq2seq)

| § | Section | Rationale |
|---|---------|-----------|
| 10.1 | Gated Recurrent Networks | Ends on "Gates beyond…" which already points straight at linearization — the 10.1→SSM splice is nearly written (`lstm.md:1028-1035` needs only its last sentence changed). |
| 10.2 | Linear Recurrence and SSMs | Unchanged content (+ Option A's G1/G4 additions). Opening loses its `sec_seq2seq` strain clause (`ssm.md:16-17`), which is one sentence. Adam's first use moves here, where the justification ("exponentials make SGD awkward") is strongest. |
| 10.3 | Selective SSMs | Unchanged (+ Option A's coda edits). The "three answers" capstone still lands with all three architectures in hand. |
| 10.4 | Encoder-Decoder Models for Sequence Transduction | Becomes the application finale: the abstraction, the dataset, the GRU translator — plus the newly available payoff **"swap the encoder for the chapter's Mamba stack"** as a taught cell or exercise. Ends on the fixed-vector bottleneck and the two escapes, i.e. the chapter's last page flows *directly* into ch. 11's Bahdanau attention, which literally continues this model. |
| index | — | Rewritten: the "second thread" paragraph becomes a "the destination application" paragraph. |

*What it fixes:* the gate→linearize→select spine runs uninterrupted, giving the
SSM material the contiguous arc concern 1 asks about; the strongest possible
ch. 11 handoff (bottleneck demo on the final pages, `save_attention_weights`
plumbing now adjacent to its consumer, making cat. #5/#6 nearly moot);
encoder-decoder gets modern encoders instead of being frozen at 2016.
*What it costs:* (i) the bottleneck no longer motivates "build a better fixed
state" *before* the SSM sections — 10.2/10.3's motivation must stand on the
compute argument alone (it can; `ssm.md` already leads with it); (ii) the
chapter ends on a translation application rather than on `mamba.md`'s
carefully built limits-and-frontier climax — the coda would need to move into
or after 10.4, weakening the "three answers" close; (iii) chronological
whiplash (1997→2024→2014) unless the seq2seq section is reframed as "the
abstraction that outlived its era", which it half-is already; (iv) mechanical:
within-chapter renumber of `CHAPTER_NUMBERING`, `_quarto.yml` order, index
rewrite, edits to both transition regions, recapture of everything edited —
several×, not order-of-magnitude, more churn than Option A.

---

## 4. Recommendation

**Adopt Option A.** The chapter's structure is fundamentally sound — the
forward-reference problem is real but *shallow* (prose name-drops and one dead
code parameter, not structural dependencies; 7 fixable items vs. ~11
appropriate handoffs), and the SSM treatment needs one added demonstration,
not a reorganization. Concretely, in priority order:

1. **Add the recurrent-inference subsection to 10.3 + Mamba stepping in 10.4**
   (G1). This is the single highest-value change: it converts the chapter's
   central claim from asserted to demonstrated, *and* it is the pure-recurrence
   efficiency argument concern 2(a) asks for.
2. **Introduce Adam once** in 10.2 with the `sec_training_recipes` backref and
   `chap_optimization` forward ref; make 10.3/10.4 refer back (cat. #7/#12).
   Do not switch the SSM sections to SGD.
3. **Label the mamba.md tail as a look-ahead coda** and fix the SSD deferral
   wording; record the SSD/linear-attention revisit as a hard obligation of
   the planned LLM chapter (G2, cat. #17).
4. **De-duplicate S4D/S4DBlock via `#@save`** (G4).
5. **Remove or de-advertise the `save_attention_weights` plumbing** in 10.2
   (cat. #6) — full removal preferred, prose-only removal acceptable.
6. Small trims: cat. #2 (lstm transformer-MLP sentence), #14 (induction
   heads), #8 (COMET promise), #16 (KV-cache gloss).

**Hold Option B in reserve.** It is the right move *if* the priority is
maximizing the SSM arc's contiguity and the ch. 11 splice — it is coherent,
costed, and does not fight any approved decision — but it trades away the
chapter's current climax and adds churn disproportionate to the
forward-reference gains, which items 1–6 capture anyway.

**Leave alone:** the SSM-before-attention placement; the 10.3/10.4 internal
sequence; the HiPPO altitude; the three-views verification; the selective-copy
experiment; the honest-limits content and its Jelassi/MQAR citations; the two
`chap_attention-and-transformers` handoff numrefs; the ELMo→BERT pointer; the
bottleneck section of 10.2 (exemplary); and `ssm.md`'s CNN-anchored
parallelism motivation, which already does exactly what the forward-reference
concern wants.
