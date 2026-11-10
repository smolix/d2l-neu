# Compute Graphs and Compilation

Source: `chapter_computational-performance/compilation.md` (lines 1 to 883, up to the slides marker). About 5,800 words outside code, of which about 1,750 are exercises. Stored outputs read: `outputs/pytorch/.../compilation.json` and `outputs/jax/.../compilation.json`.

## Verdict

A good section with one first-rate passage and two experiments that do not carry their subsections. The fusion subsection is the chapter's method at its best: it counts bytes, predicts a factor of nine and 1.15 ms against 0.13 ms, measures 1.25 ms against 0.15 ms, and then scales the same count to the billion-parameter GPT in a closed form. After that the section stops predicting. The whole-step experiment gives no expected step time, never says how much faster the compiled step is, and hides a sixfold gap between the two frameworks (PyTorch 1.71 ms, JAX 0.26 ms for the same step) that the chapter's own model explains: the step is overhead-bound, not bandwidth-bound, so default `torch.compile` barely helps it. The most important fix is to make "Compiling a Training Step" and "Kernel Replay" one measured argument: predict the step from the roofline, show that fusion alone leaves the PyTorch step far above it, and let replay close the gap. The second fix is a correctness one: the "epilogue" fusion pattern is stated as something these compilers do, and neither Inductor nor XLA does it for matrix products on a GPU; the section's own replay experiment contradicts it.

## Scores

| dimension | score |
|---|---|
| Clarity of explanation | 4 |
| Clarity of thought and focus | 3 |
| Pedagogy | 4 |
| Language | 4 |
| Ease and enjoyment | 3 |

## What works

- **The fusion argument** ("Counting Memory Traffic"). Eight named temporaries $t_1 \ldots t_8$, ten reads and eight writes, a ratio of 9, and then the measurement. The sentence after the output is the best in the section and should be protected: "The compiler made memory no faster and arithmetic no cheaper. It removed 16 of the 18 array transfers." The printed GB/s column is a good device: both versions move data at most of the card's bandwidth, which shows the regime and not only the speedup.
- **The scale-up to the GPT**, ending in $9P/(\beta d) \approx 0.72$. All numbers recompute (134 MB per array, 2.4 GB, $5.5 \times 10^{11}$ FLOPs, 3.3 ms). The remark that the ratio is independent of the token count and falls with width is the kind of judgment the voice guide asks for.
- **The honest caveat** "Real models seldom write GELU as a chain of eight operations. Libraries ship hand-fused kernels for common activations". It keeps the toy from being oversold.
- **The graph-break demonstration.** The printed two graphs make an abstract statement visible, and the prose is written to survive the random input ("the branch that this random input took, `x + 1` or `x - 1`").
- **Break-even $n^*$** is the right quantity and the worked 30 s / 10 ms = 3,000 steps is memorable.
- **Short imperative sentences with a view**: "Un-jitted JAX is for debugging; training steps are jitted." "Debug eagerly, then compile."
- **The exercises** are the strongest part: padding buckets (a real optimization with a square-root answer near $w = 46$), fusion behind a matmul ($S(n) = (n + 2970)/(n + 330)$, falling to 1.1 near $n = 26{,}000$), the streaming softmax recurrence that prepares FlashAttention, and the tracer of one's own.
- Figure `fig_fusion` (rendered) is clear and carries the count. `fig_compile_pipelines` is adequate.

## Structural findings

1. **The section predicts once and then stops.** Fusion: predicted and measured. Whole step: only the break-even formula, no step time. Replay: "a few microseconds" per kernel asserted, no predicted speedup. The chapter's method is predict, measure, classify, fix, re-measure; here the classify and re-measure steps are missing after the first experiment. See COMP-9 and COMP-12.
2. **"Compiling a Training Step" tests the wrong regime.** Its lead-in is about fusing elementwise work, but its MLP (width 1,024, batch 512) has 2 MB activations. Bandwidth is irrelevant at that size; the step is a few hundred small kernels and is overhead-bound, the subject of the next subsection. The two subsections should be reordered or merged so that the step is diagnosed first.
3. **The tabs carry the main contrast of "Graph Capture".** That Dynamo falls back to Python silently while `jax.jit` refuses is the one design difference a reader should remember, and each reader sees only half of it. One shared sentence should state both behaviours; the tabs then show the output.
4. **What a reader will look for and not find.** (a) Tracing against scripting: nothing on TorchScript (`torch.jit.trace`, `torch.jit.script`), which older tutorials and the first edition of this book taught, nor on `torch.export`. One sentence saying that `torch.compile` replaces them for training, and that a value-based branch under `torch.jit.trace` was silently frozen to the branch taken, would place the reader and would sharpen the "Branches on Values" story into three behaviours (freeze silently, refuse, fall back). (b) A profile after compilation. The section never uses the profiler of :numref:`subsec_perf-profiler` to show that the bottleneck moved. (c) Any statement of which compile modes exist beyond `reduce-overhead`. Not needed.
5. **Length is about right** (4,000 words of body). What can go: the second byte count in the introduction (COMP-1), the "Fusion Patterns" paragraph on epilogues (COMP-7), and half of each PyTorch tab note on Dynamo internals (COMP-5).
6. **Ending.** The paragraph after the bullet list in "Limits of Compilation" is a recap plus a handoff, followed by the Summary. Keep the handoff sentence, drop the recap sentence (COMP-14).

## Passage-level findings

**COMP-1** (moderate, thought: repetition). Introduction. Anchor: "the\neight kernels together move 1.15 GB to produce 64 MB of output" (grep "eight kernels together move 1.15 GB").
The 18-array count is now told four times: in `performance-model.md` ("read and write 18 tensors"), in this introduction, in "Counting Memory Traffic", and in the caption of `fig_fusion`. The introduction also gives the prediction (1.15 ms against 0.13 ms) before the subsection that derives it. Fix: let the introduction cite the count in one clause ("eighteen array transfers where two would do, :numref:`subsec_perf-regimes`") and keep the full count, with the millisecond prediction, in "Counting Memory Traffic" next to the measurement.

**COMP-2** (minor, explanation). "Computation Graphs". Anchor: "which takes 1–5 µs in the\nCUDA driver and about 5–15 µs of host time" (grep "about 5–15 µs of host time").
The 5–15 µs figure appears three times in this section (here, "Kernel Replay", exercise 7), each time as a fact of nature. It is a property of the host CPU and the framework version, `performance-model.md` uses $h = 20$ µs, and the stored replay output gives 18 µs per layer. Fix: define $h$ once as the measured host time per call from :numref:`subsec_perf-sweep`, use the symbol afterwards, and say that it is some tens of microseconds on current hosts.

**COMP-3** (minor, correctness). "Computation Graphs". Anchor: "Autograd records this graph during the forward pass so that it can walk it".
True of PyTorch. In the JAX tab nothing is recorded during an ordinary forward pass; `jax.grad` traces the function. The shared prose should say "a framework with a tape" or move the sentence into the PyTorch tab.

**COMP-4** (moderate, correctness). "Shapes and Recompilation" and Summary. Anchors: "A call with tensors\nof a new shape or dtype therefore needs a new compilation" (grep "of a new shape or dtype therefore needs a new compilation") and "a new shape requires a new\ncompilation" (grep "a new shape requires a new").
The PyTorch cell shows the opposite for the third shape: two compilations for lengths 8, 8, 16, 32, because the second compilation is symbolic in the length. The shared sentence and the Summary are true only for the JAX tab. Fix: "A compiler that specializes kernels to shapes must compile again for a new shape, unless it compiled for a symbolic size. JAX always specializes; PyTorch specializes first and generalizes on the second shape it sees." Then the two outputs (3 traces, 2 compilations) illustrate one shared statement.

**COMP-5** (moderate, correctness: statements that will date). PyTorch tab notes in "Tracing", "Branches on Values", "Shapes and Recompilation", "Compiling a Training Step". Anchors: "once a function has been compiled eight times", "`torch._dynamo.explain`, an internal function whose interface may change", "Inductor caches the kernels\nit compiles on disk" (grep "it compiles on disk").
I am confident of: Dynamo, AOTAutograd, Inductor with Triton on GPU and C++ on CPU; guards; silent graph breaks; `fullgraph=True`; static first compile then a symbolic recompile; `dynamic=False`. Unverified against PyTorch 2.11, believed correct: the limit of eight and its name `recompile_limit` (the option was called `cache_size_limit` in earlier releases); the on-disk cache being on by default; `explain` exposing `graph_count`, `graph_break_count` and `graphs`. All of these will date, and the config name already changed once. Fix: keep the mechanism in prose and move version-specific names into code comments; say "after a small number of recompilations (eight at the time of writing)". The stored first-call time of 0.7 s is a warm-cache number, which the prose should say where it reports the time, not two paragraphs later.

**COMP-6** (minor, explanation). "Counting Memory Traffic". Anchor: "about a dozen FLOPs per element,\nwhich is far less than the GPU can do while it moves the element's bytes" (grep "about a dozen FLOPs per element").
This is the ridge-point argument without the ridge point. The fused kernel moves 8 bytes per element, so it would need about $8 I^*$, more than a thousand FLOPs per element, to be compute-bound. Say so with the number; it is the chapter's central quantity. Likewise the closed form $9P/(\beta d)$ a page later is $9 I^*/d$ and should be written that way: "the unfused activation costs as much as the matrix products when the width is nine ridge points, about 1,500".

**COMP-7** (major, correctness; confidence about 85%). "Fusion Patterns" and Summary. Anchors: "An *epilogue* is a run of elementwise operations that follows a matrix\nmultiplication" (grep "An \*epilogue\* is a run of elementwise operations") and "epilogues of matrix multiplications into single kernels".
The text says "Compilers fuse three kinds of patterns" and that "The compiler applies them to each tile of the product while the tile is still in registers". To my knowledge neither Inductor nor XLA does this for matrix products on a GPU: both hand the product to a library kernel (cuBLAS) and fuse only the elementwise code around it. Fusing a bias and activation into the product is done by hand-written or library kernels (cuDNN's fused convolution, CUTLASS epilogues). The section contradicts itself on the point: two paragraphs later "A large matrix multiplication ... already runs as a single tuned library kernel; compiling the code around it leaves it as it is", in "Kernel Replay" "Fusion ... cannot merge matrix multiplications", and the replay experiment counts 120 kernels for 60 layers after compilation, which means the tanh was not fused into its product. Fix: reduce to two patterns the reader has seen evidence for (elementwise chains, reductions), and mention epilogue fusion in one sentence as something libraries do for convolutions and that general compilers do not do for matrix products. Remove it from the Summary. The reduction claim ("loads a row into on-chip memory once ... writes the row once") is also unverified for long rows; exercise 5 treats exactly that limit, so a clause "when the row fits" would align the text with the exercise.

**COMP-8** (minor, thought). "Counting Memory Traffic". Anchor: "Give it a GELU MLP of hidden width".
The running example of the chapter has a SwiGLU network of width $\tfrac{8}{3}d$ (`memory-precision.md`, and the default `act='swiglu'` of `d2l.GPT`). Here it is given a different feed-forward network for one paragraph. Either compute the example for SwiGLU (a sigmoid and two multiplications, a shorter chain, which would also show that the unfused penalty depends on how the activation is written) or say in a clause that this is a variant.

**COMP-9** (major, pedagogy and thought). "Compiling a Training Step". Anchor: "The first call takes far longer than a step, up to several seconds, and\nthe printed $n^*$ is a small fraction" (grep "The first call takes far longer than a step").
This is all the shared prose says about an output that contains the section's most interesting result. Stored: PyTorch eager 2.98 ms, compiled 1.71 ms; JAX eager 25.95 ms, compiled 0.26 ms. (a) No speedup is stated or predicted. (b) The roofline predicts about 0.2 ms for this step ($\approx 1.3 \times 10^{10}$ FLOPs at the tf32 rate, plus about 50 MB of parameter traffic). JAX reaches it; compiled PyTorch is six to eight times above it. The reason is the one the next subsection teaches: the step is a few hundred small kernels, eager costs about 150 to 200 launches at $h$, and default `torch.compile` with a graph break still issues its kernels from the host. (c) The lead-in promises fusion ("lets the compiler fuse the elementwise work of all three") and the experiment cannot show it, because 2 MB arrays are not bandwidth-bound. As written, a PyTorch reader sees a 1.7 times gain, a JAX reader a 100 times gain, and neither is told why. Fix: before the cell, predict the step from FLOPs and bytes and predict the eager time from the kernel count times $h$; after it, classify (overhead-bound) and re-measure with `mode='reduce-overhead'` in PyTorch, which merges this subsection with "Kernel Replay" into one arc. If the experiment is not changed in this pass, the prose must at least report the speedup in each tab at a robust precision and name the regime.

**COMP-10** (moderate, correctness: claim no cell shows). PyTorch tab after the whole-step cell. Anchor: "captures the step in two graphs".
Nothing printed supports "two graphs", the break at `torch.autograd.grad`, or "AOTAutograd has already derived the backward graph" (wraps a line; grep "already derived the backward graph"). I believe the description is right for current releases, but it is unverified and exactly the kind of behaviour that changes (compiled autograd is moving this boundary). One line of `torch._dynamo.explain(train_step)(net, X, Y)` would show it; exercise 6 already asks the reader to do this for the Adam version. Also "The usual sequence of `zero_grad`,\n`backward`, and an optimizer's `step` compiles as well" is vague: say that it compiles with breaks at `backward`.

**COMP-11** (minor, language). "Compiling a Training Step", JAX tab. Anchor: "The weights and biases are\ndrawn uniformly from" (grep "drawn uniformly from \$\[-1").
Repeats the docstring of `init_mlp` three lines above it. Delete.

**COMP-12** (moderate, pedagogy: experiment read for more than it shows). "Kernel Replay". Anchors: "which the\n4090 completes in a few microseconds" (grep "4090 completes in a few microseconds") and "Dividing each time\nby 120 reads off the two costs per kernel" (grep "by 120 reads off the two costs per kernel").
(a) The roofline gives about 0.1 µs for $8.4 \times 10^6$ FLOPs and under 1 µs for the bytes. The "few microseconds" is a floor per kernel that the model does not contain, the same floor `performance-model.md` attributes to idle SMs and start-up. Say that this is a measured floor, not a roofline number. (b) No speedup is predicted. With $h$ and a floor $k$ per kernel the prediction is $h/k$, here about 18/2.5; state it before the cell. (c) The division by 120 is described but not computed; print it. (d) In the PyTorch tab the comparison is eager modules against `reduce-overhead`, so the gain mixes two things: Inductor removing `nn.Module` and dispatcher overhead, and CUDA graphs removing the launches. A third timing with plain `torch.compile(deep)` would separate them and would show whether the title's mechanism is what produces the factor of seven. Without it, "makes `torch.compile` record the compiled kernels as CUDA graphs and replay them" (grep "as CUDA graphs and replay them") is an attribution the experiment does not support.

**COMP-13** (minor, correctness: unverified). "Kernel Replay", JAX tab. Anchor: "XLA calls them command buffers".
I believe XLA's GPU runtime records command buffers as CUDA graphs and that this is on by default in current releases, but I cannot confirm the default for the pinned JAX. It is a statement about an internal flag and will date. Soften to "current XLA releases" or cut the last two sentences; the tab's point (one call from Python) does not need them.

**COMP-14** (minor, pedagogy: two endings). "Limits of Compilation". Anchor: "Compilation removes memory traffic and launches from a program that fits on\none GPU" (grep "Compilation removes memory traffic and launches from a program").
The first sentence restates the section, and the Summary follows directly. Keep only the handoff ("Compilation leaves the largest consumers of memory where they are ...") or fold it into the Summary's last sentence. The five bullets above it are a list at equal weight; the first and the fourth are consequences already derived ($n^*$, compute-bound code), so the list could shrink to the three that add something (branches, shapes, debugging).

**COMP-15** (minor, language). Title and first subsection. Anchors: "# Compute Graphs and Compilation" and "### Computation Graphs".
"Compute graph" and "computation graph" alternate (title, heading, `fig_compute_graph`); "matrix multiplication" here against "matrix product" in `performance-model.md` and in this section's own exercises. Pick one of each. Units: this section uses MB and GB, its neighbours MiB and GiB.

**Pattern: voice.** The prose is clean and even. Outside the sentences quoted under "What works" there is little judgment: no statement of which of fusion and replay matters more for a real transformer step, or of how often a graph break matters in practice. One sentence of ranking in the introduction (for the GPT of this chapter fusion is worth tenths of the MLP time, replay matters only for small models) would give the section a point of view; `fast-transformer.md` presumably supplies the numbers.

## Code and experiments

- `compilation-capture-two-philosophies-1`: good. The cell ID keeps a "two philosophies" framing that the prose no longer states (see structural finding 3).
- `compilation-capture-two-philosophies-2`: outputs match the tab prose (2 compilations with `s77 : torch.SymInt`; 3 traces). The shared prose does not match the PyTorch output (COMP-4).
- `compilation-what-the-compiler-does-fusion`: PyTorch 1.25 ms and 921 GB/s against 0.15 ms and 880 GB/s, a factor 8.3. JAX 1.69 ms and 682 GB/s against 0.16 ms and 802 GB/s, a factor 10.6. "eight to ten times faster" is at the edge for JAX, and "both\nversions move data at a similar rate, most of the card's" is generous for 682 GB/s; the JAX eager chain also pays dispatch time. Write "eight to eleven" or explain the JAX surplus in a clause. The `assert allclose` is a sensible guard, not a confirm-only cell.
- `compilation-whole-step-compilation-measured`: see COMP-9. The break-even of 582 steps (PyTorch) depends on a warm compile cache; the prose correctly quotes no number, but should say that a cold compile is several times longer.
- `compilation-the-overhead-regime-capture-and-replay`: PyTorch 2.16 against 0.30 ms ("seven to eight times", holds); JAX 8.41 against 0.29 ms ("twenty to thirty times", at the upper edge at 29). These ratios are host-dependent (they are $h$ over a kernel floor) and will move on another CPU more than on another GPU; say so.
- Numbers no cell prints: "two graphs" (COMP-10), the per-kernel costs (COMP-12).
- Timings quoted in prose that are tied to the RTX 4090: 1.15 ms and 0.13 ms (as predictions from $\beta$, fine), "a few microseconds" (COMP-12). The ratios are hardware-independent where bandwidth-bound and should be the quoted quantity.

## Exercises

Eight problems, graded, each with a question. Best in the chapter so far. Notes:

- Exercise 1 (PyTorch): `torch.cond` under `torch.compile` and its capture as one graph by `explain` is unverified and marked prototype in the documentation I know; check it runs on the pinned version.
- Exercise 2 (PyTorch): `torch._dynamo.mark_dynamic` and `torch._logging.set_logs(recompiles=True)` exist as written, to my knowledge. Private names; will date.
- Exercise 4 uses $I^* \approx 165$; `performance-model.md` prints 164.
- Exercise 7 repeats "about 5–15 µs" (COMP-2).
- Exercise 8 is long but every part teaches; keep.
- Nothing to cut. One to add if COMP-9 is deferred: "Predict the time of `train_step` from its FLOPs and bytes, compare with the eager and compiled timings, and name the regime of each."

## Still open from October

The October review's chapter-level items (em dashes, hardcoded section numbers, draft history) do not occur in this file: no em dashes, no hardcoded numbers, no "exactly". Nothing specific to this section was listed.

## For the chapter review

- Order is right: the section cashes the `gelu_ish` cliffhanger of `performance-model.md` and hands memory capacity to `memory-precision.md`. It assumes `d2l.Benchmark` and asynchronous dispatch from the Builders' Guide and $h$, $P$, $\beta$, $I^*$ from the first two sections; it uses $I^*$ less than it could (COMP-6).
- Overlap: the 18-transfer count with `performance-model.md` (COMP-1); the 5–15 µs launch cost with `hardware.md`, quoted in at least four places across the chapter.
- The running example is not the same model in adjacent sections (GELU MLP here, SwiGLU in `memory-precision.md`).
- Suspected chapter-wide pattern: the shared prose is written from the PyTorch output and the tab notes carry what differs, so several shared sentences are false for one tab (COMP-3, COMP-4), and where the two frameworks differ most (a 100 times gain against a 1.7 times gain) the difference is not explained.
- Suspected chapter-wide pattern: the first experiment of a section is predicted from the roofline, later ones are only measured.
- Statements that will date are concentrated in the PyTorch tab notes; a chapter-level decision on how much Dynamo internals to name would help.
