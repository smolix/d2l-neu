# Further Reading
:label:`sec_ssm-further-reading`

Over the course of this chapter, you have built the recurrent layers that modern long-context models pair with attention, traced their roots back to the LSTM, and measured where a fixed-size state runs out of capacity. Most of this literature is less than five years old, and not all of it will last. This section surveys the extensions and open questions the chapter left out, along with reference implementations and tutorials.

## Recurrence and Attention Together

### Hybrid Models in Practice

:numref:`sec_hybrids` ended on three questions: how few attention layers are enough, where in the stack they belong, and whether sliding-window attention can replace global attention. The matched 8B-parameter comparison of :citet:`Waleffe.Byeon.Riach.ea.2024` mixes 24 Mamba-2 layers with 4 attention layers (about 8% of all layers including MLPs), places a Mamba-2 layer first so no positional embeddings are needed, and spaces the attention layers evenly. At that ratio, the hybrid matched or exceeded a Transformer across all twelve standard benchmarks, whereas pure Mamba models lagged on five-shot MMLU and phone-book lookup.

Production models vary in their exact mix: Qwen3-Next :cite:`Qwen.Team.2025` and Kimi Linear :cite:`Kimi.Team.2025b` use one attention layer in four around delta-rule recurrences (:numref:`sec_deltanet`), Jamba :cite:`Lieber.Lenz.Bata.ea.2024,Jamba.Team.2024` uses one in eight, and Nemotron-H :cite:`NVIDIA.2025` retains Mamba-2 layers. Griffin :cite:`De.Smith.Fernando.ea.2024` drops global attention entirely, interleaving two gated diagonal recurrence layers with one sliding-window attention layer (:numref:`sec_windowed-linear-attention`); as :numref:`sec_hybrids` predicts, exact lookup degrades once the context exceeds the window width.

### Distillation from Transformers

Rather than pretraining a recurrent or hybrid model from scratch, one can convert a pretrained Transformer by initializing matrix-state layers from its attention projections (:numref:`sec_matrix-state`) and distilling. :citet:`Wang.Paliotta.May.ea.2024` retain a subset of attention layers to form a hybrid, while :citet:`Bick.Li.Xing.ea.2024` progressively align attention maps, hidden states, and final outputs using 3B–5B tokens—under 1% of the original pretraining budget.

### Between a Fixed State and a Full Cache

Two designs interpolate between fixed-size state and a full KV cache. Based :cite:`Arora.Eyuboglu.Zhang.ea.2024` pairs a second-order Taylor approximation to $\exp(\mathbf{q}^\top\mathbf{k})$ with exact sliding-window attention, trading state size directly against recall. Log-linear attention :cite:`Guo.Yang.Goel.ea.2025` maintains one matrix state per dyadic history block (doubling in length further into the past), so memory grows as $O(\log T)$ and applies directly to Mamba-2 and Gated DeltaNet.

## Writing to a Matrix Memory

### Expressivity of Fixed-State Models

Some sequential tasks remain out of reach for diagonal or time-invariant linear recurrences regardless of state size. Consider tracking a ball under five cups across a sequence of swaps—composing permutations in the symmetric group $S_5$. :citet:`Merrill.Petty.Sabharwal.2024` prove that state space models with diagonal or fixed transitions (including S4 and Mamba), like Transformers, compute only functions in the constant-depth threshold circuit class $\mathsf{TC}^0$. Composing permutations of five or more elements is complete for $\mathsf{NC}^1$ (logarithmic-depth circuits) and therefore lies outside $\mathsf{TC}^0$ unless $\mathsf{TC}^0 = \mathsf{NC}^1$. (With three or four cups, the symmetric group is solvable and remains inside $\mathsf{TC}^0$.)

Because the delta rule's transition $\mathbf{I} - \beta_t \mathbf{k}_t\mathbf{k}_t^\top$ is input-dependent and non-diagonal, the $\mathsf{TC}^0$ barrier does not rule it out. However, :citet:`Grazzi.Siems.Franke.ea.2025` show that in finite precision, any linear recurrence whose transition eigenvalues lie strictly in $[0, 1]$ cannot compute parity—and therefore cannot track swaps—at arbitrary lengths. Standard `GatedDeltaNet` uses $\beta_t \in (0, 1)$ from a sigmoid and falls under this restriction. Allowing $\beta_t \in (0, 2)$ lets the transition reach negative eigenvalues: at $\beta_t = 2$ with key $(\mathbf{e}_i - \mathbf{e}_j)/\sqrt{2}$, $\mathbf{I} - 2\mathbf{k}_t\mathbf{k}_t^\top$ is a reflection that swaps coordinates $i$ and $j$ (Exercise :exref:`ex_dn-parity`).

### Generalized Delta Rules, Exponential Gates, and Network Memories

- **Generalized delta rules.** RWKV-7 :cite:`Peng.Zhang.Goldstein.ea.2025` unties the delta rule's rank-one projection by decaying each coordinate at its own rate, erasing along one key vector, and writing along a distinct key vector, yielding an asymmetric transition capable of recognizing any regular language. DeltaProduct :cite:`Siems.Carstensen.Zela.ea.2025` composes multiple reflections $\prod_i (\mathbf{I} - \beta_i\mathbf{k}_i\mathbf{k}_i^\top)$ per token so a single step can represent higher-rank permutations.
- **Exponential gates.** A sigmoid input gate caps any single token's write weight at $1$. The mLSTM of xLSTM :cite:`Beck.Poppel.Spanring.ea.2024,Beck.Poppel.Lippe.ea.2025` uses exponential input gates paired with a running normalizer state and the online log-max stabilization of :numref:`sec_attention-at-scale`.
- **Network memories.** TTT :cite:`Sun.Li.Dalal.ea.2024` and Titans :cite:`Behrouz.Zhong.Mirrokni.2025` generalize the online gradient view of :numref:`sec_deltanet` by replacing the matrix $\mathbf{S}_t$ with a linear layer or small MLP updated by per-token gradient steps (with momentum and weight decay in Titans). Whether nonlinear fast weights recall more associations per byte of state than a matrix of equal parameter count remains an open question.

## State Space Models and Long Convolutions

### Memory by Polynomial Projection

:numref:`sec_ssm` initialized S4D's rates as $\lambda_n = -n$. That choice originates in HiPPO :cite:`Gu.Dao.Ermon.ea.2020`, which maintains the coefficients of the best degree-$(N-1)$ orthogonal polynomial approximation to the input history $x(s)$ ($s \le t$). Those coefficients satisfy a linear ODE with a fixed lower-triangular HiPPO matrix $\mathbf{A}$. S4 :cite:`Gu.Goel.Re.2022` made the system time-invariant—corresponding to Legendre polynomials over an exponentially warped history :cite:`Gu.Johnson.Timalsina.ea.2023`—and decomposed $\mathbf{A}$ into a normal matrix plus a low-rank correction for fast kernel evaluation. S4D :cite:`Gu.Gupta.Goel.Re.2022` showed that keeping only the diagonal entries (complex eigenvalues or the real diagonal $-1, \ldots, -N$) recovers nearly the same performance with a fraction of the machinery. S5 :cite:`Smith.Warrington.Linderman.2023` replaced the per-channel scalar SSMs with a single multi-input, multi-output diagonal SSM evaluated via parallel scan (:numref:`subsec_parallel-scans`), and Gu's thesis :cite:`Gu.2023` synthesizes this line of work.

### Successors of Mamba, Direct Linear Recurrences, and Long Convolutions

- **Mamba-2 and Mamba-3.** Mamba-2 :cite:`Dao.Gu.2024` shares a scalar decay across each head to cast training into the matrix-multiply chunked form of :numref:`sec_matrix-state`. Mamba-3 :cite:`Lahoti.Li.Chen.ea.2026` introduces trapezoidal discretization, complex states for rotational tracking, and multi-input updates that raise arithmetic intensity during autoregressive decoding.
- **Linear Recurrent Unit (LRU).** :citet:`Orvieto.Smith.Gu.ea.2023` parameterize a discrete diagonal recurrence directly with complex eigenvalues inside the unit disk and an input normalization $\sqrt{1 - |\lambda|^2}$, matching continuous-time SSMs on the Long Range Arena and motivating the real-valued RG-LRU used in Griffin.
- **Long convolutions.** H3 :cite:`Fu.Dao.Saab.ea.2023` chained a shift SSM with a multiplicative associative SSM, and Hyena :cite:`Poli.Massaroli.Nguyen.ea.2023` replaced the SSM kernel entirely with an implicitly parameterized full-length filter evaluated in $O(T \log T)$ via the FFT. Because general long convolutions lack a compact recurrent state for $O(1)$ decoding, language models have largely favored recurrent SSMs, while long convolutions remain popular in genomics.

## Beyond Language

Outside language modeling, bidirectional scans over image patches power Vision Mamba :cite:`Zhu.Liao.Zhang.ea.2024` and VMamba :cite:`Liu.Tian.Zhao.ea.2024`; multiscale S4 stacks model raw audio waveforms in SaShiMi :cite:`Goel.Gu.Donahue.ea.2022`; and million-nucleotide genomic contexts are modeled with long convolutions in HyenaDNA :cite:`Nguyen.Poli.Faizi.ea.2023` or reverse-complement-equivariant bidirectional Mamba layers in Caduceus :cite:`Schiff.Kao.Gokaslan.ea.2024`.

## Reference Implementations and Tutorials

- **Implementations:** [The Annotated S4](https://srush.github.io/annotated-s4/) (Rush & Karamcheti, 2022) builds full S4 line by line; [state-spaces/mamba](https://github.com/state-spaces/mamba) (Gu & Dao) provides the reference Mamba and Mamba-2 `ssd_minimal` implementation, while [mamba-minimal](https://github.com/johnma2006/mamba-minimal) gives a single-file PyTorch Mamba (:numref:`sec_mamba`); [flash-linear-attention](https://github.com/fla-org/flash-linear-attention) provides Triton kernels for GLA, DeltaNet, and Gated DeltaNet (:numref:`sec_matrix-state` and :numref:`sec_deltanet`); [Gated DeltaNet for Linear Attention](https://github.com/rasbt/LLMs-from-scratch/tree/main/ch04/08_deltanet) (Raschka, 2025) offers a from-scratch tutorial on Gated DeltaNet hybrids; and [zoology](https://github.com/HazyResearch/zoology) provides the synthetic associative recall suite used in :numref:`sec_matrix-state` and :numref:`sec_hybrids`.
- **Expositions and surveys:** [On the Tradeoffs of SSMs and Transformers](https://goombalab.github.io/blog/2025/tradeoffs/) and [State Space Duality (Mamba-2), parts I–IV](https://goombalab.github.io/blog/2024/mamba2-part1-model/) (Gu & Dao, 2024–2025); [A Visual Guide to Mamba and State Space Models](https://newsletter.maartengrootendorst.com/p/a-visual-guide-to-mamba-and-state) (Grootendorst, 2024); [DeltaNet Explained, parts I–III](https://sustcsonglin.github.io/blog/2024/deltanet-1/) (Yang, 2024); and [Test-Time Regression](https://arxiv.org/abs/2501.12352) (Wang, Shi & Fox, 2025) for the unified regression view of memory at the end of :numref:`sec_deltanet`.

[Discussions](https://d2l.discourse.group/t/36786)
