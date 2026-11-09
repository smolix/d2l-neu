# Further Reading
:label:`sec_lm_compression_fr`

<!-- Author brief, copied verbatim from the Language Models part plan
(docs/llm/language_models_part_plan.md, "FR Further Reading: Compression · T0").
Blueprint: none (the plan adds this section).
The plan is a specification, not book text; this comment is replaced
when the section is written.

**Idea:** The compression methods outside the teachable core, each named with the one thing it adds to what the chapter taught.
**Opening:** A reader who has rescaled, reconstructed, rotated, retrained and pruned a model wants to know what lies below three bits, what can be compressed without any loss, how pruning is done at the scale of a model family, and how the bits-against-parameters question is settled across sizes. The closing question is which of these a new hardware format will make obsolete.
**Teaches:** Codebooks and lattices (QuIP#, AQLM): what a vector codebook buys below three bits against the scalar grids of 11.1, with the rate-distortion gap of 11.4's exercise as the bound, and what it costs in codebook memory, lookups and kernels; one short cell runs `d2l.kmeans_vq` on one rotated `down_proj` of Qwen3-1.7B against the scalar grid at equal bits. Lossless exponent coding (DFloat11, ZipNN): BF16 exponents carry about three bits of entropy, so a Huffman code over exponents saves about 30% exactly; one short cell encodes and decodes a 4-million-weight slice bit for bit with `d2l.df11_encode` and `d2l.df11_decode` and prints the whole-model ratio from exponent histograms; the decompression cost at batch 1 is named, not measured. Structured pruning plus distillation at scale (Minitron): the recipe that produces a model family from one parent with a small fraction of the training tokens, as the scale version of 11.6's Part C. The k-bit inference scaling laws: across sizes and families, four bits is close to optimal at a fixed total number of bits, the reading behind 11.1's fixed-memory sweep. Quantization-aware training recipes at scale (LLM-QAT, ParetoQ) and the two-to-three-bit transition, as what 11.5's two-minute budget could not test. Ternary models trained with the quantizer in the loop (BitNet b1.58) and their integer-only inference kernels.
**Derivation:** none.
**Notebook:** none (reading, with the two short cells named above; no training run).
**Figures:** none.
**Restate / forward:** 11.4's rotation and rate-distortion exercise; 11.6's pruning granularities. Forward: none.
**Exercises:** none.
**Writer notes:** Every citation carries a clause saying what it adds; no survey of method names. The two cells reuse 11.4's `d2l.kmeans_vq` and the exponent-coding helpers listed in 11.6's notes, both defined here with `#@save` since no section owns them any more. The DFloat11 production comparison (`DFloat11/Qwen3-4B-DF11` against `Qwen/Qwen3-4B`, greedy outputs equal on eight prompts) is optional and gated by the `dfloat11` wheel; lossless is about the weights, and bit-identical outputs additionally need the same kernels. Keep to 1,000 to 2,000 words.
**Sources:**
- Papers: [QuIP#](https://arxiv.org/abs/2402.04396) (Hadamard incoherence and lattice codebooks); [Extreme Compression of Large Language Models via Additive Quantization (AQLM)](https://arxiv.org/abs/2401.06118) (additive multi-codebook quantization); [70% Size, 100% Accuracy: Lossless LLM Compression for Efficient GPU Inference via Dynamic-Length Float (DFloat11)](https://arxiv.org/abs/2504.11651) (exponent entropy and exact reconstruction); [ZipNN: Lossless Compression for AI Models](https://arxiv.org/abs/2411.05239) (lossless coding of model files); [Minitron](https://arxiv.org/abs/2407.14679) (pruning and distillation at scale); [k-bit Inference Scaling Laws](https://arxiv.org/abs/2212.09720); [LLM-QAT](https://arxiv.org/abs/2305.17888); [ParetoQ](https://arxiv.org/abs/2502.02631); [BitNet b1.58](https://arxiv.org/abs/2402.17764).
- Code and documentation: [quip-sharp](https://github.com/Cornell-RelaxML/quip-sharp); [AQLM](https://github.com/Vahe1994/AQLM); [DFloat11](https://github.com/LeanModels/DFloat11); [Minitron](https://github.com/NVlabs/Minitron); [BitNet](https://github.com/microsoft/BitNet).
- Courses, tutorials and blogs: Smola, [Efficiency in LLMs](https://alex.smola.org/posts/45-mlss-efficiency/) (lossless coding against lossy quantization); [MIT 6.5940](https://hanlab.mit.edu/courses/2026-fall-65940) (the advanced-quantization and distillation lectures).
-->
