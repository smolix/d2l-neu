# Transformers
:label:`chap_transformers`

Chat models use the attention layer of :numref:`chap_attention`, but
attention only moves information between positions: applied to a single
word, it is a linear map. Each word an assistant writes passes up through
dozens of units of one design, and attention is one part of each. What else
does a unit need? How can a stack that deep be trained? And how can it
write fast enough for us to read along?

The unit is the *transformer block*, introduced in 2017 for machine
translation :cite:`Vaswani.Shazeer.Parmar.ea.2017`. It pairs attention with
a two-layer feed-forward network that transforms each position on its own.
Both add to a residual stream that carries the signal upward, and
normalization layers rescale what each of the two reads. With a few
settings changed, the block now runs language, vision and speech models.

Most settings answer a problem of depth or scale. A deep stack can lose its
signal at initialization or stall in training. Generating one token at a
time repeats work unless the model keeps what it has computed, and then
reading memory, not doing arithmetic, sets the speed. More parameters cost
more computation unless most sit idle for each token. And a run that costs
millions has to be sized before it starts. Other settings are there because
they worked: nobody knows why the gated feed-forward network of nearly
every open model helps.

Two sections are the core. :numref:`sec_transformer-block` builds a block
and measures what moving its normalization does to a deep stack at
initialization. :numref:`sec_gpt` stacks blocks into a language model,
trains it beside the recurrent network of :numref:`chap_rnn` at matched
size on the same bytes, and loads GPT-2 into it. Generation comes next:
caching past keys and values, then shrinking the cache. If you read
anything beyond the core, read these; they explain why a long conversation
costs more per word than a short one.

Without its causal mask, or joined to a second stack, the block becomes an
encoder or an encoder–decoder (:numref:`sec_transformer`), which
:numref:`chap_lm_encoders` builds on; fed image patches, it classifies
images, a section (:numref:`sec_vision-transformer`) that a reader who
wants only language models can skip. Size closes the chapter: the cost of a
run, mixtures of experts, and the settings of released models.

We assume the masked, rotary attention of :numref:`chap_attention` and
AdamW with warmup (:numref:`chap_optimization`). On one RTX 4090 every
notebook runs in under a quarter of an hour. Data, fine-tuning and serving
wait for the Language Models part (:numref:`chap_lm_pretrained` onward).

```toc
:maxdepth: 2

transformer-block
gpt
kv-cache
kv-cache-reduction
encoders-decoders
vision-transformer
scaling-laws
moe
modern-recipe
further-reading
```

## Resources and Further Reading {.unnumbered}

Karpathy's video and the Annotated Transformer are worth an evening each:
the first builds in real time what :numref:`sec_gpt` builds, the second
writes the wiring of :numref:`sec_transformer` line by line beside the
paper. Each entry names the section it goes with, and all are freely
accessible online. Related topics that the chapter does not teach are
collected in :numref:`sec_transformers-further-reading`.

**Tutorials and courses**

- [Let's Build GPT: From Scratch, in Code, Spelled Out](https://www.youtube.com/watch?v=kCc8FmEb1nY), Andrej Karpathy (2023): the video counterpart of :numref:`sec_gpt`, a character-level GPT assembled and trained in real time. [nanoGPT](https://github.com/karpathy/nanoGPT) is the repository form of the same model, [build-nanogpt](https://github.com/karpathy/build-nanogpt) goes with a follow-up video that reproduces GPT-2 small, and [nanochat](https://github.com/karpathy/nanochat) applies the same approach to a full chat system, with its training costs stated in dollars.
- [The Annotated Transformer](https://nlp.seas.harvard.edu/annotated-transformer/), Harvard NLP (2018, updated 2022): the original encoder–decoder, implemented line by line next to the text of the paper; the wiring of :numref:`sec_transformer`.
- [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/), Jay Alammar (2018): the same encoder–decoder drawn one step at a time, from the stacks of encoders and decoders to the masked self-attention of the decoder and the encoder–decoder attention through which it reads the source; read it with :numref:`sec_transformer`.
- [Build a Large Language Model (From Scratch)](https://github.com/rasbt/LLMs-from-scratch), Sebastian Raschka (2024): the free code of a book that covers the ground of :numref:`sec_transformer-block` through :numref:`sec_kv-cache-reduction` at length, with notebooks for grouped-query attention, sliding windows, and from-scratch ports of current open models.
- [Stanford CS336: Language Modeling from Scratch](https://cs336.stanford.edu/): a course whose first assignment builds from primitives what :numref:`sec_transformer-block` and :numref:`sec_gpt` build: RMSNorm, RoPE, SwiGLU, causal attention and the training loop. The lectures are on YouTube.
- [CMU Advanced NLP, minLlama Assignment](https://www.phontron.com/class/anlp-fall2024/): build a Llama-style decoder and load pretrained weights into it, the exercise of loading GPT-2 in :numref:`sec_gpt` at a larger scale.

**Architecture papers**

- [On Layer Normalization in the Transformer Architecture](https://arxiv.org/abs/2002.04745), Xiong et al. (2020): a mean-field analysis of the gradients of post-norm and pre-norm transformers at initialization, which argues that a post-norm model needs a learning-rate warmup. :numref:`subsec_gpt-norm-placement` trains a post-norm model with and without one, and :numref:`sec_transformer-block` measures a different failure of post-norm, the collapse of its tokens.
- [GLU Variants Improve Transformer](https://arxiv.org/abs/2002.05202), Noam Shazeer (2020): a four-page comparison of gated feed-forward networks at matched parameter counts. :numref:`subsec_gpt-run-to-run` runs one of its pairs, SwiGLU against GELU, on a small byte-level model.
- [GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints](https://arxiv.org/abs/2305.13245), Ainslie et al. (2023): grouped-query attention as :numref:`sec_kv-cache-reduction` uses it, including the uptraining recipe that converts multi-head checkpoints.
- [Efficient Streaming Language Models with Attention Sinks](https://arxiv.org/abs/2309.17453), Xiao et al. (2024): why a model that keeps only a window of recent tokens fails unless it also keeps the first few; :numref:`sec_kv-cache-reduction` measures how few on GPT-2.
- [Perceiver IO: A General Architecture for Structured Inputs and Outputs](https://arxiv.org/abs/2107.14795), Jaegle et al. (2022): learned queries that read an input of any size and write an output of any shape, the interface of :numref:`sec_transformer`.
- [An Image Is Worth 16x16 Words](https://arxiv.org/abs/2010.11929), Dosovitskiy et al. (2021): the ViT paper behind :numref:`sec_vision-transformer`. It finds that a transformer, which assumes little about images, trails a convolutional network of similar size when trained on ImageNet alone and matches or beats it when pretrained on far more images; the section asks the same question at the other end of the scale.
- [Switch Transformers](https://arxiv.org/abs/2101.03961), Fedus et al. (2022), and [DeepSeek-V3 Technical Report](https://arxiv.org/abs/2412.19437), DeepSeek-AI (2024): the two ends of the design range in :numref:`sec_moe`, one expert per token balanced by an auxiliary loss, and many small experts balanced by a per-expert bias on the router's scores.
- [The Big LLM Architecture Comparison](https://magazine.sebastianraschka.com/p/the-big-llm-architecture-comparison), Sebastian Raschka (2025, maintained): current open models compared on the same few axes, a maintained version of :numref:`tab_modern-recipe` in :numref:`sec_modern-recipe`.

**The arithmetic of scale**

- [Transformer Inference Arithmetic](https://kipp.ly/transformer-inference-arithmetic/), kipply (2022): the FLOPs of each generated token, the size of the cache, and the bandwidth limit of decoding, worked out for a large model; the accounting of :numref:`sec_kv-cache` at scale.
- [Transformer Math 101](https://blog.eleuther.ai/transformer-math/), EleutherAI (2023): the training-side companion. It derives the rule that training costs about six FLOPs per parameter and token, and what that rule predicts about memory and time; :numref:`sec_scaling-laws` checks the same count against the operations a counter records in one training step.
- [Training Compute-Optimal Large Language Models](https://arxiv.org/abs/2203.15556), Hoffmann et al. (2022): Chinchilla, whose fitted law :numref:`sec_scaling-laws` uses to split a training budget between parameters and tokens.
- [The Ultra-Scale Playbook](https://huggingface.co/spaces/nanotron/ultrascale-playbook), Hugging Face (2025): the parallelism and memory engineering that a training budget beyond one GPU requires, treated in :numref:`chap_performance` and, beyond one node, in :numref:`chap_lm_parallelism`.

<!-- slides -->

::: {.slide}
::: {.cover}
[Dive into Deep Learning · §11]{.kicker}

One block, many models<br>
**the block · GPT · KV cache · encoders and decoders · ViT · scaling laws · experts · the modern recipe**
:::
:::

::: {.slide title="Twelve Copies of One Block Make GPT-2 Small"}
[The problem]{.kicker}

- One block: attention ($4d^2$ parameters) and a feed-forward network
  ($8d^2$), each adding to a residual stream. At width $d = 768$ that is
  $12d^2 \approx 7.1$M.
- Deeper, it can fail before training starts: after 32 post-norm blocks
  with Xavier weights, all tokens agree to about one part in $10^4$.
- Generating, every token reads all the weights and a growing cache: at
  4,096 tokens a GPT-2-sized model reads 288 MiB of cache besides 471 MiB
  of weights.

**Check:** twelve blocks hold 85M parameters, but GPT-2 small has 124M.
Where are the other 39M?

::: {.fragment .fade-up}
::: {.d2l-note}
In the token embedding, $50{,}257 \times 768 \approx 38.6$M, which the
output layer reuses, and in 1,024 learned positions, 0.8M.
:::
:::
:::

::: {.slide title="By the End of This Chapter You Can"}
[Goals]{.kicker}

- build the block, and say when a deep post-norm stack maps all tokens
  toward one vector and what pre-norm changes;
- train a GPT beside a recurrent network of its size, and load GPT-2 into
  it;
- implement a KV cache, estimate the bandwidth of decoding, and shrink the
  cache by sharing key–value heads;
- count parameters and FLOPs, split a compute budget, and route tokens to
  experts;
- wire encoders, encoder–decoders and ViTs, and read a released model as
  settings of the block.

**Idea:** most problems of depth and scale are answered by a setting of
one block. **Running examples:** a 4.8M-parameter byte-level GPT; GPT-2
small (124M, 12 blocks of width 768).
:::

::: {.slide title="The Block and GPT Are the Core; the Cache Comes Next"}
[Scope and place]{.kicker}

- **Assumes:** attention with masks and rotary positions; AdamW with
  warmup.
- **Core:** the block and GPT.
- **Next:** the KV cache and how to shrink it, which explain why a long
  conversation costs more per word.
- **Can wait:** the vision transformer; encoders until the encoder chapter
  of the language-model part.
- **Cost:** one RTX 4090; every notebook under a quarter of an hour.
- **Leaves out:** data, fine-tuning, serving, quantization, kernels,
  parallelism.
:::

::: {.slide title="Build a GPT, Make It Generate, Then Rewire, Scale and Read It"}
[Chapter map]{.kicker}

| | section | question |
|--------|------------------------------|------------------------------------------|
| **build** | The Transformer Block | what does a block need besides attention? |
| | A GPT from Scratch | how do blocks become a trained language model? |
| **generate** | The KV Cache | how do we generate without recomputing? |
| | Shrinking the KV Cache | which factor of the cache can shrink? |
| **rewire** | Encoders, Decoders, and Cross-Attention | which wiring suits which task? |
| | Vision Transformer | can the same blocks classify images? |
| **scale** | Scaling Laws | what does a run cost, and how is a budget split? |
| | Mixture of Experts | how do we store many FFNs and run few? |
| **read** | The Modern Recipe | which settings do released models choose? |
| | Further Reading | what does the chapter leave out? |
:::

::: {.slide title="Every Section Runs on Models That Fit One GPU"}
[Running examples]{.kicker}

| section | example |
|------------------------------|--------------------------------------------------|
| The Transformer Block | 32 random blocks: post-LN and pre-LN with Xavier weights, post-LN at GPT-2's weight scale; one vector through LayerNorm and RMSNorm |
| A GPT from Scratch | a 4.8M-parameter GPT on WikiText-103 bytes; a recurrent network of the same size on the same bytes; GPT-2 small |
| The KV Cache | an untrained GPT-2-sized model: cached decoding, prefill, batched decode |
| Shrinking the KV Cache | the byte GPT with 8 and with 2 key–value heads; GPT-2 small: low-rank keys, sinks |
| Encoders, Decoders, and Cross-Attention | masked bytes; sequence reversal; learned queries |
| Vision Transformer | Fashion-MNIST at $96 \times 96$, 36 patches |
| Scaling Laws | fifteen GPTs: five sizes, three data budgets |
| Mixture of Experts | an 8-expert GPT against a dense one |
| The Modern Recipe | GPT-2, Llama 3 8B and Qwen3-32B as constructor calls at our model's size |
:::
