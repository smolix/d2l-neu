# Model Serving
:label:`sec_model_serving`

The language models in this book generated text in a notebook, one prompt
at a time, for the person who ran the cell. Suppose instead that a hundred
colleagues send requests to the same language model whenever they like. If
the GPU answers them one after another at two seconds each, the last
colleague waits more than three minutes. What has to change between calling
a model and serving it?

Serving makes one accelerator work on many requests at once. Generating a
token reads the weights and the key--value cache from memory
(:numref:`sec_hardware_buyers`). All requests share the weights, so an
engine that advances many requests in the same step reads the weights once
for all of them, provided memory holds the cache of every active request. We
start with the latency numbers that users notice, survey local and server
engines as of November 2026, and explain the techniques that make the
engines fast.

## Workloads and Latency Metrics

A user of a chat assistant notices two delays: the wait before the first
word appears, and the pace of the words that follow. The first is the
**time to first token** (TTFT), counted from the arrival of the request. On
an idle server it is dominated by *prefill*, which processes all prompt
tokens in parallel and is bound by compute. The second is the **time per
output token** (TPOT), dominated by *decode*, which produces one token per
sequence and step and is bound by memory bandwidth. Throughput, the tokens
per second summed over all users, trades against both.

![Prompt prefill is parallel and often compute-heavy; autoregressive decode advances one token per sequence and stresses memory movement.](../img/tools-prefill-decode.svg)
:label:`fig_tools_prefill_decode`

Which numbers matter depends on the workload, and the workload picks the
tool:

:Serving workloads and example starting points (as of November 2026)
:label:`tab_serving_workloads`

| Workload | Optimize for | Start with |
|---|---|---|
| Personal assistant on your machine | simplicity, privacy | Ollama / LM Studio (llama.cpp), MLX on Macs |
| Offline batch scoring | throughput per dollar | vLLM offline mode |
| One application's API | operability | vLLM behind a small authenticated proxy |
| Multi-user / agentic service | goodput under an SLO | vLLM or SGLang, then NVIDIA Dynamo at scale |
| Max throughput on fixed NVIDIA fleet | throughput per GPU | TensorRT-LLM |

A service level objective (SLO) sets a latency target, such as a TTFT below
1 s. *Goodput*, the rate of completed requests that also met their target,
is the production metric. Overload can raise raw throughput while goodput
collapses, because requests queue until they miss their targets.

## Local Serving
:label:`subsec_model-serving-local`

On a laptop or a workstation with one user, the questions are whether a
model fits and how fast it answers. Many local inference tools use
**llama.cpp** and its **GGUF** file format, and community repositories
commonly publish GGUF conversions of open-weight models. Users often reach
llama.cpp through a wrapper:

* **Ollama** builds on llama.cpp and, on Apple silicon, also on MLX.
  `ollama run qwen3:8b` downloads, caches, and serves a model, with an
  OpenAI-compatible API at `http://localhost:11434/v1`.
* **LM Studio** offers the same engines behind a graphical interface, with
  control over how much of the model is offloaded to the GPU.
* **MLX** is Apple's array framework. Its package `mlx-lm` runs and
  fine-tunes models on Apple silicon, and the `mlx-community` organization
  on Hugging Face publishes thousands of converted models.

Common GGUF quantizations include `Q4_K_M` (about 4.9 bits per weight),
`Q5_K_M` (about 5.7), and `Q8_0` (about 8.5). For the 8.2 billion
parameters of Qwen3-8B, `Q4_K_M` gives $8.2 \times 10^9 \times 4.9 / 8
\approx 5.0$ GB, close to the 5.2 GB of Ollama's `qwen3:8b` download. The
4.9 bits were measured on Llama 3.1 8B, and `Q4_K_M` keeps a
model-dependent share of tensors at higher precision. The cache of a full
32,768-token context in bf16 adds almost as much again. Higher-bit formats
usually retain quality better; measure the effect on your own task. Decode
speed is at most the memory bandwidth divided by the bytes read per token,
so every extra bit costs speed as well as memory. A common rule is to pick
the largest model whose quantized weights fit in memory with room for the
cache, then the highest-bit format that still fits.

## Server Engines

### Open-Source and NVIDIA Engines

A shared server needs an engine that schedules requests together. For an
open-source GPU server, **vLLM** provides continuous batching, paged
KV-cache management, prefix caching, tensor parallelism, quantized-model
support, and an OpenAI-compatible server in one command:

```bash
vllm serve Qwen/Qwen3-8B --max-model-len 32768
```

**SGLang** is its closest peer. Its *RadixAttention*
:cite:`Zheng.Yin.Xie.ea.2024` keeps cached keys and values in a radix tree
shared across requests, which helps when many requests share long prefixes:
chat with a system prompt, retrieval over the same documents, or trees of
agent calls. Both projects evolve quickly and support several accelerator
backends; benchmark both on your distributions of prompt length and
concurrency before selecting one.

NVIDIA's tools trade portability for hardware-specific optimization.
**TensorRT-LLM** runs models on PyTorch with NVIDIA-tuned kernels, which can
improve steady-state throughput at the cost of more setup. **Dynamo**
coordinates many GPUs and nodes, with vLLM, SGLang, or TensorRT-LLM as
backends. It runs prefill and decode on separate pools of GPUs and routes
each request by worker load and by how much of its prefix a worker already
caches. These tools matter at datacenter scale; on a few GPUs, vLLM or
SGLang alone is simpler.

### A Common Client API
:label:`subsec_model-serving-contract`

An application should not need rewriting when it moves from a laptop to a
server. Many of these tools implement the OpenAI chat-completions API, so
the same client code targets a local Ollama, your vLLM server, or a
commercial provider after a change of the base URL and the model name:

```text
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:8000/v1", api_key="local")
response = client.chat.completions.create(
    model="Qwen/Qwen3-8B",
    messages=[{"role": "user", "content": "Explain KV caching briefly."}],
    temperature=0,
)
print(response.choices[0].message.content)
```

Servers that accept the same requests can still answer them differently:
tokenization, sampling defaults, structured-output support, streaming
behavior, and token accounting all vary. Keep a small conformance test and
rerun it before swapping engines.

## Engine Techniques

Four composable techniques explain much of the speed of these engines.

**Continuous batching.** A static batch waits for its slowest member.
Continuous batching admits new requests and retires finished ones between
decode steps, keeping the batch nearly full:

![Two batch slots serving requests of 5, 2, 7, and 3 output tokens, as in the toy scheduler below. Static batching pads each batch to its longest member; continuous batching refills a finished slot at the next step and finishes in 9 steps instead of 12.](../img/tools-continuous-batching.svg)
:label:`fig_tools_continuous_batching`

The toy scheduler below captures the idea in a few lines.
Exercise :exref:`ex_model-serving-arrivals` extends it with arrival times and
a memory budget:

```{.python .input #model-serving-scheduler}
from collections import deque

requests = deque([5, 2, 7, 3])  # output tokens requested
capacity = 2
active, timeline = [], []
while requests or active:
    while requests and len(active) < capacity:
        active.append(requests.popleft())
    timeline.append(tuple(active))
    active = [remaining - 1 for remaining in active if remaining > 1]

timeline
```

**Paged KV cache.** Each active sequence's key--value cache grows with its
length. vLLM's PagedAttention :cite:`Kwon.Li.Zhuang.ea.2023` allocates it in
fixed-size blocks, like virtual memory, and so nearly eliminates the
fragmentation that once capped batch sizes. Capacity planning follows: GPU
memory must hold the weights *plus* the cache of every concurrent sequence.
A model that fits for one request may therefore exceed memory under high
concurrency.

**Prefix caching.** When two requests share a tokenized prefix (the same
system prompt, the same document), its cached keys and values can be
computed once and reused:

![A prefix cache reuses KV state only when tokenized prefixes match, then computes each request's unique suffix.](../img/tools-prefix-cache.svg)
:label:`fig_tools_prefix_cache`

Some commercial APIs charge less for cached input tokens, under rules that
differ by provider. Placing shared, stable content before the variable part
of a prompt then saves money as well as prefill time.

**Speculative decoding and quantization.** A small draft model proposes
several tokens, and the target model verifies them in one parallel pass.
This speeds up decoding when most proposals are accepted and preserves the
target distribution. Quantization reduces the bytes read during decode:
fewer bytes per weight give more tokens per second, until quality degrades
at the lowest bit widths. Recent NVIDIA tensor cores support FP8 (from the
Hopper and Ada generations on) and NVFP4 (from Blackwell on), and vLLM and
SGLang also load 4-bit AWQ weights. GGUF formats play the same role
locally.

## Operations

An engine listening on a port still lacks the controls that a shared
service needs. Put authentication, TLS, and rate limits in a proxy in front
of the engine, and do not expose the engine's own port. Admission control should reject or queue
requests *before* memory runs out, not after an out-of-memory error.
Propagate client cancellations so that abandoned generations stop consuming
compute. Measure TTFT, TPOT, and end-to-end latency as p50 and p95 under
realistic arrival patterns, not as averages under a closed loop of clients.
Avoid logging raw prompts by default. Pin the model, tokenizer,
quantization, and engine versions together: a faster server that answers
differently is a different system, so gate rollouts on task quality as well
as on latency.

You can now estimate whether a model and the caches of its users fit on a
card, and which engine to try first. How a server should admit and order
requests under load remains open; the simulators of the exercises explore
it.

## Summary

Serving latency splits into compute-bound prefill, which sets the time to first token (TTFT), and bandwidth-bound decode, which sets the time per output token (TPOT); choose engines to match the workload, from Ollama, LM Studio, or MLX locally to vLLM or SGLang for shared services and TensorRT-LLM with Dynamo across large NVIDIA fleets. Quantization in GGUF formats locally and AWQ, FP8, or NVFP4 on servers turns saved weight bytes directly into faster token generation until quality degrades at the lowest bit widths. Modern engines combine continuous batching, paged KV caches, prefix caching, and speculative decoding, while client prompt ordering determines how much prefix state can be reused. Although the OpenAI-compatible chat-completions API keeps client code portable across local and cloud servers, subtle behavioral differences still warrant a small conformance test before switching engines. Finally, memory capacity must hold the model weights alongside every concurrent user's KV cache, and production deployments require admission control, client cancellation propagation, tail-latency monitoring, and co-pinned model and engine versions.

## Exercises

1. **Batch size limits.** The command `vllm serve Qwen/Qwen3-8B` serves a
   model with $L = 36$ layers and $H_\textrm{kv} = 8$ key--value heads of
   dimension $d_h = 128$, whose bf16 weights occupy $B_w = 16.4$ GB. Its
   cache is stored in bf16 as well. Serve it on an H100 SXM, whose memory bandwidth is $\beta = 3.35$ TB/s and whose dense
   bf16 rate is 989 TFLOP/s (:numref:`tab_gpu_specs`).
   A decode step reads the weights once for the whole batch and the cache
   of every active sequence. For $b$ sequences that each hold $n$ cached
   tokens, the step, and hence the time per output token of every
   sequence, takes about

   $$t(b) = \frac{B_w + b\, n\, B_\textrm{kv}}{\beta},$$

   where $B_\textrm{kv}$ is the cache size per token.
    1. Every layer stores one key and one value vector per key--value head
       and token, so $B_\textrm{kv} = 2 \cdot L \cdot H_\textrm{kv} \cdot
       d_h \cdot (\textrm{bytes per element})$ (:numref:`sec_kv-cache`).
       Compute $B_\textrm{kv}$. Evaluate the TPOT and the throughput
       $b / t(b)$ at $n = 2048$ for
       $b \in \{1, 8, 64, 256\}$. Which value does the throughput approach
       as $b$ grows, and which term of $t(b)$ sets it? Count 2 FLOPs per
       parameter and token for the 8.2 billion parameters, and check with
       the bf16 rate that the step at $b = 256$ is still bound by memory
       traffic.
    1. An interactive service promises a TPOT of at most 20 ms. What is
       the largest batch it can run? Which batch would an offline scoring
       job of :numref:`tab_serving_workloads` choose instead, and why?
    1. Let the engine use 72 GB of the card for weights and cache, and let
       the sequences hold 2,048 tokens on average. Before paged caches,
       servers reserved one contiguous region per request for its maximum
       length. How many sequences fit when each one reserves the
       `--max-model-len` of 32,768 tokens? What fraction of the reserved
       cache then holds token states? :citet:`Kwon.Li.Zhuang.ea.2023`
       measured 20.4% to 38.2% in such systems. Why is your fraction lower?
       How many sequences fit when the cache is allocated in blocks of 16
       tokens? Bound the memory that partly filled blocks waste.
    1. Which limit sets the batch of the interactive service under each of
       the two allocation schemes: memory or the TPOT target?
   :exlabel:`ex_model-serving-batch-limits`

1. [code] **Conformance test.** :numref:`subsec_model-serving-contract`
   recommends a small conformance test before swapping engines.
    1. Write a function `conformance(base_url, model)` with the `openai`
       client of :numref:`subsec_model-serving-contract` that runs four
       checks and reports which of them pass. Every check also requires a
       nonempty `content`. First, two identical requests
       at temperature 0 return the same text. Second, a request with
       `max_tokens=8` ends with `finish_reason` equal to `"length"` and
       reports at most 8 completion tokens in `usage`. Third, a request that
       asks the model to count from 1 to 20 with `stop=["7"]` returns text
       without the digit 7. Fourth, a streamed request at temperature 0 with
       `stream_options={"include_usage": True}` yields chunks whose
       concatenated content and final `usage` equal those of the same
       request without streaming. Run the test against Ollama with a small
       model such as `qwen3:0.6b`, which runs on a CPU, and against a second
       OpenAI-compatible server of your choice. Which checks fail, and why?
    1. On a vLLM server, the first check passes while the server is idle
       and fails occasionally while other clients send requests to it. No
       sampling takes place at temperature 0. Explain the failure.
    1. Both servers load the same model with the same tokenizer and
       receive the same messages, yet they report values of
       `usage.prompt_tokens` that differ by a few tokens. Name two possible
       causes. Why does such a difference matter to a client beyond
       billing?

1. [code] **Prefix cache for an agent.** A coding agent calls a model 30
   times. The first prompt holds a system prompt and repository context of
   20,000 tokens. Every later prompt repeats the previous prompt and
   appends 1,000 tokens: the model's previous answer and a tool result.
    1. How many prompt tokens must the engine prefill over the 30 calls
       without prefix caching, and how many with a cache that keeps every
       earlier prompt? A provider charges a tenth of its input price for
       cached tokens. What fraction of the uncached input bill remains?
    1. vLLM caches only full blocks, and it keys each block by a hash of
       the key of the preceding block and of the block's own token ids.
       Implement this scheme with blocks of 16 tokens as a function
       `prefill_tokens(cache, tokens)` that returns the number of tokens to
       compute and adds the new full blocks to the set `cache`. Simulate the
       agent with random token ids, and compare the result with your count
       from the first sub-problem. Why does the key of a block include the
       key of the preceding block?
    1. The agent framework writes the current time, to the second, into
       the first line of the system prompt. Rerun the simulation. Where
       should the timestamp go instead?
    1. A retrieval application places three of 50 documents after a shared
       system prompt, in order of relevance. Two requests retrieve the same
       three documents in different orders. Which blocks does the second
       request reuse? Propose an ordering that raises reuse, and state what
       it gives up.

1. [code] **Arrivals and a memory budget.** ● Extend the cell that builds
   `timeline` into a simulator in which time advances in decode steps.
   Request $i$ arrives at step $a_i$, has a prompt of $P_i$ tokens, and
   generates $T_i$ tokens, one per step from the step of its admission on.
   The server runs at most 64 sequences and holds at most 32,768 tokens of
   cache, and a sequence holds its prompt plus the tokens generated so far.
   Draw the arrivals from a Poisson process with $\lambda$ requests per
   step, draw $P_i$ uniformly from 100 to 1,000, and let $T_i$ be one plus
   an exponential variable with mean 200, rounded down and capped at
   `max_tokens = 1024`. Measure the time to first token in steps, from the
   arrival to the first generated token.
    1. Admit requests first come, first served when the cache can hold the
       prompt plus `max_tokens`, the most that a request can ever need.
       Reject an arriving request when 50 requests already wait. For
       $\lambda \in \{0.05, 0.1, 0.15, 0.2\}$, report the tokens generated
       per step, the p50 and p95 TTFT, and the rejected fraction.
    1. Admit a request instead whenever the cache can hold its prompt plus
       one token. When the active sequences outgrow the cache, preempt the
       most recently admitted one, free its cache, and put it back at the
       head of the queue, so that it recomputes its cache on readmission.
       Repeat the measurements. The simulator charges no time for prefill.
       Which resource does recomputation spend on a real server, and which
       would swapping the cache to CPU memory spend instead? How would
       charging for prefill change your comparison with the first
       sub-problem?
    1. Order the queue by the number of remaining tokens, shortest first,
       and run both admission rules at $\lambda = 0.2$. Compare this order
       with first come, first served on the p95 TTFT of all requests and on
       that of the 5% of requests with the longest outputs.
    1. A server knows `max_tokens` but not $T_i$. What could it order the
       queue by instead, and how much of the gain of the third sub-problem
       would you expect it to keep?
   :exlabel:`ex_model-serving-arrivals`

1. **Speculative sampling.** ● In the speculative decoding of
   :citet:`Leviathan.Kalman.Matias.2023`, a draft model with next-token
   distribution $q$ proposes a token $x \sim q$, and the target model,
   with next-token distribution $p$, accepts it with probability
   $\min(1, p(x) / q(x))$. After a rejection, the token is drawn instead
   from the residual distribution

   $$p'(x) = \frac{\max(0, p(x) - q(x))}{\sum_{x'} \max(0, p(x') - q(x'))}.$$

    1. Show that the emitted token is distributed according to $p$, and
       that the probability of acceptance is
       $\beta = \sum_x \min(p(x), q(x))$.
    1. In every round, the draft proposes $\gamma$ tokens one after
       another, and one pass of the target scores all of them. The target
       emits the accepted tokens up to the first rejection and then one
       token of its own, drawn from $p'$ after a rejection or from $p$ when
       all $\gamma$ tokens were accepted. Let $\alpha$ be the mean of the
       acceptance probability $\beta$ of the first sub-problem over the
       positions of a sequence, and assume that every draft token is
       accepted independently with probability $\alpha$. Show that a round
       emits $(1 - \alpha^{\gamma + 1}) / (1 - \alpha)$ tokens in
       expectation. A draft step takes a fraction $c$ of the time of a
       target step, and the scoring pass takes one target step. Derive the
       expected speedup over plain decoding, and find the best $\gamma$ for
       $c = 0.05$ and $\alpha \in \{0.6, 0.8, 0.9\}$.
    1. The scoring pass takes one target step only while it is bound by
       memory traffic. A server decodes $b$ sequences together, and each
       sequence scores $\gamma + 1$ positions per round. Ignore the cache,
       store the weights in bf16, and count 2 FLOPs per parameter and token.
       Above which $b$ does the pass become compute-bound on the H100 of
       exercise :exref:`ex_model-serving-batch-limits` for $\gamma = 4$? What does this imply for
       speculative decoding on a busy server compared with a single user?

1. [extended] **Goodput under load.** ●● Turn your simulator of exercise
   :exref:`ex_model-serving-arrivals` into one whose clock runs in seconds
   instead of decode steps, for
   Qwen3-8B on the H100 of exercise :exref:`ex_model-serving-batch-limits`
   with 72 GB for weights and a paged cache. A step processes $p$ prompt
   tokens and one token for each of $b$ decoding sequences, which together
   hold $K$ cached tokens. With $N$ parameters, the dense bf16 rate $F$,
   and the notation of exercise :exref:`ex_model-serving-batch-limits`,
   the step takes

   $$t_\textrm{step} = \max\left(\frac{2N(p + b)}{F}, \frac{B_w + K B_\textrm{kv}}{\beta}\right),$$

   which ignores the FLOPs of the attention scores. Requests arrive as a
   Poisson process at $\lambda$ requests per second, with prompts uniform
   between 500 and 4,000 tokens, and output lengths of one plus an
   exponential variable with mean 250, rounded down and capped at 2,048. A
   request counts toward goodput when its TTFT is at most 1 s and its
   TPOT, the mean time between its output tokens, is at most 50 ms.
    1. Prefill every admitted prompt in full within one step. Estimate the
       saturation rate from $t_\textrm{step}$ before you simulate. Then
       sweep $\lambda$ from light load to well beyond saturation, and plot
       the throughput and the goodput in requests per second. At which rate
       does goodput peak, and how does it compare with your estimate?
    1. A full prefill of a long prompt delays the next token of every
       decoding sequence. Implement chunked prefill
       :cite:`Agrawal.Kedia.Panwar.ea.2024`, which caps the tokens of a
       step, decode tokens included, at a budget and spreads a prompt over
       several steps. Repeat the sweep for budgets of 512, 2,048, and 8,192
       tokens. Which budget maximizes goodput? How does the answer depend
       on the two SLOs?
    1. Add admission control that rejects a request as soon as its TTFT
       target can no longer be met. How does it change goodput beyond
       saturation? Explain why goodput collapses without it while
       throughput stays high.
    1. Let every client abandon its request after 10 s and send it once
       more. Compare goodput when the server keeps generating abandoned
       requests and when it cancels them.
    1. State the largest request rate at which at least 90% of all
       arriving requests meet both SLOs, counting rejected requests as
       misses, and the configuration that achieves it.

[Discussions](https://d2l.discourse.group/t/36847)

<!-- slides -->

::: {.slide}
::: {.cover}
[Dive into Deep Learning · §39.7]{.kicker}

Model Serving: Prefill vs. Decode, Continuous Batching, PagedAttention, and SLOs<br>**TTFT & TPOT · goodput · GGUF & AWQ/FP8 · vLLM & SGLang · continuous batching · PagedAttention · prefix caching · speculative decoding**
:::
:::

::: {.slide title="From Single-User Notebooks to Multi-User Serving"}
[Motivation]{.kicker}

::: {.cols .vc}
::: {.col}
- **Why serial serving collapses:** Running 100 two-second requests one after another makes the 100th user wait **over 3 minutes** while streaming weights 100 separate times ($98\%$ ALU idle).
- **Why dynamic batching transforms decode:** Advancing $b$ concurrent users together reads the $16.4\text{ GB}$ model weights **once per step** for all $b$ users, multiplying throughput $30\text{--}50\times$.
:::

::: {.col .fig .big}
@fig:ch39-tools-serial-vs-batched-serving
:::
:::
:::

::: {.slide title="Two Phases of Generation: Prefill vs. Decode"}
[Workload Anatomy]{.kicker}

Every generative request executes two physically distinct computational regimes that map directly to the two latency metrics users feel:

::: {.cols .vc}
::: {.col}
- **1. Prompt Prefill $\to$ Time to First Token (TTFT):**
  - Processes all $p$ prompt tokens in parallel to populate the initial KV cache.
  - High arithmetic intensity (**compute-bound**); sets the wait before the first word appears.
- **2. Autoregressive Decode $\to$ Time per Output Token (TPOT):**
  - Generates one new token per active sequence per step.
  - Low arithmetic intensity (**memory-bandwidth bound**); sets the streaming reading pace ($\text{tok/s} = 1/\text{TPOT}$).
:::

::: {.col .fig}
![](../img/tools-prefill-decode.svg){.nostretch width=100%}
:::
:::
:::

::: {.slide title="Latency SLOs, Raw Throughput, and Goodput"}
[Production Metrics]{.kicker}

::: {.cols .vc}
::: {.col}
- **Service Level Objectives (SLOs):** Bound both Time to First Token (e.g., $\text{TTFT} \le 1.0\text{ s}$) and Time per Output Token ($\text{TPOT} \le 50\text{ ms}$, or $\ge 20\text{ tok/s}$ per user).
- **Goodput vs. Raw Throughput:** Under overload, queue depth explodes so **goodput (SLO-compliant req/s) collapses toward zero** even while raw token/s stays pinned near $100\%$ peak.
:::

::: {.col .fig .big}
@fig:ch39-tools-throughput-vs-goodput
:::
:::
:::

::: {.slide title="Matching the Serving Stack to the Workload (Nov 2026)"}
[Engine Landscape]{.kicker}

::: {.cols .vc}
::: {.col}
- **Local & Offline Batch:** Use **Ollama / LM Studio** (`llama.cpp` GGUF) or **MLX** for local single-user privacy, and **vLLM offline mode** to maximize tokens/dollar on dataset scoring.
- **Online APIs & Datacenter Fleets:** Use **vLLM** or **SGLang** (RadixAttention prefix tree) for low-latency multi-user services, and **NVIDIA Dynamo + TensorRT-LLM** for disaggregated multi-node fleets.
:::

::: {.col .fig .big}
@fig:ch39-tools-serving-stack-selector
:::
:::
:::

::: {.slide title="Local Serving Engines and GGUF Quantization"}
[Single-User Inference]{.kicker}

::: {.cols .vc}
::: {.col}
- **Qwen3-8B ($8.2\text{B}$) GGUF footprints:** `Q4_K_M` ($\sim 4.9\text{ bits/w}$) takes $\sim 5.2\text{ GB}$, `Q5_K_M` takes $\sim 5.8\text{ GB}$, and `Q8_0` takes $\sim 8.7\text{ GB}$.
- **Capacity + speed + KV cache:** 4-bit weights nearly double batch-1 decode speed over 8-bit, and leave room on a 16 GB card for a 32K BF16 KV cache ($\sim 4.8\text{ GB}$).
:::

::: {.col .fig .big}
@fig:ch39-tools-gguf-memory-budget
:::
:::
:::

::: {.slide title="Batched Decode Arithmetic and the TPOT Ceiling"}
[Multi-User Roofline]{.kicker}

::: {.cols .vc}
::: {.col}
$$\text{TPOT}(b) \approx \frac{B_w + b \, n \, B_{\text{kv}}}{\beta}, \qquad \text{Throughput}(b) = \frac{b}{\text{TPOT}(b)}$$

- **Qwen3-8B (BF16) on H100 ($\beta = 3.35\text{ TB/s}$, $n = 2{,}048$):** Scaling $b$ from $1 \to 64$ raises TPOT by only $2.1\times$ ($5.0 \to 10.7\text{ ms}$) while multiplying throughput **$30\times$** ($201 \to 6{,}000\text{ tok/s}$).
- **SLO & KV ceiling:** A $20\text{ ms}$ TPOT SLO caps batch size at $b_{\max} = 167$ ($8{,}350\text{ tok/s}$), approaching the KV-cache ceiling $\beta/(n B_{\text{kv}}) \approx 11{,}090\text{ tok/s}$.
:::

::: {.col .fig}
@fig:ch39-tools-batched-tpot-scaling
:::
:::
:::

::: {.slide title="Server Engines: vLLM, SGLang, TensorRT-LLM, Dynamo"}
[Multi-User Architectures]{.kicker}

::: {.cols .vc}
::: {.col}
- **vLLM & SGLang:** vLLM standardizes PagedAttention, continuous batching, and OpenAI-compatible serving (`vllm serve`); SGLang's **RadixAttention** stores KV caches in a radix tree for multi-turn agents.
- **TensorRT-LLM & NVIDIA Dynamo:** TensorRT-LLM compiles hardware-tuned FP8/NVFP4 kernels, while Dynamo disaggregates **prefill and decode onto separate GPU pools** with prefix-aware routing.
:::

::: {.col .fig .big}
@fig:ch39-tools-server-engine-arch
:::
:::
:::

::: {.slide title="The OpenAI API Contract and Conformance Testing"}
[Portable Client Code]{.kicker}

Because local and server engines implement the OpenAI Chat Completions schema, client applications switch from a laptop Ollama to a cloud vLLM cluster by changing only `base_url` and `model`:

::: {.cols}
::: {.col}
```{.python #model-serving-exercises}
from openai import OpenAI

client = OpenAI(
    base_url="http://127.0.0.1:8000/v1",
    api_key="local",
)
resp = client.chat.completions.create(
    model="Qwen/Qwen3-8B",
    messages=[{"role": "user", "content": "Hi"}],
    temperature=0,
)
```
:::

::: {.col}
- **Why identical API schemas still diverge in behavior:**
  - Different chat-template Jinja whitespace/BOS tokens shift `usage.prompt_tokens` and consume context budgets differently.
  - Streaming usage chunks (`stream_options`), `stop` string truncation, and structured-output grammar engines vary by server.
  - **Batch-size non-determinism:** Even at `temperature=0`, a busy vLLM server groups requests into varying batch shapes, changing floating-point reduction order enough to flip close argmax ties.
:::
:::
:::

::: {.slide title="Continuous Batching: Eliminating Static Padding Waste"}
[Engine Technique 1]{.kicker}

Static batching locks $b$ requests together until the longest generation finishes, wasting GPU slots whenever shorter responses finish early:

::: {.cols .vc}
::: {.col}
- **Iteration-level scheduling (Continuous Batching):**
  - At the end of *every single decode step*, the engine checks for sequences that emitted `<eos>` or hit `max_tokens`.
  - Finished sequences exit immediately, freeing their KV-cache blocks, and waiting requests from the queue enter the batch on the very next step.
- **Result on requests of `[5, 2, 7, 3]` output tokens across 2 slots:** Continuous batching finishes all four requests in **9 decode steps instead of 12** ($25\%$ fewer steps; slot utilization rises from $71\%$ to $94\%$).
:::

::: {.col .fig}
![](../img/tools-continuous-batching.svg){.nostretch width=100%}
:::
:::
:::

::: {.slide title="Simulating a Continuous Batching Scheduler in Code"}
[Executable Scheduler]{.kicker}

Ten lines of Python capture the core state machine of an iteration-level continuous batching loop:

::: {.cols .vc}
::: {.col}
- **Step-by-step slot refill:**
  - Step 1 starts `(5, 2)`. After 2 steps, the 2-token request finishes (`(3,)` remaining) and the 7-token request immediately takes slot 2 (`(3, 7)`).
  - When the first request finishes at step 5, the final 3-token request joins the 7-token one, which has 4 tokens left (`(4, 3)`), leaving only 1 single-occupancy tail step at the very end.
:::

::: {.col .fig}
@model-serving-scheduler
:::
:::
:::

::: {.slide title="PagedAttention and KV Cache Capacity Planning"}
[Engine Technique 2]{.kicker}

::: {.cols .vc}
::: {.col}
- **Contiguous reservation waste (`max_len=32,768`):** Reserves $4.83\text{ GB/seq}$ out of $55.6\text{ GB}$ free H100 VRAM, fitting only **$b = 11$ sequences** while leaving $93.7\%$ of reserved KV memory empty at $n = 2{,}048$.
- **PagedAttention (16-token virtual blocks):** Allocates $2.36\text{ MB}$ blocks on demand ($< 0.7\%$ internal waste), expanding concurrent capacity **$16.7\times$ to $b = 184$ sequences** so the $20\text{ ms}$ TPOT SLO ($b = 167$) binds first.
:::

::: {.col .fig .big}
@fig:ch39-tools-paged-attention-blocks
:::
:::
:::

::: {.slide title="Prefix Caching and Prompt Engineering for Cache Hits"}
[Engine Technique 3]{.kicker}

When requests share an identical leading token sequence (system instructions, few-shot examples, or repository context), **prefix caching** reuses the precomputed KV blocks without re-running prefill:

::: {.cols .vc}
::: {.col}
- **Chained block hashing:** Each 16-token block is keyed by $\text{hash}(\text{parent\_block\_key}, \text{token\_ids})$ because causal self-attention makes token representations depend on every preceding token.
- **30-turn coding agent ($20\text{K}$ base context + $1\text{K}$ appended/turn):**
  - Uncached prefill computes $\sum_{k=0}^{29} (20 + k)\text{K} = \mathbf{1{,}035{,}000\text{ tokens}}$.
  - Prefix caching computes only $20\text{K} + 29 \times 1\text{K} = \mathbf{49{,}000\text{ tokens}}$ (**$21\times$ less prefill work**).
- **Prompt design rule:** Never put a per-second timestamp on line 1 of a system prompt—doing so invalidates the hash chain from block 0. Put volatile metadata at the *end* of the prompt.
:::

::: {.col .narrow}
![](../img/tools-prefix-cache.svg){.nostretch width=100%}
:::
:::
:::

::: {.slide title="Speculative Decoding and Server Quantization"}
[Engine Technique 4]{.kicker}

::: {.cols .vc}
::: {.col}
$$\mathbb{E}[\text{tokens/round}] = \frac{1 - \alpha^{\gamma + 1}}{1 - \alpha}$$

- **Draft & parallel verify:** A small draft model $q$ (cost $c \ll 1$) proposes $\gamma$ tokens; one target pass $p$ verifies all $\gamma+1$ tokens in parallel, preserving the exact target distribution ($2.80\times$ speedup at $\alpha=0.8, \gamma=4, c=0.05$).
- **Busy-server caveat:** Parallel verification is free only while memory-bound; on H100 ($I^* \approx 295$), any batch $b > 295/(\gamma+1) = 59$ becomes compute-bound.
:::

::: {.col .fig}
@fig:ch39-tools-speculative-decoding
:::
:::
:::

::: {.slide title="Chunked Prefill, Admission Control, and Operations"}
[Production Reliability]{.kicker}

::: {.cols .vc}
::: {.col}
- **Chunked prefill:** Splits long 16K prompts into fixed step budgets (e.g., $2{,}048$ tokens/step) and piggybacks active decode tokens so prefill bursts never spike TPOT.
- **Admission, cancellation & co-pinning:** Reject arrivals early when queue depth exceeds the TTFT SLO, abort disconnected HTTP streams, front with a TLS/auth/rate-limit proxy, and co-pin `(model_sha, tokenizer, quant, engine)`.
:::

::: {.col .fig .big}
@fig:ch39-tools-chunked-prefill-ops
:::
:::
:::

::: {.slide title="Concept Check: KV Cache, Batching, and Prefix Reuse"}
[Check Your Understanding]{.kicker}

Test your serving systems arithmetic on these three scenarios:

1. In a retrieval-augmented generation (RAG) service, a shared 2,000-token system prompt is followed by 3 retrieved documents out of a 50-document corpus. Request 1 retrieves `[Doc A, Doc B, Doc C]` and Request 2 retrieves `[Doc B, Doc A, Doc C]`. Under block-hashed prefix caching, which KV blocks does Request 2 reuse, and how can you raise cache hits?
2. On our H100 serving Qwen3-8B ($n = 2{,}048$ average context, $72\text{ GB}$ memory budget), the service enforces a $\text{TPOT} \le 20\text{ ms}$ SLO ($b_{\max,\text{SLO}} = 167$). Under contiguous `max_model_len=32768` allocation ($b_{\max,\text{mem}} = 11$) vs. PagedAttention ($b_{\max,\text{mem}} = 184$), which constraint binds the batch size in each case?

::: {.d2l-note .fragment .fade-up}
**Answers:**

1. Request 2 reuses **only the 2,000-token system prompt**. Because `Doc B` sits at position 2001 instead of `Doc A`, every block hash from token 2001 onward changes. Ordering retrieved documents in a **canonical ID order** (`[Doc A, Doc B, Doc C]`) allows full prefix reuse whenever subsets share leading documents, trading off relevance-order positional bias.
2. Under **contiguous reservation**, **memory capacity binds first** ($11 \ll 167$, leaving the GPU starved). Under **PagedAttention**, memory holds $184$ sequences, so the **$20\text{ ms}$ TPOT latency SLO binds first** at $b = 167$.
:::
:::

::: {.slide title="Concept Check: KV Cache, Batching, and Prefix Reuse (cont.)"}
[Check Your Understanding]{.kicker}

3. For speculative decoding with draft cost $c = 0.05$, acceptance rate $\alpha = 0.8$, and $\gamma = 4$ draft tokens, what is the expected speedup over standard batch-1 decoding?

::: {.d2l-note .fragment .fade-up}
**Answer:**

3. Expected tokens per round $= (1 - 0.8^5) / (1 - 0.8)$ $= 0.67232 / 0.2 = \mathbf{3.362\text{ tokens}}$ in time $1 + 4(0.05) = \mathbf{1.20\text{ target steps}}$, giving a **$\mathbf{2.80\times}$ speedup**.
:::
:::

::: {.slide title="Summary: Model Serving Systems"}
[Key Takeaways]{.kicker}

| Serving Component | Physical / Algorithmic Principle | Production Engineering Rule |
| :--- | :--- | :--- |
| **Prefill vs. Decode** | Prefill is parallel & compute-bound (sets TTFT); decode is serial & bandwidth-bound (sets TPOT) | Optimize prefill with tensor cores & prefix caching; optimize decode with batching & quantization |
| **Batched decode scaling** | $t(b) = (B_w + b \, n \, B_{\text{kv}}) / \beta$ | Batching amortizes weight reads $B_w$ until KV-cache bandwidth $b \, n \, B_{\text{kv}}$ or the TPOT SLO caps $b$ |
| **Continuous batching** | Retires finished sequences and refills slots at every decode step | Eliminates static padding bubbles and raises GPU slot occupancy |
| **PagedAttention** | Allocates KV cache in on-demand 16-token virtual blocks | Eliminates $\sim 90\%+$ contiguous reservation waste, expanding concurrent batch capacity $> 10\times$ |
| **Prefix caching** | Reuses KV blocks across matching chained block hashes | Place static system prompts and canonical context *before* dynamic timestamps or user queries |
| **Goodput & admission** | Overload inflates queues so goodput collapses while raw throughput stays high | Use chunked prefill, early TTFT-based admission control, and client cancellation propagation |
:::
