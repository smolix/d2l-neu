# Models behind an API
:label:`sec_lm_api`

<!-- execution: locked; requires: openrouter -->

<!-- Author brief, copied verbatim from the Language Models part plan
(docs/llm/language_models_part_plan.md, "1.7 Models behind an API · T0").
Blueprint: none (the plan adds this section).
The plan is a specification, not book text; this comment is replaced
when the section is written.

**Idea:** A hosted model is a service with a protocol, a price and limits that the client can measure, and what it cannot return is why the rest of the part works with open weights.
**Opening:** DeepSeek-V4.1-Flash has 763 billion parameters, more than one node of eight 40 GB A100s holds even at four bits (an 8 × 80 GB node holds it at four bits), yet it answers a GSM8K problem in about a second for a printed fraction of a cent. The section treats the client as a measuring instrument (latency, throughput under concurrency, cost, reproducibility) and closes on what survives of the chapter's toolkit when the weights are someone else's.
**Teaches:**
- The chat-completions protocol: messages, sampling fields that map onto `sec_decoding`, streaming, `usage` as the ground truth for tokens and cost, `system_fingerprint` and served model id; a field probe matrix across a router, a local server and a closed provider.
- `d2l.ChatAPI`: a request-hash replay cache (`replay`, `record`, `live`), bounded concurrency with full-jitter backoff honouring `Retry-After`, a cost cap, a manifest; the SDK's own retries turned off so 429s are counted.
- Model ids drift (aliases re-pointed, providers re-quantize, `:fastest` switches providers between calls; one set of open weights is served by dozens of providers at FP4, FP8 or MXFP4, each with its own supported fields); temperature 0 with a seed is not determinism; what an API withholds: prompt log-probabilities, hidden states, weights, the sampler.
- Reasoning effort as a request field: it moves reasoning tokens (inside `completion_tokens`, reported in `usage`) and cost, not the protocol; accuracy against effort belongs to 9.6.
- Terms of use as an engineering input: which outputs may be committed, and the check 3.5, 7.3 and 9.1 run before an API model generates training data.
- The same client against `transformers serve` for a local model, so hosted and local are measured with identical prompts and extractor.
**Derivation:** Little's law for a closed loop with c in-flight requests and service time W0 at light load: throughput X = min(c / W0, Xmax), with the knee at c* = Xmax W0, and X W = c at every level; the expected sleep of full-jitter backoff after attempt a, half of min(cap, base 2^a). Smallest numeric case: W0 = 2 s and Xmax = 8 requests per second give the knee at 16 in-flight requests; the expected total sleep over six attempts with base 1 s and cap 60 s, summed by hand.
**Notebook:**
1. Probe the field matrix on three endpoints (`/v1/models`, `logprobs`, `seed`, `response_format`, thinking controls) and print it; snapshot the router listing with prices and date.
2. Experiment 1: 100 GSM8K problems (1.4's seeded ids, 4-shot through `d2l.fewshot_prompt(template='chat_multi')`) on DeepSeek-V4.1-Flash-1017 through the router (one pinned provider), on Qwen3.8-27B, and on Qwen3-4B behind `transformers serve`; accuracy with intervals, the paired difference, latency p50 and p90, cost from `usage` reconciled with the billing page when live; the flagship run twice (replica 0 and 1) as the noise floor.
3. Experiment 2: the three longest problems, 20 replicas each at temperature 0 and a fixed seed, hosted, local alone, local under 8 background requests; count distinct completions and the first character of divergence.
4. Experiment 3: streaming on the flagship at concurrency 1, 2, 4, 8, 16, 32 requests per level, with two shuffled passes (320 requests); throughput, time to first token, 429 counts; fit min(c / W0, Xmax).
5. One cell: the flagship at three effort settings on five committed prompts, reporting reasoning tokens, completion tokens and wall-clock (an anecdote, labelled as such).
6. Claim that survives noise: X W = c within a few percent at every level, and a knee is claimed only if both passes show it; a model gap at n = 100 smaller than the replica disagreement is called unresolved with 1.5's minimum detectable effect stated; identical outputs on 60 replicas do not show determinism, and the text says so.
7. Reused pieces: 1.3's prompt builder, 1.4's extractor and problem ids, 1.5's `paired_bootstrap` and `bootstrap_ci`; a recording makes live calls under `COST_CAP_USD = 5` (about 750 calls, estimated under $2; see "Paid API calls"), and every build replays the committed caches at T0.
**Figures:** `fig_lm_api_concurrency` (plot: throughput and time to first token against concurrency, the fitted closed-loop curve, 429 counts as labels); `fig_lm_api_lost` (what the open-weights toolkit of 1.2 to 1.6 can see and what each endpoint type returns, as a table-like figure).
**Restate / forward:** `sec_decoding` (request fields are the sampler's parameters: temperature, top-p, stop); `sec_repro` (a seed promises reproducibility only for a sampler you own). Forward: 10.1 (the `tools` field).
**Exercises:** 5 to 7. Load-bearing: derive W(c) beyond the knee from X W = c and the saturated throughput (derive); measure prompt-cache hits on a shared 2,000-token prefix through `usage.cached_tokens` and the cost difference (measure); check that the request hash excludes `stream` and the API key and includes `replica`, by constructing colliding and non-colliding requests (check); open: run Experiment 1 through an FP4 and an FP8 provider of the same weights and test whether the accuracy difference exceeds the replica disagreement, or run 1.5's judge through the router on 100 pairs and compare its position bias with the local judge's.
**Writer notes:**
- Exports (`01_interfaces.md`): `d2l.ChatAPI(base_url, model, api_key=None, *, cache=None, mode='replay', price=None, cost_cap_usd=None, timeout=120.0, max_retries=6, default_params=None, http_client=None)` with `.chat(messages, *, replica=0, stream=False, **params) -> d2l.ChatResult`, `async .achat`, `.models()`, `.spent_usd`, `.manifest()`; `mode='replay'` raises `d2l.ReplayMiss` on a miss so a build can never reach the network; a live call past the cap raises `d2l.CostCapExceeded`; `d2l.chat_batch(api, requests, *, concurrency=8)` returns results in input order with failures as exceptions; `d2l.backoff_delay(attempt, base=1.0, cap=60.0, rng=None, retry_after=None)`; `d2l.PricePoint(input, output, cached_input=None, per=1_000_000)` and `d2l.api_cost(usage, price)`; `d2l.price_from_listing`. The request hash is the sha256 of canonical JSON over host, model, messages, sampling and format params and `replica`; it excludes `stream`, timeouts, headers and keys. The API key is never read from a file and never enters the hash, cache or logs.
- Wrap `openai.OpenAI` with `max_retries=0` (the SDK retries twice by default and would hide 429s from the counters); retry only connection errors, timeouts and 408, 429, 5xx; raise at once on 400, 401, 403, 404, 422.
- The flagship is DeepSeek-V4.1-Flash-1017 (MIT, open weights, 763B parameters in a sparse mixture of experts), always by its dated snapshot id, never the re-pointable alias `deepseek-v4.1-flash`; the second hosted arm is Qwen3.8-27B (Apache-2.0, open weights a T2 pair can serve, the larger sibling of the local arm's family); the local arm is Qwen3-4B behind `transformers serve` (not Qwen3.5-9B, as the blueprint has it; GPU stage 2026-11-13: not Qwen3.5-4B either, because continuous batching in Transformers 5.22.2 rejects Qwen3.5's `linear_attention` layers and the server falls back to one request at a time, about 22 tokens/s). Pin one provider per model through the router's provider preferences with fallbacks off, choosing one that exposes `logprobs`, `seed` and `structured_outputs` (on 2026-11-06 Morph, Nebius and Novita did for the flagship; DeepSeek's own route lacked `seed` and doubles its price in two weekday windows, 01:00 to 04:00 and 06:00 to 10:00 UTC); record the served model id and provider with every response; `:fastest` switches providers between calls. Fallback if the snapshot leaves the router: the undated alias with the served id recorded, else Qwen3.8-2.4T-A95B at about ten times the price. The release ships no chat template, only a Python prompt encoder, so the hosted arm also withholds how the prompt is encoded.
- `transformers serve` is documented as experimental and rejects `logprobs`, `n`, `response_format`; the probe matrix is the evidence; fallback for the local arm is vLLM in its own environment. The `d2l.ChatAPI` is a measurement and replay client; 13.1's load-testing client is a different object and neither imports the other.
- Terms and costs: live runs go through OpenRouter (or a similar aggregator; the Hugging Face router needs a Hub token with the Inference Providers permission) with the token held outside the repository and a fixed per-build budget set as a notebook constant (the cost cap); prices in the listing carry no stated unit, so reconcile with the billing page and use it when they disagree; outputs of the MIT and Apache-2.0 models are committed with attribution, a closed provider's outputs never.
- Pitfalls to state: timing whole responses and calling it latency; counting streamed chunks as tokens; a `max_tokens` that hidden reasoning consumes (reasoning tokens are inside completion tokens unless the provider says otherwise, asserted from `usage`); comparing local and hosted under different prompts, thinking modes or extractors.
- Committed artifacts: `lm-api-gsm8k-v1`, `lm-api-determinism-v1`, `lm-api-sweep-v1`, `lm-api-judge-v1`, `lm-api-effort-v1` (JSONL caches), `lm-api-listing-v1` (the `/v1/models` rows with date), `manifest.json` per cache, `lm-api-fields.csv`.
- Keep the section short (about 2,000 words); the derivation and three experiments are each one `##` subsection.
- Confirm at authoring the snapshot id, its licence file and the pinned providers' supported fields, and re-snapshot the listing with prices and date.
**Sources:**
- Papers: [Model Equality Testing: Which Model Is This API Serving?](https://arxiv.org/abs/2410.20247) (statistical tests that an endpoint serves the model it claims); [Are You Getting What You Pay For? Auditing Model Substitution in LLM APIs](https://arxiv.org/abs/2504.04715) (substitution and quantization at providers).
- Code and documentation: Hugging Face, [Inference Providers](https://huggingface.co/docs/inference-providers/index) (the router, provider policies and the `model:provider` suffix), its [chat-completion specification](https://huggingface.co/docs/inference-providers/tasks/chat-completion) and [pricing and billing](https://huggingface.co/docs/inference-providers/pricing); [openai-python](https://github.com/openai/openai-python) (clients, `usage` types, default retries); Transformers, [Serving](https://huggingface.co/docs/transformers/main/en/serving) (`transformers serve`, its endpoints and limits); vLLM, [OpenAI-compatible server](https://docs.vllm.ai/en/latest/serving/online_serving/) (the local fallback); [DeepSeek-V4.1-Flash-1017 model card](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017) (the flagship: licence, recommended sampling, the prompt encoder); [Qwen3.8 README](https://github.com/QwenLM/Qwen3.8) (the second arm's controls); OpenRouter, [provider selection](https://openrouter.ai/docs/guides/routing/provider-selection) (pinning a provider with fallbacks off) and [reasoning tokens](https://openrouter.ai/docs/guides/best-practices/reasoning-tokens) (the `reasoning` field; reasoning tokens count against the output limit); [RFC 6585](https://www.rfc-editor.org/rfc/rfc6585) (HTTP 429).
- Courses, tutorials and blogs: AWS Architecture Blog, [Exponential Backoff and Jitter](https://aws.amazon.com/blogs/architecture/exponential-backoff-and-jitter/) (the schedules behind `backoff_delay`); Thinking Machines, [Defeating Nondeterminism in LLM Inference](https://thinkingmachines.ai/blog/defeating-nondeterminism-in-llm-inference/) (why temperature 0 is not deterministic); [lm-evaluation-harness API guide](https://github.com/EleutherAI/lm-evaluation-harness/blob/main/docs/API_guide.md) (evaluating through chat completions).
-->

```{.python .input #api-imports}
%%tab pytorch
import math
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from d2l import torch as d2l
```

```{.python .input #api-imports-extra}
%%tab pytorch
import asyncio
import collections
import contextlib
import datetime
import gc
import hashlib
import heapq
import json
import os
import pandas as pd
import random
import re
import sys
import time
from scipy import stats
from transformers.cli.serving.chat_completion import (
    UNUSED_CHAT_COMPLETION_FIELDS)
```

<!-- Placeholder (stage 2): the switches of a recording, the models and their pinned providers, and where the caches live. -->

```{.python .input #api-config}
%%tab pytorch
API_MODE = os.environ.get('D2L_API_MODE', 'replay')  # or record, live
assert API_MODE in ('replay', 'record', 'live'), API_MODE
if API_MODE != 'replay' and not os.environ.get('OPENROUTER_API_KEY'):
    print(f'D2L_API_MODE={API_MODE} needs OPENROUTER_API_KEY: replaying')
    API_MODE = 'replay'
COST_CAP_USD = 5.0  # for all paid calls of one recording
RECORD_DIR = os.path.join('..', 'data', 'lm-api', 'record')
D2L_API_LOG_DIR = os.environ.get(  # logs outside the repository
    'D2L_API_LOG_DIR', os.path.expanduser('~/.local/share/d2l/lm-api'))
FLAGSHIP = 'deepseek/deepseek-v4.1-flash-1017'  # a dated snapshot id
SECOND = 'qwen/qwen3.8-27b'
CLOSED = 'openai/gpt-5-nano'  # probe only; its outputs are never committed
PROVIDERS = {FLAGSHIP: 'novita', SECOND: 'parasail', CLOSED: 'openai'}
LISTED = [FLAGSHIP, SECOND, CLOSED]  # the router models of the listing
LOCAL = 'Qwen/Qwen3-4B'  # behind transformers serve on our GPU
LOCAL_MODEL = f'{LOCAL}@{d2l.HF_REVISIONS[LOCAL]}'  # requests name id@rev
LOCAL_LIVE = API_MODE != 'replay' and torch.cuda.is_available()
LOCAL_MODE = API_MODE if LOCAL_LIVE else 'replay'
API_SHA1 = {  # the committed caches (zips), recorded 2026-11-13
    'lm-api-listing-v1': '4916ce668be31ed37947dac80bc094a70ee75efc',
    'lm-api-fields-v2': '9a12dce5a344f3e746de0c915fa3d2f505ee0f71',
    'lm-api-probe-v2': 'e9aea3523b8ced2383500d3772bfb82893733354',
    'lm-api-sweep-v1': '63250dccbf90eae6143b5affe74ae25972f514b1',
    'lm-api-gsm8k-v1': '1fbb38c4cf5f65fb69da5ba3719cd850ba768dc6',
    'lm-api-determinism-v1': '448e39446d960dac298a5db3a4e6c0b95051683b',
    'lm-api-effort-v1': '7480d44c1bc2b6f8d2ac90aadf9f50689b569b9b'}
for key, sha1 in API_SHA1.items():
    if sha1 is not None:
        d2l.DATA_HUB[key] = (d2l.LM_DATA_URL + key + '.zip', sha1)
rec = d2l.APIRecording(API_MODE, RECORD_DIR, COST_CAP_USD, PROVIDERS)

def usd(api, r):  # cost of one response: the provider's, else the listing's
    cost = r.usage.get('cost')
    return d2l.api_cost(r.usage, api.price) if cost is None else cost

print(f'mode {API_MODE}, cost cap {COST_CAP_USD} USD, local server '
      f'{"live" if LOCAL_LIVE else "replayed"}')
```

<!-- Placeholder (stage 2): a dated snapshot of the router listing: providers (the tag names the quantization), prices and supported fields (`json` is the listing's `structured_outputs`). -->

```{.python .input #api-listing}
%%tab pytorch
WEIGHTS = {  # the open weights behind each model, at a pinned commit
    FLAGSHIP: ('deepseek-ai/DeepSeek-V4.1-Flash-1017',
               '1bae0e1088993b9b82de08f59de5be5a34e42170'),
    SECOND: ('Qwen/Qwen3.8-27B', '37561816e66b5b9153b1b64c10968c18c1e6f4a0'),
    LOCAL: (LOCAL, d2l.HF_REVISIONS[LOCAL])}
LISTING = rec.snapshot_listing(LISTED, WEIGHTS)  # taken when recording
rows = []
for m in LISTED:
    e = d2l.listing_endpoint(LISTING, m, PROVIDERS[m])
    price = d2l.price_from_listing(LISTING, m, PROVIDERS[m])
    rows.append({'model': m.split('/')[1], 'provider': e['tag'],
                 'context': e['context_length'], 'in': price.input,
                 'out': price.output, 'cached': price.cached_input,
                 **{f: f in e['supported_parameters'] for f in (
                     'seed', 'logprobs', 'reasoning')},
                 'json': 'structured_outputs' in e['supported_parameters']})
print(f'listing retrieved {LISTING["retrieved"]}; prices in USD per million '
      f'tokens')
print(pd.DataFrame(rows).to_string(index=False))
```

<!-- Placeholder (stage 2): the open models people use most do not fit our GPUs, even at four bits per weight. The flagship at four bits (382 GB) exceeds a node of eight 40 GB A100s but fits eight 80 GB cards: say "40 GB" wherever an A100 node is named. -->

```{.python .input #api-weights-arith}
%%tab pytorch
PRECISIONS = (('bf16', 2), ('fp8', 1), ('4 bits', 0.5))  # bytes/weight

def best_fit(n, gb):  # the highest precision whose weights fit in gb
    return next((p for p, b in PRECISIONS if n * b <= gb * 1e9), 'none')

rows = []
for m in (FLAGSHIP, SECOND, LOCAL):
    n = LISTING['hf'][m]['params']
    rows.append({'weights': LISTING['hf'][m]['repo'].split('/')[1],
                 'params (B)': n / 1e9,
                 **{f'{p} (GB)': n * b / 1e9 for p, b in PRECISIONS},
                 '24GB card': best_fit(n, 24), '8x40GB': best_fit(n, 8 * 40),
                 '8x80GB': best_fit(n, 8 * 80)})
print(pd.DataFrame(rows).round(1).to_string(index=False))
```

<!-- Placeholder (stage 2): one request to the flagship: a GSM8K problem in, text, a served id and a usage block out. -->

```{.python .input #api-first-call}
%%tab pytorch
gsm_train = pd.read_parquet(d2l.download_keyed('gsm8k-train'))
gsm_test = pd.read_parquet(d2l.download_keyed('gsm8k-test'))
samples = pd.read_json(d2l.download_keyed('lm-gsm8k-samples-v1'),
                       lines=True)
# the 100 test problems of the scoring section, by its seeded rule
GSM_IDS = [int(i) for i in dict.fromkeys(samples.problem_id)]
assert GSM_IDS == list(np.random.default_rng(0).permutation(
    len(gsm_test))[:100])
flagship = rec.hosted(FLAGSHIP, rec.cache('lm-api-probe-v2', 'flagship'),
                      share=0.05, reasoning={'enabled': False})
question = [{'role': 'user', 'content': gsm_test.question[GSM_IDS[0]]}]
r = flagship.chat(question, temperature=0, seed=0, max_tokens=512)
print(r.text)
u = r.usage
print(f'finish {r.finish_reason}; {u["prompt_tokens"]} prompt and '
      f'{u["completion_tokens"]} completion tokens ({u["reasoning_tokens"]} '
      f'reasoning, {u["cached_tokens"]} cached)')
print(f'{r.t_total:.2f} s; served {r.model} by {r.provider}; '
      f'system_fingerprint {r.system_fingerprint}')
print(f'{u["cost"]:.6f} USD billed, {d2l.api_cost(u, flagship.price):.6f} '
      f'USD from the listing')
```

<!-- Placeholder (stage 2): hidden reasoning is billed as completion tokens and can use up the whole output budget (on the problem with the longest reference solution: on the first problem Qwen3.8-27B reasoned for 39 tokens and answered inside 64). With thinking off the same problem needs about 870 tokens: at 512 it was cut off as well (lm-api-probe-v1), so the contrast runs at 1,024 and finishes with `stop`. -->

```{.python .input #api-reasoning-trap}
%%tab pytorch
second = rec.hosted(SECOND, rec.cache('lm-api-probe-v2', 'second'),
                    share=0.05)
LONGEST = sorted(GSM_IDS, key=lambda i: -len(gsm_test.answer[i]))
hard = [{'role': 'user', 'content': gsm_test.question[LONGEST[0]]}]
trap = second.chat(hard, temperature=0, seed=0, max_tokens=64)  # thinks
u = trap.usage
print(f'content {trap.text!r}, finish {trap.finish_reason}, '
      f'{len(trap.reasoning or "")} characters of reasoning')
print(f'usage: {u["prompt_tokens"]} prompt, {u["completion_tokens"]} '
      f'completion, {u["reasoning_tokens"]} reasoning tokens')
assert trap.text == '' and trap.finish_reason == 'length'
assert u['reasoning_tokens'] == u['completion_tokens'] == 64
answer = second.chat(hard, temperature=0, seed=0, max_tokens=1024,
                     extra_body={'reasoning': {'enabled': False}})
u, last = answer.usage, answer.text.strip().splitlines()[-1]
print(f'thinking off: last line {last!r} (reference '
      f'{d2l.extract_answer(gsm_test.answer[LONGEST[0]])}), '
      f'{u["completion_tokens"]} completion tokens, finish '
      f'{answer.finish_reason}')
assert not u['reasoning_tokens']
assert answer.finish_reason == 'stop', 'thinking off did not finish either'
```

<!-- Placeholder (stage 2): the effort field moves reasoning tokens and cost, not the protocol (an anecdote on five problems). Quote the paired per-problem counts, not the medians: high is above low and medium on every problem, while low and medium are indistinguishable on these five (medium is below low on three). -->

```{.python .input #api-effort}
%%tab pytorch
effort_api = rec.hosted(FLAGSHIP, rec.cache('lm-api-effort-v1', 'flagship'),
                        share=0.2)
EFFORTS = ('low', 'medium', 'high')
batch = [{'messages': [{'role': 'user', 'content': gsm_test.question[i]}],
          'temperature': 0, 'seed': 0, 'max_tokens': 16384,
          'extra_body': {'reasoning': {'effort': e}}}
         for e in EFFORTS for i in LONGEST[:5]]
results = await d2l.chat_batch(effort_api, batch, concurrency=15)
assert not any(isinstance(r, Exception) for r in results), results
effort = pd.DataFrame({
    'effort': [e for e in EFFORTS for _ in range(5)],
    'reasoning tokens': [r.usage['reasoning_tokens'] for r in results],
    'completion tokens': [r.usage['completion_tokens'] for r in results],
    'seconds': [r.t_total for r in results],
    'USD': [usd(effort_api, r) for r in results]})
medians = effort.groupby('effort', sort=False).median()
print('an anecdote: n = 5 problems per effort; medians')
print(medians.round({'reasoning tokens': 0, 'completion tokens': 0,
                     'seconds': 2, 'USD': 6}))
paired = effort['reasoning tokens'].to_numpy().reshape(3, 5)  # by problem
for e, t in zip(EFFORTS, paired):
    print(f'{e:6s} reasoning tokens per problem: {t.tolist()}')
for i, j in ((0, 1), (1, 2), (0, 2)):  # paired, problem by problem
    print(f'{EFFORTS[j]} above {EFFORTS[i]} on '
          f'{(paired[j] > paired[i]).sum()} of 5 problems')
```

<!-- Placeholder (stage 2): the same protocol from a server on our own GPU: transformers serve with continuous batching. -->

```{.python .input #api-local-start}
%%tab pytorch
NO_THINK_LOCAL = {'extra_body': {'chat_template_kwargs': {
    'enable_thinking': False}}}
STORY_LOCAL = [{'role': 'user', 'content': 'Write a long story.'}]
LOCAL_HTTP = None  # the local endpoint's HTTP client (None: real HTTP)
if LOCAL_LIVE:
    server = d2l.LocalChatServer([
        os.path.join(os.path.dirname(sys.executable), 'transformers'),
        'serve', LOCAL_MODEL, '--continuous-batching', '--dtype', 'bfloat16',
        '--model-timeout', '-1', '--cb-max-memory-percent', '0.2',
        '--host', '127.0.0.1', '--port', '{port}'])
    LOCAL_URL = server.start()
    warm = d2l.ChatAPI(LOCAL_URL, LOCAL_MODEL, mode='live',
                       default_params=NO_THINK_LOCAL)
    tic = time.monotonic()
    warm.chat([{'role': 'user', 'content': 'Hello'}], max_tokens=8)
    print(f'{LOCAL_URL}: ready after {server.load_time:.0f} s, warm-up '
          f'request {time.monotonic() - tic:.1f} s')
    r = warm.chat(STORY_LOCAL, max_tokens=256)  # one request alone
    print(f'{r.usage["completion_tokens"]} tokens in {r.t_total:.1f} s: '
          f'{r.usage["completion_tokens"] / r.t_total:.1f} tokens/s')
else:  # the request hash keeps the host only, so any port replays
    server, LOCAL_URL = None, 'http://127.0.0.1/v1'
    print(f'{LOCAL_URL}: replayed from the committed caches')

def local(cache, mode=LOCAL_MODE):
    """A client of the local server, thinking switched off."""
    return rec.add(d2l.ChatAPI(LOCAL_URL, LOCAL_MODEL, cache=cache, mode=mode,
                               http_client=LOCAL_HTTP,
                               default_params=NO_THINK_LOCAL))
```

<!-- Placeholder (stage 2): a request field a provider accepts is not necessarily one it honours: one detection rule per field. With require_parameters, any field an endpoint does not list is rejected, so each test sends only the field it probes (the seed test samples at the default temperature where temperature is not listed). gpt-5-nano's usage under max_completion_tokens=16 reports more reasoning than completion tokens, so claim 7 (reasoning inside completion tokens) is limited to the experiment arms. -->

```{.python .input #api-probe}
%%tab pytorch
STORY = [{'role': 'user', 'content':
          'Write a short paragraph about a lighthouse keeper.'}]
COUNT = [{'role': 'user', 'content':
          'Count from 1 to 10, separated by commas.'}]

def seed_honoured(a, b, c):  # STORY with seed 7 twice (replicas), then 8
    return a.text == b.text != c.text

def stop_honoured(a):  # COUNT with stop=['5']
    return '4' in a.text and '5' not in a.text

FIELDS_FILE = os.path.join(rec.folder('lm-api-fields-v2'),
                           'lm-api-fields.csv')
LOCAL_NAME = 'Qwen3-4B (local server)'
if API_MODE != 'replay':  # replay reads the committed table
    off, on = ({'reasoning': {'enabled': s}} for s in (False, True))
    local_off, local_on = ({'chat_template_kwargs': {'enable_thinking': s}}
                           for s in (False, True))
    closed = rec.hosted(CLOSED, os.path.join(D2L_API_LOG_DIR, 'scratch',
                                             'closed-probe.jsonl'),
                        share=0.05)
    endpoints = {  # client, body of every test, thinking on and off,
        # output budget, the id /v1/models should list
        'DeepSeek-V4.1-Flash (Novita)': (flagship, off, on, off, 64,
                                         FLAGSHIP),
        'Qwen3.8-27B (Parasail)': (second, off, on, off, 64, SECOND),
        'gpt-5-nano (OpenAI)': (closed, None, on, off, 2048,
                                CLOSED)}  # a closed model thinks by default
    if LOCAL_LIVE:
        endpoints[LOCAL_NAME] = (local(rec.cache('lm-api-probe-v2', 'local')),
                                 local_off, local_on, local_off, 64, LOCAL)
    await d2l.record_field_probe(
        FIELDS_FILE, endpoints, local_name=LOCAL_NAME, listing=LISTING,
        providers=PROVIDERS, story=STORY, count=COUNT,
        rules={'seed': seed_honoured, 'stop': stop_honoured})
```

<!-- Placeholder (stage 2): the probe matrix, one status per endpoint and field, and for each OpenAI field whether transformers serve declares it unused. The local server ignores `stop` although it is not declared unused (Transformers maps it to `stop_strings`, which the continuous-batching path never reads); `seed` is not declared and is honoured. Its `/v1/models` "ignored" is a listing artifact: the server lists cached models by branch ref, and Qwen3-4B was fetched by commit. -->

```{.python .input #api-probe-table}
%%tab pytorch
SHORT = {'DeepSeek-V4.1-Flash (Novita)': 'flagship',
         'Qwen3.8-27B (Parasail)': 'Qwen3.8-27B',
         'gpt-5-nano (OpenAI)': 'gpt-5-nano', LOCAL_NAME: 'local'}
SENT = {'seed': ['seed'], 'n=2': ['n'],  # the OpenAI fields each test sends
        'logprobs, top 5': ['logprobs', 'top_logprobs'], "stop=['5']":
        ['stop'], 'response_format': ['response_format'],
        'max_completion_tokens=16': ['max_completion_tokens'],
        'stream usage': ['stream_options']}
fields = pd.read_csv(FIELDS_FILE).replace({'endpoint': SHORT})
table = fields.pivot(index='field', columns='endpoint', values='status')
table = table.loc[fields.field.unique(), fields.endpoint.unique()]
table['serve declares unused'] = [
    ('yes' if set(SENT[f]) <= UNUSED_CHAT_COMPLETION_FIELDS else 'no')
    if f in SENT else '' for f in table.index]
print(table.to_string())
batching = fields[(fields.endpoint == 'local') &
                  (fields.field == '4 concurrent requests')].iloc[0]
LOCAL_BATCHES = batching.status != 'serialized'
print(f'local server: {batching.status} ({batching.evidence})')
```

<!-- Placeholder (stage 2): the request hash keys a replay cache on everything that can change the output and nothing secret. -->

```{.python .input #api-request-hash}
%%tab pytorch
#@save
def request_hash(base_url, model, messages, params, replica=0):
    """sha256 of the canonical JSON (sorted keys, no whitespace, UTF-8) of
    what can change a response: the endpoint's hostname, the model, the
    messages, the merged parameters (None values dropped) without the
    transport fields stream, stream_options, timeout and extra_headers, and
    the client-side replica number."""
    from urllib.parse import urlsplit  # not in the d2l library preamble
    transport = ('stream', 'stream_options', 'timeout', 'extra_headers')
    params = d2l._chat_merge(params, {})  # drops None values at every level
    params = {k: v for k, v in params.items() if k not in transport}
    key = {'host': urlsplit(base_url).hostname, 'model': model,
           'messages': messages, 'params': params, 'replica': replica}
    blob = json.dumps(key, sort_keys=True, separators=(',', ':'),
                      ensure_ascii=False)
    return hashlib.sha256(blob.encode('utf-8')).hexdigest()
```

<!-- Placeholder (stage 2): what changes the hash and what does not. -->

```{.python .input #api-hash-demo}
%%tab pytorch
url = 'https://openrouter.ai/api/v1'
model = 'deepseek/deepseek-v4.1-flash-1017'
messages = [{'role': 'user', 'content': 'What is 17 * 23?'}]
pin = {'only': ['novita'], 'allow_fallbacks': False,
       'require_parameters': True}
params = {'temperature': 0, 'seed': 0, 'max_tokens': 64,
          'extra_body': {'provider': pin}}
variants = {
    'request': (url, params, 0),
    'stream=True': (url, {**params, 'stream': True}, 0),
    'another API key': (url, {**params, 'extra_headers': {
        'Authorization': 'Bearer another-key'}}, 0),
    'replica=1': (url, params, 1),
    'seed=1': (url, {**params, 'seed': 1}, 0),
    'provider parasail': (url, {**params, 'extra_body': {
        'provider': {**pin, 'only': ['parasail']}}}, 0)}
for name, (u, p, replica) in variants.items():
    print(f'{name:18s}', request_hash(u, model, messages, p, replica)[:12])
```

<!-- Placeholder (stage 2): cost is arithmetic on usage and a dated price. -->

```{.python .input #api-price}
%%tab pytorch
#@save
PricePoint = collections.namedtuple(  # USD per `per` tokens
    'PricePoint', ['input', 'output', 'cached_input', 'per'],
    defaults=(None, 1_000_000))

#@save
def api_cost(usage, price):
    """Cost in USD of one response, ((p - c) input + c cached_input +
    o output) / per, with p prompt, c cached and o completion tokens from
    usage (reasoning tokens are inside o). cached_input None means no
    discount; missing cached tokens count as 0; missing prompt or
    completion tokens raise ValueError."""
    p, o = usage.get('prompt_tokens'), usage.get('completion_tokens')
    if p is None or o is None:
        raise ValueError(f'usage lacks prompt or completion tokens: {usage}')
    c = usage.get('cached_tokens') or 0
    cached = price.input if price.cached_input is None else price.cached_input
    return ((p - c) * price.input + c * cached + o * price.output) / price.per
```

<!-- Placeholder (stage 2): prices are dated inputs: the pinned provider, and a provider that prices by time window. -->

```{.python .input #api-cost-example}
%%tab pytorch
novita = d2l.price_from_listing(LISTING, FLAGSHIP, 'novita')
print(f'Novita, USD per million tokens: {novita.input} in, {novita.output} '
      f'out, {novita.cached_input} cached in')
usage = {'prompt_tokens': 1000, 'completion_tokens': 300}
cost = d2l.api_cost(usage, novita)
assert abs(cost - (1000 * novita.input + 300 * novita.output) / 1e6) < 1e-15
print(f'1,000 prompt and 300 completion tokens: {cost:.6f} USD')
assert f'{cost:.6f}' == '0.000363', 'the text quotes 0.000363 USD: update it'
for hour in (2, 12):  # a Thursday, inside and outside a peak window
    when = datetime.datetime(2026, 11, 12, hour, tzinfo=datetime.timezone.utc)
    price = d2l.price_from_listing(LISTING, FLAGSHIP, 'deepseek', when=when)
    print(f'DeepSeek at {when:%a %H:%M} UTC: {price.input:.3f} in, '
          f'{price.output:.3f} out')
```

<!-- Placeholder (stage 2): full-jitter backoff, the formula of the derivation. -->

```{.python .input #api-backoff}
%%tab pytorch
#@save
def backoff_delay(attempt, base=1.0, cap=60.0, rng=None, retry_after=None):
    """Full-jitter sleep after failed attempt `attempt` (0, 1, ...): uniform
    on [0, min(cap, base 2^attempt)], and at least the server's
    Retry-After."""
    rng = rng or random
    return max(retry_after or 0, rng.uniform(0, min(cap, base * 2**attempt)))
```

<!-- Placeholder (stage 2): one call: cache by mode, cost reservation, the retry loop, settling and recording. -->

```{.python .input #api-achat}
%%tab pytorch
@d2l.add_to_class(d2l.ChatAPI)  #@save
async def achat(self, messages, *, replica=0, stream=False, **params):
    """One chat completion as a d2l.ChatResult. The cache answers first when
    the mode allows; a live call reserves its worst-case cost, retries
    connection errors, timeouts, 408, 429 and 5xx responses and error
    events inside a stream with full-jitter backoff (Retry-After as a
    floor), and raises at once on any other status."""
    import asyncio  # not in the d2l library preamble
    params = d2l._chat_merge(self.default_params, params)
    h = d2l.request_hash(self.base_url, self.model, messages, params, replica)
    self._stats['calls'] += 1
    hit = self._lookup(h, messages)  # raises ReplayMiss in replay mode
    if hit is not None:
        return hit
    reserved = self._reserve(messages, params)  # raises CostCapExceeded
    t_send, n_retries, n_429 = time.monotonic(), 0, 0
    created = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
    try:
        for attempt in range(self.max_retries + 1):
            try:
                out = await self._send(messages, params, stream, h, replica,
                                       attempt)
                break
            except Exception as e:
                retry, status, retry_after = self._retry_info(e)
                n_429 += status == 429
                if not retry or attempt == self.max_retries:
                    e.n_retries, e.n_429 = n_retries, n_429
                    raise
                n_retries += 1
                await asyncio.sleep(d2l.backoff_delay(
                    attempt, self.backoff_base, self.backoff_cap, self.rng,
                    retry_after))
    except BaseException:
        self._settle(reserved, 0.0)
        raise
    return self._record(h, messages, params, replica, out, t_send, created,
                        n_retries, n_429, reserved)
```

<!-- Placeholder (stage 2): many requests with bounded concurrency, in input order. -->

```{.python .input #api-chat-batch}
%%tab pytorch
#@save
async def chat_batch(api, requests, *, concurrency=8):
    """Send requests (dicts with 'messages', optionally 'replica' and request
    parameters) through api with at most `concurrency` in flight; results
    come in input order, a failed request as its exception. A replay miss
    or a capped call is not a result: once every request has finished (the
    paid ones are cached), the first d2l.ReplayMiss or d2l.CostCapExceeded
    is raised, with the whole list as its `results` attribute."""
    import asyncio  # not in the d2l library preamble
    semaphore = asyncio.Semaphore(concurrency)
    async def one(request):
        async with semaphore:
            try:
                request = dict(request)
                return await api.achat(request.pop('messages'), **request)
            except Exception as e:
                return e
    results = await asyncio.gather(*(one(r) for r in requests))
    for r in results:
        if isinstance(r, (d2l.ReplayMiss, d2l.CostCapExceeded)):
            r.results = results
            raise r
    return results
```

<!-- Placeholder (stage 2): the SDK's own retries are off, so the client counts every 429. -->

```{.python .input #api-retry-mock}
%%tab pytorch
hello = [{'role': 'user', 'content': 'Hello'}]
fake = d2l.FakeChatServer(0.01, 1, script=[429, 429, 200])
api = d2l.ChatAPI('http://fake.local/v1', 'fake-model', mode='live',
                  http_client=fake.transport())
r = api.chat(hello, max_tokens=8)
print(f'attempts {fake.n_requests}, 429s {r.n_429}, retries {r.n_retries}')
assert (fake.n_requests, r.n_429, r.n_retries) == (3, 2, 2)
fake = d2l.FakeChatServer(0.01, 1, script=[401])
api = d2l.ChatAPI('http://fake.local/v1', 'fake-model', mode='live',
                  http_client=fake.transport())
try:
    api.chat(hello, max_tokens=8)
except Exception as e:
    error = e
else:
    raise AssertionError('the 401 was not raised')
print(f'{type(error).__name__} after {fake.n_requests} attempt')
assert getattr(error, 'status_code', None) == 401 and fake.n_requests == 1
```

<!-- Placeholder (stage 2): the derivation's mean sleep, checked by a Monte Carlo. -->

```{.python .input #api-backoff-check}
%%tab pytorch
p, K, rng = 0.3, 6, random.Random(0)
def total_sleep():
    sleep = 0.0
    for a in range(K):
        if rng.random() >= p:
            break
        sleep += d2l.backoff_delay(a, 1.0, 60.0, rng)
    return sleep
draws = np.array([total_sleep() for _ in range(200_000)])
formula = sum(p**(a + 1) * min(60, 2**a) / 2 for a in range(K))
print(f'E[S] = {formula:.4f} s; simulated {draws.mean():.4f} '
      f'+- {3 * draws.std() / np.sqrt(len(draws)):.4f} s (3 sigma)')
```

<!-- Placeholder (stage 2): 64 clients that failed together. Lead with the finish time (over 200 seeds, jitter finishes in about 15 s against 63 s, the unjittered sum 1 + 2 + ... + 32); full jitter also makes fewer attempts on almost every seed, but only about 4% fewer (a fixed-window server admits a whole wave each time), so do not build on the attempt count. -->

```{.python .input #api-retry-storm}
%%tab pytorch
def retry_storm(n=64, rate=10, jitter=True, seed=0):
    """Simulated time: n clients send at t = 0 to a server that admits at
    most `rate` requests in any one-second window and rejects the rest; a
    rejected client sleeps and retries. Returns attempts and finish time."""
    rng, admitted = random.Random(seed), collections.deque()
    events, attempts, t_done = [(0.0, i, 0) for i in range(n)], 0, 0.0
    while events:
        t, i, a = heapq.heappop(events)
        attempts += 1
        while admitted and admitted[0] <= t - 1:
            admitted.popleft()
        if len(admitted) < rate:
            admitted.append(t)
            t_done = max(t_done, t)
        else:
            d = d2l.backoff_delay(a, rng=rng) if jitter else min(60, 2**a)
            heapq.heappush(events, (t + d, i, a + 1))
    return attempts, t_done

storms = {jitter: np.array([retry_storm(jitter=jitter, seed=k)
                            for k in range(200)]) for jitter in (False, True)}
print('over 200 seeds: median (min to max)')
for jitter, name in ((False, 'no jitter:  '), (True, 'full jitter:')):
    (a_lo, t_lo), (a_mid, t_mid), (a_hi, t_hi) = np.percentile(
        storms[jitter], [0, 50, 100], axis=0)
    print(f'{name} all done at {t_mid:.1f} s ({t_lo:.1f} to {t_hi:.1f}), '
          f'{a_mid:.0f} attempts ({a_lo:.0f} to {a_hi:.0f})')
fewer = (storms[True][:, 0] < storms[False][:, 0]).sum()
print(f'full jitter makes fewer attempts on {fewer} of 200 seeds')
```

<!-- Placeholder (stage 2): Little's law on the client's own timestamps, measured on the steady window. -->

```{.python .input #api-closed-loop}
%%tab pytorch
#@save
def closed_loop_stats(results, c):
    """Throughput and latency of a closed loop that kept c requests in
    flight; exceptions count as n_failed. The steady window runs from the
    c-th to the (N - c)-th of the N completions. X is the completions per
    second inside it (X_tokens their completion tokens), W the mean latency
    and ttft_mean the mean time to first token of the requests wholly inside
    it, and Lbar the mean number in flight; little = X W / c, sustained =
    Lbar / c. ttft_p50 and ttft_p90 are over all N."""
    d2l.check_closed_loop(results, c)  # one loop of c, from one run
    ok = [r for r in results if not isinstance(r, BaseException)]
    N = len(ok)
    if N <= 2 * c:
        raise ValueError(f'{N} results at c = {c}; need more than 2c')
    s = np.array([r.t_send for r in ok])
    f = s + np.array([r.t_total for r in ok])
    t_a, t_b = np.sort(f)[[c - 1, N - c - 1]]
    window = t_b - t_a
    if window <= 0:
        raise ValueError('the steady window is empty')
    done = (f > t_a) & (f <= t_b)  # completions inside the window
    inside = (s >= t_a) & (f <= t_b)  # requests wholly inside it
    tokens = [r.usage.get('completion_tokens') for r, d in zip(ok, done) if d]
    X = done.sum() / window
    W = (f - s)[inside].mean() if inside.any() else float('nan')
    Lbar = np.clip(np.minimum(f, t_b) - np.maximum(s, t_a), 0, None).sum()
    # the first token: the first content when the reasoning is only
    # whitespace (the empty think block of a server with thinking off)
    firsts = [r.t_first if (r.reasoning or '').strip() or (
        r.t_first_content is None) else r.t_first_content for r in ok]
    ttft = [None if t is None else t - r.t_send for r, t in zip(ok, firsts)]
    known = [t for t in ttft if t is not None]
    p50, p90 = (map(float, np.percentile(known, [50, 90])) if known
                else (None, None))
    steady = [t for t, i in zip(ttft, inside) if i and t is not None]
    X_tokens = None if None in tokens else float(sum(tokens) / window)
    return {'X': float(X), 'X_tokens': X_tokens,
            'W': float(W), 'Lbar': float(Lbar / window),
            'little': float(X * W / c), 'sustained': float(Lbar / window / c),
            'ttft_p50': p50, 'ttft_p90': p90,
            'ttft_mean': float(np.mean(steady)) if steady else None,
            'n_429': sum(getattr(r, 'n_429', 0) or 0 for r in results),
            'n_retries': sum(getattr(r, 'n_retries', 0) or 0 for r in results),
            'n_failed': len(results) - N, 'window': float(window),
            'n_window': int(done.sum())}
```

<!-- Placeholder (stage 2): the fake server has a known W_0 and X_max, so the measured curve can be checked against the formula. -->

```{.python .input #api-littles-fake}
%%tab pytorch
service_time, capacity = 0.1, 8  # X_max = 80 requests per second
api = d2l.ChatAPI('http://fake.local/v1', 'fake-model', mode='live',
                  http_client=d2l.FakeChatServer(service_time,
                                                 capacity).transport())
rows = []
gc.collect()
gc.freeze()  # no collector pauses inside the timed requests
for c in (1, 2, 4, 8, 16, 32):
    batch = [{'messages': [{'role': 'user', 'content': f'question {i}'}],
              'max_tokens': 16} for i in range(8 * c)]
    stats_c = closed_loop_stats(await chat_batch(api, batch,
                                                 concurrency=c), c)
    rows.append({'c': c, 'X': stats_c['X'], 'W': stats_c['W'],
                 'little': stats_c['little'],
                 'sustained': stats_c['sustained']})
table = pd.DataFrame(rows)
W_0 = table['W'][0]  # as the client sees it: service time plus its own work
table['formula X'] = np.minimum(table['c'] / W_0, capacity / service_time)
gc.unfreeze()
assert (abs(table['X'] / table['formula X'] - 1) < 0.05).all(), table
assert (abs(table[['little', 'sustained']] - 1) < 0.02).all().all(), table
table.round(3)
```

<!-- Placeholder (stage 2): Experiment 1: a closed-loop sweep against the local server, two passes in shuffled order. Time to first token is measured to the first content token: with thinking off, transformers serve streams the empty think block at once, before it schedules the request, and the first capture timed that chunk (TTFT looked flat; it grows past the knee). Exercise 5 (t_total - t_first, chunk counts) must use the same convention. The `in decode` column is Little's law applied to the decode stage alone, X (W - ttft_mean): it levels off at about 45 from c = 64 while time to first token jumps, so beyond the knee requests wait for a batch slot. The knee is this server's admission capacity at the KV budget the notebook sets (`--cb-max-memory-percent 0.2` in `api-local-start`), on one RTX 4090; it is not where batching stops paying (the median gap between tokens grows only from about 41 to 56 ms) and not the GPU's capacity. -->

```{.python .input #api-sweep-local}
%%tab pytorch
# 64 tokens keep the recording near its 30-minute target
SWEEP = {'max_tokens': 64, 'temperature': 0, 'stream': True}

async def closed_loop(api, c, first, **params):
    """One level of a closed loop: 8c distinct GSM8K train questions from
    index `first` on, c in flight; closed_loop_stats of the results."""
    batch = [{'messages': [{'role': 'user', 'content': q}], **params}
             for q in gsm_train.question.iloc[first:first + 8 * c]]
    results = await chat_batch(api, batch, concurrency=c)
    failed = [r for r in results if isinstance(r, Exception)]
    assert not failed, (  # a failure is not cached: replay would miss it
        f'c = {c}: {len(failed)} failed, first {failed[0]!r:.100}; delete '
        f'{api.cache_path} and record the arm again')
    assert not any(d2l.chat_thinks(r) for r in results), f'c = {c}: thinking'
    return {'c': c, **closed_loop_stats(results, c)}

def check_little(table, little, sustained):
    for _, row in table.iterrows():
        assert abs(row.little - 1) < little and abs(row.sustained - 1) < (
            sustained), (f'c = {row.c}: little {row.little:.3f}, sustained '
                         f'{row.sustained:.3f}, {row.n_retries} retries, '
                         f'{row.n_window} completions in the window')

def show(table):
    """Print a sweep: X in requests/s, X_tokens in tokens/s, times in s, and
    'in decode' = X (W - ttft_mean), Little's law for the decode stage."""
    table = table.assign(decode=table.X * (table.W - table.ttft_mean))
    columns = [k for k in ('pass', 'c', 'X', 'X_tokens', 'W', 'ttft_p50',
                           'ttft_p90', 'decode', 'little', 'sustained')
               if k in table]
    counts = ['n_429', 'n_retries', 'n_failed']
    if table[counts].to_numpy().any():
        columns += counts
    print(table[columns].rename(columns={'decode': 'in decode'}).to_string(
        index=False, formatters={'X': '{:.2f}'.format,
                                 'X_tokens': '{:.0f}'.format,
                                 'in decode': '{:.1f}'.format},
        float_format='{:.3f}'.format))
    if not table[counts].to_numpy().any():
        print('no 429s, retries or failed requests at any level')

LEVELS = [1, 2, 4, 8, 16, 32, 64, 128] if LOCAL_BATCHES else [1, 2, 4, 8]
sweep_api = local(rec.cache('lm-api-sweep-v1', 'local'))
rng, order = random.Random(0), []
for p in (1, 2):
    levels = list(LEVELS)
    rng.shuffle(levels)
    order += [(p, c) for c in levels]
first, rows = 4, []  # train items 4, 5, ... in execution order
for p, c in order:
    rows.append({'pass': p, **await closed_loop(sweep_api, c, first,
                                                 **SWEEP)})
    first += 8 * c
sweep = pd.DataFrame(rows).sort_values(['pass', 'c'], ignore_index=True)
check_little(sweep, 0.05, 0.03)
show(sweep)
```

<!-- Placeholder (stage 2): the fitted closed-loop curve and the knee rule, per pass. -->

```{.python .input #api-sweep-fit}
%%tab pytorch
def fit(table):
    """W_0, X_max, c* and the knee: the smallest level from which on the
    efficiency X W_0 / c stays below 0.7."""
    table = table.sort_values('c')
    W_0, X_max = table.W[table.c == 1].median(), table.X.max()
    E = (table.X * W_0 / table.c).to_numpy()
    knee = next((c for i, c in enumerate(table.c) if (E[i:] < 0.7).all()),
                None)
    return W_0, X_max, X_max * W_0, knee

knees = []
for p, table in sweep.groupby('pass'):
    W_0, X_max, c_star, knee = fit(table)
    knees.append(knee)
    print(f'pass {p}: W_0 {W_0:.2f} s, X_max {X_max:.1f} requests/s, '
          f'c* {c_star:.1f}, efficiency below 0.7 from c = {knee}')
both = knees[0] is not None and knees[0] == knees[1]
print('knee in both passes:', 'yes' if both else 'no')
assert both, "no knee common to both passes: set the server's batch cap"
W_0, X_max, _, _ = fit(sweep)  # pooled, for the fitted curves
tokens = (sweep.X_tokens / sweep.X).mean()  # completion tokens per request
grid = np.geomspace(1, max(LEVELS), 100)
passes = [sweep[sweep['pass'] == p] for p in (1, 2)]
fig, axes = d2l.plt.subplots(1, 2, figsize=(7.4, 2.6),
                             gridspec_kw={'wspace': 0.4})
d2l.plot([t.c.to_numpy() for t in passes] + [grid],
         [t.X_tokens.to_numpy() for t in passes] +
         [np.minimum(grid / W_0, X_max) * tokens], 'requests in flight c',
         'completion tokens/s', legend=['pass 1', 'pass 2', 'fit'],
         xscale='log', fmts=('o', 's', '-'), axes=axes[0])
d2l.plot([t.c.to_numpy() for t in passes] + [grid],
         [t.W.to_numpy() for t in passes] + [np.maximum(W_0, grid / X_max)],
         'requests in flight c', 'mean latency W (s)',
         legend=['pass 1', 'pass 2', 'fit'], xscale='log',
         fmts=('o', 's', '-'), axes=axes[1])
```

<!-- Placeholder (stage 2): the same client against the hosted flagship: one client does not reach its capacity. -->

```{.python .input #api-sweep-hosted}
%%tab pytorch
contrast = rec.hosted(FLAGSHIP, rec.cache('lm-api-sweep-v1', 'flagship'),
                      share=0.1, reasoning={'enabled': False})
first, rows = 4084, []  # train items after the local sweep's
for c in (1, 8, 32):
    rows.append(await closed_loop(contrast, c, first, **SWEEP))
    first += 8 * c
hosted_sweep = pd.DataFrame(rows)
check_little(hosted_sweep, 0.10, 0.05)
X_c = hosted_sweep.set_index('c').X
print(f'X(32) / X(1) = {X_c[32] / X_c[1]:.1f}')
show(hosted_sweep)
```

<!-- Placeholder (stage 2): Experiment 2: identical 4-shot prompts, decoding, thinking mode and extractor for every arm. -->

```{.python .input #api-gsm8k-setup}
%%tab pytorch
GSM_DEMOS = [(q, re.sub(r'<<[^>]*>>', '', a))  # calculator notes removed
             for q, a in zip(gsm_train.question[:4], gsm_train.answer[:4])]
GSM_MESSAGES = [d2l.fewshot_prompt(GSM_DEMOS, gsm_test.question[i],
                                   template='chat_multi') for i in GSM_IDS]
GSM_GOLD = [d2l.extract_answer(gsm_test.answer[i]) for i in GSM_IDS]
GSM_PARAMS = {'temperature': 0, 'seed': 0, 'max_tokens': 512}
for m in GSM_MESSAGES[0]:
    print(f'{m["role"]:9s} {m["content"][:64]!r}')
print('parameters:', GSM_PARAMS)
print('thinking off: reasoning.enabled=False (router), '
      'enable_thinking=False (local)')
```

<!-- Placeholder (stage 2): four arms at once; per arm accuracy with its interval, tokens and latency. -->

```{.python .input #api-gsm8k-run}
%%tab pytorch
gsm_flagship = rec.hosted(FLAGSHIP, rec.cache('lm-api-gsm8k-v1', 'flagship'),
                          share=0.2, reasoning={'enabled': False})
gsm_second = rec.hosted(SECOND, rec.cache('lm-api-gsm8k-v1', 'second'),
                        share=0.2, reasoning={'enabled': False})
gsm_local = local(rec.cache('lm-api-gsm8k-v1', 'local'))
ARMS = {'flagship r0': (gsm_flagship, 0, 8),
        'flagship r1': (gsm_flagship, 1, 8),
        'Qwen3.8-27B': (gsm_second, 0, 8),
        'Qwen3-4B local': (gsm_local, 0, 8 if LOCAL_BATCHES else 1)}
outs = await asyncio.gather(*(chat_batch(api, [
    {'messages': m, 'replica': k, **GSM_PARAMS} for m in GSM_MESSAGES],
    concurrency=c) for api, k, c in ARMS.values()))
RESULTS, CORRECT, rows = dict(zip(ARMS, outs)), {}, []
for arm, results in RESULTS.items():
    assert not any(isinstance(r, Exception) for r in results), arm
    assert not any(d2l.chat_thinks(r) for r in results), f'{arm}: thinking'
    correct = np.array([d2l.extract_answer(r.text) == g
                        for r, g in zip(results, GSM_GOLD)], dtype=float)
    CORRECT[arm], ci = correct, d2l.bootstrap_ci(correct)
    p50, p90 = np.percentile([r.t_total for r in results], [50, 90])
    rows.append({'arm': arm, 'concurrency': ARMS[arm][2],
                 'accuracy': ci.mean, 'lo': ci.ci[0], 'hi': ci.ci[1],
                 'prompt tokens': np.mean([r.usage['prompt_tokens']
                                           for r in results]),
                 'completion tokens': np.mean([r.usage['completion_tokens']
                                               for r in results]),
                 'p50 s': p50, 'p90 s': p90})
gsm = pd.DataFrame(rows).set_index('arm')
print(gsm.to_string(formatters={
    'prompt tokens': '{:.0f}'.format, 'completion tokens': '{:.0f}'.format,
    'p50 s': '{:.1f}'.format, 'p90 s': '{:.1f}'.format},
    float_format='{:.2f}'.format))
```

<!-- Placeholder (stage 2): the price of a correct answer, as billed and at list price, and the local arm's GPU-seconds. Part of the cost-per-correct ratio is prompt caching at Novita (the 4-shot prefix is billed at the cached price on most flagship calls; Parasail caches none): quote both the billed and the list-price ratio, "about 4 to 6 times", not one figure. The local arm's GPU-seconds per correct are measured at concurrency 8, below the knee (about 32): say so here and in Exercise 3. No cell measures the figure near the knee, so quote no factor for it. -->

```{.python .input #api-gsm8k-cost}
%%tab pytorch
rows = []
for arm, (api, _, _) in ARMS.items():
    results, n_correct = RESULTS[arm], max(CORRECT[arm].sum(), 1)
    if api.price is not None:  # hosted: dollars, billed and at list price
        cost = sum(usd(api, r) for r in results)
        listed = sum(d2l.api_cost({**r.usage, 'cached_tokens': 0}, api.price)
                     for r in results)  # as if no prompt token were cached
        cached = sum(r.usage.get('cached_tokens') or 0 for r in results)
        rows.append({'arm': arm, 'cached share': cached / sum(
                         r.usage['prompt_tokens'] for r in results),
                     'USD per 100': 100 * cost / len(results),
                     'USD per 1,000 correct': 1000 * cost / n_correct,
                     'at list price': 1000 * listed / n_correct})
    else:  # local: the arm's wall-clock span on one GPU
        span = (max(r.t_send + r.t_total for r in results) -
                min(r.t_send for r in results))
        rows.append({'arm': arm, 'GPU-s per correct': span / n_correct})
cost = pd.DataFrame(rows).set_index('arm')
ratio = cost.loc['Qwen3.8-27B'] / cost.loc['flagship r0']
print(f'Qwen3.8-27B / flagship: accuracy x '
      f'{gsm.accuracy["Qwen3.8-27B"] / gsm.accuracy["flagship r0"]:.2f}; '
      f'USD per correct x {ratio["USD per 1,000 correct"]:.2f} billed, '
      f'x {ratio["at list price"]:.2f} at list price')
print(cost.to_string(na_rep='-', formatters={
    'cached share': '{:.2f}'.format, 'USD per 100': '{:.4f}'.format,
    'USD per 1,000 correct': '{:.3f}'.format,
    'at list price': '{:.3f}'.format, 'GPU-s per correct': '{:.2f}'.format}))
```

<!-- Placeholder (stage 2): paired comparisons with exact McNemar tests, Holm-adjusted, against the replica noise floor. -->

```{.python .input #api-gsm8k-compare}
%%tab pytorch
def holm(p):  # Holm's step-down adjustment of p-values
    adjusted, running = np.empty(len(p)), 0.0
    for rank, i in enumerate(np.argsort(p)):
        running = max(running, min(1.0, (len(p) - rank) * p[i]))
        adjusted[i] = running
    return adjusted

PAIRS = {'flagship - 27B': ('flagship r0', 'Qwen3.8-27B'),
         'flagship - local': ('flagship r0', 'Qwen3-4B local'),
         '27B - local': ('Qwen3.8-27B', 'Qwen3-4B local'),
         'r0 - r1': ('flagship r0', 'flagship r1')}  # the noise floor
rows = []
for name, (a, b) in PAIRS.items():
    x, y = CORRECT[a], CORRECT[b]
    res = d2l.paired_bootstrap(x, y, n_boot=10000, seed=0)
    n_ab, n_ba = int((x > y).sum()), int((x < y).sum())  # discordant
    p = (stats.binomtest(min(n_ab, n_ba), n_ab + n_ba, 0.5).pvalue
         if n_ab + n_ba else 1.0)
    rows.append({'comparison': name, 'difference': res.mean,
                 'lo': res.ci[0], 'hi': res.ci[1], 'b': n_ab, 'c': n_ba,
                 'McNemar p': p, 'MDE': d2l.min_detectable_effect(
                     np.std(x - y, ddof=1), len(x))})
compare = pd.DataFrame(rows)
compare['Holm p'] = list(holm(compare['McNemar p'][:3].to_numpy())) + [None]
floor = compare.iloc[3]
if floor.lo > 0 or floor.hi < 0:
    print('replicas disagree systematically: no verdicts')
else:
    compare['verdict'] = [
        'resolved' if h < 0.05 and (lo > 0 or hi < 0)
        else 'unresolved at n = 100' for h, lo, hi in zip(
            compare['Holm p'][:3], compare.lo[:3], compare.hi[:3])] + [
        'noise floor']
print(compare.to_string(index=False, na_rep='', formatters={
    'difference': '{:.2f}'.format, 'lo': '{:.2f}'.format,
    'hi': '{:.2f}'.format, 'McNemar p': '{:.4f}'.format,
    'MDE': '{:.2f}'.format, 'Holm p': '{:.4f}'.format}))
```

<!-- Placeholder (stage 2): Experiment 3: identical requests at temperature 0 with a seed: twenty to the hosted flagship, five each to the local server alone and under load. The cause of the local-alone divergence was checked on the GPU (stage 2b, verify-18.7-outputs.md): with prefix block sharing off (ContinuousBatchingConfig.allow_block_sharing=False; no CLI flag) five sequential repeats agree on all three problems, with it on replica 0 differs from 1 to 4 on two of three (later replicas reuse the cached 4-shot prompt blocks); under 8 background lanes they disagree with sharing on or off, so batch composition alone also breaks determinism. First divergence is counted in characters, not tokens. -->

```{.python .input #api-determinism}
%%tab pytorch
DET_IDS = LONGEST[:3]  # the longest reference solutions
DET_MESSAGES = [GSM_MESSAGES[GSM_IDS.index(i)] for i in DET_IDS]

def replicas(first, n):  # n replicas of each of the three requests
    return [{'messages': m, 'replica': k, **GSM_PARAMS}
            for m in DET_MESSAGES for k in range(first, first + n)]

N_LOCAL = 5  # 30 local replies of 13-18 s stay inside the 30-min target
det_flagship = rec.hosted(FLAGSHIP, rec.cache('lm-api-determinism-v1',
                                              'flagship'),
                          share=0.1, reasoning={'enabled': False})
det_local = local(rec.cache('lm-api-determinism-v1', 'local'))
on_flagship, alone = await asyncio.gather(
    chat_batch(det_flagship, replicas(0, 20), concurrency=4),
    chat_batch(det_local, replicas(0, N_LOCAL), concurrency=1))
BACKGROUND = [i for i in range(len(gsm_test)) if i not in GSM_IDS][:64]
LANES = 8 if LOCAL_BATCHES else 1  # more would queue at a serial server
print(f'under load: {LANES} background lane(s); the local server '
      f'{"batches" if LOCAL_BATCHES else "runs one request at a time"}')

def background(j):  # request j of the load: 64 other problems, cycled
    return {'messages': d2l.fewshot_prompt(GSM_DEMOS, gsm_test.question[
        BACKGROUND[j % 64]], template='chat_multi'), 'replica': j // 64,
            **GSM_PARAMS}

load = (d2l.BackgroundLoad(local(rec.cache('lm-api-determinism-v1',
                                           'background'), mode='live'),
                           background, LANES)
        if LOCAL_LIVE else contextlib.nullcontext())
async with load:  # the load's responses are logged, never replayed
    under_load = await chat_batch(det_local, replicas(100, N_LOCAL),
                                  concurrency=1)
if LOCAL_LIVE:
    print(f'background: {len(load.failures)} failed requests')
rows = []
for condition, results in (('flagship', on_flagship), ('local alone', alone),
                           ('local under load', under_load)):
    assert not any(isinstance(r, Exception) or d2l.chat_thinks(r)
                   for r in results)
    n = len(results) // len(DET_IDS)  # replicas per problem
    for k, i in enumerate(DET_IDS):
        texts = [r.text for r in results[n * k:n * k + n]]
        modal = collections.Counter(texts).most_common(1)[0][0]
        firsts = [len(os.path.commonprefix([t, modal]))  # in characters
                  for t in texts if t != modal]
        rows.append({'condition': condition, 'problem': i, 'replicas': n,
                     'distinct': len(set(texts)),
                     'diverges at (min)': min(firsts, default=None),
                     '(median)': np.median(firsts) if firsts else None,
                     'answers': len({d2l.extract_answer(t) for t in texts})})
print(pd.DataFrame(rows).to_string(index=False, na_rep='-', formatters={
    'diverges at (min)': '{:.0f}'.format, '(median)': '{:g}'.format}))
```

<!-- Placeholder (stage 2): the local server is stopped and its port freed. -->

```{.python .input #api-local-stop}
%%tab pytorch
if server is not None:
    print('server exit code', server.stop())
else:
    print('no local server in replay')
```

<!-- Placeholder (stage 2): what every client's recording did over all its sessions: live calls, retries, 429s, failed attempts and spend; one provider and one served id per arm; the openai SDK version of the recording. Quote the recording's totals from the committed-caches line (it includes the background lanes and the local probe); the closed-model probe's spend is outside the committed caches. -->

```{.python .input #api-manifest}
%%tab pytorch
rows, ids = [], set()
for api in {api.calls_path: api for api in rec.clients}.values():
    t = d2l.api_recorded(api)  # totals of every session of the recording
    key = next((k for k in API_SHA1 if k in api.cache_path), 'scratch')
    arm = os.path.splitext(os.path.basename(api.cache_path))[0]
    rows.append({'cache': key.replace('lm-api-', '') + '/' + arm,
                 **{k: t.get(k, 0) for k in (
                     'live calls', 'retries', '429s', 'failed', 'USD',
                     'usage cost', 'listing cost')}})
    if api.price is not None:  # one provider and one served id per arm
        assert len(t['served']) <= 1 and len(t['providers']) <= 1, t
    ids.update((re.sub(r'@([0-9a-f]{8})[0-9a-f]+', r'@\1', s),
                'local' if p == 'None' else p)
               for s, p in zip(t['served'], t['providers']))
manifest = pd.DataFrame(rows)
spent = sum(api.spent_usd for api in rec.clients)  # this run's live calls
total = manifest.USD.sum()  # every session of the recording
assert spent <= COST_CAP_USD and total <= COST_CAP_USD, (spent, total)
reported = manifest['usage cost'].sum()  # where the provider sent a cost
if reported:
    print('cost from the listing / cost the provider reported: '
          f'{manifest["listing cost"].sum() / reported:.4f}')
if API_MODE != 'replay':  # sessions accumulate in each artifact's manifest
    d2l.write_api_manifests(rec.clients, RECORD_DIR)
if os.environ.get('OPENROUTER_API_KEY'):  # in no cache, log or manifest
    n = d2l.assert_key_absent([RECORD_DIR, os.path.join(
        D2L_API_LOG_DIR, 'scratch')], os.environ['OPENROUTER_API_KEY'])
    print(f'key material: none in {n} files')
print(f'clients of this notebook: {manifest["live calls"].sum()} live '
      f'calls, {total:.4f} USD of {COST_CAP_USD}; this run {spent:.4f} USD')
committed = d2l.api_committed([rec.folder(k) for k in API_SHA1])
print(f'committed caches: {committed["live calls"]} live calls, '
      f'{committed["USD"]:.4f} USD, openai SDK {", ".join(committed["sdk"])}')
print('(the background lanes and the local probe included, the '
      'closed-model probe not)')
for s, p in sorted(ids):
    print(f'served id {s} by {p}')
print(manifest.drop(columns=['usage cost', 'listing cost']).to_string(
    index=False, formatters={'USD': '{:.6f}'.format}))
```
