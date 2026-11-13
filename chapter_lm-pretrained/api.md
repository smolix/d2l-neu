# Models behind an API
:label:`sec_lm_api`

<!-- execution: locked; requires: openrouter -->

<!-- Author brief, copied verbatim from the Language Models part plan
(docs/llm/language_models_part_plan.md, "1.7 Models behind an API · T0").
Blueprint: none (the plan adds this section).
The plan is a specification, not book text; this comment is replaced
when the section is written.

**Idea:** A hosted model is a service with a protocol, a price and limits that the client can measure, and what it cannot return is why the rest of the part works with open weights.
**Opening:** DeepSeek-V4.1-Flash has 763 billion parameters, more than one A100 node holds even at four bits, yet it answers a GSM8K problem in about a second for a printed fraction of a cent. The section treats the client as a measuring instrument (latency, throughput under concurrency, cost, reproducibility) and closes on what survives of the chapter's toolkit when the weights are someone else's.
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
import datetime
import hashlib
import heapq
import gc
import json
import os
import random
import re
import shutil
import sys
import time
import zipfile
import huggingface_hub
import pandas as pd
import requests
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
# Development, smoke and closed-model logs stay outside the repository
D2L_API_LOG_DIR = os.environ.get('D2L_API_LOG_DIR', os.path.expanduser(
    '~/.local/share/d2l/lm-api'))
FLAGSHIP = 'deepseek/deepseek-v4.1-flash-1017'  # a dated snapshot id
SECOND = 'qwen/qwen3.8-27b'
CLOSED = 'openai/gpt-5-nano'  # probe only; its outputs are never committed
PROVIDERS = {FLAGSHIP: 'novita', SECOND: 'parasail', CLOSED: 'openai'}
LISTED = [FLAGSHIP, SECOND, CLOSED]  # the router models of the listing
# Qwen3-4B, not Qwen3.5-4B: continuous batching in Transformers 5.22.2
# has no cache for Qwen3.5's linear-attention layers, so it would serve
# one request at a time
LOCAL = 'Qwen/Qwen3-4B'  # behind transformers serve on our GPU
WEIGHTS = {  # the open weights behind each model, at a pinned commit
    FLAGSHIP: ('deepseek-ai/DeepSeek-V4.1-Flash-1017',
               '1bae0e1088993b9b82de08f59de5be5a34e42170'),
    SECOND: ('Qwen/Qwen3.8-27B', '37561816e66b5b9153b1b64c10968c18c1e6f4a0'),
    LOCAL: (LOCAL, d2l.HF_REVISIONS[LOCAL])}
LOCAL_MODEL = f'{LOCAL}@{d2l.HF_REVISIONS[LOCAL]}'  # requests name id@rev
LOCAL_LIVE = API_MODE != 'replay' and torch.cuda.is_available()
LOCAL_MODE = API_MODE if LOCAL_LIVE else 'replay'
API_SHA1 = {  # the committed caches (zips), recorded 2026-11-13
    'lm-api-listing-v1': '4916ce668be31ed37947dac80bc094a70ee75efc',
    'lm-api-fields-v1': '1978c17699e6266ab84072db23a5699dfa2d42e2',
    'lm-api-probe-v1': '18ffdf9df2c8aa0de8809f598f041c6d5fe43a56',
    'lm-api-sweep-v1': '63250dccbf90eae6143b5affe74ae25972f514b1',
    'lm-api-gsm8k-v1': '1fbb38c4cf5f65fb69da5ba3719cd850ba768dc6',
    'lm-api-determinism-v1': '448e39446d960dac298a5db3a4e6c0b95051683b',
    'lm-api-effort-v1': '7480d44c1bc2b6f8d2ac90aadf9f50689b569b9b'}
for key, sha1 in API_SHA1.items():
    if sha1 is not None:
        d2l.DATA_HUB[key] = (d2l.LM_DATA_URL + key + '.zip', sha1)
RECORD_DIR = os.path.join('..', 'data', 'lm-api', 'record')

def unpack(key):
    """Download a registered zip (sha1 checked) and unpack it once, into a
    folder named by the sha1 (a re-registered key gets a fresh folder)."""
    path = d2l.download_keyed(key)
    folder = os.path.join(os.path.dirname(path),
                          'files-' + d2l.DATA_HUB[key][1][:12])
    if not os.path.isdir(folder):
        with zipfile.ZipFile(path) as z:
            z.extractall(folder + '.part')
        os.replace(folder + '.part', folder)
    return folder

def api_dir(key):
    """The folder of one artifact: the unpacked zip in replay; otherwise
    RECORD_DIR/key, seeded once from the zip when it is registered (the
    downloaded file itself is never appended to)."""
    if API_MODE == 'replay':
        if key not in d2l.DATA_HUB:
            raise KeyError(f'{key} is not registered yet: record it first '
                           '(D2L_API_MODE=record)')
        return unpack(key)
    folder = os.path.join(RECORD_DIR, key)
    if not os.path.isdir(folder):
        try:
            shutil.copytree(unpack(key), folder)
        except Exception as e:  # not registered, or the download failed
            os.makedirs(folder, exist_ok=True)
            print(f'{key}: recording into an empty cache ({e!r:.60})')
    return folder

def cache_file(key, arm):
    return os.path.join(api_dir(key), arm + '.jsonl')

def calls_log(api):
    """The live attempts in a client's calls log: every session of the
    recording (a torn last line, from a killed run, is skipped)."""
    if api.calls_path is None or not os.path.exists(api.calls_path):
        return []
    calls = []
    with open(api.calls_path, encoding='utf-8') as f:
        for line in f:
            try:
                calls.append(json.loads(line))
            except ValueError:
                pass
    return calls

CLIENTS, SHARES = [], {}  # every client with a cache; budget per cache

def hosted(model, cache, *, share, **kwargs):
    """A client of model at its pinned provider that may spend the fixed
    fraction `share` of COST_CAP_USD over the whole recording, less what
    earlier sessions spent on this cache. The shares sum to at most 1, so
    concurrent clients cannot pass the cap together."""
    SHARES[cache] = share
    assert sum(SHARES.values()) <= 1 + 1e-9, SHARES
    api = d2l.openrouter_api(model, PROVIDERS[model], listing=LISTING,
                             cache=cache, mode=API_MODE,
                             cost_cap_usd=share * COST_CAP_USD, **kwargs)
    api.cost_cap_usd = max(0.0, api.cost_cap_usd - sum(
        c.get('cost') or 0.0 for c in calls_log(api)))
    CLIENTS.append(api)
    return api

def usd(api, r):  # cost of one response: the provider's, else the listing's
    cost = r.usage.get('cost')
    return d2l.api_cost(r.usage, api.price) if cost is None else cost

print(f'mode {API_MODE}, cost cap {COST_CAP_USD} USD, local server '
      f'{"live" if LOCAL_LIVE else "replayed"}')
```

<!-- Placeholder (stage 2): a dated snapshot of the router listing: providers, quantization, prices and supported fields. -->

```{.python .input #api-listing}
%%tab pytorch
listing_file = os.path.join(api_dir('lm-api-listing-v1'), 'listing.json')
if not os.path.exists(listing_file):  # record: snapshot the free listing
    url = 'https://openrouter.ai/api/v1/models'
    def get(u):
        r = requests.get(u, timeout=60)
        r.raise_for_status()
        return r.json()['data']
    rows = {m['id']: m for m in get(url)}
    now = datetime.datetime.now(datetime.timezone.utc)
    snapshot = {'retrieved': now.strftime('%Y-%m-%dT%H:%M:%SZ'),
                'models': {m: rows[m] for m in LISTED},
                'endpoints': {m: get(f'{url}/{m}/endpoints')['endpoints']
                              for m in LISTED},
                'hf': {}}
    for m, (repo, rev) in WEIGHTS.items():  # parameter totals of the Hub
        info = huggingface_hub.HfApi().model_info(repo, revision=rev)
        snapshot['hf'][m] = {'repo': repo, 'revision': rev,
                             'params': info.safetensors.total}
    with open(listing_file, 'w') as f:
        json.dump(snapshot, f, indent=1)
with open(listing_file) as f:
    LISTING = json.load(f)

def endpoint(model):  # the row of the pinned provider, as priced below
    rows, p = LISTING['endpoints'][model], PROVIDERS[model]
    return next((e for e in rows if e['tag'] == p), None) or next(
        e for e in rows if e['tag'].split('/')[0] == p)

rows = []
for m in LISTED:
    e, price = endpoint(m), d2l.price_from_listing(LISTING, m, PROVIDERS[m])
    rows.append({'model': m, 'provider': e['tag'],
                 'quantization': e['quantization'],
                 'context': e['context_length'], 'in $/M': price.input,
                 'out $/M': price.output, 'cached $/M': price.cached_input,
                 **{f: f in e['supported_parameters'] for f in (
                     'seed', 'logprobs', 'structured_outputs', 'reasoning')}})
print('listing retrieved', LISTING['retrieved'])
pd.DataFrame(rows)
```

<!-- Placeholder (stage 2): the open models people use most do not fit our GPUs, even at four bits per weight. -->

```{.python .input #api-weights-arith}
%%tab pytorch
PRECISIONS = (('bf16', 2), ('fp8', 1), ('4 bits', 0.5))  # bytes/weight

def best_fit(n, gb):  # the highest precision whose weights fit in gb
    return next((p for p, b in PRECISIONS if n * b <= gb * 1e9), 'none')

rows = []
for m in (FLAGSHIP, SECOND, LOCAL):
    n = LISTING['hf'][m]['params']
    rows.append({'weights': LISTING['hf'][m]['repo'], 'params (B)': n / 1e9,
                 **{f'{p} (GB)': n * b / 1e9 for p, b in PRECISIONS},
                 '24 GB card': best_fit(n, 24),
                 '8 x 40 GB node': best_fit(n, 8 * 40)})
pd.DataFrame(rows).round(1)
```

<!-- Placeholder (stage 2): one request to the flagship: a GSM8K problem in, text, a served id and a usage block out. -->

```{.python .input #api-first-call}
%%tab pytorch
gsm_train = pd.read_parquet(d2l.download_keyed('gsm8k-train'))
gsm_test = pd.read_parquet(d2l.download_keyed('gsm8k-test'))
samples = pd.read_json(d2l.download_keyed('lm-gsm8k-samples-v1'),
                       lines=True)
GSM_IDS = [int(i) for i in dict.fromkeys(samples.problem_id)]  # 18.4's
assert GSM_IDS == list(np.random.default_rng(0).permutation(
    len(gsm_test))[:100])  # its seeded rule
flagship = hosted(FLAGSHIP, cache_file('lm-api-probe-v1', 'flagship'),
                  share=0.05, reasoning={'enabled': False})
question = [{'role': 'user', 'content': gsm_test.question[GSM_IDS[0]]}]
r = flagship.chat(question, temperature=0, seed=0, max_tokens=512)
print(r.text)
print(f'finish {r.finish_reason}; usage {r.usage}')
print(f'{r.t_total:.2f} s; served {r.model} by {r.provider}; '
      f'{d2l.api_cost(r.usage, flagship.price):.6f} USD from the listing')
```

<!-- Placeholder (stage 2): hidden reasoning is billed as completion tokens and can use up the whole output budget (on the problem with the longest reference solution: on the first problem Qwen3.8-27B reasoned for 39 tokens and answered inside 64). -->

```{.python .input #api-reasoning-trap}
%%tab pytorch
second = hosted(SECOND, cache_file('lm-api-probe-v1', 'second'),
                share=0.05)
LONGEST = sorted(GSM_IDS, key=lambda i: -len(gsm_test.answer[i]))
hard = [{'role': 'user', 'content': gsm_test.question[LONGEST[0]]}]
trap = second.chat(hard, temperature=0, seed=0, max_tokens=64)  # thinks
print(f'content {trap.text!r}, finish {trap.finish_reason}, '
      f'{len(trap.reasoning or "")} characters of reasoning; {trap.usage}')
assert trap.text == '' and trap.finish_reason == 'length'
assert trap.usage['reasoning_tokens'] == trap.usage['completion_tokens'] == 64
answer = second.chat(hard, temperature=0, seed=0, max_tokens=512,
                     extra_body={'reasoning': {'enabled': False}})
print(f'thinking off: {answer.text[-80:]!r}; {answer.usage}')
assert not answer.usage['reasoning_tokens']
```

<!-- Placeholder (stage 2): the effort field moves reasoning tokens and cost, not the protocol (an anecdote on five problems). -->

```{.python .input #api-effort}
%%tab pytorch
effort_api = hosted(FLAGSHIP, cache_file('lm-api-effort-v1', 'flagship'),
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
print(medians.round(4))
ordered = (np.diff(medians['reasoning tokens']) > 0).all()
print('more reasoning tokens at higher effort:', 'yes' if ordered else 'no')
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

def local(cache, **kwargs):
    """A client of the local server, thinking switched off."""
    api = d2l.ChatAPI(LOCAL_URL, LOCAL_MODEL, cache=cache, mode=LOCAL_MODE,
                      http_client=LOCAL_HTTP, default_params=NO_THINK_LOCAL,
                      **kwargs)
    CLIENTS.append(api)
    return api
```

<!-- Placeholder (stage 2): a request field a provider accepts is not necessarily one it honours: one detection rule per field. -->

```{.python .input #api-probe}
%%tab pytorch
STORY = [{'role': 'user', 'content':
          'Write a short paragraph about a lighthouse keeper.'}]
COUNT = [{'role': 'user', 'content':
          'Count from 1 to 10, separated by commas.'}]
JSON_Q = [{'role': 'user', 'content': 'What is 6 times 7? Answer in JSON.'}]
SCHEMA = {'type': 'json_schema', 'json_schema': {
    'name': 'answer', 'strict': True, 'schema': {
        'type': 'object', 'properties': {'answer': {'type': 'integer'}},
        'required': ['answer'], 'additionalProperties': False}}}

def is_answer(text):  # exactly {"answer": <integer>}
    try:
        out = json.loads(text)
    except ValueError:
        return False
    return (isinstance(out, dict) and set(out) == {'answer'} and
            type(out['answer']) is int)

def n_choices(api, a):  # further choices are kept in the cache's extra
    return max(1, len(api.entries(a.request_hash)[0]['extra'].get(
        'choices', [])))

def covered(api, a):  # top-5 log-probabilities for every generated token
    n, out = len(a.logprobs or []), a.usage['completion_tokens']
    full = all(len(t) == 5 for _, _, t in a.logprobs or [])
    status = ('honoured' if full and n >= out - 1 else  # the end token
              'partial' if n else 'ignored')
    return status, f'{n} of {out} tokens'

def thinks(r):  # a streamed empty think block arrives as '\n\n'
    return bool(r.usage['reasoning_tokens'] or (r.reasoning or '').strip())

def echoed(api, a):  # prompt log-probabilities, or the prompt echoed
    lp = 'prompt_logprobs' in api.entries(a.request_hash)[0]['extra']
    echo = COUNT[0]['content'] in a.text
    return lp or echo, f'prompt_logprobs: {lp}, echo: {echo}'

def tests(base, on, off, max_tokens):
    """field -> (requests as (messages, replica, params), rule mapping the
    client and the results to (honoured?, evidence)); base is the request
    body every test sends, on and off the bodies of the thinking switch.
    Evidence holds flags, counts and lengths, never response text: the
    closed model's outputs may not be committed."""
    def p(body=base, **kw):
        return {'max_tokens': max_tokens, **kw,
                **({'extra_body': body} if body else {})}
    return {
        'seed': ([(STORY, 0, p(temperature=1, seed=7)),
                  (STORY, 1, p(temperature=1, seed=7)),
                  (STORY, 0, p(temperature=1, seed=8))],
                 lambda api, a, b, c: (a.text == b.text != c.text,
                                       f'7 = 7: {a.text == b.text}, '
                                       f'8 differs: {c.text != a.text}')),
        'n=2': ([(COUNT, 0, p(n=2))], lambda api, a: (
            n_choices(api, a) == 2, f'{n_choices(api, a)} choices returned')),
        'logprobs, top 5': ([(COUNT, 0, p(logprobs=True, top_logprobs=5))],
                            covered),
        "stop=['5']": ([(COUNT, 0, p(stop=['5']))], lambda api, a: (
            '4' in a.text and '5' not in a.text,
            f"has 4: {'4' in a.text}, has 5: {'5' in a.text}")),
        'response_format': ([(JSON_Q, 0, p(response_format=SCHEMA))],
                            lambda api, a: (is_answer(a.text), (
                                f'parses as {{answer: int}}: '
                                f'{is_answer(a.text)}, '
                                f'{len(a.text)} characters'))),
        'thinking switch': ([(COUNT, 0, p(body=on, max_tokens=256)),
                             (COUNT, 0, p(body=off))], lambda api, a, b: (
            thinks(a) and not thinks(b),
            f'on {thinks(a)}, off {thinks(b)}')),
        'max_completion_tokens=16': (
            [(STORY, 0, p(max_tokens=None, max_completion_tokens=16))],
            lambda api, a: (a.usage['completion_tokens'] <= 16,
                            f'{a.usage["completion_tokens"]} tokens')),
        'prompt log-probabilities': (  # vLLM's two fields
            [(COUNT, 0, p(body={**(base or {}), 'prompt_logprobs': 1,
                                'echo': True}))],
            echoed),
        'stream usage': ([(COUNT, 0, p(stream=True))], lambda api, a: (
            a.usage['completion_tokens'] is not None, 'final chunk'))}

TODAY = time.strftime('%Y-%m-%d', time.gmtime())

def day(results):  # UTC date of the newest live call behind the results
    return max(r.created for r in results)[:10]

def failure(e):  # (status, evidence) of a request that failed
    status = getattr(e, 'status_code', None)
    kind = 'rejected' if status and 400 <= status < 500 else 'error'
    return kind, f'{status or type(e).__name__}: {e}'[:80]

async def check(api, requests_, rule):
    try:
        results = [await api.achat(m, replica=k, **q)
                   for m, k, q in requests_]
    except (d2l.ReplayMiss, d2l.CostCapExceeded):
        raise
    except Exception as e:  # after the client's retries; never cached
        return (*failure(e), TODAY)
    honoured, evidence = rule(api, *results)  # a bool, or a status
    if not isinstance(honoured, str):
        honoured = 'honoured' if honoured else 'ignored'
    return honoured, evidence[:80], day(results)

async def probe(name, api, base, on, off, max_tokens, model_id):
    rows = []
    for field, (requests_, rule) in tests(base, on, off,
                                          max_tokens).items():
        status, evidence, date = await check(api, requests_, rule)
        rows.append({'endpoint': name, 'field': field, 'status': status,
                     'evidence': evidence, 'date': date})
    try:  # a live GET, never cached
        listed = any(i.startswith(model_id) for i in await asyncio.to_thread(
            api.models))
        status, evidence = ('honoured' if listed else 'ignored',
                            f'lists {model_id}: {listed}')
    except d2l.ReplayMiss:  # a replayed client cannot list: no verdict
        raise
    except Exception as e:
        status, evidence = failure(e)
    rows.append({'endpoint': name, 'field': '/v1/models', 'status': status,
                 'evidence': evidence, 'date': TODAY})
    params = {'max_tokens': max_tokens,
              **({'extra_body': base} if base else {})}
    one = await api.achat(COUNT, replica=10, **params)
    four = await d2l.chat_batch(api, [
        {'messages': COUNT, 'replica': k, **params} for k in range(11, 15)],
        concurrency=4)
    ratio = max(r.t_total for r in four) / one.t_total
    n_429 = sum(r.n_429 or 0 for r in [one, *four])  # retries add waits
    verdict = ('rate-limited' if n_429 else 'batched' if ratio < 2
               else 'serialized' if ratio > 3 else 'partial')
    rows.append({'endpoint': name, 'field': '4 concurrent requests',
                 'status': verdict,
                 'evidence': f'latency x {ratio:.2f}, {n_429} 429s',
                 'date': day([one, *four])})
    return rows

fields_file = os.path.join(api_dir('lm-api-fields-v1'), 'lm-api-fields.csv')
LOCAL_NAME = 'Qwen3-4B (local server)'
if API_MODE != 'replay':  # replay reads the committed table
    router_off, router_on = ({'reasoning': {'enabled': s}}
                             for s in (False, True))
    local_off, local_on = ({'chat_template_kwargs': {'enable_thinking': s}}
                           for s in (False, True))
    closed = hosted(CLOSED, os.path.join(D2L_API_LOG_DIR, 'scratch',
                                         'closed-probe.jsonl'), share=0.05)
    ENDPOINTS = {  # client, body of every test, thinking on and off,
        # output budget, the id /v1/models should list
        'DeepSeek-V4.1-Flash (Novita)': (flagship, router_off, router_on,
                                         router_off, 64, FLAGSHIP),
        'Qwen3.8-27B (Parasail)': (second, router_off, router_on,
                                   router_off, 64, SECOND),
        'gpt-5-nano (OpenAI)': (closed, None, router_on, router_off, 2048,
                                CLOSED)}  # a closed model thinks by default
    if LOCAL_LIVE:
        ENDPOINTS[LOCAL_NAME] = (
            local(cache_file('lm-api-probe-v1', 'local')), local_off,
            local_on, local_off, 64, LOCAL)
    rows = await asyncio.gather(*(probe(name, *args)
                                  for name, args in ENDPOINTS.items()))
    fields = pd.DataFrame([r for rs in rows for r in rs])
    if not LOCAL_LIVE:  # no server here: keep the recorded local column
        old = (pd.read_csv(fields_file) if os.path.exists(fields_file)
               else pd.DataFrame({'endpoint': []}))
        old = old[old.endpoint == LOCAL_NAME]
        assert len(old), 'no recorded local column: record on a GPU host'
        print('local server column kept from', old.date.max())
        fields = pd.concat([fields, old], ignore_index=True)
    fields.to_csv(fields_file, index=False)
fields = pd.read_csv(fields_file)
table = fields.pivot(index='field', columns='endpoint', values='status')
print(table.loc[fields.field.unique(), fields.endpoint.unique()])
batching = fields[(fields.endpoint == LOCAL_NAME) &
                  (fields.field == '4 concurrent requests')].iloc[0]
LOCAL_BATCHES = batching.status != 'serialized'
print(f'local server: {batching.status} ({batching.evidence})')
print('fields transformers serve ignores:', sorted(
    UNUSED_CHAT_COMPLETION_FIELDS & {
        'n', 'logprobs', 'top_logprobs', 'response_format',
        'max_completion_tokens', 'stream_options', 'seed'}))
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
    params = {k: v for k, v in d2l._chat_merge(params, {}).items()
              if k not in transport}
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
print('Novita, USD per million tokens:', novita)
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

<!-- Placeholder (stage 2): the derivation's mean sleep, and 64 clients that failed together. -->

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
for jitter in (False, True):
    attempts, t_done = retry_storm(jitter=jitter)
    print(f'jitter {jitter!s:5}: {attempts} attempts, all done at '
          f'{t_done:.1f} s')
```

<!-- Placeholder (stage 2): Little's law on the client's own timestamps, measured on the steady window. -->

```{.python .input #api-closed-loop}
%%tab pytorch
#@save
def closed_loop_stats(results, c):
    """Throughput and latency of a closed loop that kept c requests in
    flight. Exceptions count as n_failed and are excluded. For the N
    remaining results let s_i = t_send, f_i = s_i + t_total and f_(1) <= ...
    <= f_(N) the sorted completion times; N <= 2c raises ValueError. The
    steady window runs from t_a = f_(c) to t_b = f_(N-c). X counts the
    completions with t_a < f_i <= t_b per second of the window and X_tokens
    their completion tokens; W is the mean of f_i - s_i over requests with
    s_i >= t_a and f_i <= t_b; Lbar = sum_i max(0, min(f_i, t_b) -
    max(s_i, t_a)) / (t_b - t_a) is the mean number in flight. little =
    X W / c checks the client's bookkeeping, sustained = Lbar / c that it
    kept c in flight. X_tokens is None when a counted completion has no
    usage. ttft_p50 and ttft_p90 (from t_first) are over all N. The results
    must come from one closed loop: a d2l.ReplayMiss or d2l.CostCapExceeded
    among them, a mix of cached and live results, more than c requests in
    flight at once, or send times whose wall and monotonic clocks disagree
    by more than 2 s (two recording runs) raise ValueError."""
    d2l._check_closed_loop(results, c)  # one loop of c, from one run
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
    ttft = [r.t_first - r.t_send for r in ok if r.t_first is not None]
    p50, p90 = (map(float, np.percentile(ttft, [50, 90])) if ttft
                else (None, None))
    X_tokens = None if None in tokens else float(sum(tokens) / window)
    return {'X': float(X), 'X_tokens': X_tokens,
            'W': float(W), 'Lbar': float(Lbar / window),
            'little': float(X * W / c), 'sustained': float(Lbar / window / c),
            'ttft_p50': p50, 'ttft_p90': p90,
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

<!-- Placeholder (stage 2): Experiment 1: a closed-loop sweep against the local server, two passes in shuffled order. -->

```{.python .input #api-sweep-local}
%%tab pytorch
SWEEP = {'max_tokens': 64, 'temperature': 0, 'stream': True}  # 30-min run

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
    assert not any(thinks(r) for r in results), f'c = {c}: thinking'
    return {'c': c, **closed_loop_stats(results, c)}

def check_little(table, little, sustained):
    for _, row in table.iterrows():
        assert abs(row.little - 1) < little and abs(row.sustained - 1) < (
            sustained), (f'c = {row.c}: little {row.little:.3f}, sustained '
                         f'{row.sustained:.3f}, {row.n_retries} retries, '
                         f'{row.n_window} completions in the window')

COLUMNS = ['c', 'X', 'X_tokens', 'W', 'ttft_p50', 'ttft_p90', 'n_429',
           'n_retries', 'n_failed', 'little', 'sustained']
LEVELS = [1, 2, 4, 8, 16, 32, 64, 128] if LOCAL_BATCHES else [1, 2, 4, 8]
sweep_api = local(cache_file('lm-api-sweep-v1', 'local'))
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
sweep[['pass'] + COLUMNS].round(3)
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
assert both, 'no knee common to both passes (claim 2: set the batch cap)'
W_0, X_max, _, _ = fit(sweep)  # pooled, for the fitted curves
tokens = (sweep.X_tokens / sweep.X).mean()  # completion tokens per request
grid = np.geomspace(1, max(LEVELS), 100)
passes = [sweep[sweep['pass'] == p] for p in (1, 2)]
fig, axes = d2l.plt.subplots(1, 2, figsize=(7, 2.6))
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
contrast = hosted(FLAGSHIP, cache_file('lm-api-sweep-v1', 'flagship'),
                  share=0.1, reasoning={'enabled': False})
first, rows = 4084, []  # train items after the local sweep's
for c in (1, 8, 32):
    rows.append(await closed_loop(contrast, c, first, **SWEEP))
    first += 8 * c
hosted_sweep = pd.DataFrame(rows)
check_little(hosted_sweep, 0.10, 0.05)
X_c = hosted_sweep.set_index('c').X
print(f'X(32) / X(1) = {X_c[32] / X_c[1]:.1f}')
hosted_sweep[COLUMNS].round(3)
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
print('parameters:', GSM_PARAMS, '; thinking off: reasoning.enabled = '
      'False (router), chat_template_kwargs.enable_thinking = False (local)')
```

<!-- Placeholder (stage 2): four arms at once; per arm accuracy, latency, and the price of a correct answer. -->

```{.python .input #api-gsm8k-run}
%%tab pytorch
gsm_flagship = hosted(FLAGSHIP, cache_file('lm-api-gsm8k-v1', 'flagship'),
                      share=0.2, reasoning={'enabled': False})
gsm_second = hosted(SECOND, cache_file('lm-api-gsm8k-v1', 'second'),
                    share=0.2, reasoning={'enabled': False})
gsm_local = local(cache_file('lm-api-gsm8k-v1', 'local'))
ARMS = {'flagship r0': (gsm_flagship, 0, 8),
        'flagship r1': (gsm_flagship, 1, 8),
        'Qwen3.8-27B': (gsm_second, 0, 8),
        'Qwen3-4B local': (gsm_local, 0, 8 if LOCAL_BATCHES else 1)}
outs = await asyncio.gather(*(chat_batch(api, [
    {'messages': m, 'replica': k, **GSM_PARAMS} for m in GSM_MESSAGES],
    concurrency=c) for api, k, c in ARMS.values()))
CORRECT, rows = {}, []
for (arm, (api, _, _)), results in zip(ARMS.items(), outs):
    assert not any(isinstance(r, Exception) for r in results), arm
    assert not any(thinks(r) for r in results), f'{arm}: thinking'
    correct = np.array([d2l.extract_answer(r.text) == g
                        for r, g in zip(results, GSM_GOLD)], dtype=float)
    CORRECT[arm], ci = correct, d2l.bootstrap_ci(correct)
    latency = np.percentile([r.t_total for r in results], [50, 90])
    row = {'arm': arm, 'accuracy': ci.mean, 'lo': ci.ci[0], 'hi': ci.ci[1],
           'prompt tokens': np.mean([r.usage['prompt_tokens']
                                     for r in results]),
           'completion tokens': np.mean([r.usage['completion_tokens']
                                         for r in results]),
           'p50 s': latency[0], 'p90 s': latency[1]}
    n_correct = max(correct.sum(), 1)
    if api.price is not None:  # hosted: dollars
        cost = sum(usd(api, r) for r in results)
        row.update({'USD per 100': 100 * cost / len(results),
                    'USD per correct': cost / n_correct})
    else:  # local: the arm's wall-clock span on one GPU
        span = (max(r.t_send + r.t_total for r in results) -
                min(r.t_send for r in results))
        row['GPU-s per correct'] = span / n_correct
    rows.append(row)
gsm = pd.DataFrame(rows).set_index('arm')
per_correct = gsm['USD per correct']
print(f'Qwen3.8-27B / flagship: accuracy x '
      f'{gsm.accuracy["Qwen3.8-27B"] / gsm.accuracy["flagship r0"]:.2f}, '
      f'USD per correct x '
      f'{per_correct["Qwen3.8-27B"] / per_correct["flagship r0"]:.2f}')
gsm.round(4).assign(**{'USD per correct': gsm['USD per correct'].round(6)})
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

PAIRS = [('flagship r0', 'Qwen3.8-27B'), ('flagship r0', 'Qwen3-4B local'),
         ('Qwen3.8-27B', 'Qwen3-4B local'), ('flagship r0', 'flagship r1')]
rows = []
for a, b in PAIRS:
    x, y = CORRECT[a], CORRECT[b]
    res = d2l.paired_bootstrap(x, y, n_boot=10000, seed=0)
    n_ab, n_ba = int((x > y).sum()), int((x < y).sum())  # discordant
    p = (stats.binomtest(min(n_ab, n_ba), n_ab + n_ba, 0.5).pvalue
         if n_ab + n_ba else 1.0)
    rows.append({'comparison': f'{a} - {b}', 'difference': res.mean,
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
compare.round(4)
```

<!-- Placeholder (stage 2): Experiment 3: identical requests at temperature 0 with a seed: twenty to the hosted flagship, five each to the local server alone and under load. -->

```{.python .input #api-determinism}
%%tab pytorch
DET_IDS = LONGEST[:3]  # the longest reference solutions
DET_MESSAGES = [GSM_MESSAGES[GSM_IDS.index(i)] for i in DET_IDS]

def replicas(first, n):  # n replicas of each of the three requests
    return [{'messages': m, 'replica': k, **GSM_PARAMS}
            for m in DET_MESSAGES for k in range(first, first + n)]

N_LOCAL = 5  # a local reply takes about 18 s: 30 fit the 30-minute run

det_flagship = hosted(FLAGSHIP, cache_file('lm-api-determinism-v1',
                                           'flagship'),
                      share=0.1, reasoning={'enabled': False})
det_local = local(cache_file('lm-api-determinism-v1', 'local'))
on_flagship, alone = await asyncio.gather(
    chat_batch(det_flagship, replicas(0, 20), concurrency=4),
    chat_batch(det_local, replicas(0, N_LOCAL), concurrency=1))
# Under load: background lanes cycle through 64 other test problems; their
# responses are logged, never replayed (replay skips them). A server that
# runs one request at a time gets one lane: more would only queue, and no
# lane can change the composition of a batch.
BACKGROUND = [i for i in range(len(gsm_test)) if i not in GSM_IDS][:64]
LANES = 8 if LOCAL_BATCHES else 1
print(f'under load: {LANES} background lane(s); the local server '
      f'{"batches" if LOCAL_BATCHES else "runs one request at a time"}')
done, failures = asyncio.Event(), []

async def lanes():
    background = d2l.ChatAPI(
        LOCAL_URL, LOCAL_MODEL, mode='live', http_client=LOCAL_HTTP,
        cache=cache_file('lm-api-determinism-v1', 'background'),
        default_params=NO_THINK_LOCAL)
    CLIENTS.append(background)
    async def lane(j):
        while not done.is_set():
            messages = d2l.fewshot_prompt(GSM_DEMOS, gsm_test.question[
                BACKGROUND[j % 64]], template='chat_multi')
            try:
                await background.achat(messages, replica=j // 64,
                                       **GSM_PARAMS)
            except Exception as e:
                failures.append(e)
            j += LANES
    await asyncio.gather(*(lane(j) for j in range(LANES)))

load = asyncio.ensure_future(lanes()) if LOCAL_LIVE else None
if load is not None:
    await asyncio.sleep(2)  # let the lanes fill
under_load = await chat_batch(det_local, replicas(100, N_LOCAL),
                              concurrency=1)
done.set()
if load is not None:
    await load
    print(f'background: {len(failures)} failed requests')
rows = []
for condition, results in (('flagship', on_flagship), ('local alone', alone),
                           ('local under load', under_load)):
    assert not any(isinstance(r, Exception) or thinks(r) for r in results)
    n = len(results) // len(DET_IDS)  # replicas per problem
    for k, i in enumerate(DET_IDS):
        texts = [r.text for r in results[n * k:n * k + n]]
        modal = collections.Counter(texts).most_common(1)[0][0]
        firsts = [len(os.path.commonprefix([t, modal]))
                  for t in texts if t != modal]
        rows.append({'condition': condition, 'problem': i, 'replicas': n,
                     'distinct completions': len(set(texts)),
                     'first divergence, min': min(firsts, default=None),
                     'median': np.median(firsts) if firsts else None,
                     'distinct answers': len({d2l.extract_answer(t)
                                              for t in texts})})
pd.DataFrame(rows)
```

<!-- Placeholder (stage 2): the local server is stopped and its port freed. -->

```{.python .input #api-local-stop}
%%tab pytorch
if server is not None:
    print('server exit code', server.stop())
else:
    print('no local server in replay')
```

<!-- Placeholder (stage 2): what every client did: calls, cache hits, retries, spend, and one provider and one served id per arm. -->

```{.python .input #api-manifest}
%%tab pytorch
def recorded(api):
    """Totals of the whole recording from a client's calls log: live calls,
    retries, 429s, failed attempts, USD, served ids and providers, and the
    listing's and the provider's cost of the responses with usage.cost."""
    t, served, providers = collections.Counter(), set(), set()
    for call in calls_log(api):
        ok = call.get('status') == 200
        t['live calls'] += call['attempt'] == 0
        t['retries'] += call['attempt'] > 0
        t['429s'] += call.get('status') == 429
        t['failed'] += not ok
        if not ok:
            continue
        t['USD'] += call.get('cost') or 0.0
        served.add(str(call.get('served_model')))
        providers.add(str(call.get('provider')))
        cost = call['usage'].get('cost')
        if cost is not None and api.price is not None:
            t['usage cost'] += cost
            t['listing cost'] += d2l.api_cost(call['usage'], api.price)
    return t, sorted(served), sorted(providers)

rows, by_folder, seen = [], collections.defaultdict(dict), set()
for api in CLIENTS:
    if api.calls_path in seen:
        continue
    seen.add(api.calls_path)
    m, (t, served, providers) = api.manifest(), recorded(api)
    folder = os.path.dirname(os.path.abspath(api.cache_path))
    by_folder[folder][m['cache']] = (m, {**t, 'served': served,
                                         'providers': providers})
    key = next((k for k in API_SHA1 if k in folder), 'scratch')
    rows.append({'cache': os.path.join(key, m['cache']), 'calls': m['calls'],
                 'cache hits': m['cache_hits'], **{k: t[k] for k in (
                     'live calls', 'retries', '429s', 'failed', 'USD')},
                 'usage cost': t['usage cost'],
                 'listing cost': t['listing cost'],
                 'served': ', '.join(served),
                 'providers': ', '.join(providers)})
    if api.price is not None:  # claim 8: one provider, one served id
        assert len(served) <= 1 and len(providers) <= 1, (m['cache'], t)
manifest = pd.DataFrame(rows)
spent = sum(api.spent_usd for api in CLIENTS)  # this run's live calls
total = sum(row['USD'] for row in rows)  # every session of the recording
assert spent <= COST_CAP_USD and total <= COST_CAP_USD, (spent, total)
usage_sum = sum(row['usage cost'] for row in rows)
if usage_sum:  # only responses whose provider reported usage.cost
    listing_sum = sum(row['listing cost'] for row in rows)
    print(f'cost from the listing / cost the provider reported: '
          f'{listing_sum / usage_sum:.4f}')
if API_MODE != 'replay':  # sessions accumulate in each artifact's manifest
    now = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
    for folder, arms in by_folder.items():
        if os.path.abspath(RECORD_DIR) not in folder:
            continue
        path, old = os.path.join(folder, 'manifest.json'), {}
        if os.path.exists(path):
            with open(path) as f:
                old = json.load(f)
        sessions = old.get('sessions', []) + [{
            'created': now, 'clients': {a: m for a, (m, _) in arms.items()}}]
        totals = {**old.get('recorded', {}),  # arms this run did not use
                  **{a: r for a, (_, r) in arms.items()}}
        with open(path, 'w') as f:
            json.dump({'updated': now, 'recorded': totals,
                       'sessions': sessions}, f, indent=1,
                      ensure_ascii=False)
key = os.environ.get('OPENROUTER_API_KEY')
if key:  # the key may occur in no cache, calls log or manifest
    files = [os.path.join(d, f) for root in (
        RECORD_DIR, os.path.join(D2L_API_LOG_DIR, 'scratch'))
             for d, _, fs in os.walk(root) for f in fs]
    hits = []
    for name in files:
        with open(name, 'rb') as f:
            if key.encode() in f.read():
                hits.append(name)
    assert not hits, f'the API key occurs in {len(hits)} files'
    print(f'key material: none in {len(files)} files')
print(f'the recording: {sum(row["live calls"] for row in rows)} live calls, '
      f'{total:.4f} USD of {COST_CAP_USD}; this run {spent:.4f} USD')
manifest.drop(columns=['usage cost', 'listing cost'], errors='ignore')
```
