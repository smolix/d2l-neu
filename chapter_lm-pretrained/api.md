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
2. Experiment 1: 100 GSM8K problems (1.4's seeded ids, 4-shot through `d2l.fewshot_prompt(template='chat_multi')`) on DeepSeek-V4.1-Flash-1017 through the router (one pinned provider), on Qwen3.8-27B, and on Qwen3.5-4B behind `transformers serve`; accuracy with intervals, the paired difference, latency p50 and p90, cost from `usage` reconciled with the billing page when live; the flagship run twice (replica 0 and 1) as the noise floor.
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
- The flagship is DeepSeek-V4.1-Flash-1017 (MIT, open weights, 763B parameters in a sparse mixture of experts), always by its dated snapshot id, never the re-pointable alias `deepseek-v4.1-flash`; the second hosted arm is Qwen3.8-27B (Apache-2.0, open weights a T2 pair can serve, the larger sibling of the local arm's family); the local arm is Qwen3.5-4B behind `transformers serve` (not Qwen3.5-9B, as the blueprint has it). Pin one provider per model through the router's provider preferences with fallbacks off, choosing one that exposes `logprobs`, `seed` and `structured_outputs` (on 2026-11-06 Morph, Nebius and Novita did for the flagship; DeepSeek's own route lacked `seed` and doubles its price in two weekday windows, 01:00 to 04:00 and 06:00 to 10:00 UTC); record the served model id and provider with every response; `:fastest` switches providers between calls. Fallback if the snapshot leaves the router: the undated alias with the served id recorded, else Qwen3.8-2.4T-A95B at about ten times the price. The release ships no chat template, only a Python prompt encoder, so the hosted arm also withholds how the prompt is encoded.
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
import hashlib
import heapq
import json
import os
import random
import time
import pandas as pd
from scipy import stats
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
fake = d2l.FakeChatServer(0.01, 1, script=[401])
api = d2l.ChatAPI('http://fake.local/v1', 'fake-model', mode='live',
                  http_client=fake.transport())
try:
    api.chat(hello, max_tokens=8)
except Exception as e:
    print(f'{type(e).__name__} after {fake.n_requests} attempt')
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
for c in (1, 2, 4, 8, 16, 32):
    requests = [{'messages': [{'role': 'user', 'content': f'question {i}'}],
                 'max_tokens': 16} for i in range(8 * c)]
    stats_c = closed_loop_stats(await chat_batch(api, requests,
                                                 concurrency=c), c)
    rows.append({'c': c, 'X': stats_c['X'], 'W': stats_c['W'],
                 'little': stats_c['little'],
                 'sustained': stats_c['sustained']})
table = pd.DataFrame(rows)
W_0 = table['W'][0]  # as the client sees it: service time plus its own work
table['formula X'] = np.minimum(table['c'] / W_0, capacity / service_time)
assert (abs(table['X'] / table['formula X'] - 1) < 0.05).all()
assert (abs(table[['little', 'sustained']] - 1) < 0.02).all().all()
table.round(3)
```
