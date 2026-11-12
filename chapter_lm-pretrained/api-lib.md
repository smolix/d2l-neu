# API Client Scaffolding (build-only, unlisted)

<!--
This file is not listed in `_quarto.yml`/`CHAPTER_NUMBERING` and is never
rendered or executed as a notebook: `make lib` (`tools/build_lib.py`) only
scans it for `#@save` blocks, and `gen_notebooks.py` excludes it via
`LIB_ONLY_FILES` (appended after the numbered chapters, not early: nothing
uses these symbols at import time). It holds the scaffolding of
:numref:`sec_lm_api` that the section does not show: the record
`ChatResult`, the errors `ReplayMiss` and `CostCapExceeded`, the parameter
merge, response parsing and `Retry-After` helpers, the `ChatAPI` class body
(the section shows its `async def achat`, merged in by
`@d2l.add_to_class`), `price_from_listing`, `openrouter_api`,
`LocalChatServer` and `FakeChatServer`. Every block is tagged
`%%tab pytorch`; `openai`, `httpx`, `asyncio` and the other modules outside
the library preamble are imported inside the functions that use them, so
`import d2l.torch` works without them.
-->

<!-- Not rendered; it sets the "Defined in" suffix of the docstrings. -->
:label:`sec_lm_api`

```{.python .input #api-lib-records}
%%tab pytorch
#@save
ChatResult = collections.namedtuple('ChatResult', [  # one chat completion
    'text', 'finish_reason', 'usage', 'logprobs', 'model', 'provider',
    'system_fingerprint', 'reasoning', 'request_hash', 'from_cache',
    't_send', 't_first', 't_first_content', 't_total', 'itl', 'n_retries',
    'n_429', 'created'])

#@save
class ReplayMiss(KeyError):
    """A request that is not in the cache of a client in replay mode: a
    build that replays never reaches the network."""
    def __str__(self):
        return str(self.args[0]) if self.args else ''

#@save
class CostCapExceeded(RuntimeError):
    """A live call whose worst-case cost would cross the client's cost cap;
    raised before any HTTP request is made."""
```

```{.python .input #api-lib-helpers}
%%tab pytorch
#@save
def _chat_merge(base, update):
    """Merge request parameters: keys of update win, nested dicts merge, and
    keys whose value is None are dropped at every level (so None unsets a
    default)."""
    out = {}
    for k in list(base) + [k for k in update if k not in base]:
        v = d2l._chat_plain(update[k] if k in update else base[k])
        if isinstance(v, dict):
            inner = base.get(k) if k in update else None
            v = d2l._chat_merge(inner if isinstance(inner, dict) else {}, v)
        if v is not None:
            out[k] = v
    return out

#@save
def _chat_plain(v):
    """v with NumPy (or torch) scalars, also inside lists and tuples, turned
    into Python numbers, so that they hash and serialize like them."""
    if isinstance(v, (list, tuple)):
        return [d2l._chat_plain(a) for a in v]
    if getattr(v, 'ndim', None) == 0 and hasattr(v, 'item'):
        return v.item()
    return v

#@save
def _chat_usage(u):
    """The usage block of a response as prompt, completion, cached and
    reasoning tokens and the provider's cost; missing values are None,
    never 0."""
    u = u or {}
    prompt = u.get('prompt_tokens_details') or {}
    completion = u.get('completion_tokens_details') or {}
    return {'prompt_tokens': u.get('prompt_tokens'),
            'completion_tokens': u.get('completion_tokens'),
            'cached_tokens': prompt.get('cached_tokens'),
            'reasoning_tokens': completion.get(
                'reasoning_tokens', u.get('reasoning_tokens')),
            'cost': u.get('cost')}

#@save
def _chat_logprobs(lp):
    """Generated-token log-probabilities as (token, logprob, top) triples,
    top a list of (token, logprob); None when the response has none."""
    if not lp or not lp.get('content'):
        return None
    return [(t['token'], t['logprob'],
             [(a['token'], a['logprob']) for a in t.get('top_logprobs') or []])
            for t in lp['content']]

#@save
def _chat_parse(r):
    """The fields of a (non-streamed) chat-completions response dict that
    d2l.ChatResult keeps; unknown top-level fields (such as prompt
    log-probabilities) and the texts of further choices go to 'extra'."""
    choices = r.get('choices') or [{}]
    choice, known = choices[0], {'id', 'object', 'created', 'model', 'usage',
                                 'choices', 'provider', 'system_fingerprint'}
    message = choice.get('message') or {}
    extra = {k: v for k, v in r.items() if k not in known and v is not None}
    if len(choices) > 1:
        extra['choices'] = [{'text': (c.get('message') or {}).get('content'),
                             'finish_reason': c.get('finish_reason')}
                            for c in choices]
    return {'text': message.get('content') or '',
            'reasoning': (message.get('reasoning')
                          or message.get('reasoning_content')),
            'finish_reason': choice.get('finish_reason'),
            'logprobs': d2l._chat_logprobs(choice.get('logprobs')),
            'usage': d2l._chat_usage(r.get('usage')), 'model': r.get('model'),
            'provider': r.get('provider'),
            'system_fingerprint': r.get('system_fingerprint'), 'extra': extra,
            't_first': None, 't_first_content': None, 'itl': None,
            'n_chunks': None}

#@save
def _retry_after(value):
    """Seconds in a Retry-After header (delta seconds or an HTTP date), or
    None when the header is absent or unreadable."""
    if value is None:
        return None
    try:
        return max(0.0, float(value))
    except ValueError:
        import datetime  # not in the d2l library preamble
        import email.utils
        try:
            t = email.utils.parsedate_to_datetime(value)
            now = datetime.datetime.now(datetime.timezone.utc)
            return max(0.0, (t - now).total_seconds())
        except (TypeError, ValueError):
            return None

#@save
def _check_closed_loop(results, c):
    """Raise ValueError unless results can be one closed loop of c recorded
    in one run: no d2l.ReplayMiss or d2l.CostCapExceeded among them, not a
    mix of cached and live responses, never more than c in flight, and
    t_send minus the wall-clock `created` the same within 2 s (a monotonic
    clock of another boot or a suspended machine shifts it)."""
    import calendar  # not in the d2l library preamble
    bad = [r for r in results
           if isinstance(r, (d2l.ReplayMiss, d2l.CostCapExceeded))]
    if bad:
        raise ValueError(f'{len(bad)} requests got no response: {bad[0]!r}')
    ok = [r for r in results if not isinstance(r, BaseException)]
    if len({r.from_cache for r in ok} - {None}) > 1:
        raise ValueError('the results mix cached and live responses; record '
                         'the level in one run, into a fresh cache')
    s = np.array([r.t_send for r in ok], dtype=float)
    f = s + np.array([r.t_total for r in ok], dtype=float)
    step = np.concatenate([np.ones(len(ok)), -np.ones(len(ok))])
    order = np.lexsort((step, np.concatenate([s, f])))  # ends before starts
    if len(ok) and np.cumsum(step[order]).max() > c:
        raise ValueError(f'more than c = {c} requests were in flight at once: '
                         f'not one closed loop')
    lag = [t - calendar.timegm(time.strptime(r.created, '%Y-%m-%dT%H:%M:%SZ'))
           for t, r in zip(s, ok) if r.created is not None]
    if lag and max(lag) - min(lag) > 2:
        raise ValueError('the send times come from more than one run '
                         '(monotonic and wall clocks disagree)')
```

```{.python .input #api-lib-chat-api}
%%tab pytorch
#@save
class ChatAPI:
    """A measuring and replaying client for OpenAI-compatible chat
    completions. Every request is keyed by d2l.request_hash. Mode 'replay'
    answers from the JSONL cache and raises d2l.ReplayMiss on a miss,
    'record' answers from the cache and goes live on a miss, 'live' always
    goes live and appends; a replay returns the first recorded entry. Every
    live HTTP attempt, failed or not, is appended to <cache stem>.calls.jsonl.
    The SDK client (openai.AsyncOpenAI, max_retries=0) is created at the
    first live call, so constructing and replaying never imports openai or
    reads a key. api_key is a string, a function returning one (called at
    the first live call) or None (a placeholder for local servers); it never
    enters the hash, cache, calls log or manifest. With price set, a live
    call needs max_tokens (or max_completion_tokens); with cost_cap_usd it
    reserves its worst-case cost before sending and settles to the real
    cost after. http_client may also be an httpx transport (such as
    d2l.FakeChatServer().transport()). backoff_base, backoff_cap and rng
    (None: the random module) set the retry schedule."""
    RETRY_STATUS = (408, 429)  # and every 5xx

    def __init__(self, base_url, model, api_key=None, *, cache=None,
                 mode='replay', price=None, cost_cap_usd=None, timeout=120.0,
                 max_retries=6, default_params=None, http_client=None):
        import threading  # not in the d2l library preamble
        if mode not in ('replay', 'record', 'live'):
            raise ValueError(f"mode must be 'replay', 'record' or 'live', "
                             f'not {mode!r}')
        if cost_cap_usd is not None and price is None:
            raise ValueError('a cost cap needs a price')
        self.base_url, self.model, self.api_key = base_url, model, api_key
        self.mode, self.price, self.cost_cap_usd = mode, price, cost_cap_usd
        self.timeout, self.max_retries = timeout, max_retries
        self.default_params = d2l._chat_merge(default_params or {}, {})
        self.http_client, self.cache_path = http_client, cache
        self.calls_path = (None if cache is None else
                           os.path.splitext(cache)[0] + '.calls.jsonl')
        self.backoff_base, self.backoff_cap, self.rng = 1.0, 60.0, None
        self._lock, self._cache = threading.Lock(), {}
        self._sdk = self._loop = self._sdk_version = None
        self._sdks = {}  # event loop -> SDK client
        self._spent = self._reserved = 0.0
        self._stats, self._costs = collections.Counter(), collections.Counter()
        self._served, self._providers = (collections.Counter(),
                                         collections.Counter())
        self._trimmed = set()  # files whose torn last line was dealt with
        if cache is not None and os.path.exists(cache):
            import warnings  # not in the d2l library preamble
            with open(cache, 'rb') as f:
                lines = f.read().split(b'\n')
            for i, line in enumerate(lines):
                try:
                    entry = json.loads(line) if line.strip() else None
                except ValueError:
                    if i < len(lines) - 1:  # a whole line: a damaged cache
                        raise ValueError(f'{cache}: line {i + 1} is not JSON')
                    warnings.warn(f'{cache}: skipping its torn last line (a '
                                  f'run killed while writing it)')
                    continue
                if entry is not None:
                    self._cache.setdefault(entry['hash'], []).append(entry)

    @property
    def spent_usd(self):
        """USD settled by this client's live calls in this session."""
        return self._spent

    def entries(self, request_hash):
        """The cache entries (dicts) recorded for one request hash."""
        return self._cache.get(request_hash, [])

    def chat(self, messages, *, replica=0, stream=False, **params):
        """Synchronous d2l.ChatAPI.achat, run on the client's private event
        loop in a worker thread (so it also works inside Jupyter)."""
        return self._run(self.achat(messages, replica=replica, stream=stream,
                                    **params))

    def models(self):
        """The model ids the endpoint lists (GET /models; never replayed)."""
        if self.mode == 'replay':
            raise d2l.ReplayMiss(f'{self.model}: models() needs a live '
                                 f'endpoint, the client is in replay mode')
        async def ids():
            return [m.id async for m in self._client().models.list()]
        return self._run(ids())

    def manifest(self):
        """This session's counts (calls, cache hits, live calls, HTTP
        attempts, retries, 429s, failed attempts), the spent and reconciled
        cost (sum of api_cost against sum of usage.cost over the results
        returned), and the served model ids and providers of the results."""
        from urllib.parse import urlsplit  # not in the d2l library preamble
        keys = ('calls', 'cache_hits', 'live_calls', 'attempts', 'retries',
                'n_429', 'failed_attempts')
        return {'host': urlsplit(self.base_url).hostname,
                'model': self.model, 'mode': self.mode,
                'cache': (None if self.cache_path is None else
                          os.path.basename(self.cache_path)),
                'cache_entries': sum(map(len, self._cache.values())),
                **{k: self._stats[k] for k in keys},
                'spent_usd': self._spent, 'cost_cap_usd': self.cost_cap_usd,
                'price': None if self.price is None else self.price._asdict(),
                **{k: self._costs[k] for k in (
                    'n_api_cost', 'sum_api_cost', 'n_usage_cost',
                    'sum_usage_cost')},
                'served_models': dict(self._served),
                'providers': dict(self._providers), 'sdk': self._sdk_version}

    def _run(self, coro):
        import asyncio  # not in the d2l library preamble
        import threading
        with self._lock:
            if self._loop is None:
                self._loop = asyncio.new_event_loop()
                threading.Thread(target=self._loop.run_forever,
                                 daemon=True).start()
        return asyncio.run_coroutine_threadsafe(coro, self._loop).result()

    def _client(self):
        """The SDK client of the running event loop, created on first use:
        one per open loop (the private loop of chat() and a notebook's), the
        clients of closed loops dropped."""
        import asyncio  # not in the d2l library preamble
        loop = asyncio.get_running_loop()
        with self._lock:
            for old in [l for l in self._sdks if l.is_closed()]:
                del self._sdks[old]
            if loop in self._sdks:
                return self._sdks[loop]
            import openai  # not in the d2l library preamble
            http = self.http_client
            if http is not None and hasattr(http, 'handle_async_request'):
                import httpx2  # the httpx flavour of openai >= 3
                http = httpx2.AsyncClient(transport=http)
            key = self.api_key() if callable(self.api_key) else self.api_key
            self._sdk = self._sdks[loop] = openai.AsyncOpenAI(
                base_url=self.base_url, api_key=key or 'none', max_retries=0,
                timeout=self.timeout, http_client=http)
            self._sdk_version = openai.__version__
            return self._sdk

    def _lookup(self, h, messages):
        """The first cached result of h unless the mode is 'live'; a miss
        raises d2l.ReplayMiss in replay mode and returns None otherwise."""
        if self.mode != 'live' and self._cache.get(h):
            self._stats['cache_hits'] += 1
            return self._result(self._cache[h][0], from_cache=True)
        if self.mode == 'replay':
            last = next((m.get('content') for m in reversed(messages)
                         if m.get('role') == 'user'), '')
            raise d2l.ReplayMiss(f'{self.model}: request {h[:12]} is not in '
                                 f'the cache (last user message '
                                 f'{str(last)[:80]!r})')
        return None

    def _reserve(self, messages, params):
        """Reserve the worst case: one prompt token per UTF-8 byte of the
        messages' JSON (a byte-level BPE token covers at least one byte, and
        the JSON's quoting outweighs a chat template's few special tokens per
        message) and max_tokens completion tokens for each of the n
        choices."""
        if self.price is None:
            self._stats['live_calls'] += 1
            return 0.0
        out = params.get('max_tokens', params.get('max_completion_tokens'))
        if out is None:
            raise ValueError('a priced live call needs max_tokens')
        n = len(json.dumps(messages, ensure_ascii=False).encode('utf-8'))
        worst = d2l.api_cost({'prompt_tokens': n, 'completion_tokens': out * (
            params.get('n') or 1)}, self.price)
        with self._lock:
            if (self.cost_cap_usd is not None and self._spent + self._reserved
                    + worst > self.cost_cap_usd):
                raise d2l.CostCapExceeded(
                    f'{self.model}: spent {self._spent:.6f} + reserved '
                    f'{self._reserved:.6f} + this call {worst:.6f} USD would '
                    f'cross the cap of {self.cost_cap_usd} USD')
            self._reserved += worst
        self._stats['live_calls'] += 1
        return worst

    def _settle(self, reserved, cost):
        with self._lock:
            self._reserved -= reserved
            self._spent += cost

    def _cost(self, usage, reserved=None):
        """Real cost of a response: the larger of usage.cost and api_cost
        from the price; the reservation when neither is known."""
        costs = [usage.get('cost')]
        if self.price is not None:
            try:
                costs.append(d2l.api_cost(usage, self.price))
            except ValueError:
                pass
        costs = [c for c in costs if c is not None]
        return max(costs) if costs else (reserved or 0.0)

    def _retry_info(self, e):
        """(retry?, HTTP status, Retry-After seconds) of a failed attempt:
        connection errors, timeouts, RETRY_STATUS and every 5xx are retried,
        and so is an error event inside a stream (the SDK raises a bare
        APIError; its numeric code, if any, is treated as the status)."""
        def retry(status):
            return status in self.RETRY_STATUS or 500 <= status < 600
        status = getattr(e, 'status_code', None)
        if status is not None:
            headers = getattr(getattr(e, 'response', None), 'headers', {})
            return (retry(status), status,
                    d2l._retry_after(headers.get('retry-after')))
        openai = sys.modules.get('openai')
        if openai is not None and isinstance(e, openai.APIConnectionError):
            return True, None, None
        if openai is not None and type(e) is openai.APIError:  # mid-stream
            code = str(getattr(e, 'code', None) or '')
            if code.isdigit():
                return retry(int(code)), int(code), None
            return True, None, None
        transport = [getattr(sys.modules.get(m), 'TransportError', None)
                     for m in ('httpx2', 'httpx')]
        return any(t is not None and isinstance(e, t)
                   for t in transport), None, None

    async def _send(self, messages, params, stream, h, replica, attempt):
        """One HTTP attempt, logged to the calls log whatever its outcome;
        returns the parsed response with its chunk times."""
        kwargs = {k: v for k, v in params.items()
                  if k not in ('stream', 'stream_options')}
        log = {'created': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
               'hash': h, 'model': self.model, 'replica': replica,
               'attempt': attempt, 'stream': stream}
        self._stats['attempts'] += 1
        self._stats['retries'] += attempt > 0
        t0 = time.monotonic()
        try:
            create = self._client().chat.completions.create
            if stream:
                options = d2l._chat_merge({'include_usage': True},
                                          params.get('stream_options') or {})
                out = await self._read_stream(await create(
                    model=self.model, messages=messages, stream=True,
                    stream_options=options, **kwargs))
            else:
                response = await create(model=self.model, messages=messages,
                                        **kwargs)
                out = d2l._chat_parse(response.model_dump())
                out['t_end'] = time.monotonic()
        except Exception as e:
            _, status, retry_after = self._retry_info(e)
            self._stats['failed_attempts'] += 1
            self._stats['n_429'] += status == 429
            self._append(self.calls_path, {
                **log, 'status': status, 'retry_after': retry_after,
                'error': f'{type(e).__name__}: {e}'[:200],
                'latency': time.monotonic() - t0})
            raise
        self._append(self.calls_path, {
            **log, 'status': 200, 'latency': out['t_end'] - t0,
            'finish_reason': out['finish_reason'], 'usage': out['usage'],
            'cost': self._cost(out['usage']), 'served_model': out['model'],
            'provider': out['provider']})
        return out

    async def _read_stream(self, chunks):
        """Concatenate a streamed response; t_first is the first chunk with
        reasoning or content, t_first_content the first with content, and
        itl the gaps between such chunks."""
        out = dict.fromkeys(('finish_reason', 'usage', 'model', 'provider',
                             'system_fingerprint', 't_first',
                             't_first_content'))
        text, reasoning, logprobs, times = [], [], [], []
        now = time.monotonic()
        async for chunk in chunks:
            c, now = chunk.model_dump(), time.monotonic()
            for k in ('model', 'provider', 'system_fingerprint', 'usage'):
                out[k] = c.get(k) or out[k]
            for choice in c.get('choices') or []:
                if choice.get('index', 0) != 0:
                    continue
                delta = choice.get('delta') or {}
                r = delta.get('reasoning') or delta.get('reasoning_content')
                if r or delta.get('content'):
                    times.append(now)
                    if out['t_first'] is None:
                        out['t_first'] = now
                if delta.get('content'):
                    text.append(delta['content'])
                    if out['t_first_content'] is None:
                        out['t_first_content'] = now
                if r:
                    reasoning.append(r)
                logprobs += d2l._chat_logprobs(choice.get('logprobs')) or []
                out['finish_reason'] = (choice.get('finish_reason')
                                        or out['finish_reason'])
        out.update(text=''.join(text), reasoning=''.join(reasoning) or None,
                   logprobs=logprobs or None, extra={}, t_end=now,
                   usage=d2l._chat_usage(out['usage']), n_chunks=len(times),
                   itl=[b - a for a, b in zip(times, times[1:])])
        return out

    def _record(self, h, messages, params, replica, out, t_send, created,
                n_retries, n_429, reserved):
        """Settle the cost of a live response, append it to the cache and
        return it as a d2l.ChatResult."""
        from urllib.parse import urlsplit  # not in the d2l library preamble
        self._settle(reserved, self._cost(out['usage'], reserved))
        fields = ('text', 'finish_reason', 'usage', 'logprobs', 'model',
                  'provider', 'system_fingerprint', 'reasoning', 't_first',
                  't_first_content', 'itl')
        result = {**{k: out[k] for k in fields}, 'request_hash': h,
                  't_send': t_send, 't_total': out['t_end'] - t_send,
                  'n_retries': n_retries, 'n_429': n_429, 'created': created}
        transport = ('stream', 'stream_options', 'timeout', 'extra_headers')
        request = {'host': urlsplit(self.base_url).hostname,
                   'model': self.model, 'messages': messages,
                   'replica': replica, 'params': {
                       k: v for k, v in params.items() if k not in transport}}
        entry = json.loads(json.dumps(  # what a reload of the file gives
            {'hash': h, 'request': request, 'result': result,
             'extra': out['extra'], 'n_chunks': out['n_chunks'],
             'sdk': self._sdk_version}, ensure_ascii=False))
        self._cache.setdefault(h, []).append(entry)
        self._append(self.cache_path, entry)
        return self._result(entry, from_cache=False)

    def _result(self, entry, from_cache):
        r = dict(entry['result'], from_cache=from_cache)
        if r['logprobs'] is not None:
            r['logprobs'] = [(t, lp, [tuple(a) for a in top])
                             for t, lp, top in r['logprobs']]
        result = d2l.ChatResult(**r)
        self._served[result.model] += 1
        self._providers[result.provider] += 1
        if result.usage.get('cost') is not None:
            self._costs['n_usage_cost'] += 1
            self._costs['sum_usage_cost'] += result.usage['cost']
        if self.price is not None:
            try:
                self._costs['sum_api_cost'] += d2l.api_cost(result.usage,
                                                            self.price)
                self._costs['n_api_cost'] += 1
            except ValueError:
                pass
        return result

    def _append(self, path, obj):
        """Append one JSON line (the key scrubbed, should it ever appear)."""
        if path is None:
            return
        line = json.dumps(obj, ensure_ascii=False, sort_keys=True)
        key = getattr(self._sdk, 'api_key', None)
        if isinstance(key, str) and len(key) >= 8:
            line = line.replace(key, '[api key]')
        with self._lock:
            os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
            if path not in self._trimmed:
                d2l._jsonl_trim_torn(path)
                self._trimmed.add(path)
            with open(path, 'a', encoding='utf-8') as f:
                f.write(line + '\n')

#@save
def _jsonl_trim_torn(path):
    """End a JSONL file with a newline before appending to it: a torn last
    line (no newline, as a run killed mid-write leaves it) is cut off unless
    it is complete JSON, which only gets its newline."""
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        return
    with open(path, 'rb+') as f:
        f.seek(-1, os.SEEK_END)
        if f.read(1) == b'\n':
            return
        f.seek(0)
        data = f.read()
        cut = data.rfind(b'\n') + 1
        try:
            json.loads(data[cut:])
            f.write(b'\n')
        except ValueError:
            f.truncate(cut)
```

```{.python .input #api-lib-openrouter}
%%tab pytorch
#@save
def _api_window(override, day, hhmm):
    """Whether a pricing.overrides row covers a UTC weekday and HHMM time;
    utc_end 0 means midnight, and a window may wrap past it."""
    if override.get('utc_days') and day not in override['utc_days']:
        return False
    start, end = override.get('utc_start'), override.get('utc_end')
    if start is None and end is None:
        return True
    start, end = start or 0, end or 2400
    return (start <= hhmm < end) if start < end else (
        hhmm >= start or hhmm < end)

#@save
def price_from_listing(listing, model_id, provider, *, when=None):
    """The d2l.PricePoint (USD per million tokens) of one provider's endpoint
    of model_id in an OpenRouter listing snapshot: listing['endpoints']
    [model_id] holds the endpoint rows of /api/v1/models/<id>/endpoints (or
    that whole response). provider matches a row's tag exactly, else its
    provider_name or the slug before the tag's '/'. Listed prices are USD
    per token with the row's `discount` already applied (Novita's 0.165 per
    million at discount 0.45 is 0.3 x 0.55), so it is not applied again.
    Rows with time-window pricing.overrides need when, an aware datetime
    whose UTC weekday and time select the window."""
    import datetime  # not in the d2l library preamble
    import decimal
    rows = listing['endpoints'][model_id]
    if isinstance(rows, dict):
        rows = rows['data']['endpoints']
    p = provider.lower()
    match = [r for r in rows if str(r.get('tag', '')).lower() == p] or [
        r for r in rows if p in (str(r.get('provider_name', '')).lower(),
                                 str(r.get('tag', '')).lower().split('/')[0])]
    if len(match) != 1:
        raise ValueError(f'{len(match)} endpoints of {model_id} match '
                         f'{provider!r}: {[r.get("tag") for r in match]}')
    pricing = match[0]['pricing']
    if pricing.get('overrides'):
        if when is None or when.tzinfo is None:
            raise ValueError(f'{provider} prices {model_id} by time window: '
                             f'pass when, an aware datetime')
        t = when.astimezone(datetime.timezone.utc)
        day = ('monday', 'tuesday', 'wednesday', 'thursday', 'friday',
               'saturday', 'sunday')[t.weekday()]
        window = next((o for o in pricing['overrides'] if d2l._api_window(
            o, day, 100 * t.hour + t.minute)), {})
        pricing = {**pricing, **window}
    def per_million(key):
        if pricing.get(key) is None:
            return None
        value = decimal.Decimal(str(pricing[key])) * 1_000_000
        if value < 0:
            raise ValueError(f'{provider}: no fixed {key} price ({value})')
        return float(value)
    return d2l.PricePoint(per_million('prompt'), per_million('completion'),
                          per_million('input_cache_read'))

#@save
def _openrouter_key():
    """The OpenRouter key, read from the environment only (never a file)."""
    key = os.environ.get('OPENROUTER_API_KEY')
    if not key:
        raise RuntimeError('a live OpenRouter call needs OPENROUTER_API_KEY '
                           'in the environment')
    return key

#@save
def openrouter_api(model, provider, *, listing, cache=None, mode='replay',
                   cost_cap_usd=None, reasoning=None, when=None, **kwargs):
    """A d2l.ChatAPI for one model on OpenRouter with one provider pinned
    (no fallbacks, require_parameters on), priced from the listing snapshot
    (None: unpriced). reasoning (e.g. {'enabled': False}) is sent only when
    given. The key is read from OPENROUTER_API_KEY at the first live call;
    further keyword arguments go to d2l.ChatAPI."""
    body = {'provider': {'only': [provider], 'allow_fallbacks': False,
                         'require_parameters': True}}
    if reasoning is not None:
        body['reasoning'] = reasoning
    defaults = d2l._chat_merge({'extra_body': body},
                               kwargs.pop('default_params', None) or {})
    kwargs.setdefault('api_key', d2l._openrouter_key)
    price = (None if listing is None else
             d2l.price_from_listing(listing, model, provider, when=when))
    return d2l.ChatAPI('https://openrouter.ai/api/v1', model, cache=cache,
                       mode=mode, price=price, cost_cap_usd=cost_cap_usd,
                       default_params=defaults, **kwargs)
```

```{.python .input #api-lib-servers}
%%tab pytorch
#@save
class LocalChatServer:
    """An OpenAI-compatible server (such as `transformers serve`) run as a
    subprocess in its own process group. '{port}' in cmd becomes port (a
    free one when None); start() polls http://host:port/v1/models until it
    answers 200 and returns the base URL, or kills the server and raises
    with the last 20 log lines; stop() terminates the whole group, deletes
    the temporary log and returns the exit code. Also a context manager."""
    def __init__(self, cmd, *, port=None, host='127.0.0.1',
                 ready_timeout=900.0):
        self.cmd, self.port, self.host = list(cmd), port, host
        self.ready_timeout = ready_timeout
        self.proc = self.base_url = self.log_path = self.load_time = None

    def start(self):
        import atexit  # not in the d2l library preamble
        import socket
        import subprocess
        import tempfile
        import urllib.request
        if self.proc is not None:
            return self.base_url
        if self.port is None:
            with socket.socket() as s:
                s.bind((self.host, 0))
                self.port = s.getsockname()[1]
        cmd = [str(a).replace('{port}', str(self.port)) for a in self.cmd]
        fd, self.log_path = tempfile.mkstemp(prefix='d2l-server-',
                                             suffix='.log')
        t0 = time.monotonic()
        with os.fdopen(fd, 'wb') as log:
            self.proc = subprocess.Popen(
                cmd, stdout=log, stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL, start_new_session=True)
        atexit.register(self.stop)
        url = f'http://{self.host}:{self.port}/v1'
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        while True:
            if self.proc.poll() is not None:  # its children may live on
                code, tail = self.proc.returncode, self._tail()
                self.stop()
                raise RuntimeError(f'server exited with code {code} before '
                                   f'it was ready:\n{tail}')
            left = self.ready_timeout - (time.monotonic() - t0)
            if left <= 0:
                tail = self._tail()
                self.stop()
                raise TimeoutError(f'server not ready after '
                                   f'{self.ready_timeout} s:\n{tail}')
            try:
                with opener.open(url + '/models',
                                 timeout=max(0.5, min(5.0, left))) as r:
                    if r.status == 200:
                        break
            except Exception:
                pass
            time.sleep(min(0.5, max(left, 0.0)))
        self.load_time, self.base_url = time.monotonic() - t0, url
        return url

    def stop(self):
        import atexit  # not in the d2l library preamble
        import signal
        import subprocess
        if self.proc is None:
            return None
        proc, self.proc = self.proc, None
        atexit.unregister(self.stop)
        try:
            os.killpg(proc.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        try:
            code = proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGKILL)
            code = proc.wait()
        try:  # children that outlived the leader
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        if self.log_path is not None and os.path.exists(self.log_path):
            os.remove(self.log_path)
        return code

    def _tail(self, n=20):
        with open(self.log_path, encoding='utf-8', errors='replace') as f:
            return ''.join(f.readlines()[-n:])

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *exc):
        self.stop()

#@save
class FakeChatServer:
    """An in-process chat-completions server for tests. Each request takes
    service_time seconds in one of capacity slots, so a closed loop has
    W_0 = service_time and X_max = capacity / service_time; a busy slot
    starts its next request when the previous one ended (not when the event
    loop gets round to it), so the capacity is exact. script lists
    statuses returned in order before normal behaviour (429s carry
    Retry-After: 0); fail_rate answers 503 at random; rate_limit answers 429
    beyond that many requests per second. A request with thinking switched
    on gets reasoning first ('reasoning', or 'reasoning_content' when it
    carries chat_template_kwargs, as a local server); max_tokens, stop, n,
    logprobs and stream_options.include_usage are honoured, and requests
    with a provider pin get a provider and usage.cost. transport() plugs it
    into the SDK's http_client; n_requests, statuses, bodies and
    max_in_flight record what it saw."""
    PRICE = (0.1, 0.4)  # USD per million prompt and completion tokens

    def __init__(self, service_time, capacity, *, fail_rate=0.0,
                 rate_limit=None, script=None, seed=0):
        self.service_time, self.capacity = service_time, capacity
        self.fail_rate, self.rate_limit = fail_rate, rate_limit
        self.script, self.rng = list(script or []), random.Random(seed)
        self.n_requests = self.in_flight = self.max_in_flight = 0
        self.statuses, self.bodies = [], []
        self._admitted = collections.deque()
        self._sem = self._sem_loop = self._free = None

    def transport(self):
        """An httpx.AsyncBaseTransport of the flavour the SDK uses."""
        try:
            import httpx2 as httpx  # openai >= 3; not in the d2l preamble
        except ImportError:
            import httpx
        server = self
        class Transport(httpx.AsyncBaseTransport):
            async def handle_async_request(self, request):
                return await server._handle(request, httpx)
        return Transport()

    async def _acquire(self, arrival):
        """Wait for a slot; the service starts at arrival or when the slot's
        previous request ended, whichever is later."""
        import asyncio  # not in the d2l library preamble
        import heapq
        loop = asyncio.get_running_loop()
        if self._sem_loop is not loop:
            self._sem, self._sem_loop = asyncio.Semaphore(self.capacity), loop
            self._free = [0.0] * self.capacity
        await self._sem.acquire()
        return max(arrival, heapq.heappop(self._free))

    def _release(self, end):
        import heapq  # not in the d2l library preamble
        heapq.heappush(self._free, end)
        self._sem.release()

    def _status(self):
        if self.script:
            return self.script.pop(0), True
        if self.rate_limit is not None:
            now = time.monotonic()
            while self._admitted and now - self._admitted[0] >= 1.0:
                self._admitted.popleft()
            if len(self._admitted) >= self.rate_limit:
                return 429, False
            self._admitted.append(now)
        if self.fail_rate and self.rng.random() < self.fail_rate:
            return 503, False
        return 200, False

    def _reply(self, body):
        """Tokens ((kind, text) pairs), finish reason and usage of a reply."""
        messages = body.get('messages') or [{}]
        last = str(messages[-1].get('content', ''))
        words = (f'Fake reply to {len(last)} characters: the answer is '
                 f'{len(last) % 10}.').split(' ')
        content, finish = [w if i == 0 else ' ' + w
                           for i, w in enumerate(words)], 'stop'
        stops = body.get('stop') or []
        for i in range(len(content)):
            text = ''.join(content[:i + 1])
            cut = min((text.find(s) for s in ([stops] if isinstance(
                stops, str) else stops) if s in text), default=None)
            if cut is not None:
                prefix = text[:cut]
                content = [prefix] if prefix else []
                break
        r = body.get('reasoning')
        think = (r.get('enabled', True) if isinstance(r, dict) else bool(
            (body.get('chat_template_kwargs') or {}).get('enable_thinking')))
        tokens = ([('r', w) for w in ('Let', ' me', ' think.')] if think
                  else []) + [('c', w) for w in content]
        limit = body.get('max_tokens', body.get('max_completion_tokens'))
        if limit is not None and len(tokens) > limit:
            tokens, finish = tokens[:limit], 'length'
        n_prompt = sum(len(str(m.get('content', '')).split())
                       for m in messages)
        usage = {'prompt_tokens': n_prompt, 'completion_tokens': len(tokens),
                 'total_tokens': n_prompt + len(tokens),
                 'prompt_tokens_details': {'cached_tokens': 0},
                 'completion_tokens_details': {'reasoning_tokens': sum(
                     k == 'r' for k, _ in tokens)}}
        if 'provider' in body:
            usage['cost'] = (n_prompt * self.PRICE[0]
                             + len(tokens) * self.PRICE[1]) / 1e6
        return tokens, finish, usage

    def _logprobs(self, body, words):
        if not body.get('logprobs'):
            return None
        k = body.get('top_logprobs') or 0
        return {'content': [
            {'token': w, 'logprob': -0.1 * (i + 1), 'top_logprobs': [
                {'token': w if j == 0 else f'{w}#{j}',
                 'logprob': -0.1 * (i + 1) - j} for j in range(k)]}
            for i, w in enumerate(words)]}

    async def _handle(self, request, httpx):
        import asyncio  # not in the d2l library preamble
        arrival = time.monotonic()
        self.n_requests += 1
        if request.method == 'GET' and request.url.path.endswith('/models'):
            self.statuses.append(200)
            return httpx.Response(200, json={'object': 'list', 'data': [
                {'id': 'fake-model', 'object': 'model', 'created': 0,
                 'owned_by': 'd2l'}]})
        body = json.loads((await request.aread()) or b'{}')
        self.bodies.append(body)
        self.in_flight += 1
        self.max_in_flight = max(self.max_in_flight, self.in_flight)
        status, scripted = self._status()
        self.statuses.append(status)
        if status != 200:
            self.in_flight -= 1
            scripted_429 = scripted and status == 429
            headers = {'retry-after': '0'} if scripted_429 else {}
            return httpx.Response(status, headers=headers, json={'error': {
                'message': f'fake status {status}', 'code': status}})
        tokens, finish, usage = self._reply(body)
        head = {'id': 'fake-1', 'created': int(time.time()),
                'model': body.get('model'), 'system_fingerprint': 'fp_fake'}
        if 'provider' in body:
            head['provider'] = (body['provider'].get('only') or ['Fake'])[0]
        field = ('reasoning_content' if 'chat_template_kwargs' in body
                 else 'reasoning')
        if body.get('stream'):
            return httpx.Response(
                200, headers={'content-type': 'text/event-stream'},
                content=self._stream(arrival, body, head, field, tokens,
                                     finish, usage))
        try:
            end = await self._acquire(arrival) + self.service_time
            try:
                await asyncio.sleep(max(0.0, end - time.monotonic()))
            finally:
                self._release(end)
        finally:
            self.in_flight -= 1
        reasoning = ''.join(w for k, w in tokens if k == 'r')
        words = [w for k, w in tokens if k == 'c']
        message = {'role': 'assistant', 'content': ''.join(words)}
        if reasoning:
            message[field] = reasoning
        choice = {'message': message, 'finish_reason': finish,
                  'logprobs': self._logprobs(body, words)}
        return httpx.Response(200, json={
            **head, 'object': 'chat.completion', 'usage': usage,
            'choices': [{'index': i, **choice}
                        for i in range(body.get('n') or 1)]})

    async def _stream(self, arrival, body, head, field, tokens, finish,
                      usage):
        """Server-sent events: a role chunk, the first token half the
        service time after the start, the rest spread over the other half."""
        import asyncio  # not in the d2l library preamble
        def event(choices, **more):
            return 'data: {}\n\n'.format(json.dumps({
                **head, 'object': 'chat.completion.chunk', 'choices': choices,
                **more})).encode()
        async def until(t):
            await asyncio.sleep(max(0.0, t - time.monotonic()))
        try:
            start = await self._acquire(arrival)
            gap = self.service_time / 2 / max(1, len(tokens))
            try:
                yield event([{'index': 0, 'finish_reason': None,
                              'delta': {'role': 'assistant', 'content': ''}}])
                await until(start + self.service_time / 2)
                for i, (kind, w) in enumerate(tokens):
                    lp = self._logprobs(body, [w]) if kind == 'c' else None
                    key = 'content' if kind == 'c' else field
                    yield event([{'index': 0, 'finish_reason': None,
                                  'logprobs': lp, 'delta': {key: w}}])
                    await until(start + self.service_time / 2 + (i + 1) * gap)
                yield event([{'index': 0, 'delta': {},
                              'finish_reason': finish}])
                if (body.get('stream_options') or {}).get('include_usage'):
                    yield event([], usage=usage)
                yield b'data: [DONE]\n\n'
            finally:
                self._release(start + self.service_time)
        finally:
            self.in_flight -= 1
```
