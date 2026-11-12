#!/usr/bin/env python3
"""CPU tests for sec_lm_api (chapter_lm-pretrained/api.md and its build-only
companion api-lib.md).

Library code (spec 18.7 part 5), numbered as the spec's unit tests:
1.  `request_hash`: invariant to key order, `stream`, `stream_options`,
    `timeout`, the API key and the port; changes with `seed`, `max_tokens`,
    `model`, the provider pin, one message character and `replica`.
2.  `backoff_delay`: draws in [0, min(cap, base 2^a)] with mean d/2 (3 sigma);
    `retry_after` is a floor.
3.  `FakeChatServer(script=[429, 429, 200])`: 3 attempts, `n_429 == 2`,
    `n_retries == 2`, no hidden SDK retries (transport request count).
4.  400 and 401 raise after one attempt; `fail_rate=1.0` with
    `max_retries=3` makes 4 attempts; connection errors and timeouts retry.
5.  `chat_batch`: 50 requests at concurrency 5 with one failing: at most 5 in
    flight, 49 results and 1 exception in input order.
6.  The cache: record hits without HTTP, replay misses raise `ReplayMiss`,
    live appends and replay returns the first entry, a reload replays the
    same results, and no key string reaches the cache, calls log or manifest.
7.  The calls log: one line per HTTP attempt, append-only.
8.  Streaming: the same text as a non-streamed call, the timestamp order,
    `t_first` from a reasoning delta, `usage` from the final chunk.
9.  `api_cost` on hand-computed cases (1e-12).
10. `price_from_listing`: per-token strings to per-million, the
    `pricing.overrides` windows of the DeepSeek row, errors.
11. The cost cap: no HTTP request past the cap; 20 concurrent calls never
    overshoot it.
12. `closed_loop_stats`: a discrete-event simulation (exact), N = 2c raises,
    `chat_batch` against `FakeChatServer(0.1, 8)`, and the D1 Monte Carlo.
13. `LocalChatServer` with a stub HTTP server: start, process group gone after
    stop, a server that never becomes ready, one that exits early.
14. Replay without a key and with `openai` blocked from import (subprocess).
15. The parameter merge: a per-call `extra_body` keeps the provider pin and
    changes the hash; `reasoning=None` sends no `reasoning` key.
Also: the spec's signatures, `ChatResult` fields, `models()`, the sync
`chat` inside a running event loop, `n=2` and logprobs parsing,
`Retry-After` parsing, and `import d2l.torch` without `openai`.
The Review* classes at the end are regression tests for the review of the
18.7 library code: replay misses and capped calls raise out of `chat_batch`,
`closed_loop_stats` refuses results that are not one closed loop of one run
and reports `X_tokens` None without usage, every 5xx and stream error events
retry, the reservation covers `n` and one token per prompt byte, torn cache
lines, NumPy parameters, one SDK client per open event loop, and the local
server's log and process group after a failed start.

No network, no GPU: every live call goes to `d2l.FakeChatServer` through
the SDK's own httpx transport. Run:
`.venv-lm/bin/python -m unittest tools/test_lm_api.py`, or `make test-lm`.
"""

import asyncio
import datetime
import gc
import inspect
import json
import math
import os
import random
import signal
import subprocess
import sys
import tempfile
import textwrap
import time
import unittest
from unittest import mock

import numpy as np
from d2l import torch as d2l

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
URL = 'http://fake.local/v1'
KEY = 'sk-test-SECRET-0123456789abcdef'
HELLO = [{'role': 'user', 'content': 'Hello'}]
FLAGSHIP = 'deepseek/deepseek-v4.1-flash-1017'

# Two endpoint rows of /api/v1/models/deepseek/deepseek-v4.1-flash-1017/
# endpoints as listed on 2026-11-12 (trimmed to the fields the code reads).
NOVITA = {'provider_name': 'Novita', 'tag': 'novita/fp8', 'quantization': 'fp8',
          'pricing': {'prompt': '0.000000165', 'completion': '0.00000066',
                      'input_cache_read': '0.0000000033', 'discount': 0.45}}
_WEEKDAYS = ['monday', 'tuesday', 'wednesday', 'thursday', 'friday']
_OFF = {'prompt': '0.00000015', 'completion': '0.0000006',
        'input_cache_read': '0.000000003'}
_PEAK = {'prompt': '0.0000003', 'completion': '0.0000012',
         'input_cache_read': '0.000000006'}
DEEPSEEK = {'provider_name': 'DeepSeek', 'tag': 'deepseek',
            'quantization': 'unknown', 'pricing': {
                **_PEAK, 'discount': 0, 'overrides': [
                    {'utc_days': ['saturday', 'sunday'], **_OFF},
                    {'utc_start': 0, 'utc_end': 100, 'utc_days': _WEEKDAYS,
                     **_OFF},
                    {'utc_start': 100, 'utc_end': 400, 'utc_days': _WEEKDAYS,
                     **_PEAK},
                    {'utc_start': 400, 'utc_end': 600, 'utc_days': _WEEKDAYS,
                     **_OFF},
                    {'utc_start': 600, 'utc_end': 1000,
                     'utc_days': _WEEKDAYS, **_PEAK},
                    {'utc_start': 1000, 'utc_end': 0, 'utc_days': _WEEKDAYS,
                     **_OFF}]}}
LISTING = {'retrieved': '2026-11-12T07:50:00Z',
           'endpoints': {FLAGSHIP: [NOVITA, DEEPSEEK]}}


def utc(*args):
    return datetime.datetime(*args, tzinfo=datetime.timezone.utc)


def fake_api(fake=None, **kwargs):
    """A live client on a FakeChatServer with a fast retry schedule."""
    fake = fake or d2l.FakeChatServer(0.01, 4)
    kwargs.setdefault('mode', 'live')
    api = d2l.ChatAPI(URL, 'fake-model', http_client=fake.transport(),
                      **kwargs)
    api.backoff_base, api.rng = 0.001, random.Random(0)
    return fake, api


def result(t_send, t_total, t_first=None, completion_tokens=10):
    """A ChatResult with only the fields closed_loop_stats reads set."""
    fields = dict.fromkeys(d2l.ChatResult._fields)
    fields.update(t_send=t_send, t_total=t_total, t_first=t_first,
                  usage={'completion_tokens': completion_tokens},
                  n_retries=0, n_429=0)
    return d2l.ChatResult(**fields)


def closed_loop_sim(c, n, capacity, service):
    """Discrete-event closed loop: c clients, a FIFO server of `capacity`
    slots with constant service time; a client resends when its request
    completes, until n requests were sent. Times are integer ticks times
    `service` (a dyadic number, so equal times are equal floats)."""
    queue, sent, running, out, t = list(range(c)), c, [], [], 0
    starts = {i: 0 for i in range(c)}  # request id -> send tick
    while queue or running:
        while queue and len(running) < capacity:
            running.append(queue.pop(0))
        t += 1
        for i in running:
            out.append(result(starts[i] * service, (t - starts[i]) * service))
            if sent < n:
                starts[sent] = t
                queue.append(sent)
                sent += 1
        running = []
    return out


class Signatures(unittest.TestCase):
    """The signatures and records fixed by spec 18.7 part 5."""

    def test_signatures(self):
        expect = {
            d2l.request_hash: '(base_url, model, messages, params, replica=0)',
            d2l.backoff_delay: ('(attempt, base=1.0, cap=60.0, rng=None, '
                                'retry_after=None)'),
            d2l.api_cost: '(usage, price)',
            d2l.price_from_listing: ('(listing, model_id, provider, *, '
                                     'when=None)'),
            d2l.ChatAPI: ("(base_url, model, api_key=None, *, cache=None, "
                          "mode='replay', price=None, cost_cap_usd=None, "
                          "timeout=120.0, max_retries=6, default_params=None, "
                          "http_client=None)"),
            d2l.ChatAPI.chat: ('(self, messages, *, replica=0, stream=False, '
                               '**params)'),
            d2l.ChatAPI.achat: ('(self, messages, *, replica=0, stream=False, '
                                '**params)'),
            d2l.chat_batch: '(api, requests, *, concurrency=8)',
            d2l.openrouter_api: (
                "(model, provider, *, listing, cache=None, mode='replay', "
                "cost_cap_usd=None, reasoning=None, when=None, **kwargs)"),
            d2l.LocalChatServer: ("(cmd, *, port=None, host='127.0.0.1', "
                                  "ready_timeout=900.0)"),
            d2l.FakeChatServer: ('(service_time, capacity, *, fail_rate=0.0, '
                                 'rate_limit=None, script=None, seed=0)'),
            d2l.closed_loop_stats: '(results, c)'}
        for fn, sig in expect.items():
            self.assertEqual(str(inspect.signature(fn)), sig, fn.__name__)
        self.assertTrue(inspect.iscoroutinefunction(d2l.ChatAPI.achat))
        self.assertTrue(inspect.iscoroutinefunction(d2l.chat_batch))
        self.assertIsInstance(d2l.ChatAPI.spent_usd, property)

    def test_records(self):
        self.assertEqual(d2l.ChatResult._fields, (
            'text', 'finish_reason', 'usage', 'logprobs', 'model', 'provider',
            'system_fingerprint', 'reasoning', 'request_hash', 'from_cache',
            't_send', 't_first', 't_first_content', 't_total', 'itl',
            'n_retries', 'n_429', 'created'))
        self.assertEqual(d2l.PricePoint._fields,
                         ('input', 'output', 'cached_input', 'per'))
        self.assertEqual(d2l.PricePoint(1, 2), (1, 2, None, 1_000_000))
        self.assertTrue(issubclass(d2l.ReplayMiss, KeyError))
        self.assertTrue(issubclass(d2l.CostCapExceeded, RuntimeError))
        self.assertEqual(str(d2l.ReplayMiss('plain text')), 'plain text')

    def test_achat_merged_into_class(self):
        # make lib merges the shown `@d2l.add_to_class` method into the body
        with open(os.path.join(REPO, 'd2l', 'torch.py'), encoding='utf-8') as f:
            src = f.read()
        body = src[src.index('class ChatAPI:'):]
        body = body[:body.index('\nclass ', 1)]
        self.assertIn('    async def achat(self, messages, *', body)
        self.assertNotIn('@d2l.add_to_class(d2l.ChatAPI)', src)

    def test_constructor_errors(self):
        with self.assertRaises(ValueError):
            d2l.ChatAPI(URL, 'm', mode='replays')
        with self.assertRaises(ValueError):
            d2l.ChatAPI(URL, 'm', cost_cap_usd=1.0)


class RequestHash(unittest.TestCase):
    """Test 1."""
    model = FLAGSHIP
    pin = {'provider': {'only': ['novita'], 'allow_fallbacks': False,
                        'require_parameters': True}}

    def h(self, params, *, url='https://openrouter.ai/api/v1', model=None,
          messages=HELLO, replica=0):
        return d2l.request_hash(url, model or self.model, messages, params,
                                replica)

    def test_invariances(self):
        p = {'temperature': 0, 'seed': 0, 'max_tokens': 64,
             'extra_body': self.pin}
        ref = self.h(p)
        self.assertRegex(ref, r'^[0-9a-f]{64}$')
        reordered = {'extra_body': {'provider': dict(reversed(list(
            self.pin['provider'].items())))}, 'max_tokens': 64, 'seed': 0,
            'temperature': 0}
        same = [reordered, {**p, 'stream': True},
                {**p, 'stream_options': {'include_usage': True}},
                {**p, 'timeout': 5.0},
                {**p, 'extra_headers': {'Authorization': f'Bearer {KEY}'}},
                {**p, 'top_k': None}]
        for q in same:
            self.assertEqual(self.h(q), ref, q)
        self.assertEqual(self.h(p, url='https://openrouter.ai:8443/api/v1'),
                         ref)
        self.assertEqual(self.h(p, url='https://OpenRouter.ai/api/v1/'), ref)
        local = d2l.request_hash('http://127.0.0.1:8000/v1', 'm', HELLO, p)
        self.assertEqual(
            d2l.request_hash('http://127.0.0.1:41234/v1', 'm', HELLO, p), local)

    def test_changes(self):
        p = {'temperature': 0, 'seed': 0, 'max_tokens': 64,
             'extra_body': self.pin}
        ref = self.h(p)
        other_pin = {'provider': {**self.pin['provider'], 'only': ['nebius']}}
        changed = [
            self.h({**p, 'seed': 1}), self.h({**p, 'max_tokens': 65}),
            self.h(p, model='qwen/qwen3.8-27b'),
            self.h({**p, 'extra_body': other_pin}),
            self.h({**p, 'extra_body': {}}),
            self.h(p, messages=[{'role': 'user', 'content': 'Hellp'}]),
            self.h(p, messages=[{'role': 'system', 'content': 'Hello'}]),
            self.h(p, replica=1),
            self.h(p, url='https://api.deepseek.com/v1')]
        self.assertEqual(len(set(changed + [ref])), len(changed) + 1)

    def test_key_not_in_hash_through_the_client(self):
        fake = d2l.FakeChatServer(0.0, 4)
        hashes = []
        for key in ('key-A-0123456789', 'key-B-0123456789'):
            _, api = fake_api(fake, api_key=key)
            hashes.append(api.chat(HELLO, max_tokens=8).request_hash)
        hashes.append(fake_api(fake)[1].chat(HELLO, max_tokens=8,
                                             stream=True).request_hash)
        self.assertEqual(len(set(hashes)), 1)
        self.assertEqual(hashes[0], d2l.request_hash(URL, 'fake-model', HELLO,
                                                     {'max_tokens': 8}))


class Backoff(unittest.TestCase):
    """Test 2 and Retry-After parsing."""

    def test_full_jitter(self):
        rng, n = random.Random(0), 100_000
        for attempt, base, cap in ((0, 1.0, 60.0), (3, 1.0, 60.0),
                                   (7, 1.0, 60.0), (2, 0.5, 1.5)):
            d = min(cap, base * 2**attempt)
            x = np.array([d2l.backoff_delay(attempt, base, cap, rng)
                          for _ in range(n)])
            self.assertGreaterEqual(x.min(), 0.0)
            self.assertLessEqual(x.max(), d)
            sigma = d / math.sqrt(12 * n)
            self.assertLess(abs(x.mean() - d / 2), 3 * sigma, (attempt, d))

    def test_retry_after_floor(self):
        rng = random.Random(0)
        draws = [d2l.backoff_delay(0, rng=rng, retry_after=3)
                 for _ in range(1000)]
        self.assertTrue(all(x == 3 for x in draws))  # U(0, 1) < 3 always
        self.assertGreaterEqual(d2l.backoff_delay(5, rng=random.Random(0),
                                                  retry_after=3), 3)
        random.seed(0)  # rng=None uses the module's random
        a = d2l.backoff_delay(4)
        random.seed(0)
        self.assertEqual(d2l.backoff_delay(4), a)

    def test_retry_after_header(self):
        self.assertIsNone(d2l._retry_after(None))
        self.assertEqual(d2l._retry_after('0'), 0.0)
        self.assertEqual(d2l._retry_after('2.5'), 2.5)
        self.assertEqual(d2l._retry_after('-3'), 0.0)
        self.assertIsNone(d2l._retry_after('soon'))
        future = (datetime.datetime.now(datetime.timezone.utc)
                  + datetime.timedelta(seconds=30))
        date = future.strftime('%a, %d %b %Y %H:%M:%S GMT')
        self.assertTrue(25 < d2l._retry_after(date) <= 30)
        self.assertEqual(d2l._retry_after('Wed, 21 Oct 2015 07:28:00 GMT'),
                         0.0)


class Retries(unittest.TestCase):
    """Tests 3 and 4."""

    def test_two_429s(self):
        fake, api = fake_api(d2l.FakeChatServer(0.01, 1,
                                                script=[429, 429, 200]))
        with mock.patch.object(d2l, 'backoff_delay',
                               wraps=d2l.backoff_delay) as backoff:
            r = api.chat(HELLO, max_tokens=8)
        self.assertEqual(fake.n_requests, 3)  # no hidden SDK retries
        self.assertEqual(fake.statuses, [429, 429, 200])
        self.assertEqual((r.n_429, r.n_retries), (2, 2))
        self.assertEqual(r.finish_reason, 'length')
        self.assertEqual([c.args[0] for c in backoff.call_args_list], [0, 1])
        self.assertEqual([c.args[4] for c in backoff.call_args_list],
                         [0.0, 0.0])  # Retry-After: 0 passed as the floor
        m = api.manifest()
        self.assertEqual((m['attempts'], m['retries'], m['n_429'],
                          m['failed_attempts'], m['live_calls']),
                         (3, 2, 2, 2, 1))

    def test_no_retry(self):
        import openai
        for status, exc in ((400, openai.BadRequestError),
                            (401, openai.AuthenticationError),
                            (403, openai.PermissionDeniedError),
                            (404, openai.NotFoundError),
                            (422, openai.UnprocessableEntityError)):
            fake, api = fake_api(d2l.FakeChatServer(0.0, 1, script=[status]))
            with self.assertRaises(exc):
                api.chat(HELLO, max_tokens=8)
            self.assertEqual(fake.n_requests, 1, status)

    def test_retry_statuses(self):
        for status in (408, 429, 500, 502, 503, 504):
            fake, api = fake_api(d2l.FakeChatServer(0.0, 1,
                                                    script=[status, 200]))
            r = api.chat(HELLO, max_tokens=8)
            self.assertEqual((fake.n_requests, r.n_retries, r.n_429),
                             (2, 1, int(status == 429)), status)

    def test_fail_rate_exhausts_retries(self):
        import openai
        fake, api = fake_api(d2l.FakeChatServer(0.0, 1, fail_rate=1.0),
                             max_retries=3)
        with self.assertRaises(openai.InternalServerError) as cm:
            api.chat(HELLO, max_tokens=8)
        self.assertEqual(fake.n_requests, 4)
        self.assertEqual(set(fake.statuses), {503})
        self.assertEqual((cm.exception.n_retries, cm.exception.n_429), (3, 0))

    def test_connection_errors_and_timeouts_retry(self):
        import httpx2
        fake = d2l.FakeChatServer(0.0, 1)
        inner, errors = fake.transport(), [httpx2.ConnectError('refused'),
                                           httpx2.ReadTimeout('slow')]

        class Flaky(httpx2.AsyncBaseTransport):
            async def handle_async_request(self, request):
                if errors:
                    raise errors.pop(0)
                return await inner.handle_async_request(request)

        api = d2l.ChatAPI(URL, 'fake-model', mode='live', http_client=Flaky())
        api.backoff_base = 0.001
        r = api.chat(HELLO, max_tokens=8)
        self.assertEqual((r.n_retries, r.n_429, fake.n_requests), (2, 0, 1))

    def test_client_side_error_not_retried(self):
        fake, api = fake_api()
        with self.assertRaises(TypeError):  # the SDK rejects the keyword
            api.chat(HELLO, max_tokens=8, not_a_field=1)
        self.assertEqual(fake.n_requests, 0)

    def test_rate_limit(self):
        fake, api = fake_api(d2l.FakeChatServer(0.0, 8, rate_limit=2),
                             max_retries=0)
        out = asyncio.run(d2l.chat_batch(api, [
            {'messages': HELLO, 'max_tokens': 8, 'replica': i}
            for i in range(5)], concurrency=5))
        self.assertEqual(sum(isinstance(r, Exception) for r in out), 3)
        self.assertEqual(sorted(fake.statuses), [200, 200, 429, 429, 429])


class Batch(unittest.TestCase):
    """Test 5."""

    def test_order_concurrency_failure(self):
        fake, api = fake_api(d2l.FakeChatServer(0.02, 50))
        requests = [{'messages': [{'role': 'user', 'content': 'x' * i}],
                     'max_tokens': 16} for i in range(50)]
        requests[17]['not_a_field'] = 1  # fails client-side (TypeError)
        out = asyncio.run(d2l.chat_batch(api, requests, concurrency=5))
        self.assertEqual(len(out), 50)
        self.assertLessEqual(fake.max_in_flight, 5)
        self.assertEqual(fake.max_in_flight, 5)
        errors = [i for i, r in enumerate(out) if isinstance(r, Exception)]
        self.assertEqual(errors, [17])
        self.assertIsInstance(out[17], TypeError)
        for i, r in enumerate(out):
            if i != 17:
                self.assertIn(f'reply to {i} characters', r.text)

    def test_default_concurrency_and_replicas(self):
        fake, api = fake_api(d2l.FakeChatServer(0.02, 50))
        out = asyncio.run(d2l.chat_batch(
            api, [{'messages': HELLO, 'replica': i, 'max_tokens': 4}
                  for i in range(20)]))
        self.assertEqual(fake.max_in_flight, 8)
        self.assertEqual(len({r.request_hash for r in out}), 20)


class Cache(unittest.TestCase):
    """Tests 6 and 7."""

    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.cache = os.path.join(self.dir.name, 'sub', 'cache.jsonl')
        self.calls = os.path.join(self.dir.name, 'sub', 'cache.calls.jsonl')

    def tearDown(self):
        self.dir.cleanup()

    def read(self, path):
        with open(path, encoding='utf-8') as f:
            return f.read()

    def test_modes(self):
        fake, api = fake_api(mode='record', cache=self.cache, api_key=KEY)
        r1 = api.chat(HELLO, max_tokens=8)
        r2 = api.chat(HELLO, max_tokens=8)
        self.assertEqual(fake.n_requests, 1)
        self.assertFalse(r1.from_cache)
        self.assertTrue(r2.from_cache)
        self.assertEqual(r1._replace(from_cache=True), r2)
        # replay: a hit returns the recorded entry, a miss raises
        replay = d2l.ChatAPI(URL, 'fake-model', cache=self.cache)
        self.assertEqual(replay.chat(HELLO, max_tokens=8), r2)
        long = 'Q' * 100
        with self.assertRaises(d2l.ReplayMiss) as cm:
            replay.chat([{'role': 'user', 'content': long}], max_tokens=8)
        msg = str(cm.exception)
        h = d2l.request_hash(URL, 'fake-model',
                             [{'role': 'user', 'content': long}],
                             {'max_tokens': 8})
        self.assertIn('fake-model', msg)
        self.assertIn(h[:12], msg)
        self.assertIn('Q' * 80, msg)
        self.assertNotIn('Q' * 81, msg)
        with self.assertRaises(d2l.ReplayMiss):
            replay.chat(HELLO, max_tokens=8, replica=1)
        with self.assertRaises(d2l.ReplayMiss):
            replay.models()
        # live appends a second entry; replay still returns the first
        _, live = fake_api(fake, cache=self.cache, api_key=KEY)
        r3 = live.chat(HELLO, max_tokens=8)
        self.assertFalse(r3.from_cache)
        self.assertNotEqual(r3.t_send, r1.t_send)
        self.assertEqual(fake.n_requests, 2)
        replay = d2l.ChatAPI(URL, 'fake-model', cache=self.cache)
        self.assertEqual(len(replay.entries(r1.request_hash)), 2)
        self.assertEqual(replay.chat(HELLO, max_tokens=8), r2)
        self.assertIsNone(replay._sdk)
        # no key in any file or the manifest
        for text in (self.read(self.cache), self.read(self.calls),
                     json.dumps(api.manifest()), json.dumps(live.manifest()),
                     json.dumps(replay.manifest())):
            self.assertNotIn(KEY, text)
        m = replay.manifest()
        self.assertEqual((m['calls'], m['cache_hits'], m['live_calls'],
                          m['attempts'], m['cache_entries']), (1, 1, 0, 0, 2))

    def test_entry_contents(self):
        fake, api = fake_api(mode='record', cache=self.cache, api_key=KEY)
        r = api.chat(HELLO, max_tokens=8, temperature=0, seed=0,
                     timeout=30.0, extra_headers={'X-Key': KEY})
        (line,) = self.read(self.cache).splitlines()
        entry = json.loads(line)
        self.assertEqual(entry['hash'], r.request_hash)
        self.assertEqual(entry['request'], {
            'host': 'fake.local', 'model': 'fake-model', 'messages': HELLO,
            'replica': 0, 'params': {'max_tokens': 8, 'temperature': 0,
                                     'seed': 0}})
        self.assertEqual(entry['sdk'], api.manifest()['sdk'])
        self.assertEqual(set(entry['result']),
                         set(d2l.ChatResult._fields) - {'from_cache'})
        self.assertRegex(r.created, r'^\d{4}-\d\d-\d\dT\d\d:\d\d:\d\dZ$')
        self.assertNotIn(KEY, self.read(self.cache) + self.read(self.calls))

    def test_key_scrubbed_from_error_text(self):
        import httpx2

        class Echo(httpx2.AsyncBaseTransport):  # a server that echoes the key
            async def handle_async_request(self, request):
                return httpx2.Response(401, json={'error': {
                    'message': 'bad key ' + request.headers['authorization']}})

        api = d2l.ChatAPI(URL, 'm', KEY, cache=self.cache, mode='live',
                          http_client=Echo())
        with self.assertRaises(Exception):
            api.chat(HELLO, max_tokens=8)
        log = self.read(self.calls)
        self.assertIn('[api key]', log)
        self.assertNotIn(KEY, log)

    def test_calls_log_append_only(self):
        fake, api = fake_api(d2l.FakeChatServer(0.0, 1, script=[
            503, 200, 429, 200, 400]), cache=self.cache, api_key=KEY)
        api.chat(HELLO, max_tokens=8)
        before = self.read(self.calls)
        self.assertEqual(len(before.splitlines()), 2)
        api.chat(HELLO, max_tokens=8, replica=1)
        with self.assertRaises(Exception):
            api.chat(HELLO, max_tokens=8, replica=2)
        after = self.read(self.calls)
        self.assertTrue(after.startswith(before))
        rows = [json.loads(line) for line in after.splitlines()]
        self.assertEqual(len(rows), fake.n_requests)
        self.assertEqual([r['status'] for r in rows], [503, 200, 429, 200, 400])
        self.assertEqual([r['attempt'] for r in rows], [0, 1, 0, 1, 0])
        self.assertEqual([r['replica'] for r in rows], [0, 0, 1, 1, 2])
        self.assertTrue(all('latency' in r for r in rows))
        self.assertIn('error', rows[4])
        self.assertEqual(rows[1]['served_model'], 'fake-model')
        # the cache holds the two successes only
        self.assertEqual(len(self.read(self.cache).splitlines()), 2)
        # a replay appends nothing
        replay = d2l.ChatAPI(URL, 'fake-model', cache=self.cache)
        replay.chat(HELLO, max_tokens=8)
        self.assertEqual(self.read(self.calls), after)

    def test_replay_reload_identical(self):
        fake, api = fake_api(mode='record', cache=self.cache)
        live = [api.chat([{'role': 'user', 'content': 'x' * i}],
                         max_tokens=16, stream=i % 2 == 1, logprobs=True,
                         top_logprobs=2) for i in range(4)]
        replay = d2l.ChatAPI(URL, 'fake-model', cache=self.cache)
        again = [replay.chat([{'role': 'user', 'content': 'x' * i}],
                             max_tokens=16, logprobs=True, top_logprobs=2)
                 for i in range(4)]
        for a, b in zip(live, again):
            self.assertEqual(a._replace(from_cache=True), b)
            self.assertIsInstance(b.logprobs[0], tuple)
            self.assertIsInstance(b.logprobs[0][2][0], tuple)

    def test_sync_chat_inside_running_loop(self):
        fake, api = fake_api()

        async def notebook_cell():  # a running loop, as in Jupyter
            first = api.chat(HELLO, max_tokens=8)
            second = await api.achat(HELLO, max_tokens=8, replica=1)
            return first, second

        first, second = asyncio.run(notebook_cell())
        self.assertEqual((first.from_cache, second.from_cache), (False, False))
        self.assertEqual(api.models(), ['fake-model'])


class Streaming(unittest.TestCase):
    """Test 8."""

    def test_stream_equals_plain(self):
        fake, api = fake_api(d2l.FakeChatServer(0.05, 4))
        messages = [{'role': 'user', 'content': 'How many apples?'}]
        plain = api.chat(messages, max_tokens=32)
        streamed = api.chat(messages, max_tokens=32, stream=True)
        self.assertEqual(streamed.text, plain.text)
        self.assertEqual(streamed.finish_reason, plain.finish_reason)
        self.assertEqual(streamed.usage, plain.usage)  # from the final chunk
        self.assertIsNone(plain.t_first)
        self.assertIsNone(plain.itl)
        self.assertLessEqual(streamed.t_send, streamed.t_first)
        self.assertLessEqual(streamed.t_first,
                             streamed.t_send + streamed.t_total)
        self.assertEqual(streamed.t_first, streamed.t_first_content)
        self.assertGreaterEqual(streamed.t_first - streamed.t_send, 0.025)
        self.assertEqual(len(streamed.itl), streamed.usage[
            'completion_tokens'] - 1)  # one chunk per fake token
        self.assertTrue(all(g >= 0 for g in streamed.itl))
        self.assertEqual(api.entries(streamed.request_hash)[-1]['n_chunks'],
                         streamed.usage['completion_tokens'])
        self.assertEqual(fake.bodies[-1]['stream_options'],
                         {'include_usage': True})

    def test_reasoning_first(self):
        for body, field in (({'reasoning': {'enabled': True}}, 'reasoning'),
                            ({'chat_template_kwargs': {
                                'enable_thinking': True}},
                             'reasoning_content')):
            fake, api = fake_api(d2l.FakeChatServer(0.05, 4))
            r = api.chat(HELLO, max_tokens=32, stream=True, extra_body=body)
            self.assertEqual(r.reasoning, 'Let me think.')
            self.assertLess(r.t_first, r.t_first_content)
            self.assertEqual(r.usage['reasoning_tokens'], 3)
            self.assertLessEqual(r.t_first_content, r.t_send + r.t_total)
            plain = api.chat(HELLO, max_tokens=32, extra_body=body, replica=1)
            self.assertEqual((plain.reasoning, plain.text), (r.reasoning,
                                                             r.text))

    def test_reasoning_trap(self):
        """Hidden reasoning spends max_tokens: empty content, 'length'."""
        for stream in (False, True):
            _, api = fake_api()
            r = api.chat(HELLO, max_tokens=3, stream=stream,
                         extra_body={'reasoning': {'enabled': True}})
            self.assertEqual((r.text, r.finish_reason), ('', 'length'))
            self.assertIsNone(r.t_first_content)
            self.assertEqual(r.usage['reasoning_tokens'],
                             r.usage['completion_tokens'])

    def test_no_usage_without_include_usage(self):
        _, api = fake_api()
        r = api.chat(HELLO, max_tokens=8, stream=True,
                     stream_options={'include_usage': False})
        self.assertEqual(set(r.usage.values()), {None})


class Parsing(unittest.TestCase):
    """Response normalization."""

    def test_usage_none_never_zero(self):
        self.assertEqual(d2l._chat_usage(None), dict.fromkeys(
            ('prompt_tokens', 'completion_tokens', 'cached_tokens',
             'reasoning_tokens', 'cost')))
        u = d2l._chat_usage({'prompt_tokens': 0, 'completion_tokens': 5,
                             'cost': 0.0})
        self.assertEqual((u['prompt_tokens'], u['cost'], u['cached_tokens']),
                         (0, 0.0, None))

    def test_parse_response(self):
        r = d2l._chat_parse({
            'model': 'served', 'provider': 'Novita', 'prompt_logprobs': [1],
            'system_fingerprint': 'fp', 'metadata': None,
            'choices': [{'message': {'content': None,
                                     'reasoning_content': 'hmm'},
                         'finish_reason': 'length', 'logprobs': {
                             'content': [{'token': 'a', 'logprob': -1.0,
                                          'top_logprobs': [{'token': 'a',
                                                            'logprob': -1.0}]}
                                         ]}}],
            'usage': {'prompt_tokens': 3, 'completion_tokens': 2,
                      'prompt_tokens_details': {'cached_tokens': 1},
                      'completion_tokens_details': {'reasoning_tokens': 2},
                      'cost': 1e-6}})
        self.assertEqual((r['text'], r['reasoning'], r['provider']),
                         ('', 'hmm', 'Novita'))
        self.assertEqual(r['logprobs'], [('a', -1.0, [('a', -1.0)])])
        self.assertEqual(r['usage'], {'prompt_tokens': 3,
                                      'completion_tokens': 2,
                                      'cached_tokens': 1,
                                      'reasoning_tokens': 2, 'cost': 1e-6})
        self.assertEqual(r['extra'], {'prompt_logprobs': [1]})

    def test_n_choices_and_logprobs_through_the_sdk(self):
        fake, api = fake_api()
        r = api.chat(HELLO, max_tokens=8, n=2, logprobs=True, top_logprobs=5)
        extra = api.entries(r.request_hash)[0]['extra']
        self.assertEqual(len(extra['choices']), 2)
        self.assertEqual(len(r.logprobs), 8)
        self.assertEqual(len(r.logprobs[0][2]), 5)
        self.assertEqual(r.system_fingerprint, 'fp_fake')

    def test_stop_and_max_completion_tokens(self):
        _, api = fake_api()
        r = api.chat([{'role': 'user', 'content': 'count'}], max_tokens=32,
                     stop=['characters'])
        self.assertEqual((r.text, r.finish_reason),
                         ('Fake reply to 5 ', 'stop'))
        r = api.chat(HELLO, max_completion_tokens=4)
        self.assertEqual(r.usage['completion_tokens'], 4)


class Cost(unittest.TestCase):
    """Tests 9 and 10."""

    def test_api_cost(self):
        novita = d2l.PricePoint(0.165, 0.66, 0.0033)
        cases = [
            ({'prompt_tokens': 1000, 'completion_tokens': 300},
             d2l.PricePoint(0.165, 0.66), 0.000363),
            ({'prompt_tokens': 1000, 'completion_tokens': 300,
              'cached_tokens': 400}, novita,
             (600 * 0.165 + 400 * 0.0033 + 300 * 0.66) / 1e6),
            ({'prompt_tokens': 1000, 'completion_tokens': 300,
              'cached_tokens': 400}, d2l.PricePoint(0.165, 0.66), 0.000363),
            ({'prompt_tokens': 1000, 'completion_tokens': 300,
              'cached_tokens': None, 'reasoning_tokens': 200, 'cost': 9.9},
             novita, 0.000363),
            ({'prompt_tokens': 50, 'completion_tokens': 10,
              'cached_tokens': 50}, d2l.PricePoint(2.0, 8.0, 0.0, per=1000),
             (0 + 50 * 0.0 + 10 * 8.0) / 1000)]
        for usage, price, want in cases:
            self.assertAlmostEqual(d2l.api_cost(usage, price), want,
                                   delta=1e-12)
        for usage in ({'prompt_tokens': None, 'completion_tokens': 3},
                      {'prompt_tokens': 3, 'completion_tokens': None},
                      {'prompt_tokens': 3}):
            with self.assertRaises(ValueError):
                d2l.api_cost(usage, novita)

    def test_listing_flat(self):
        price = d2l.price_from_listing(LISTING, FLAGSHIP, 'novita')
        self.assertEqual(price, d2l.PricePoint(0.165, 0.66, 0.0033))
        for provider in ('Novita', 'novita/fp8'):
            self.assertEqual(d2l.price_from_listing(LISTING, FLAGSHIP,
                                                    provider), price)
        raw = {'endpoints': {FLAGSHIP: {'data': {'endpoints': [NOVITA]}}}}
        self.assertEqual(d2l.price_from_listing(raw, FLAGSHIP, 'novita'),
                         price)
        self.assertEqual(d2l.api_cost(
            {'prompt_tokens': 1000, 'completion_tokens': 300}, price),
            0.000363)

    def test_listing_windows(self):
        with self.assertRaises(ValueError):
            d2l.price_from_listing(LISTING, FLAGSHIP, 'deepseek')
        with self.assertRaises(ValueError):  # naive datetime
            d2l.price_from_listing(LISTING, FLAGSHIP, 'deepseek',
                                   when=datetime.datetime(2026, 11, 12, 2))
        peak = d2l.PricePoint(0.3, 1.2, 0.006)
        off = d2l.PricePoint(0.15, 0.6, 0.003)
        cases = [(utc(2026, 11, 12, 2, 0), peak),     # Thursday 02:00
                 (utc(2026, 11, 12, 5, 0), off),      # Thursday 05:00
                 (utc(2026, 11, 12, 9, 59), peak),
                 (utc(2026, 11, 12, 10, 0), off),
                 (utc(2026, 11, 12, 0, 30), off),
                 (utc(2026, 11, 12, 23, 59), off),
                 (utc(2026, 11, 14, 2, 0), off),      # Saturday
                 (utc(2026, 11, 16, 7, 0), peak)]     # Monday
        for when, want in cases:
            self.assertEqual(d2l.price_from_listing(
                LISTING, FLAGSHIP, 'deepseek', when=when), want, when)
        pacific = datetime.timezone(datetime.timedelta(hours=-8))
        wed_evening = datetime.datetime(2026, 11, 11, 18, 0, tzinfo=pacific)
        self.assertEqual(d2l.price_from_listing(
            LISTING, FLAGSHIP, 'deepseek', when=wed_evening), peak)
        # a row without overrides ignores when
        self.assertEqual(d2l.price_from_listing(
            LISTING, FLAGSHIP, 'novita', when=utc(2026, 11, 12, 2)).input,
            0.165)

    def test_listing_errors(self):
        with self.assertRaises(ValueError):
            d2l.price_from_listing(LISTING, FLAGSHIP, 'parasail')
        twin = dict(NOVITA, tag='novita/bf16')
        listing = {'endpoints': {FLAGSHIP: [NOVITA, twin]}}
        with self.assertRaises(ValueError):  # ambiguous by name
            d2l.price_from_listing(listing, FLAGSHIP, 'novita')
        self.assertEqual(d2l.price_from_listing(
            listing, FLAGSHIP, 'novita/bf16').output, 0.66)
        variable = {'provider_name': 'X', 'tag': 'x', 'pricing': {
            'prompt': '-1', 'completion': '-1'}}
        with self.assertRaises(ValueError):
            d2l.price_from_listing({'endpoints': {'m': [variable]}}, 'm', 'x')
        with self.assertRaises(KeyError):
            d2l.price_from_listing(LISTING, 'other/model', 'novita')


class CostCap(unittest.TestCase):
    """Test 11."""
    price = d2l.PricePoint(1000.0, 1000.0)  # 0.001 USD per token

    def worst(self, messages, max_tokens, n=1):
        size = len(json.dumps(messages, ensure_ascii=False).encode('utf-8'))
        return (size + n * max_tokens) * 0.001

    def test_single_call_over_cap(self):
        fake, api = fake_api(price=self.price,
                             cost_cap_usd=self.worst(HELLO, 64) - 1e-9)
        with self.assertRaises(d2l.CostCapExceeded):
            api.chat(HELLO, max_tokens=64)
        self.assertEqual(fake.n_requests, 0)
        r = api.chat(HELLO, max_tokens=8)  # this one fits
        self.assertEqual(api.spent_usd, d2l.api_cost(r.usage, self.price))
        self.assertEqual(api._reserved, 0.0)

    def test_price_needs_max_tokens(self):
        fake, api = fake_api(price=self.price)
        with self.assertRaises(ValueError):
            api.chat(HELLO)
        self.assertEqual(fake.n_requests, 0)
        fake_api(fake)[1].chat(HELLO)  # unpriced: fine

    def test_concurrent_calls_never_overshoot(self):
        cap = 0.5
        fake, api = fake_api(d2l.FakeChatServer(0.05, 20), price=self.price,
                             cost_cap_usd=cap)
        requests = [{'messages': HELLO, 'max_tokens': 100, 'replica': i}
                    for i in range(20)]
        with self.assertRaises(d2l.CostCapExceeded) as cm:  # after all ran
            asyncio.run(d2l.chat_batch(api, requests, concurrency=20))
        out = cm.exception.results
        capped = [r for r in out if isinstance(r, d2l.CostCapExceeded)]
        done = [r for r in out if isinstance(r, d2l.ChatResult)]
        self.assertEqual(len(capped) + len(done), 20)
        self.assertGreater(len(capped), 0)
        self.assertEqual(len(done), int(cap // self.worst(HELLO, 100)))
        self.assertEqual(fake.n_requests, len(done))
        self.assertLessEqual(api.spent_usd, cap)
        self.assertAlmostEqual(api.spent_usd, sum(
            d2l.api_cost(r.usage, self.price) for r in done), delta=1e-12)
        self.assertAlmostEqual(api._reserved, 0.0, delta=1e-12)
        # the settled slack admits later calls
        with self.assertRaises(d2l.CostCapExceeded) as cm:
            asyncio.run(d2l.chat_batch(api, requests, concurrency=20))
        later = cm.exception.results
        self.assertLessEqual(api.spent_usd, cap)
        self.assertGreater(sum(isinstance(r, d2l.ChatResult)
                               for r in later), 0)

    def test_failed_call_releases_reservation(self):
        fake, api = fake_api(d2l.FakeChatServer(0.0, 1, script=[400]),
                             price=self.price, cost_cap_usd=1.0)
        with self.assertRaises(Exception):
            api.chat(HELLO, max_tokens=8)
        self.assertEqual((api._reserved, api.spent_usd), (0.0, 0.0))

    def test_usage_cost_settles_when_larger(self):
        fake, api = fake_api(price=d2l.PricePoint(0.0, 0.0))
        r = api.chat(HELLO, max_tokens=8,
                     extra_body={'provider': {'only': ['fake']}})
        self.assertGreater(r.usage['cost'], 0)
        self.assertEqual(api.spent_usd, r.usage['cost'])
        m = api.manifest()
        self.assertEqual((m['n_api_cost'], m['sum_api_cost'],
                          m['n_usage_cost'], m['sum_usage_cost']),
                         (1, 0.0, 1, r.usage['cost']))


class ClosedLoop(unittest.TestCase):
    """Test 12."""
    W_0, capacity = 0.125, 8  # dyadic service time: exact float timestamps

    def test_simulation_exact(self):
        X_max = self.capacity / self.W_0
        for c in (1, 2, 4, 8, 16, 32):
            out = closed_loop_sim(c, 8 * c, self.capacity, self.W_0)
            st = d2l.closed_loop_stats(out, c)
            self.assertAlmostEqual(st['X'], min(c / self.W_0, X_max),
                                   delta=1e-9)
            self.assertAlmostEqual(st['little'], 1.0, delta=1e-9)
            self.assertAlmostEqual(st['sustained'], 1.0, delta=1e-9)
            self.assertAlmostEqual(st['W'], max(self.W_0, c / X_max),
                                   delta=1e-9)
            self.assertAlmostEqual(st['Lbar'], c, delta=1e-9)
            self.assertAlmostEqual(st['X_tokens'], 10 * st['X'], delta=1e-6)
            self.assertEqual((st['n_failed'], st['n_429'], st['n_retries']),
                             (0, 0, 0))
            self.assertIsNone(st['ttft_p50'])

    def test_window_bounds(self):
        out = closed_loop_sim(4, 8, self.capacity, self.W_0)  # N = 2c
        with self.assertRaises(ValueError):
            d2l.closed_loop_stats(out, 4)
        out = closed_loop_sim(4, 9, self.capacity, self.W_0)
        d2l.closed_loop_stats(out, 4)
        # exceptions are counted and excluded, their retries and 429s kept
        err = RuntimeError('boom')
        err.n_retries, err.n_429 = 6, 4
        st = d2l.closed_loop_stats(closed_loop_sim(2, 16, 8, 0.125)
                                   + [err], 2)
        self.assertEqual((st['n_failed'], st['n_retries'], st['n_429']),
                         (1, 6, 4))
        with self.assertRaises(ValueError):
            d2l.closed_loop_stats([err] * 10 + closed_loop_sim(2, 4, 8, 0.125),
                                  2)

    def test_steady_window_and_ttft(self):
        # the contract by hand: c = 1, five sequential requests of 1 s
        out = [result(i, 1.0, t_first=i + 0.1 * (i + 1),
                      completion_tokens=i) for i in range(5)]
        st = d2l.closed_loop_stats(out, 1)
        # f = 1..5, t_a = f_(1) = 1, t_b = f_(4) = 4: completions 2, 3, 4
        self.assertEqual((st['window'], st['n_window']), (3.0, 3))
        self.assertAlmostEqual(st['X'], 1.0)
        self.assertAlmostEqual(st['X_tokens'], (1 + 2 + 3) / 3)
        self.assertAlmostEqual(st['W'], 1.0)
        self.assertAlmostEqual(st['ttft_p50'], 0.3)
        self.assertAlmostEqual(st['ttft_p90'],
                               float(np.percentile([.1, .2, .3, .4, .5], 90)))

    def test_through_fake_server(self):
        # Real time: a full collection of what earlier tests left (the build
        # of test_build_lib_async) inside the c = 32 level skews it by 5%.
        gc.collect()
        gc.freeze()
        self.addCleanup(gc.unfreeze)
        fake = d2l.FakeChatServer(0.1, 8)
        api = d2l.ChatAPI(URL, 'fake-model', mode='live',
                          http_client=fake.transport())
        W_0 = None  # measured at c = 1: service time plus client work
        for c in (1, 2, 32):
            requests = [{'messages': [{'role': 'user', 'content': f'q{i}'}],
                         'max_tokens': 16} for i in range(8 * c)]
            out = asyncio.run(d2l.chat_batch(api, requests, concurrency=c))
            st = d2l.closed_loop_stats(out, c)
            W_0 = W_0 or st['W']
            self.assertTrue(0.1 <= W_0 < 0.115, W_0)
            self.assertLess(abs(st['X'] / min(c / W_0, 80) - 1), 0.05, c)
            self.assertLess(abs(st['little'] - 1), 0.02, c)
            self.assertLess(abs(st['sustained'] - 1), 0.02, c)

    def test_backoff_monte_carlo(self):
        """D1: E[S] = sum_a p^(a+1) d_a / 2 = 0.3575 s at p = 0.3, K = 6."""
        p, K, rng = 0.3, 6, random.Random(0)
        formula = sum(p**(a + 1) * min(60, 2**a) / 2 for a in range(K))
        self.assertAlmostEqual(formula, 0.15 * (1 - 0.6**6) / 0.4, 12)
        self.assertAlmostEqual(formula, 0.357504, 6)

        def total_sleep():
            sleep = 0.0
            for a in range(K):
                if rng.random() >= p:
                    break
                sleep += d2l.backoff_delay(a, 1.0, 60.0, rng)
            return sleep

        draws = np.array([total_sleep() for _ in range(200_000)])
        sigma = draws.std() / math.sqrt(len(draws))
        self.assertLess(abs(draws.mean() - formula), 3 * sigma)
        # unjittered, six failures sleep 63 s; jittered, 31.5 s on average
        self.assertEqual(sum(min(60, 2**a) for a in range(6)), 63)


STUB = textwrap.dedent('''\
    import http.server, subprocess, sys, time
    mode, port = sys.argv[1], int(sys.argv[2])
    print('stub starting', mode, flush=True)
    if mode == 'never':
        time.sleep(120)
    if mode == 'exit':
        sys.exit(3)
    child = subprocess.Popen([sys.executable, '-c',
                              'import time; time.sleep(120)'])
    class H(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            ok = self.path == '/v1/models'
            self.send_response(200 if ok else 404)
            self.end_headers()
            self.wfile.write(b'{"object": "list", "data": [{"id": "stub"}]}')
        def log_message(self, *args):
            pass
    time.sleep(0.5)  # not ready at once
    http.server.HTTPServer(('127.0.0.1', port), H).serve_forever()
    ''')


class LocalServer(unittest.TestCase):
    """Test 13."""

    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.stub = os.path.join(self.dir.name, 'stub.py')
        with open(self.stub, 'w') as f:
            f.write(STUB)

    def tearDown(self):
        self.dir.cleanup()

    def group_gone(self, pgid, timeout=10.0):
        t0 = time.monotonic()
        while time.monotonic() - t0 < timeout:
            try:
                os.killpg(pgid, 0)
            except ProcessLookupError:
                return True
            time.sleep(0.05)
        return False

    def test_start_stop(self):
        server = d2l.LocalChatServer([sys.executable, self.stub, 'serve',
                                      '{port}'], ready_timeout=30)
        url = server.start()
        self.assertEqual(url, f'http://127.0.0.1:{server.port}/v1')
        self.assertEqual(server.start(), url)  # idempotent
        self.assertGreater(server.load_time, 0.4)
        pgid = server.proc.pid
        self.assertEqual(os.getpgid(pgid), pgid)  # its own process group
        self.assertNotEqual(os.getpgid(pgid), os.getpgrp())
        code = server.stop()
        self.assertEqual(code, -signal.SIGTERM)
        self.assertTrue(self.group_gone(pgid))
        self.assertIsNone(server.stop())

    def test_context_manager_and_port(self):
        with d2l.LocalChatServer([sys.executable, self.stub, 'serve',
                                  '{port}'], ready_timeout=30) as server:
            pgid = server.proc.pid
            self.assertTrue(server.base_url.endswith(f':{server.port}/v1'))
        self.assertIsNone(server.proc)
        self.assertTrue(self.group_gone(pgid))

    def test_never_ready(self):
        server = d2l.LocalChatServer([sys.executable, self.stub, 'never',
                                      '{port}'], ready_timeout=2)
        t0 = time.monotonic()
        with self.assertRaises(TimeoutError) as cm:
            server.start()
        self.assertLess(time.monotonic() - t0, 10)
        self.assertIn('stub starting never', str(cm.exception))
        self.assertIsNone(server.proc)

    def test_early_exit(self):
        server = d2l.LocalChatServer([sys.executable, self.stub, 'exit',
                                      '{port}'], ready_timeout=30)
        with self.assertRaises(RuntimeError) as cm:
            server.start()
        self.assertIn('code 3', str(cm.exception))
        self.assertIn('stub starting exit', str(cm.exception))


REPLAY_CHILD = textwrap.dedent('''\
    import json, os, sys
    class Block:
        def find_spec(self, name, path=None, target=None):
            if name.split('.')[0] in ('openai', 'httpx2', 'httpx',
                                      'transformers'):
                raise ImportError('blocked: ' + name)
    sys.meta_path.insert(0, Block())
    for k in ('OPENAI_API_KEY', 'OPENROUTER_API_KEY'):
        assert k not in os.environ, k
    from d2l import torch as d2l
    listing = json.loads(sys.argv[2])
    api = d2l.openrouter_api(%r, 'novita', listing=listing, cache=sys.argv[1],
                             reasoning={'enabled': False})
    r = api.chat([{'role': 'user', 'content': 'Hello'}], max_tokens=8,
                 temperature=0)
    assert r.from_cache and api._sdk is None, r
    assert 'openai' not in sys.modules
    print(json.dumps({'text': r.text, 'provider': r.provider,
                      'hash': r.request_hash}))
    ''') % FLAGSHIP


class ReplayWithoutKey(unittest.TestCase):
    """Test 14 and the router defaults of test 15."""

    def test_replay_in_a_clean_process(self):
        with tempfile.TemporaryDirectory() as d:
            cache = os.path.join(d, 'cache.jsonl')
            fake = d2l.FakeChatServer(0.0, 1)
            api = d2l.openrouter_api(
                FLAGSHIP, 'novita', listing=LISTING, cache=cache,
                mode='record', reasoning={'enabled': False}, api_key=KEY,
                http_client=fake.transport())
            rec = api.chat(HELLO, max_tokens=8, temperature=0)
            self.assertEqual(rec.provider, 'novita')
            env = {k: v for k, v in os.environ.items()
                   if k not in ('OPENAI_API_KEY', 'OPENROUTER_API_KEY')}
            env['CUDA_VISIBLE_DEVICES'] = ''
            child = os.path.join(d, 'child.py')
            with open(child, 'w') as f:
                f.write(REPLAY_CHILD)
            out = subprocess.run(
                [sys.executable, child, cache, json.dumps(LISTING)],
                env=env, capture_output=True, text=True, timeout=300)
            self.assertEqual(out.returncode, 0, out.stderr[-2000:])
            got = json.loads(out.stdout.strip().splitlines()[-1])
            self.assertEqual(got, {'text': rec.text, 'provider': 'novita',
                                   'hash': rec.request_hash})
            with open(cache, encoding='utf-8') as f:
                self.assertNotIn(KEY, f.read())

    def test_key_read_at_first_live_call_only(self):
        fake = d2l.FakeChatServer(0.0, 1)
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop('OPENROUTER_API_KEY', None)
            api = d2l.openrouter_api(FLAGSHIP, 'novita', listing=LISTING,
                                     mode='live',
                                     http_client=fake.transport())
            self.assertEqual(api.price, d2l.PricePoint(0.165, 0.66, 0.0033))
            with self.assertRaises(RuntimeError):  # no key: no request
                api.chat(HELLO, max_tokens=8)
            self.assertEqual(fake.n_requests, 0)
            os.environ['OPENROUTER_API_KEY'] = KEY
            api.chat(HELLO, max_tokens=8)
        self.assertEqual(fake.n_requests, 1)
        self.assertNotIn(KEY, json.dumps(api.manifest()))

    def test_unpriced_without_listing(self):
        api = d2l.openrouter_api(FLAGSHIP, 'deepseek', listing=None)
        self.assertIsNone(api.price)
        with self.assertRaises(ValueError):  # windowed row needs when
            d2l.openrouter_api(FLAGSHIP, 'deepseek', listing=LISTING)
        api = d2l.openrouter_api(FLAGSHIP, 'deepseek', listing=LISTING,
                                 when=utc(2026, 11, 12, 2))
        self.assertEqual(api.price.input, 0.3)
        self.assertEqual(api.base_url, 'https://openrouter.ai/api/v1')


class Merge(unittest.TestCase):
    """Test 15 and the merge rule."""

    def test_merge_rule(self):
        base = {'temperature': 0, 'seed': 0, 'extra_body': {
            'provider': {'only': ['a'], 'allow_fallbacks': False},
            'reasoning': {'enabled': False}}}
        frozen = json.dumps(base, sort_keys=True)
        update = {'seed': None, 'max_tokens': 8, 'extra_body': {
            'provider': {'only': ['b'], 'sort': None},
            'reasoning': None, 'top_k': 5}}
        self.assertEqual(d2l._chat_merge(base, update), {
            'temperature': 0, 'max_tokens': 8, 'extra_body': {
                'provider': {'only': ['b'], 'allow_fallbacks': False},
                'top_k': 5}})
        self.assertEqual(json.dumps(base, sort_keys=True), frozen)
        self.assertEqual(d2l._chat_merge({'a': {'b': None, 'c': 1}}, {}),
                         {'a': {'c': 1}})
        self.assertEqual(d2l._chat_merge({'a': 1}, {'a': {'b': 2}}),
                         {'a': {'b': 2}})
        self.assertEqual(d2l._chat_merge({'a': {'b': 2}}, {'a': 3}),
                         {'a': 3})

    def test_per_call_extra_body_keeps_the_pin(self):
        fake = d2l.FakeChatServer(0.0, 1)
        api = d2l.openrouter_api(FLAGSHIP, 'novita', listing=LISTING,
                                 mode='live', api_key=KEY,
                                 http_client=fake.transport())
        plain = api.chat(HELLO, max_tokens=8)
        body = fake.bodies[-1]
        self.assertNotIn('reasoning', body)  # reasoning=None sends none
        pin = {'only': ['novita'], 'allow_fallbacks': False,
               'require_parameters': True}
        self.assertEqual(body['provider'], pin)
        off = api.chat(HELLO, max_tokens=8,
                       extra_body={'reasoning': {'enabled': False}})
        body = fake.bodies[-1]
        self.assertEqual(body['provider'], pin)
        self.assertEqual(body['reasoning'], {'enabled': False})
        self.assertNotEqual(off.request_hash, plain.request_hash)
        self.assertEqual(off.request_hash, d2l.request_hash(
            'https://openrouter.ai/api/v1', FLAGSHIP, HELLO,
            {'max_tokens': 8, 'extra_body': {'provider': pin, 'reasoning': {
                'enabled': False}}}))
        # the constructor's reasoning switch lands in the same place
        api = d2l.openrouter_api(FLAGSHIP, 'novita', listing=LISTING,
                                 mode='live', api_key=KEY,
                                 reasoning={'enabled': False},
                                 http_client=fake.transport())
        same = api.chat(HELLO, max_tokens=8)
        self.assertEqual(fake.bodies[-1]['reasoning'], {'enabled': False})
        self.assertEqual(same.request_hash, off.request_hash)
        # a per-call provider field merges into the pin
        api.chat(HELLO, max_tokens=8,
                 extra_body={'provider': {'quantizations': ['fp8']}})
        self.assertEqual(fake.bodies[-1]['provider'],
                         {**pin, 'quantizations': ['fp8']})

    def test_default_params(self):
        fake = d2l.FakeChatServer(0.0, 1)
        api = d2l.ChatAPI(URL, 'fake-model', mode='live',
                          http_client=fake.transport(),
                          default_params={'temperature': 0, 'seed': 0,
                                          'max_tokens': 16})
        r = api.chat(HELLO, seed=None, max_tokens=8)
        self.assertEqual({k: fake.bodies[-1][k] for k in fake.bodies[-1]
                          if k not in ('model', 'messages')},
                         {'temperature': 0, 'max_tokens': 8})
        self.assertEqual(r.request_hash, d2l.request_hash(
            URL, 'fake-model', HELLO, {'temperature': 0, 'max_tokens': 8}))


class ImportWithoutSDK(unittest.TestCase):
    """`import d2l.torch` and replay need neither openai nor httpx."""

    def test_import(self):
        code = textwrap.dedent('''\
            import sys
            class Block:
                def find_spec(self, name, path=None, target=None):
                    if name.split('.')[0] in ('openai', 'httpx2', 'httpx'):
                        raise ImportError(name)
            sys.meta_path.insert(0, Block())
            from d2l import torch as d2l
            api = d2l.ChatAPI('http://127.0.0.1/v1', 'm')
            try:
                api.chat([{'role': 'user', 'content': 'x'}])
            except d2l.ReplayMiss as e:
                print('miss', e)
            ''')
        env = dict(os.environ, CUDA_VISIBLE_DEVICES='')
        out = subprocess.run([sys.executable, '-c', code], env=env,
                             capture_output=True, text=True, timeout=300)
        self.assertEqual(out.returncode, 0, out.stderr[-2000:])
        self.assertIn('miss m: request', out.stdout)


# ---------------------------------------------------------------------------
# Regression tests for the review of the 18.7 library code (stage 2a): each
# one failed before its fix.


def numbered(n, **params):
    return [{'messages': [{'role': 'user', 'content': f'q{i}'}], **params}
            for i in range(n)]


class ReviewRetry(unittest.TestCase):
    """Finding 3: every 5xx and an error event inside a stream retry."""

    def test_every_5xx_retries(self):
        for status in (501, 520, 522, 524, 529):
            fake, api = fake_api(d2l.FakeChatServer(0.0, 1,
                                                    script=[status, 200]))
            r = api.chat(HELLO, max_tokens=8)
            self.assertEqual((fake.n_requests, r.n_retries), (2, 1), status)

    def midstream(self, error):
        """A transport that answers 200 and then an SSE error event once."""
        import httpx2
        first = (b'data: {"id":"x","object":"chat.completion.chunk",'
                 b'"created":0,"model":"m","choices":[{"index":0,'
                 b'"delta":{"content":"Hi"},"finish_reason":null}]}\n\n'
                 b'data: ' + json.dumps(error).encode() + b'\n\n')
        good = (b'data: {"id":"x","object":"chat.completion.chunk",'
                b'"created":0,"model":"m","choices":[{"index":0,'
                b'"delta":{"content":"Hi"},"finish_reason":"stop"}]}\n\n'
                b'data: [DONE]\n\n')
        seen = []

        class Transport(httpx2.AsyncBaseTransport):
            async def handle_async_request(self, request):
                seen.append(1)
                return httpx2.Response(
                    200, headers={'content-type': 'text/event-stream'},
                    content=first if len(seen) == 1 else good)

        api = d2l.ChatAPI(URL, 'm', mode='live', http_client=Transport())
        api.backoff_base = 0.001
        return api, seen

    def test_midstream_error_event_retries(self):
        for error in ({'error': {'code': 502, 'message': 'Provider error'}},
                      {'error': {'code': 'server_error', 'message': 'gone'},
                       'choices': [{'index': 0, 'delta': {'content': ''},
                                    'finish_reason': 'error'}]}):
            api, seen = self.midstream(error)
            r = api.chat(HELLO, max_tokens=8, stream=True)
            self.assertEqual((r.text, r.n_retries, len(seen)), ('Hi', 1, 2))

    def test_midstream_client_error_does_not_retry(self):
        import openai
        api, seen = self.midstream({'error': {'code': 400, 'message': 'no'}})
        with self.assertRaises(openai.APIError):
            api.chat(HELLO, max_tokens=8, stream=True)
        self.assertEqual(len(seen), 1)


class ReviewCostCap(unittest.TestCase):
    """Finding 4: the reservation covers n choices and digit-heavy prompts."""

    def test_n_choices_reserved(self):
        class FakeN(d2l.FakeChatServer):
            def _reply(self, body):  # usage of all n choices, as real servers
                tokens, finish, usage = super()._reply(body)
                k = body.get('n') or 1
                return tokens, finish, dict(usage,
                                            completion_tokens=k * len(tokens))
        price = d2l.PricePoint(1.0, 1000.0)
        one = d2l.api_cost({'prompt_tokens': 0, 'completion_tokens': 8}, price)
        fake, api = fake_api(FakeN(0.01, 4), price=price,
                             cost_cap_usd=1.5 * one + 1e-4)
        with self.assertRaises(d2l.CostCapExceeded):
            api.chat(HELLO, max_tokens=8, n=2)
        self.assertEqual(fake.n_requests, 0)
        api.cost_cap_usd = 2 * one + 1e-3
        api.chat(HELLO, max_tokens=8, n=2)
        self.assertLessEqual(api.spent_usd, api.cost_cap_usd)

    def test_prompt_bound_holds_for_digits(self):
        """One token per byte bounds a digit-heavy prompt that ceil(bytes/2)
        undercounts (Qwen3 splits numbers into single digits)."""
        price = d2l.PricePoint(1000.0, 0.0)
        digits = [{'role': 'user', 'content': ' '.join(
            str(random.Random(i).randrange(10**8)) for i in range(50))}]
        size = len(json.dumps(digits).encode('utf-8'))
        digit_tokens = sum(ch.isdigit() for ch in digits[0]['content'])
        self.assertGreater(digit_tokens, math.ceil(size / 2))
        _, api = fake_api(price=price, cost_cap_usd=1e-3 * size + 1e-9)
        api.chat(digits, max_tokens=1)  # fits: size bytes, output free
        with self.assertRaises(d2l.CostCapExceeded):
            fake_api(price=price, cost_cap_usd=1e-3 * size - 1e-6)[1].chat(
                digits, max_tokens=1)


class ReviewBatch(unittest.TestCase):
    """Findings 1 and 6: replay misses raise; malformed requests and NumPy
    parameters."""

    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.cache = os.path.join(self.dir.name, 'cache.jsonl')

    def tearDown(self):
        self.dir.cleanup()

    def test_replay_miss_raises_after_the_batch(self):
        reqs = numbered(16, max_tokens=8)
        _, rec = fake_api(d2l.FakeChatServer(0.01, 8), mode='record',
                          cache=self.cache)
        asyncio.run(d2l.chat_batch(rec, reqs[:-2], concurrency=2))
        replay = d2l.ChatAPI(URL, 'fake-model', cache=self.cache)
        with self.assertRaises(d2l.ReplayMiss) as cm:
            asyncio.run(d2l.chat_batch(replay, reqs, concurrency=2))
        out = cm.exception.results
        self.assertEqual(len(out), 16)
        self.assertEqual([isinstance(r, d2l.ReplayMiss) for r in out],
                         [False] * 14 + [True] * 2)
        with self.assertRaises(ValueError):  # and never into statistics
            d2l.closed_loop_stats(out, 2)

    def test_malformed_request_returned(self):
        fake, api = fake_api()
        out = asyncio.run(d2l.chat_batch(
            api, [{'messages': HELLO, 'max_tokens': 8}, {'max_tokens': 8}],
            concurrency=2))
        self.assertIsInstance(out[0], d2l.ChatResult)
        self.assertIsInstance(out[1], KeyError)

    def test_numpy_scalar_params(self):
        _, api = fake_api()
        r = api.chat(HELLO, max_tokens=np.int64(8),
                     temperature=np.float32(0.5), stop=[np.str_('x')])
        self.assertEqual(r.request_hash, d2l.request_hash(
            URL, 'fake-model', HELLO,
            {'max_tokens': 8, 'temperature': 0.5, 'stop': ['x']}))
        self.assertEqual(d2l.request_hash(URL, 'm', HELLO,
                                          {'logit_bias': {'5': np.int8(-1)}}),
                         d2l.request_hash(URL, 'm', HELLO,
                                          {'logit_bias': {'5': -1}}))


class ReviewCacheFile(unittest.TestCase):
    """Finding 5: a torn last line (a run killed mid-write)."""

    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.cache = os.path.join(self.dir.name, 'cache.jsonl')

    def tearDown(self):
        self.dir.cleanup()

    def test_torn_last_line(self):
        _, rec = fake_api(mode='record', cache=self.cache)
        r = rec.chat(HELLO, max_tokens=8)
        with open(self.cache, 'a', encoding='utf-8') as f:
            f.write('{"hash": "abc", "request": {"mod')
        with self.assertWarns(UserWarning):
            replay = d2l.ChatAPI(URL, 'fake-model', cache=self.cache)
        self.assertEqual(replay.chat(HELLO, max_tokens=8).text, r.text)
        with self.assertWarns(UserWarning):  # recording again cuts it off
            _, rec = fake_api(mode='record', cache=self.cache)
        rec.chat(HELLO, max_tokens=8, replica=1)
        with open(self.cache, encoding='utf-8') as f:
            lines = f.read().split('\n')
        self.assertEqual(lines[-1], '')
        self.assertEqual([json.loads(l)['result']['created'] is not None
                          for l in lines[:-1]], [True, True])
        replay = d2l.ChatAPI(URL, 'fake-model', cache=self.cache)
        self.assertEqual(replay.manifest()['cache_entries'], 2)

    def test_complete_last_line_without_newline_kept(self):
        _, rec = fake_api(mode='record', cache=self.cache)
        rec.chat(HELLO, max_tokens=8)
        with open(self.cache, 'rb+') as f:
            f.truncate(os.path.getsize(self.cache) - 1)  # drop the newline
        _, rec = fake_api(mode='record', cache=self.cache)
        self.assertTrue(rec.chat(HELLO, max_tokens=8).from_cache)
        rec.chat(HELLO, max_tokens=8, replica=1)
        replay = d2l.ChatAPI(URL, 'fake-model', cache=self.cache)
        self.assertEqual(replay.manifest()['cache_entries'], 2)

    def test_damaged_middle_line_raises(self):
        _, rec = fake_api(mode='record', cache=self.cache)
        rec.chat(HELLO, max_tokens=8)
        with open(self.cache, 'a', encoding='utf-8') as f:
            f.write('{"torn\n')
        rec.chat(HELLO, max_tokens=8, replica=1)
        with self.assertRaises(ValueError):
            d2l.ChatAPI(URL, 'fake-model', cache=self.cache)


class ReviewClosedLoop(unittest.TestCase):
    """Findings 2 and 7: one loop of c from one run; X_tokens None."""

    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.cache = os.path.join(self.dir.name, 'cache.jsonl')

    def tearDown(self):
        self.dir.cleanup()

    def test_cached_and_live_refused(self):
        c = 4
        _, api = fake_api(d2l.FakeChatServer(0.05, 8), mode='record',
                          cache=self.cache)
        first = asyncio.run(d2l.chat_batch(api, numbered(8 * c,
                                                         max_tokens=8),
                                           concurrency=c))
        self.assertLess(abs(d2l.closed_loop_stats(first, c)['little'] - 1),
                        0.05)
        _, api = fake_api(d2l.FakeChatServer(0.05, 8), mode='record',
                          cache=self.cache)
        doubled = asyncio.run(d2l.chat_batch(api, numbered(16 * c,
                                                           max_tokens=8),
                                             concurrency=c))
        self.assertEqual(sum(r.from_cache for r in doubled), 8 * c)
        with self.assertRaises(ValueError) as cm:
            d2l.closed_loop_stats(doubled, c)
        self.assertIn('cached and live', str(cm.exception))

    def test_more_than_c_in_flight_refused(self):
        out = closed_loop_sim(4, 32, 8, 0.125)
        d2l.closed_loop_stats(out, 4)
        with self.assertRaises(ValueError):  # two loops of 4 overlaid
            d2l.closed_loop_stats(out + closed_loop_sim(4, 32, 8, 0.125), 4)

    def test_two_runs_refused(self):
        def stamped(results, epoch):
            return [r._replace(created=time.strftime(
                '%Y-%m-%dT%H:%M:%SZ', time.gmtime(epoch + r.t_send)))
                for r in results]
        run = closed_loop_sim(2, 16, 8, 0.125)
        d2l.closed_loop_stats(stamped(run, 1.8e9), 2)
        # the second half recorded after a reboot: same monotonic range,
        # an hour later on the wall clock
        mixed = stamped(run[:8], 1.8e9) + stamped(run[8:], 1.8e9 + 3600)
        with self.assertRaises(ValueError) as cm:
            d2l.closed_loop_stats(mixed, 2)
        self.assertIn('more than one run', str(cm.exception))

    def test_missing_usage_gives_none(self):
        c = 2
        _, api = fake_api(d2l.FakeChatServer(0.02, 8))
        out = asyncio.run(d2l.chat_batch(api, numbered(
            8 * c, max_tokens=8, stream=True,
            stream_options={'include_usage': False}), concurrency=c))
        st = d2l.closed_loop_stats(out, c)
        self.assertGreater(st['X'], 0)
        self.assertIsNone(st['X_tokens'])


class ReviewClients(unittest.TestCase):
    """Finding 8: SDK clients per event loop; the server's log and group."""

    def test_one_client_per_open_loop(self):
        _, api = fake_api()

        async def notebook_cell(i):
            return await api.achat(HELLO, max_tokens=8, replica=i)

        loop = asyncio.new_event_loop()  # the notebook's loop
        try:
            for i in range(3):
                api.chat(HELLO, max_tokens=8, replica=10 + i)
                loop.run_until_complete(notebook_cell(i))
            self.assertEqual(len(api._sdks), 2)
        finally:
            loop.close()
        for i in range(3):  # scripts: asyncio.run makes a loop per call
            asyncio.run(notebook_cell(20 + i))
        self.assertLessEqual(len(api._sdks), 2)

    def test_listing_discount_not_applied_twice(self):
        """Novita lists 0.165 / 0.66 at discount 0.45: the price of the
        undiscounted rows (0.3 / 1.2) times 0.55, so already discounted."""
        self.assertEqual(NOVITA['pricing']['discount'], 0.45)
        price = d2l.price_from_listing(LISTING, FLAGSHIP, 'novita')
        self.assertAlmostEqual(price.input, 0.3 * (1 - 0.45), places=12)
        self.assertAlmostEqual(price.output, 1.2 * (1 - 0.45), places=12)


ORPHAN_STUB = textwrap.dedent('''\
    import subprocess, sys
    child = subprocess.Popen([sys.executable, '-c',
                              'import time; time.sleep(120)'])
    print('stub orphaning', child.pid, flush=True)
    sys.exit(3)
    ''')


class ReviewLocalServer(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.stub = os.path.join(self.dir.name, 'stub.py')
        with open(self.stub, 'w') as f:
            f.write(ORPHAN_STUB)

    def tearDown(self):
        self.dir.cleanup()

    def test_early_exit_kills_group_and_log(self):
        server = d2l.LocalChatServer([sys.executable, self.stub, '{port}'],
                                     ready_timeout=30)
        with self.assertRaises(RuntimeError) as cm:
            server.start()
        self.assertIn('stub orphaning', str(cm.exception))
        child = int(str(cm.exception).split('stub orphaning ')[1].split()[0])
        t0 = time.monotonic()
        while time.monotonic() - t0 < 10:
            try:
                os.kill(child, 0)
            except ProcessLookupError:
                break
            time.sleep(0.05)
        else:
            os.kill(child, signal.SIGKILL)
            self.fail('the server\'s child outlived a failed start')
        self.assertFalse(os.path.exists(server.log_path))

    def test_stop_deletes_log(self):
        with open(self.stub, 'w') as f:
            f.write(STUB)
        with d2l.LocalChatServer([sys.executable, self.stub, 'serve',
                                  '{port}'], ready_timeout=30) as server:
            self.assertTrue(os.path.exists(server.log_path))
        self.assertFalse(os.path.exists(server.log_path))


if __name__ == '__main__':
    unittest.main()
