# A Released Checkpoint in the Book's Own Code
:label:`sec_lm_d2l_checkpoint`

<!-- Author brief, copied verbatim from the Language Models part plan
(docs/llm/language_models_part_plan.md, "1.2 A Released Checkpoint in the Book's Own Code · T1").
Blueprint: docs/llm/blueprints/ch01_pretrained.md section 1.2.
The plan is a specification, not book text; this comment is replaced
when the section is written.

**Idea:** The same tensors run in a subclass of `d2l.GPT` and match the reference layer by layer.
**Opening:** Qwen3-0.6B's first block almost fits the constructor the reader already has: grouped-query attention, RMSNorm, SwiGLU, RoPE. The three differences (an explicit head dimension wider than the residual, QK-norm, a RoPE base of one million) are a small diff. The section writes the diff, maps the weights, and closes on what a layer-by-layer match exposes that a final-logit check hides.
**Teaches:**
- The three subclasses `d2l.Qwen3Attention(GQAAttention)` (explicit `head_dim`, `qk_norm`, `rope_theta`, a cached `forward_step`), `d2l.SwiGLU(FeedForward)` (explicit width) and `d2l.Qwen3GPT(GPT)` (assembled through the `attn_factory` and `ffn_factory` hooks of `sec_transformer-block`; untied head for 8B), built on the meta device and loaded with `assign=True`; the book's classes are unchanged (blueprint README Decision 4).
- The weight map as strict accounting: every Hugging Face tensor consumed, every target key produced; the RoPE layout as the silent breakage (Transformers stores the two halves, the book interleaves; the permutation also applies to `q_norm` and `k_norm`), fused versus separate projections, the duplicated `lm_head` in post-trained files.
- Localizing a mismatch: relative error per layer and sublayer against hooks on the reference, and the rule that a divergence is a layer whose error exceeds ten times the clean port's error at that layer.
- `d2l.generate` as the part's batched sampler over any model that returns logits, with `d2l.Generation` records and the cached path through `forward_cached`.
- Why the book reimplements Qwen3 and not Qwen3.5: the hybrid layers need `chap_modern_rnn`'s code, which 14.6 adds.
**Derivation:** none.
**Notebook:**
1. Build `d2l.Qwen3GPT` from the configuration through `d2l.qwen3_gpt_kwargs` and load Qwen3-0.6B with `d2l.load_qwen3`; assert the parameter count equals the shape-based count of 1.1.
2. Capture `attn_out`, `resid_mid`, `mlp_out`, `resid_post` per layer from the port and from Transformers (eager attention, TF32 off) on 32 committed prompts of 64 tokens; plot relative error per layer in fp32 and bf16 against the reference's own bf16-versus-fp32 floor.
3. Inject five porting bugs (missing RoPE permutation, wrong theta, swapped gate and up projections, missing QK-norm, wrong epsilon) and show where `d2l.first_divergence` localizes each; the epsilon bug is the one a final-logit check misses.
4. Greedy-decode 64 tokens on the 32 prompts with `d2l.generate` on the port and with Transformers; report the first index of divergence per prompt.
5. Claim that survives noise: fp32 logits agree within a stated absolute tolerance (provisional 1e-3, replaced by three times the reference's measured CPU-versus-GPU difference if larger); bf16 mean absolute logit error of the port is at most 1.5 times the reference's own bf16 error; greedy tokens are identical for 64 steps on all but near-tie prompts, counted. The computation is deterministic with TF32 off: run twice and assert bitwise-equal captures.
6. Reused pieces: `d2l.GQAAttention`, `d2l.FeedForward`, `d2l.TransformerBlock`, `d2l.GPT.generate` (unchanged), the saved `generate_cached`; the GPT-2 loading cell of `subsec_gpt-loading` recapped in one paragraph.
**Figures:** `fig_lm_weight_map` (Hugging Face names and layouts on the left, `d2l.Qwen3GPT` module names on the right, the RoPE permutation drawn on one head); `fig_lm_layer_divergence` (plot: relative error per layer for the clean port in fp32 and bf16 and for the five bugs).
**Restate / forward:** `subsec_gpt-loading` (GPT-2 weights were mapped by name and shape and checked by loss and greedy tokens; this is the second instance); `sec_transformer-block` (QK-norm normalizes queries and keys per head before RoPE); `sec_positional-information` (RoPE rotates pairs of coordinates; only the pairing convention differs here).
**Exercises:** 5 to 7. Load-bearing: derive the parameter count of Qwen3-8B from its safetensors index and explain which tensors the tied 0.6B file lacks (derive); measure the drift of a port with the wrong RoPE layout as a function of position (measure); check that `d2l.generate` on the port and the Transformers model agree under top-p sampling with a shared seed, and say why batch composition can break the agreement (check).
**Writer notes:**
- The exported names and module layout are relied on by Chapters 4, 6, 11, 12, 13 and 14 and must be exactly: `model.token_emb`; `model.blks[i].attention.{W_q, W_k, W_v, W_o, q_norm, k_norm}` with attributes `head_dim, num_heads, num_kv_heads, rope_theta`; `model.blks[i].ffn.{W_g, W_1, W_2}` computing `W_2(silu(W_g x) * W_1 x)`; `model.norm`; `model.lm_head` only when untied. There are no `W_gate/W_up/W_down`. `isinstance(model, d2l.GPT)` holds.
- The subclasses skip their parents' `__init__` to avoid allocating default-width layers (gigabytes at 8B) and therefore depend on the parents' attribute names (`act`, `pos`, `max_len`, `blks`, `token_emb`, `norm`); the GPT-to-`Qwen3GPT` equivalence test exists to catch a book-side rename.
- `d2l.generate(model, tokenizer, prompts, max_new_tokens=None, *, strategy='greedy', temperature=1.0, k=None, p=None, min_p=None, seed=0, eos_token_ids=None, stop=None, chat=False, enable_thinking=None, return_logprobs=False, batch_size=8) -> list[d2l.Generation]` accepts a `d2l.GPT` (cached path), a Transformers model (own loop with `past_key_values` and left padding) or any module with `model(input_ids) -> logits`; `d2l.GPT` batches only equal-length prompts (no attention mask), so it buckets by length. Qwen3's tokenizer adds no BOS; adding one shifts every logit. Post-trained Qwen3 stops on `<|im_end|>` and `<|endoftext|>`.
- The book marks `sec_kv-cache`'s `generate_cached` cell `#@save` (the blueprint README lists it as optional; this plan requires it), so `Qwen3GPT` overrides `forward` and `forward_cached` and inherits `generate` and `generate_cached`. If the edit has not landed when authoring starts, keep the blueprint's 12-line override and delete it afterwards.
- Reference and tolerance: Transformers with `attn_implementation='eager'` for the layer match (SDPA changes rounding); fp32 with TF32 off; hooks rather than `output_hidden_states` (its last entry may be post-norm in some versions); PyTorch's `nn.RMSNorm` epsilon default depends on dtype, so pass it explicitly.
- Memory: two fp32 copies of 0.6B are about 5 GB; 4B loads in bf16 (8 GB) for a logits check; 8B (16.4 GB) exceeds a 15 GiB slot, so the build exercises it by header and mapping dry run only, and an optional cell guarded by `torch.cuda.mem_get_info()` loads it where a larger card exists (the only real-weights test of `tie_embeddings=False`).
- Do not re-teach: name mapping and parameter reconstruction in general (`subsec_gpt-loading`); RoPE; the KV cache arithmetic.
- Committed artifacts: `qwen3_0.6b_logit_reference.pt` (fp32 reference logits on the 32 prompts, stored as top-100 per position plus full last-position logits, about 2 MB); the layer-error table as CSV; the three hidden corrupted state dicts for A1 live outside the notebook.
- `d2l.qwen3_gpt_kwargs` raises on `use_sliding_window`, `rope_scaling` or `attention_bias`; confirm at authoring that no Qwen3 dense checkpoint in the ladder sets them.
**Sources:**
- Papers: [Qwen3 Technical Report](https://arxiv.org/abs/2505.09388) (architecture paragraph: GQA, RoPE base, QK-norm replacing the QKV bias).
- Code and documentation: [Qwen3-0.6B model card](https://huggingface.co/Qwen/Qwen3-0.6B) (configuration, generation defaults, licence); [safetensors documentation](https://huggingface.co/docs/safetensors/index); [`modeling_qwen3.py`](https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen3/modeling_qwen3.py) (the reference being matched; the oracle for the RoPE layout); [PyTorch serialization notes](https://docs.pytorch.org/docs/stable/notes/serialization.html) (`load_state_dict(assign=True)` on meta-device modules).
- Courses, tutorials and blogs: Sebastian Raschka, [LLMs from Scratch](https://github.com/rasbt/LLMs-from-scratch) and his [from-scratch materials](https://www.sebastianraschka.com/from-scratch/) (the closest sibling: the same checkpoint and the same permutation question); [Hugging Face LLM Course, chapter 2](https://huggingface.co/learn/llm-course/chapter2/1) (the Auto-class machinery the port bypasses).
-->

```{.python .input #d2l-checkpoint-imports}
%%tab pytorch
import math
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from d2l import torch as d2l
```

```{.python .input #d2l-checkpoint-imports-extra}
%%tab pytorch
import collections
import json
import sys
```

<!-- Placeholder (stage 2): QK-norm, a wider head and the reference's RoPE angles. -->

```{.python .input #d2l-checkpoint-qwen3-attention}
%%tab pytorch
#@save
def qwen3_rope_tables(n, head_dim, theta, offset=0, device=None):
    """Cosines and sines, each (n, head_dim / 2) in fp32, of the RoPE angles
    of positions offset, ..., offset + n - 1, with the frequencies computed
    as Transformers computes them."""
    # 1 / theta^(2i/d) on the CPU, as Transformers builds its buffer; the
    # book's theta^(-2i/d) is equal in exact arithmetic but not in fp32
    inv = 1.0 / float(theta) ** (torch.arange(
        0, head_dim, 2, dtype=torch.float32, device='cpu') / head_dim)
    pos = torch.arange(n, dtype=torch.float32, device=device) + offset
    angles = pos[:, None] * inv.to(pos.device)[None, :]
    return torch.cos(angles), torch.sin(angles)

#@save
class Qwen3Attention(d2l.CausalAttention):
    """Causal attention with separate projections, an explicit head width,
    optional RMSNorm of queries and keys per head before RoPE (QK-norm),
    and a step that reads and writes a key-value cache."""
    def __init__(self, num_hiddens, num_heads, num_kv_heads, *, head_dim=None,
                 qk_norm=False, rope_theta=10000.0, norm_eps=1e-6, bias=False,
                 rope=True, causal=True):
        nn.Module.__init__(self)  # CausalAttention's would allocate W_qkv
        if not causal:
            raise ValueError('Qwen3Attention is causal only')
        self.num_heads, self.num_kv_heads = num_heads, num_kv_heads
        self.head_dim = head_dim or num_hiddens // num_heads
        self.qk_norm, self.rope, self.rope_theta = qk_norm, rope, rope_theta
        self.kv_hiddens = num_kv_heads * self.head_dim
        q_hiddens = num_heads * self.head_dim  # may differ from num_hiddens
        self.W_q = nn.Linear(num_hiddens, q_hiddens, bias=bias)
        self.W_k = nn.Linear(num_hiddens, self.kv_hiddens, bias=bias)
        self.W_v = nn.Linear(num_hiddens, self.kv_hiddens, bias=bias)
        self.W_o = nn.Linear(q_hiddens, num_hiddens, bias=bias)
        if qk_norm:  # one gain per coordinate of a head, shared by the heads
            self.q_norm = nn.RMSNorm(self.head_dim, eps=norm_eps)
            self.k_norm = nn.RMSNorm(self.head_dim, eps=norm_eps)

    def qkv(self, X, offset=0, tables=None):
        B, T, _ = X.shape
        q = self.W_q(X).reshape(B, T, self.num_heads, self.head_dim)
        k = self.W_k(X).reshape(B, T, self.num_kv_heads, self.head_dim)
        v = self.W_v(X).reshape(B, T, self.num_kv_heads, self.head_dim)
        if self.qk_norm:  # normalize each head's query and key
            q, k = self.q_norm(q), self.k_norm(k)
        q, k, v = (u.transpose(1, 2) for u in (q, k, v))  # (B, heads, T, d_h)
        if self.rope:
            if tables is None:
                tables = qwen3_rope_tables(T, self.head_dim, self.rope_theta,
                                           offset, X.device)
            q, k = (d2l.rope(u, tables=tables) for u in (q, k))
        return q, k, v

    def forward_step(self, X, cache, tables=None):
        """Attention output for X at positions t, ..., t + T - 1, where cache
        = (ks, vs, t) holds this block's buffers, each (batch, kv heads,
        max_len, head_dim); writes the keys and values of X into them."""
        ks, vs, t = cache
        T = X.shape[1]
        q, k, v = self.qkv(X, t, tables)
        ks[:, :, t:t + T], vs[:, :, t:t + T] = k, v
        Y = F.scaled_dot_product_attention(
            q, ks[:, :, :t + T], vs[:, :, :t + T],
            attn_mask=d2l.causal_mask(T, t + T, X.device),
            enable_gqa=self.num_kv_heads < self.num_heads)
        return self.W_o(Y.transpose(1, 2).flatten(2))
```

<!-- Placeholder (stage 2): the feed-forward width and the assembled model. -->

```{.python .input #d2l-checkpoint-swiglu-gpt}
%%tab pytorch
#@save
class SwiGLU(d2l.FeedForward):
    """SwiGLU feed-forward network with an explicit hidden width."""
    def __init__(self, num_hiddens, hidden, bias=False):
        nn.Module.__init__(self)  # FeedForward's would allocate width 8d/3
        self.act = 'swiglu'
        self.W_g = nn.Linear(num_hiddens, hidden, bias=bias)
        self.W_1 = nn.Linear(num_hiddens, hidden, bias=bias)
        self.W_2 = nn.Linear(hidden, num_hiddens, bias=bias)

#@save
class Qwen3GPT(d2l.GPT):
    """d2l.GPT with Qwen3's head width, QK-norm, RoPE base, feed-forward
    width and normalization epsilon, and an optional untied output head."""
    def __init__(self, vocab_size, num_hiddens, num_heads, num_kv_heads,
                 num_blks, *, head_dim=None, qk_norm=True, rope_theta=1e6,
                 ffn_hidden=None, norm_eps=1e-6, tie_embeddings=True,
                 max_len=40960, bias=False):
        head_dim = head_dim or num_hiddens // num_heads
        ffn_hidden = ffn_hidden or round(8 * num_hiddens / 3)
        # read by CausalAttention below while GPT builds the blocks; a plain
        # attribute may be set before nn.Module.__init__ runs
        self.attn_options = dict(head_dim=head_dim, qk_norm=qk_norm,
                                 rope_theta=rope_theta, norm_eps=norm_eps)
        super().__init__(
            vocab_size, num_hiddens=num_hiddens, num_heads=num_heads,
            num_blks=num_blks, max_len=max_len, bias=bias,
            num_kv_heads=num_kv_heads,
            ffn_factory=lambda: SwiGLU(num_hiddens, ffn_hidden, bias))
        for blk in self.blks:
            blk.norm1.eps = blk.norm2.eps = norm_eps
        self.norm.eps = norm_eps
        self.head_width, self.rope_theta = head_dim, rope_theta
        if not tie_embeddings:
            self.lm_head = nn.Linear(num_hiddens, vocab_size, bias=False)
            nn.init.normal_(self.lm_head.weight, std=0.02)

    def CausalAttention(self, d, H, *, bias, rope, num_kv_heads):
        """The attention GPT builds in each block (a method, not a lambda,
        so that the model can be pickled)."""
        return Qwen3Attention(d, H, num_kv_heads, bias=bias, rope=rope,
                              **self.attn_options)

    def forward(self, X, logits_to_keep=None, tables=None):
        if tables is None:  # RoPE angles of positions 0, ..., T - 1
            tables = qwen3_rope_tables(X.shape[1], self.head_width,
                                       self.rope_theta, device=X.device)
        H = self.token_emb(X)
        for blk in self.blks:
            H = blk(H, tables=tables)
        if logits_to_keep is not None:  # the head only where logits are read
            H = H[:, -logits_to_keep:]
        H = self.norm(H)
        if hasattr(self, 'lm_head'):
            return self.lm_head(H)
        return F.linear(H, self.token_emb.weight)

    def init_cache(self, batch_size, max_len):
        """Zeroed keys and values for every block, each (blocks, batch, kv
        heads, max_len, head_dim)."""
        attn = self.blks[0].attention
        shape = (len(self.blks), batch_size, attn.num_kv_heads, max_len,
                 attn.head_dim)
        p = self.token_emb.weight
        return (torch.zeros(shape, device=p.device, dtype=p.dtype),
                torch.zeros(shape, device=p.device, dtype=p.dtype))

    @torch.inference_mode()
    def forward_cached(self, X, ks, vs, t):
        """Logits after the last of the new tokens X, which sit at positions
        t, t + 1, ...; writes their keys and values into the cache."""
        tables = qwen3_rope_tables(X.shape[1], self.head_width,
                                   self.rope_theta, t, X.device)
        H = self.token_emb(X)
        for i, blk in enumerate(self.blks):  # the modules of blk.forward
            H = H + blk.attention.forward_step(blk.norm1(H), (ks[i], vs[i], t),
                                               tables)
            H = H + blk.ffn(blk.norm2(H))
        H = self.norm(H[:, -1])
        if hasattr(self, 'lm_head'):
            return self.lm_head(H), ks, vs
        return F.linear(H, self.token_emb.weight), ks, vs

    @torch.inference_mode()
    def generate_cached(self, prefix, num_tokens, temperature=1.0, top_k=None):
        """Sample as generate does, but never recompute the prefix."""
        assert len(prefix) + num_tokens <= self.max_len, 'cache overflow'
        X = torch.tensor(prefix, device=self.token_emb.weight.device)[None]
        ks, vs = self.init_cache(1, len(prefix) + num_tokens)
        logits, ks, vs = self.forward_cached(X, ks, vs, 0)
        tokens = []
        for t in range(len(prefix), len(prefix) + num_tokens):
            tokens.append(d2l.sample_logits(logits, 'sample', temperature,
                                            k=top_k))
            if t + 1 < len(prefix) + num_tokens:  # logits for the next draw
                logits, ks, vs = self.forward_cached(tokens[-1][:, None], ks,
                                                     vs, t)
        return list(prefix) + torch.cat(tokens).tolist()

#@save
def qwen3_gpt_kwargs(cfg):
    """Qwen3GPT's arguments from a Qwen3 configuration (a dict or a
    Transformers config); raises on settings the port does not implement.
    A dict without head_dim, tie_word_embeddings or a RoPE base gets
    Transformers' Qwen3Config defaults (128, untied, 10,000)."""
    c = cfg if isinstance(cfg, dict) else cfg.to_dict()
    kind = c.get('model_type')
    if kind == 'qwen3_moe':
        raise NotImplementedError('qwen3_moe: expert layers are not ported')
    if kind and kind.startswith('qwen3_5'):
        raise NotImplementedError(
            f'{kind}: its Gated DeltaNet layers are ported in '
            'sec_lm_hybrids_at_scale')
    if kind != 'qwen3':
        raise ValueError(f'not a Qwen3 configuration: {kind!r}')
    rope = c.get('rope_parameters') or {}  # where Transformers 5 keeps theta
    for name, value in [('use_sliding_window', c.get('use_sliding_window')),
                        ('rope_scaling', c.get('rope_scaling')),
                        ('rope_type', rope.get('rope_type', 'default')
                         != 'default'),
                        ('attention_bias', c.get('attention_bias'))]:
        if value:
            raise ValueError(f'{name} is not implemented in Qwen3GPT')
    return dict(
        vocab_size=c['vocab_size'], num_hiddens=c['hidden_size'],
        num_heads=c['num_attention_heads'],
        num_kv_heads=c['num_key_value_heads'],
        num_blks=c['num_hidden_layers'], head_dim=c.get('head_dim', 128),
        qk_norm=True, rope_theta=float(c.get('rope_theta') or rope.get(
            'rope_theta', 1e4)),
        ffn_hidden=c['intermediate_size'], norm_eps=c['rms_norm_eps'],
        tie_embeddings=c.get('tie_word_embeddings', False),
        max_len=c['max_position_embeddings'], bias=False)
```

<!-- Placeholder (stage 2): the RoPE pairing convention as a permutation. -->

```{.python .input #d2l-checkpoint-rope-permutation}
%%tab pytorch
#@save
def qwen3_rope_permutation(head_dim):
    """Index idx with idx[2i] = i and idx[2i + 1] = i + head_dim / 2: taking
    row idx[j] of a head's rows as row j turns Transformers' split-half
    RoPE pairs into the book's interleaved pairs."""
    half = torch.arange(head_dim // 2)
    return torch.stack([half, half + head_dim // 2], 1).flatten()
```

<!-- Placeholder (stage 2): the weight map as strict accounting, and loading. -->

```{.python .input #d2l-checkpoint-map-load}
%%tab pytorch
#@save
def map_qwen3_state_dict(hf_state, cfg, rope_layout='permute'):
    """A Qwen3GPT state dict from a Transformers Qwen3 state dict. Renames
    every tensor and, per head, permutes the rows of q_proj and k_proj and
    the gains of q_norm and k_norm into the book's RoPE pairing
    (rope_layout='none' skips this). A KeyError lists every tensor not
    consumed and every key not produced."""
    assert rope_layout in ('permute', 'none'), rope_layout
    kw = qwen3_gpt_kwargs(cfg)
    d_h = kw['head_dim'] or kw['num_hiddens'] // kw['num_heads']
    idx = qwen3_rope_permutation(d_h)

    def permute(W):  # W: (heads * d_h, ...) rows, or one head's (d_h,) gains
        if rope_layout == 'none':
            return W
        return W.unflatten(0, (-1, d_h))[:, idx.to(W.device)].flatten(0, 1)

    keep = lambda W: W
    layer = {'input_layernorm': ('norm1', keep),
             'self_attn.q_proj': ('attention.W_q', permute),
             'self_attn.k_proj': ('attention.W_k', permute),
             'self_attn.v_proj': ('attention.W_v', keep),
             'self_attn.o_proj': ('attention.W_o', keep),
             'self_attn.q_norm': ('attention.q_norm', permute),
             'self_attn.k_norm': ('attention.k_norm', permute),
             'post_attention_layernorm': ('norm2', keep),
             'mlp.gate_proj': ('ffn.W_g', keep),
             'mlp.up_proj': ('ffn.W_1', keep),
             'mlp.down_proj': ('ffn.W_2', keep)}
    names = {'model.embed_tokens': ('token_emb', keep)}
    for i in range(kw['num_blks']):
        names.update({f'model.layers.{i}.{hf}': (f'blks.{i}.{ours}', f)
                      for hf, (ours, f) in layer.items()})
    names['model.norm'] = ('norm', keep)
    if not kw['tie_embeddings']:
        names['lm_head'] = ('lm_head', keep)
    left, state, missing = dict(hf_state), {}, []
    head = left.get('lm_head.weight')
    emb = left.get('model.embed_tokens.weight')
    if kw['tie_embeddings'] and head is not None:  # a stored copy of emb
        if not (head.is_meta or emb is None or torch.equal(head, emb)):
            raise ValueError('lm_head.weight differs from the tied embedding')
        del left['lm_head.weight']
    for hf, (ours, f) in names.items():
        if hf + '.weight' in left:
            state[ours + '.weight'] = f(left.pop(hf + '.weight'))
        else:
            missing.append(f'{ours}.weight (from {hf}.weight)')
    if left or missing:
        raise KeyError(f'not consumed: {sorted(left)}; '
                       f'not produced: {missing}')
    return state

#@save
def _qwen3_files(model_id, revision):
    """Local paths of config.json, generation_config.json (None if the
    repository has none) and the safetensors files of a Hub checkpoint at a
    revision, downloaded into the Hugging Face cache."""
    import huggingface_hub
    # the Hub answered that the file is absent; LocalEntryNotFoundError
    # (offline, not cached) is a sibling subclass and must propagate
    from huggingface_hub.errors import RemoteEntryNotFoundError

    def get(filename, optional=False):
        try:
            return huggingface_hub.hf_hub_download(model_id, filename,
                                                   revision=revision)
        except RemoteEntryNotFoundError:
            if optional:
                return None
            raise

    index = get('model.safetensors.index.json', optional=True)
    if index is None:
        weights = [get('model.safetensors')]
    else:  # a sharded checkpoint lists its shards in the index
        with open(index) as fh:
            shards = sorted(set(json.load(fh)['weight_map'].values()))
        weights = [get(shard) for shard in shards]
    return (get('config.json'), get('generation_config.json', optional=True),
            weights)

#@save
def load_qwen3(model_id='Qwen/Qwen3-0.6B', revision=None, *,
               dtype=torch.bfloat16, device=None):
    """A pinned Qwen3 checkpoint from the Hugging Face Hub as a Qwen3GPT in
    eval mode: built on the meta device, then its mapped weights assigned.
    The tokenizer is loaded separately, at model.revision."""
    from safetensors.torch import load_file  # lazy, as in load_gpt2
    revision = revision or d2l.HF_REVISIONS[model_id]
    config_file, generation_file, weight_files = _qwen3_files(model_id,
                                                              revision)
    with open(config_file) as fh:
        cfg = json.load(fh)
    with torch.device('meta'):  # shapes only: no memory is allocated
        model = Qwen3GPT(**qwen3_gpt_kwargs(cfg))
    hf_state = {}
    for path in weight_files:
        hf_state.update(load_file(path))
    device = device or d2l.try_gpu()
    state = {name: W.to(device, dtype)
             for name, W in map_qwen3_state_dict(hf_state, cfg).items()}
    # assign=True: meta tensors cannot be copied into, so take the tensors
    model.load_state_dict(state, strict=True, assign=True)
    model.hf_config, model.revision = cfg, revision
    model.generation_config = None
    if generation_file is not None:
        import transformers
        with open(generation_file) as fh:
            model.generation_config = transformers.GenerationConfig.from_dict(
                json.load(fh))
    return model.eval()
```

<!-- Placeholder (stage 2): the batched sampler of the part. -->

```{.python .input #d2l-checkpoint-sampler}
%%tab pytorch
#@save
Generation = collections.namedtuple(  # finish_reason: 'eos', 'stop', 'length'
    'Generation', 'text token_ids logprobs finish_reason')

#@save
def truncated_probs(logits, temperature=1.0, k=None, p=None, min_p=None):
    """Rows of logits (batch, vocab) -> the temperature-scaled distribution
    truncated by top-k, top-p and min-p and renormalized; ties in rank go
    to the smaller token id. As in Transformers and vLLM (and unlike
    sec_decoding's sample_next), top-p measures mass within the top k."""
    probs = F.softmax(logits.float() / temperature, -1)
    ranked, order = torch.sort(probs, dim=-1, descending=True, stable=True)
    keep = torch.full_like(order[:, :1], probs.shape[-1])  # tokens kept
    if k is not None:  # top-k: a fixed count
        keep = keep.clamp(max=k)
    if p is not None:  # top-p: the smallest head of mass at least p of
        cum = ranked.double().cumsum(-1)  # the top-k mass (1 without k)
        head = (cum < p * cum.gather(-1, keep - 1)).sum(-1, keepdim=True) + 1
        keep = torch.minimum(keep, head)
    if min_p is not None:  # min-p: a fraction of the top probability
        keep = torch.minimum(keep, (ranked >= min_p * ranked[:, :1]).sum(
            -1, keepdim=True))
    ranked = ranked * (torch.arange(probs.shape[-1], device=probs.device)
                       < keep)
    probs = torch.zeros_like(probs).scatter(-1, order, ranked)
    return probs / probs.sum(-1, keepdim=True)

#@save
def sample_logits(logits, strategy='greedy', temperature=1.0, k=None,
                  p=None, min_p=None, generator=None):
    """One token id per row of logits (batch, vocab): the argmax (ties to
    the smaller id), or a draw from truncated_probs by inverting its
    distribution function at a uniform from generator (a torch.Generator,
    or one per row)."""
    if strategy == 'greedy':
        return logits.argmax(-1)
    if strategy != 'sample':
        raise ValueError(f'unknown strategy: {strategy!r}')
    cdf = truncated_probs(logits, temperature, k, p, min_p).double().cumsum(-1)
    if isinstance(generator, (list, tuple)):  # one uniform per row
        u = torch.cat([torch.rand(1, generator=g, dtype=torch.float64)
                       for g in generator])
    else:
        u = torch.rand(len(cdf), generator=generator, dtype=torch.float64)
    u = u.to(cdf.device)[:, None] * cdf[:, -1:]  # in [0, total mass)
    return torch.searchsorted(cdf, u, right=True)[:, 0]
```

<!-- Placeholder (stage 2): two paths through a model: a cache, or recomputation. -->

```{.python .input #d2l-checkpoint-generate-paths}
%%tab pytorch
#@save
def _lm_cached_steps(model, ids, num_new):
    """Prefill equal-length prompts ids (lists of token ids) into the
    model's cache; returns the last logits and step(next_ids) -> logits."""
    X = torch.tensor(ids, device=model.token_emb.weight.device)
    ks, vs = model.init_cache(len(ids), X.shape[1] + num_new)
    t = X.shape[1]
    logits = model.forward_cached(X, ks, vs, 0)[0]

    def step(next_ids):  # next_ids: (batch,), at position t
        nonlocal t
        logits = model.forward_cached(next_ids[:, None], ks, vs, t)[0]
        t += 1
        return logits
    return logits, step

#@save
def _lm_recompute_steps(model, ids):
    """For any model mapping token ids to logits: each step runs the whole
    sequence again."""
    X = torch.tensor(ids, device=next(model.parameters()).device)

    def step(next_ids):
        nonlocal X
        X = torch.cat([X, next_ids[:, None]], 1)
        return model(X)[:, -1]
    return model(X)[:, -1], step
```

<!-- Placeholder (stage 2): Transformers' own cache, with left padding. -->

```{.python .input #d2l-checkpoint-generate-hf}
%%tab pytorch
#@save
def _lm_hf_steps(model, ids, pad_id):
    """Prompts of any lengths through a Transformers model and its cache:
    padded on the left, with positions counted from each prompt's first
    token, and logits computed at the last position only."""
    T = max(map(len, ids))
    X = torch.tensor([[pad_id] * (T - len(r)) + r for r in ids],
                     device=model.device)
    mask = torch.tensor([[0] * (T - len(r)) + [1] * len(r) for r in ids],
                        device=model.device)
    cache = None

    def run(X):
        nonlocal cache
        pos = (mask.cumsum(-1) - 1).masked_fill(mask == 0, 1)[:, -X.shape[1]:]
        out = model(input_ids=X, attention_mask=mask, position_ids=pos,
                    past_key_values=cache, use_cache=True, logits_to_keep=1)
        cache = out.past_key_values
        return out.logits[:, -1]

    def step(next_ids):
        nonlocal mask
        mask = torch.cat([mask, mask.new_ones(len(ids), 1)], 1)
        return run(next_ids[:, None])
    return run(X), step
```

<!-- Placeholder (stage 2): one sampler for the port and for Transformers. -->

```{.python .input #d2l-checkpoint-generate}
%%tab pytorch
#@save
def _lm_finish(tokens, logprobs, tokenizer, eos_ids, stop):
    """One row's Generation: a final end token is dropped, and the text is
    cut before the first stop string."""
    reason = 'length'
    if tokens and tokens[-1] in eos_ids:
        tokens, reason = tokens[:-1], 'eos'
    text = tokenizer.decode(tokens)
    cuts = [text.find(s) for s in stop or () if s in text]
    if cuts:
        text, reason = text[:min(cuts)], 'stop'
    if logprobs is not None:
        logprobs = logprobs[:len(tokens)]
    return Generation(text, tokens, logprobs, reason)

#@save
def _lm_unwrap(model):
    """The model inside a torch.compile or PEFT wrapper (which pass every
    keyword argument through to it), or model itself."""
    while True:
        inner = getattr(model, '_orig_mod', None)
        if inner is None and hasattr(model, 'get_base_model'):
            inner = model.get_base_model()
        if inner is None or inner is model:
            return model
        model = inner

#@save
def _lm_bos_ids(tokenizer):
    """The special tokens the tokenizer puts before a text (a BOS, or
    none). Those it appends after a text, such as an EOS, are left out:
    a prompt continues where its text ends."""
    with_sp, plain = (tokenizer(['a'], add_special_tokens=s)['input_ids'][0]
                      for s in (True, False))
    return next((with_sp[:k] for k in range(len(with_sp) - len(plain) + 1)
                 if with_sp[k:k + len(plain)] == plain), [])

#@save
@torch.inference_mode()
def generate(model, tokenizer, prompts, max_new_tokens=None, *,
             strategy='greedy', temperature=1.0, k=None, p=None, min_p=None,
             seed=0, eos_token_ids=None, stop=None, chat=False,
             enable_thinking=None, return_logprobs=False, batch_size=8):
    """Continue each prompt (a string, or with chat=True also a list of
    messages) and return one Generation per prompt, in input order.
    strategy is 'greedy' or 'sample' (truncated_probs's temperature, k, p,
    min_p). stop and eos_token_ids take one item or a list. seed is an
    int or a torch.Generator, either one stream for the call, or a list
    of one int per prompt: an item's random draws then depend only on its
    seed; the
    sampled text is batch-invariant only up to bf16 kernel drift, because
    the logits depend on batch and padding composition
    (:numref:`subsec_lm_nondeterminism`)."""
    config = getattr(model, 'generation_config', None)
    if max_new_tokens is None:
        max_new_tokens = getattr(config, 'max_new_tokens', None) or 256
    if eos_token_ids is None:  # the tokenizer's end token and the model's
        eos = getattr(config, 'eos_token_id', None)
        eos_token_ids = [tokenizer.eos_token_id,
                         *(eos if isinstance(eos, list) else [eos])]
    if isinstance(eos_token_ids, (int, np.integer)):
        eos_token_ids = [eos_token_ids]
    eos_ids = {int(i) for i in eos_token_ids if i is not None}
    if isinstance(stop, str):  # one string, not its characters
        stop = [stop]
    if chat:  # a string is one user message; the template adds no BOS
        opts = {} if enable_thinking is None else {
            'enable_thinking': enable_thinking}
        prompts = [tokenizer.apply_chat_template(
            [{'role': 'user', 'content': x}] if isinstance(x, str) else x,
            tokenize=False, add_generation_prompt=True, **opts)
                   for x in prompts]
    head = [] if chat else _lm_bos_ids(tokenizer)  # a BOS, never an EOS
    ids = [head + tokenizer(x, add_special_tokens=False)['input_ids']
           for x in prompts]
    if not all(ids):  # Qwen3's tokenizer adds no BOS to predict from
        raise ValueError('an empty prompt has no position to predict from')
    hf = sys.modules.get('transformers')  # loaded if model is one of its
    if hf is not None and isinstance(_lm_unwrap(model), hf.PreTrainedModel):
        pad = getattr(tokenizer, 'pad_token_id', None) or 0
        start = lambda rows: _lm_hf_steps(model, rows, pad)
        groups = [list(range(len(ids)))]  # left padding: input order
    else:  # no attention mask: batch only prompts of equal length
        if hasattr(model, 'forward_cached'):
            start = lambda rows: _lm_cached_steps(model, rows, max_new_tokens)
        else:
            start = lambda rows: _lm_recompute_steps(model, rows)
        groups = [[i for i in range(len(ids)) if len(ids[i]) == n]
                  for n in sorted({len(r) for r in ids})]
    shared = isinstance(seed, (int, np.integer, torch.Generator))
    if isinstance(seed, torch.Generator):
        generator = seed
    elif shared:
        generator = torch.Generator().manual_seed(int(seed))
    else:
        assert len(seed) == len(ids), 'one seed per prompt'
        generators = [torch.Generator().manual_seed(int(s)) for s in seed]
    out = [None] * len(ids)
    for rows in (g[j:j + batch_size] for g in groups
                 for j in range(0, len(g), batch_size)):
        logits, step = start([ids[i] for i in rows])
        gen = generator if shared else [generators[i] for i in rows]
        tokens = [[] for _ in rows]
        logprobs = [[] for _ in rows] if return_logprobs else None
        active = set(range(len(rows)))
        for n in range(max_new_tokens):
            nxt = sample_logits(logits, strategy, temperature, k, p, min_p,
                                gen)
            if return_logprobs:  # under the untruncated distribution
                lp = F.log_softmax(logits.float(), -1).gather(
                    -1, nxt[:, None])[:, 0].tolist()
            for j, token in enumerate(nxt.tolist()):
                if j in active:
                    tokens[j].append(token)
                    if return_logprobs:
                        logprobs[j].append(lp[j])
                    text = tokenizer.decode(tokens[j]) if stop else ''
                    if token in eos_ids or any(s in text for s in stop or ()):
                        active.discard(j)
            if not active or n == max_new_tokens - 1:
                break
            logits = step(nxt)
        for j, i in enumerate(rows):
            out[i] = _lm_finish(tokens[j], logprobs and logprobs[j],
                                tokenizer, eos_ids, stop)
    return out
```

<!-- Placeholder (stage 2): four capture points per layer, and where a port diverges. -->

```{.python .input #d2l-checkpoint-hooks}
%%tab pytorch
#@save
_HIDDEN_POINTS = ('attn_out', 'resid_mid', 'mlp_out', 'resid_post')

#@save
def collect_hiddens(model, input_ids):
    """Per layer, in execution order: the attention output, the residual
    stream after it, the FFN output, and the stream after the block; each
    (layers, batch, T, d) on the CPU, fp64 for an fp64 model, else fp32.
    model is a d2l.GPT or a Transformers Qwen3 model."""
    if isinstance(model, d2l.GPT):
        layers = [(b.attention, b.norm2, b.ffn, b) for b in model.blks]
    else:  # Transformers: attention, second norm and MLP of each layer
        layers = [(b.self_attn, b.post_attention_layernorm, b.mlp, b)
                  for b in model.model.layers]
    dtype = (torch.float64 if next(model.parameters()).dtype == torch.float64
             else torch.float32)
    caps = {name: [] for name in _HIDDEN_POINTS}

    def save(name, x):  # Transformers modules may return tuples
        x = x[0] if isinstance(x, tuple) else x
        caps[name].append(x.detach().to('cpu', dtype))

    handles = []
    try:
        for attn, norm2, ffn, blk in layers:
            handles += [
                attn.register_forward_hook(
                    lambda m, a, y: save('attn_out', y)),
                norm2.register_forward_pre_hook(  # the stream entering norm2
                    lambda m, a: save('resid_mid', a[0])),
                ffn.register_forward_hook(lambda m, a, y: save('mlp_out', y)),
                blk.register_forward_hook(
                    lambda m, a, y: save('resid_post', y))]
        with torch.inference_mode():
            model(input_ids)
    finally:
        for handle in handles:
            handle.remove()
    return {name: torch.stack(caps[name]) for name in _HIDDEN_POINTS}

#@save
def layer_errors(ref, ours, skip_first=1):
    """Relative Frobenius error of each capture of collect_hiddens against a
    reference over positions skip_first, ..., T - 1, per layer, capture
    point and prompt: an array (layers, 4, batch)."""
    errors = []
    for name in _HIDDEN_POINTS:
        r, o = (u[:, :, skip_first:].double().flatten(2)
                for u in (ref[name], ours[name]))
        errors.append(((o - r).norm(dim=-1) / r.norm(dim=-1)).numpy())
    return np.stack(errors, 1)

#@save
def first_divergence(errors, floor, factor=10.0):
    """The first (layer, capture point) in execution order whose median
    error over prompts exceeds factor times the median floor there (the
    clean port's error); None if there is none."""
    above = np.median(errors, -1) > factor * np.median(floor, -1)
    if not above.any():
        return None
    layer, point = np.argwhere(above)[0]  # row-major: layer by layer
    return int(layer), _HIDDEN_POINTS[point]
```
