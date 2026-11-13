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
import gc
import huggingface_hub
import itertools
import json
import os
import sys
import transformers
from safetensors.torch import load_file
```

<!-- Placeholder (stage 2): the fixed prompts of the comparison (made by tools/lm/make_port_fixtures.py) and the fp64 logit reference of assignment A1 (registered here, never read by this notebook). -->

```{.python .input #d2l-checkpoint-data-hub}
%%tab pytorch
#@save
d2l.DATA_HUB['lm-qwen3-port-prompts-v1'] = (
    d2l.LM_DATA_URL + 'lm-qwen3-port-prompts-v1.json',
    'b5840fad4ad3d8d1897c12d522def18f6eb8fa06')
d2l.DATA_HUB['lm-qwen3-0p6b-logit-ref-v1'] = (
    d2l.LM_DATA_URL + 'lm-qwen3-0p6b-logit-ref-v1.pt',
    '2fa5449638a8a8924bc6b35384dc996c23513cb6')
```

<!-- Placeholder (stage 2): the three checkpoints, the fp32 setting the comparison assumes, and the fixed prompts (8 categories of 4, the 28 that are text, one long prompt). -->

```{.python .input #d2l-checkpoint-setup}
%%tab pytorch
QWEN, QWEN_17B_BASE, QWEN_8B = (
    'Qwen/Qwen3-0.6B', 'Qwen/Qwen3-1.7B-Base', 'Qwen/Qwen3-8B')
device = d2l.try_gpu()
SMALL = device.type == 'cpu'  # no GPU: 3 prompts of 32 tokens
# A job given a GPU (CUDA_VISIBLE_DEVICES set) must not fall back to SMALL
assert not (SMALL and os.environ.get('CUDA_VISIBLE_DEVICES')), (
    'a GPU is assigned, but PyTorch sees none')
# The tolerance rule below assumes IEEE fp32 matrix products (no TF32),
# which is PyTorch's default
assert torch.get_float32_matmul_precision() == 'highest'
transformers.logging.set_verbosity_error()  # no notices or progress bars
transformers.logging.disable_progress_bar()
huggingface_hub.utils.disable_progress_bars()
huggingface_hub.utils.logging.set_verbosity_error()
with open(d2l.download('lm-qwen3-port-prompts-v1',
                       os.path.join('..', 'data', 'lm'))) as f:
    prompts = json.load(f)
ids = torch.tensor(prompts['ids'])  # (32, 64): 8 categories of 4 prompts
texts = prompts['texts']  # the 28 prompts that are text (not random ids)
long_ids = torch.tensor(prompts['long']['ids'])
if SMALL:  # the first prompt of three categories, cut to 32 tokens
    rows = [0, 4, 8]
    texts = [texts[prompts['text_rows'].index(r)] for r in rows]
    ids, long_ids = ids[rows, :32], None
NEW_TOKENS = 16 if SMALL else 64  # generated per prompt
print(f'{len(ids)} prompts of {ids.shape[1]} tokens, {len(texts)} of them '
      f'as text; long prompt: '
      f'{"none" if long_ids is None else f"{len(long_ids):,} tokens"}')
```

<!-- Placeholder (stage 2): the book's GPT at Qwen3-0.6B's widths against the checkpoint's header, layer 0, role by role. -->

```{.python .input #d2l-checkpoint-recipe-diff}
%%tab pytorch
with torch.device('meta'):  # shapes only: no memory is allocated
    book = d2l.GPT(151936, num_hiddens=1024, num_heads=16, num_blks=28,
                   num_kv_heads=8)
book_shapes = {k: tuple(v.shape) for k, v in book.state_dict().items()}
header = d2l.safetensors_header(QWEN)  # name -> (shape, dtype)
roles = [('query, key, value', ['attention.W_qkv'],
          ['self_attn.q_proj', 'self_attn.k_proj', 'self_attn.v_proj']),
         ('attention output', ['attention.W_o'], ['self_attn.o_proj']),
         ('query, key norms', [], ['self_attn.q_norm', 'self_attn.k_norm']),
         ('FFN gate', ['ffn.W_g'], ['mlp.gate_proj']),
         ('FFN up', ['ffn.W_1'], ['mlp.up_proj']),
         ('FFN down', ['ffn.W_2'], ['mlp.down_proj']),
         ('norms', ['norm1', 'norm2'],
          ['input_layernorm', 'post_attention_layernorm'])]
print(f'{"role":19}{"d2l.GPT":21}Qwen3-0.6B (layer 0)')
for role, book_names, qwen_names in roles:
    left = [f'{n.split(".")[-1]} {book_shapes[f"blks.0.{n}.weight"]}'
            for n in book_names] or ['(no counterpart)']
    right = [f'{n.split(".")[-1]} {header[f"model.layers.0.{n}.weight"][0]}'
             for n in qwen_names]
    for i, (a, b) in enumerate(itertools.zip_longest(left, right,
                                                     fillvalue='')):
        print(f'{role if i == 0 else "":19}{a:21}{b}')
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

<!-- Placeholder (stage 2): with every Qwen3 feature off, the subclass is the book's GPT: same logits, same cached path, same greedy tokens. -->

```{.python .input #d2l-checkpoint-generalization}
%%tab pytorch
torch.manual_seed(0)
gpt = d2l.GPT(100, num_hiddens=64, num_heads=4, num_blks=2).eval()
port = Qwen3GPT(100, 64, 4, 4, 2, qk_norm=False, rope_theta=1e4,
                norm_eps=1e-5, ffn_hidden=round(8 * 64 / 3)).eval()
state = {}
for name, W in gpt.state_dict().items():  # the fused W_qkv, split by rows
    if 'W_qkv' in name:
        for part, rows in zip(('W_q', 'W_k', 'W_v'), W.split(64)):
            state[name.replace('W_qkv', part)] = rows
    else:
        state[name] = W
port.load_state_dict(state, strict=True)
X = torch.randint(0, 100, (4, 32))
with torch.no_grad():
    book_tables = d2l.rope_tables(32, port.head_width)  # GPT's angles
    full = (port(X, tables=book_tables) - gpt(X)).abs().max().item()
    ks, vs = port.init_cache(4, 32)  # 16 tokens at once, then one by one
    steps = [port.forward_cached(X[:, :16], ks, vs, 0)[0]] + [
        port.forward_cached(X[:, t:t + 1], ks, vs, t)[0]
        for t in range(16, 32)]
    cached = (torch.stack(steps, 1) - port(X)[:, 15:]).abs().max().item()
prefix = X[0, :8].tolist()
same = (port.generate_cached(prefix, 16, top_k=1)
        == gpt.generate(prefix, 16, top_k=1))
print(f'max |logit difference| from d2l.GPT: {full:.1e}; forward_cached '
      f'against forward: {cached:.1e}; same greedy tokens: {same}')
assert full <= 1e-5 and cached <= 1e-5 and same
```

<!-- Placeholder (stage 2): the port's parameter count against the configuration count of sec_lm_hf_models and the checkpoint header, for 0.6B and the untied 8B. -->

```{.python .input #d2l-checkpoint-param-count}
%%tab pytorch
def config_json(model_id):
    """The configuration file of a pinned Hub checkpoint, as a dict."""
    with open(huggingface_hub.hf_hub_download(
            model_id, 'config.json',
            revision=d2l.HF_REVISIONS[model_id])) as f:
        return json.load(f)

cfg, cfg_8b = config_json(QWEN), config_json(QWEN_8B)
header_8b = d2l.safetensors_header(QWEN_8B)  # an index and 5 shards
print(f'{"model":7}{"port":>15}{"configuration":>15}{"header":>15}  head')
for name, c, hdr in (('0.6B', cfg, header), ('8B', cfg_8b, header_8b)):
    kwargs = qwen3_gpt_kwargs(c)
    with torch.device('meta'):
        meta_port = Qwen3GPT(**kwargs)
    in_file = sum(math.prod(shape) for shape, _ in hdr.values())
    if kwargs['tie_embeddings'] and 'lm_head.weight' in hdr:
        in_file -= math.prod(hdr['lm_head.weight'][0])  # a stored copy
    counts = d2l.num_params(meta_port), d2l.config_params(c), in_file
    print(f'{name:7}' + ''.join(f'{n:>15,}' for n in counts) + '  '
          + ('tied' if kwargs['tie_embeddings'] else 'lm_head (untied)'))
    assert counts[0] == counts[1] == counts[2]
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

# Check: the book's rotation of permuted queries and keys gives the scores
# of Transformers' rotation, if the QK-norm gain is permuted too
def rotate_half(x, cos, sin):  # Transformers' RoPE: pairs (i, i + d_h / 2)
    x1, x2 = x.chunk(2, -1)
    cos, sin = torch.cat([cos, cos], -1), torch.cat([sin, sin], -1)
    return x * cos + torch.cat([-x2, x1], -1) * sin

print('idx for d_h = 8:', qwen3_rope_permutation(8).tolist())
torch.manual_seed(0)
d_h, T = 128, 64
idx = qwen3_rope_permutation(d_h)
q, k = torch.randn(T, d_h), torch.randn(T, d_h)
gamma = 1 + 0.5 * torch.randn(d_h)  # a QK-norm gain
norm = lambda x, g: F.rms_norm(x, (d_h,), g, 1e-6)
tables = d2l.qwen3_rope_tables(T, d_h, 1e6)  # positions 0, ..., 63
hf = (rotate_half(norm(q, gamma), *tables)
      @ rotate_half(norm(k, gamma), *tables).T)
book = lambda g: (d2l.rope(norm(q[:, idx], g), tables=tables)
                  @ d2l.rope(norm(k[:, idx], g), tables=tables).T)
scale = hf.abs().max()
with_p, without = ((book(g) - hf).abs().max() / scale
                   for g in (gamma[idx], gamma))
print(f'max |score difference| / max |score|: rows and gain permuted '
      f'{with_p:.1e}, gain not permuted {without:.1e}')
assert with_p <= 1e-5 and without > 1e-2
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

<!-- Placeholder (stage 2): strict accounting on the post-trained 0.6B file: every tensor consumed, every key produced, and a missing tensor named. -->

```{.python .input #d2l-checkpoint-strict-accounting}
%%tab pytorch
hf_state = load_file(huggingface_hub.hf_hub_download(
    QWEN, 'model.safetensors', revision=d2l.HF_REVISIONS[QWEN]))
state = map_qwen3_state_dict(hf_state, cfg)
print(f'consumed {len(hf_state)} tensors, produced {len(state)} keys')
print('stored lm_head.weight equals the embedding (dropped):', torch.equal(
    hf_state['lm_head.weight'], hf_state['model.embed_tokens.weight']))
assert (len(hf_state), len(state)) == (311, 310)
del hf_state['model.layers.3.self_attn.k_norm.weight']
try:
    map_qwen3_state_dict(hf_state, cfg)
    raise AssertionError('a missing tensor went unnoticed')
except KeyError as e:
    print('KeyError:', e.args[0])
del hf_state, state
```

<!-- Placeholder (stage 2): the port in fp32, and Transformers in fp32 (eager attention) and fp64 (the reference). -->

```{.python .input #d2l-checkpoint-load-models}
%%tab pytorch
def release():  # return the memory of deleted models to the GPU
    gc.collect()
    torch.cuda.empty_cache()

ours32 = load_qwen3(QWEN, dtype=torch.float32, device=device)
hf32, tok = d2l.load_hf(QWEN, dtype=torch.float32)
hf32.set_attn_implementation('eager')
hf64, _ = d2l.load_hf(QWEN, dtype=torch.float64)  # SDPA keeps fp64 softmax
print(f'port: {d2l.num_params(ours32):,} parameters, a d2l.GPT: '
      f'{isinstance(ours32, d2l.GPT)}; Transformers: '
      f'{hf32.num_parameters():,}')
assert d2l.num_params(ours32) == hf32.num_parameters() == 596_049_920
```

<!-- Placeholder (stage 2): a provisional paired bootstrap for the ratio of two mean errors, the rule of eq_lm_port_rule, and two helpers that read logits. -->

```{.python .input #d2l-checkpoint-ratio-interval}
%%tab pytorch
def ratio_interval(a, b, num_samples=2000, seed=0):
    """a.mean() / b.mean() for paired per-item errors a, b, and the 95%
    percentile interval of the ratio over items resampled in pairs."""
    a, b = np.asarray(a), np.asarray(b)
    rows = np.random.default_rng(seed).integers(0, len(a),
                                                (num_samples, len(a)))
    ratios = a[rows].mean(1) / b[rows].mean(1)
    return a.mean() / b.mean(), np.percentile(ratios, [2.5, 97.5])

def port_rule(e_ours, e_ref, factor=1.5):
    """The rule: passes if the whole interval of the ratio is at most
    factor, fails if it is above; a summary string and the verdict."""
    ratio, (lo, hi) = ratio_interval(e_ours, e_ref)
    verdict = ('passes' if hi <= factor else 'fails' if lo > factor
               else 'undecided')
    return f'{ratio:.3g} [{lo:.3g}, {hi:.3g}] {verdict}', verdict

def logits(model, X, **kwargs):
    """Logits of the port or of a Transformers model for the ids X."""
    with torch.inference_mode():
        out = model(X.to(next(model.parameters()).device), **kwargs)
    return getattr(out, 'logits', out)

def cached_logits(model, X, keep, chunk=1024):
    """The last keep logits of a Transformers model for the ids X, fed
    through its own cache chunk tokens at a time."""
    cache = None
    with torch.inference_mode():
        for s in range(0, X.shape[1], chunk):
            out = model(X[:, s:s + chunk].to(model.device), use_cache=True,
                        past_key_values=cache, logits_to_keep=keep)
            cache = out.past_key_values
    return out.logits

def mean_abs(L, R):
    """Per prompt: mean |L - R| over positions and vocabulary, in fp64."""
    return (L.double() - R.double()).abs().mean((1, 2)).cpu().numpy()

def batched_errors(model, ref, ids, batch_size):
    """mean_abs per prompt, model in batches of batch_size, ref in 4s."""
    e = []
    for X in ids.split(batch_size):
        L = logits(model, X)
        e += [mean_abs(L[j:j + 4], logits(ref, X[j:j + 4]))
              for j in range(0, len(X), 4)]
    return np.concatenate(e)
```

<!-- Placeholder (stage 2): the fp32 rule on the 32 prompts, determinism, the two frequency formulas, and the long prompt with each set of angles, against the reference and against the reference with exact angles (the rule tests agreement with Transformers' angle rounding, not angle accuracy). -->

```{.python .input #d2l-checkpoint-fp32-check}
%%tab pytorch
e32_hf, e32_ours, delta32, gap = [], [], 0, 0
for X in ids.split(4):  # batches of 4 prompts
    L64, L_hf, L_ours = (logits(m, X) for m in (hf64, hf32, ours32))
    e32_hf.append(mean_abs(L_hf, L64))
    e32_ours.append(mean_abs(L_ours, L64))
    delta32 = max(delta32, (L_hf - L64).abs().max().item())
    gap = max(gap, (L_ours - L_hf).abs().max().item())
del L64, L_hf, L_ours  # per-batch logits: free their GPU memory
e32_hf, e32_ours = np.concatenate(e32_hf), np.concatenate(e32_ours)
text, verdict = port_rule(e32_ours, e32_hf)
L1, L2 = logits(ours32, ids[:4]), logits(ours32, ids[:4])  # twice
same, rerun = torch.equal(L1, L2), (L1 - L2).abs().max().item()
del L1, L2
print(f'fp32: e_HF {e32_hf.mean():.2e}, e_ours {e32_ours.mean():.2e}, '
      f'ratio {text}\nmax |L_HF - L_64| {delta32:.1e}, max |L_ours - L_HF| '
      f'{gap:.1e}, deterministic: '
      + ('True' if same else f'to within {rerun:.1e}'))
batch_rules = {b: port_rule(batched_errors(ours32, hf64, ids, b), e32_hf)
               for b in (1, 4, 16)}  # batch composition alone
print('ratio with the port in batches of\n' + '\n'.join(
    f'{b:4}: {text}' for b, (text, _) in batch_rules.items()))
two_i = torch.arange(0, 128, 2, dtype=torch.float32) / 128
inv_hf, inv_book = 1.0 / 1e6 ** two_i, 1e6 ** -two_i  # equal in exact math
print(f'frequencies that differ: {(inv_hf != inv_book).sum()} of 64; '
      'largest angle difference at position ' + ', '.join(
          f'{p:,}: {p * (inv_hf.double() - inv_book).abs().max():.1e}'
          for p in (63, 4095, 32767)))
if long_ids is not None:  # the last 256 of 4,096 positions, 16 blocks
    X = long_ids[None]
    blocks = lambda L, R: (L.double() - R).abs().mean(-1).reshape(
        16, 16).mean(1).cpu().numpy()
    inv64 = 1 / 1e6 ** (torch.arange(0, 128, 2, dtype=torch.float64) / 128)
    exact = torch.arange(4096, dtype=torch.float64)[:, None] * inv64
    angles = {'reference': None,  # the book's formula on the CPU, as above
              'book': d2l.rope_tables(4096, 128, base=1e6),
              'exact': (exact.cos().float(), exact.sin().float())}
    L_ours = {name: logits(ours32, X, logits_to_keep=256, tables=None
                           if t is None else tuple(u.to(device) for u in t))
              for name, t in angles.items()}
    L_hf = cached_logits(hf32, X, 256)
    L64 = cached_logits(hf64, X, 256, chunk=512)  # Transformers: fp32 angles

    class ExactRotary(torch.nn.Module):  # Transformers' interface, but
        def forward(self, x, position_ids):  # the angles in fp64
            a = position_ids[..., None].double() * inv64.to(x.device)
            a = torch.cat((a, a), -1)
            return a.cos().to(x.dtype), a.sin().to(x.dtype)

    rotary, hf64.model.rotary_emb = hf64.model.rotary_emb, ExactRotary()
    L64_exact = cached_logits(hf64, X, 256, chunk=512)
    hf64.model.rotary_emb = rotary  # restore the reference
    refs = (L64, L64_exact)
    e_hf = [blocks(L_hf, R).mean() for R in refs]
    print(f'long prompt: e_HF {e_hf[0]:.1e} against the reference, '
          f'{e_hf[1]:.1e} against it with exact angles')
    print(f'{"port angles":12}{"ratio, reference":28}'
          'ratio, reference with exact angles')
    for name, L in L_ours.items():
        rules = [port_rule(blocks(L, R), blocks(L_hf, R)) for R in refs]
        print(f'{name:12}{rules[0][0]:28}{rules[1][0]}')
        if name == 'reference':
            assert rules[0][1] == 'passes'
    del L_ours, L_hf, L64, L64_exact
if not SMALL:
    assert verdict == 'passes'
    assert all(v == 'passes' for _, v in batch_rules.values())
```

<!-- Placeholder (stage 2): the same rule in bf16, the fp32 reference as a stand-in for fp64, how far batch composition alone moves the ratio, and the ratio against Transformers' default SDPA attention instead of eager. -->

```{.python .input #d2l-checkpoint-bf16-check}
%%tab pytorch
del ours32
release()
ours16 = load_qwen3(QWEN, dtype=torch.bfloat16, device=device)
hf16, _ = d2l.load_hf(QWEN, dtype=torch.bfloat16)
hf16.set_attn_implementation('eager')
e16_hf, e16_ours, e16_ours_32 = [], [], []
for X in ids.split(4):
    L64, L32, L_hf, L_ours = (logits(m, X)
                              for m in (hf64, hf32, hf16, ours16))
    e16_hf.append(mean_abs(L_hf, L64))
    e16_ours.append(mean_abs(L_ours, L64))
    e16_ours_32.append(mean_abs(L_ours, L32))
del L64, L32, L_hf, L_ours
e16_hf, e16_ours, e16_ours_32 = map(np.concatenate,
                                    (e16_hf, e16_ours, e16_ours_32))
text, verdict = port_rule(e16_ours, e16_hf)
change = e16_ours_32.mean() / e16_ours.mean() - 1
print(f'bf16: e_HF {e16_hf.mean():.2e}, e_ours {e16_ours.mean():.2e}, '
      f'ratio {text}\ne_ours against the fp32 instead of the fp64 '
      f'reference: relative change {change:.1e}')

print('ratio with the port in batches of ' + ', '.join(
    f'{b}: {batched_errors(ours16, hf64, ids, b).mean() / e16_hf.mean():.3f}'
    for b in (1, 4, 16)))
hf16.set_attn_implementation('sdpa')  # Transformers' default attention
e16_sdpa = np.concatenate([mean_abs(logits(hf16, X), logits(hf64, X))
                           for X in ids.split(4)])
hf16.set_attn_implementation('eager')  # back to the yardstick above
print(f'Transformers bf16 with SDPA attention: e_HF {e16_sdpa.mean():.2e}, '
      f'ratio {port_rule(e16_ours, e16_sdpa)[0]}')
if not SMALL:
    assert verdict == 'passes'
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

<!-- Placeholder (stage 2): greedy continuations of the port and of Transformers in fp32 and bf16, where they first differ, and the reference's top-2 gap there. -->

```{.python .input #d2l-checkpoint-greedy}
%%tab pytorch
del hf64
release()
ours32 = load_qwen3(QWEN, dtype=torch.float32, device=device)
models = {'ours32': ours32, 'hf32': hf32, 'ours16': ours16, 'hf16': hf16}
gen = {name: [g.token_ids for g in generate(m, tok, texts, NEW_TOKENS)]
       for name, m in models.items()}
eos, agree = set(hf32.generation_config.eos_token_id), 0
for s in range(0, len(texts), 8):  # Transformers' own greedy loop
    batch = tok(texts[s:s + 8], padding=True, padding_side='left',
                return_tensors='pt').to(device)
    with torch.inference_mode():
        out = hf32.generate(**batch, do_sample=False,
                            max_new_tokens=NEW_TOKENS)
    for row, ours in zip(out[:, batch['input_ids'].shape[1]:].tolist(),
                         gen['hf32'][s:s + 8]):
        agree += row[:next((i for i, t in enumerate(row) if t in eos),
                           len(row))] == ours
print(f'generate on hf32 equals hf32.generate: {agree} of {len(texts)}')
assert agree == len(texts)

def first_split(a, b):
    """Index of the first token where two continuations differ, or None."""
    return None if a == b else next(
        (i for i, (x, y) in enumerate(zip(a, b)) if x != y), min(len(a),
                                                                len(b)))

def top2_gap(model, tokenizer, text, tokens):
    """The gap between the two largest logits after text and tokens."""
    X = torch.tensor([tokenizer(text, add_special_tokens=False).input_ids
                      + tokens])
    top = logits(model, X)[0, -1].float().topk(2).values
    return (top[0] - top[1]).item()

ref = gen['hf32']
for name in ('ours32', 'ours16', 'hf16'):
    split = [first_split(g, r) for g, r in zip(gen[name], ref)]
    diverged = [(i, n) for i, n in enumerate(split) if n is not None]
    gaps = [top2_gap(hf32, tok, texts[i], ref[i][:n])
            for i, n in diverged]
    others = '' if not diverged else (  # median over those that diverge
        f'; the others first differ at token '
        f'{np.median([n for _, n in diverged]):g} (median), top-2 gaps '
        'there: ' + ', '.join(f'{g:.1e}' for g in gaps))
    print(f'{name}: {len(texts) - len(diverged)} of {len(texts)} identical '
          f'to hf32 over {NEW_TOKENS} tokens{others}')
    if name == 'ours32':  # in fp32 only a near-tie may flip a token
        for (i, n), g in zip(diverged, gaps):
            print(f'  prompt {i}: token {n}, near-tie {g < 10 * delta32}')
        assert all(g < 10 * delta32 for g in gaps)
```

<!-- Placeholder (stage 2): a short chat with the port, using the repository's sampling defaults and thinking off. -->

```{.python .input #d2l-checkpoint-chat}
%%tab pytorch
config = ours16.generation_config  # temperature, top-k, top-p of the repo
questions = ['What does a key-value cache store, in one sentence?',
             'Name two differences between a base model and a chat model.']
for q, g in zip(questions, generate(
        ours16, tok, questions, 32 if SMALL else 128, chat=True,
        enable_thinking=False, strategy='sample',
        temperature=config.temperature, k=config.top_k, p=config.top_p,
        seed=0)):
    print(f'{q}\n  {g.text!r} ({g.finish_reason})')
```

<!-- Placeholder (stage 2): the same loader on Qwen3-1.7B-Base (bf16 rule against an fp32 reference) and a dry run of Qwen3-8B from its header. -->

```{.python .input #d2l-checkpoint-ladder}
%%tab pytorch
del models, ours32, hf32, ours16, hf16  # models held them too
release()
if not SMALL:  # Qwen3-1.7B-Base: fp32 reference logits kept on the CPU
    hf32, _ = d2l.load_hf(QWEN_17B_BASE, dtype=torch.float32)
    hf32.set_attn_implementation('eager')
    ref = [logits(hf32, X).cpu() for X in ids.split(4)]
    del hf32
    release()
    hf16, _ = d2l.load_hf(QWEN_17B_BASE, dtype=torch.bfloat16)
    hf16.set_attn_implementation('eager')
    ours16 = load_qwen3(QWEN_17B_BASE, dtype=torch.bfloat16, device=device)
    e = {name: np.concatenate([mean_abs(logits(m, X), R.to(device))
                               for X, R in zip(ids.split(4), ref)])
         for name, m in (('hf', hf16), ('ours', ours16))}
    text, verdict = port_rule(e['ours'], e['hf'])
    print(f'Qwen3-1.7B-Base, bf16, {len(ids)} prompts: ratio {text}')
    assert verdict == 'passes'
    del hf16, ours16, ref
    release()
# Qwen3-8B: meta tensors with the header's shapes, mapped and assigned
dtypes = {'BF16': torch.bfloat16, 'F16': torch.float16, 'F32': torch.float32}
meta = {name: torch.empty(shape, dtype=dtypes[dtype], device='meta')
        for name, (shape, dtype) in header_8b.items()}
with torch.device('meta'):
    port_8b = Qwen3GPT(**qwen3_gpt_kwargs(cfg_8b))
state_8b = map_qwen3_state_dict(meta, cfg_8b)
port_8b.load_state_dict(state_8b, strict=True, assign=True)
print(f'Qwen3-8B dry run: {len(meta)} tensors consumed, {len(state_8b)} '
      f'keys produced, untied lm_head: {"lm_head.weight" in state_8b}, '
      f'all on meta: {all(p.is_meta for p in port_8b.parameters())}')
assert len(meta) == len(state_8b) and hasattr(port_8b, 'lm_head')
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

<!-- Placeholder (stage 2): relative error per layer and capture point against Transformers in fp64, for the port and for Transformers, in fp32 and bf16. -->

```{.python .input #d2l-checkpoint-layer-errors}
%%tab pytorch
def errors_against(ref_model, models, ids, batch_size=4):
    """layer_errors of each model against ref_model on the ids, in batches
    of batch_size: a dict of arrays (layers, capture points, prompts)."""
    errors = {name: [] for name in models}
    for X in ids.split(batch_size):
        X = X.to(next(ref_model.parameters()).device)
        ref = collect_hiddens(ref_model, X)
        for name, m in models.items():
            errors[name].append(layer_errors(ref, collect_hiddens(m, X)))
    return {name: np.concatenate(e, -1) for name, e in errors.items()}

hf64, _ = d2l.load_hf(QWEN, dtype=torch.float64)
err, other = {}, {}  # other: the port in batches of 1 and 16, SDPA
for dtype, bits in ((torch.float32, 32), (torch.bfloat16, 16)):
    ours = load_qwen3(QWEN, dtype=dtype, device=device)
    hf, _ = d2l.load_hf(QWEN, dtype=dtype)
    hf.set_attn_implementation('eager')
    err.update(errors_against(hf64, {f'ours{bits}': ours, f'hf{bits}': hf},
                              ids))
    for b in (1, 16):  # batch composition alone
        other[bits, b] = errors_against(hf64, {'': ours}, ids, b)['']
    if bits == 16:  # Transformers' default attention instead of eager
        hf.set_attn_implementation('sdpa')
        other['sdpa'] = errors_against(hf64, {'': hf}, ids)['']
    del ours, hf
    release()
median = {k: np.median(e, -1) for k, e in err.items()}  # (layers, points)
layers = [0, 7, 14, 21, len(median['ours32']) - 1]
print('median relative error of resid_post at layer')
print(' ' * 6 + ''.join(f'{l:>9}' for l in layers))
for k, m in median.items():
    print(f'{k:6}' + ''.join(f'{m[l, 3]:9.1e}' for l in layers))

def ratio_range(e_ours, e_hf):
    """Port over Transformers, medians over prompts: its range over every
    layer and point, and the layers where it leaves [0.5, 2]."""
    r = np.median(e_ours, -1) / np.median(e_hf, -1)
    out = sorted(set(np.nonzero((r < 0.5) | (r > 2))[0].tolist()))
    return f'{r.min():.2f} to {r.max():.2f}, outside [0.5, 2]: {out}'

print('port / Transformers (eager) over every layer and point')
for bits, name in ((32, 'fp32'), (16, 'bf16')):
    for b, e in ((1, other[bits, 1]), (4, err[f'ours{bits}']),
                 (16, other[bits, 16])):
        print(f'{name}, port in batches of {b:2}: '
              + ratio_range(e, err[f'hf{bits}']))
print('bf16, Transformers with SDPA:   '
      + ratio_range(err['ours16'], other['sdpa']))
```

<!-- Placeholder (stage 2): five porting bugs that strict accounting passes: where each first diverges, and what the two logit rules say about each. -->

```{.python .input #d2l-checkpoint-bugs}
%%tab pytorch
def build(state, cfg, dtype, device, **changes):
    """A Qwen3GPT with some arguments changed and its weights assigned from
    state; load_state_dict(strict=True) does the strict accounting."""
    with torch.device('meta'):
        model = Qwen3GPT(**{**qwen3_gpt_kwargs(cfg), **changes})
    model.load_state_dict({k: v.to(device, dtype) for k, v in state.items()},
                          strict=True, assign=True)
    return model.eval()

def mean_square(m, key):  # a hook: mean of x^2 per token, its median
    def hook(module, args):  # over the tokens after the first
        m[key] = (args[0][:, 1:].float() ** 2).mean(-1).median().item()
    return hook

hf_state = load_file(huggingface_hub.hf_hub_download(
    QWEN, 'model.safetensors', revision=d2l.HF_REVISIONS[QWEN]))
clean = map_qwen3_state_dict(hf_state, cfg)
order = sorted(range(cfg['num_hidden_layers']), key=str)  # 0, 1, 10, ...
relabel = lambda f: {f(k.split('.')): v for k, v in clean.items()}
bugs = {  # name: (state dict, changed arguments of Qwen3GPT)
    'B1 RoPE rows not permuted': (
        map_qwen3_state_dict(hf_state, cfg, rope_layout='none'), {}),
    'B2 rope_theta 1e4': (clean, {'rope_theta': 1e4}),
    'B3 gate and up swapped': (relabel(lambda p: '.'.join(
        {'W_g': 'W_1', 'W_1': 'W_g'}.get(x, x) for x in p)), {}),
    'B4 layers in string order': (relabel(lambda p: '.'.join(  # block j
        p[:1] + [str(order.index(int(p[1])))] + p[2:] if p[0] == 'blks'
        else p)), {}),  # gets the weights of layer order[j]
    'B5 every epsilon 1e-5': (clean, {'norm_eps': 1e-5})}
del hf_state
clean32, m = build(clean, cfg, torch.float32, device), {}
for layer in (0, len(clean32.blks) - 1):  # the input of every norm
    for n in ('norm1', 'attention.q_norm', 'attention.k_norm', 'norm2'):
        clean32.blks[layer].get_submodule(n).register_forward_pre_hook(
            mean_square(m, f'{layer}.{n}'))
logits(clean32, ids[:4])
del clean32
print('mean square of the input, clean port: ' + ', '.join(
    f'{k} {v:.1e}' for k, v in m.items()))
floor, curves, found = err['ours32'], {}, {}
for name, (state, changes) in bugs.items():
    bug32, bug16 = (build(state, cfg, dtype, device, **changes)
                    for dtype in (torch.float32, torch.bfloat16))
    errors, e32, e16, top1 = [], [], [], 0
    for X in ids.split(4):
        X = X.to(device)
        errors.append(layer_errors(collect_hiddens(hf64, X),
                                   collect_hiddens(bug32, X)))
        L64, L32 = logits(hf64, X), logits(bug32, X)
        e32.append(mean_abs(L32, L64))
        e16.append(mean_abs(logits(bug16, X), L64))
        top1 += (L32.argmax(-1) == L64.argmax(-1)).sum().item()
    del bug32, bug16
    release()
    errors = np.concatenate(errors, -1)
    curves[name] = np.median(errors[:, 3], -1)  # resid_post
    where = first_divergence(errors, floor)
    there = 'none'
    if where is not None:
        l, j = where[0], _HIDDEN_POINTS.index(where[1])
        times = np.median(errors[l, j]) / np.median(floor[l, j])
        there = f'{where}, {times:.3g} x the clean error there'
    rules = [port_rule(np.concatenate(e), e_hf)
             for e, e_hf in ((e32, e32_hf), (e16, e16_hf))]
    found[name] = (where, np.array_equal(errors[:2], floor[:2]), rules,
                   np.concatenate(e16))
    print(f'{name}: accounting passes; first divergence {there}\n'
          f'  fp32 rule {rules[0][0]}; bf16 rule {rules[1][0]}; top-1 '
          f'agreement with fp64 {top1} of {ids.numel()}')
del hf64, L64, L32
release()
where, exact, _, _ = found['B4 layers in string order']
print(f'B4: blocks 0 and 1 bitwise equal to the clean port: {exact}')
ratio, (lo, hi) = ratio_interval(found['B5 every epsilon 1e-5'][3], e16_ours)
print(f"B5: bf16 logit error over the clean port's {ratio:.3g} "
      f'[{lo:.3g}, {hi:.3g}]')
# blocks 0 and 1 are correct; bitwise only on a deterministic device
assert where == (2, 'attn_out') and (exact or not same)
if not SMALL:  # B1-B4 fail both logit rules
    assert all(r[1] == 'fails' for name, (_, _, rules, _) in found.items()
               if not name.startswith('B5') for r in rules)
```

<!-- Placeholder (stage 2): the plot below; caption sentence from spec fig_lm_layer_divergence. -->

```{.python .input #d2l-checkpoint-divergence-plot}
%%tab pytorch
resid_post = lambda e: np.median(e[:, 3], -1)  # median over prompts
precision = [resid_post(err[k]) for k in ('ours32', 'hf32', 'ours16', 'hf16')]
porting = list(curves.values()) + [resid_post(floor)]
names = [name.split()[0] for name in curves] + ['correct']
ylim = (min(map(min, precision + porting)) / 2,  # one scale for both
        max(map(max, precision + porting)) * 2)
_, axes = d2l.plt.subplots(1, 2, figsize=(7, 2.6), sharey=True)
with d2l.plt.rc_context({'legend.fontsize': 'x-small'}):
    d2l.plot(np.arange(len(floor)), precision, 'layer', 'relative error',
             ['port fp32', 'Transformers fp32', 'port bf16',
              'Transformers bf16'], ylim=ylim, yscale='log',
             fmts=('C0-', 'C0--', 'C1-', 'C1--'), axes=axes[0])
    d2l.plot(np.arange(len(floor)), porting, 'layer', None, names,
             ylim=ylim, yscale='log',
             fmts=('C0-', 'C1-', 'C2-', 'C3-', 'C4-', 'C7--'), axes=axes[1])
    axes[1].legend(names, ncol=3, loc='center', bbox_to_anchor=(0.58, 0.27))
```
