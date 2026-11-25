# The Pretraining Data Path
:label:`sec_lm_data_path`

<!-- Author brief, copied verbatim from the Language Models part plan
(docs/llm/language_models_part_plan.md, "4.1 The Pretraining Data Path · T1").
Blueprint: docs/llm/blueprints/ch04_pretraining.md section 4.1.
The plan is a specification, not book text; this comment is replaced
when the section is written.

**Idea:** A data path is exact and resumable, so that the batch at step $s$ is a pure function of the manifest, the seed and $s$, or the run is not reproducible.
**Opening:** The reader's afternoon run in `sec_gpt` read a text file into a tensor and sliced it. A run that dies at step 4,000 and is restarted with a re-seeded shuffle sees a batch the dead run never would have, and a pair of short documents packed into one window attend to each other unless something stops them. A first cell packs two documents into a 32-token window, prints the attention mask, and shows that without it the second document's logits change. What remains is to ask what the run itself, in 4.2, is made of once its input is exact.
**Teaches:**
- Pre-tokenized sharded streams with a manifest: the flat token shards of 3.6, their document index, the layout hash that `load_state_dict` checks, and why dataset reproducibility (the same tokens in the same order) is not run reproducibility (deterministic arithmetic, `sec_repro`).
- Packing with per-document positions and a block-diagonal causal mask (`doc_ids`, `pos_ids` derived from the EOS that ends each document; the mask restated from `sec_attention-scoring-functions` and FlexAttention's `create_block_mask` from `sec_windowed-linear-attention`, with the SDPA boolean mask as the CPU path); the fraction of tokens that a window boundary cuts, predicted from the document-length histogram.
- Token-weighted loss across unequal documents and micro-batches, with the chunked logit computation that keeps the vocabulary projection out of peak memory (`d2l.chunked_lm_loss`), and the DDP form where the valid-token count is all-reduced before the backward pass.
- Resumable cursors that name the next batch: stateless addressing of the global batch at step $s$ as a seeded permutation sliced per rank, `state_dict` as the consumed count, elasticity across world sizes, and a `skip` set that removes a bad batch without changing the others.
- The book's `d2l.DataModule` and `d2l.BPETokenizer` stack as the main flow; Hugging Face `datasets` streaming and `torchdata` named once as the production equivalents; input starvation measured as data-wait time per step.
**Derivation:** The token-weighted reduction. For micro-batches with valid counts $n_k$ and mean losses $\bar\ell_k$, the mean of micro-batch means differs from the token-weighted loss by $\tilde L - L = -\textrm{Cov}_k(n_k / \bar n, \bar\ell_k)$; two micro-batches with counts 1,024 and 300 and mean losses 3.0 and 4.0 give a token-weighted loss of 3.23 and a mean of means of 3.50. The notebook checks accumulation over eight micro-batches with counts [1024, 900, 512, 1024, 300, 1000, 700, 1024] against a single pass to `1e-6` and shows the naive mean differing by the predicted covariance. A second small result, in the exercises: the cut fraction $f_{\textrm{cut}}(T) = \mathbb{E}[\ell \min(1, (\ell - 1)/T)] / \bar\ell$ over document lengths $\ell$, checked on the validation shard within 2% of measured packing at $T \in \{256, \ldots, 8192\}$.
**Notebook:**
1. Question: do packed batches with document masks give the same per-document logits as unpacked batches, and does an interrupted run resume on the same batches?
2. Model and data: `d2l.PackedGPT` at the `t1` preset (width 384, 8 blocks, 6 query and 2 key-value heads, 25.2M parameters) on the 100M-token FineWeb-Edu shards and the validation shard, tokenizer `d2l.r2_tokenizer()`; CPU fp32 for the exactness tests, one 16 GB GPU for the throughput and training cells.
3. Check: for documents of 5 and 9 tokens in one window (seed 0, width 64, two blocks), the masked per-document logits equal the alone logits to `1e-5` with RoPE, with and without position resets; the unmasked difference is at least `1e-3`; a learned position table needs the reset to pass; the `flex` backend matches `sdpa` within `2e-2` in bf16; the fixture `[5, 6, EOS, 7, EOS, EOS, 8]` gives `doc_ids = [0,0,0,1,1,2,3]` and `pos_ids = [0,1,2,0,1,0,0]`; a stream resumed at step 37 reproduces 50 batches bit-exactly, world sizes 1, 2 and 4 give the same multiset, and `skip=(3,)` makes logical step 3 equal physical step 4.
4. Claim that survives noise: the exactness checks are exact. The one training comparison (masked against unmasked, 25M tokens at $T = 1024$, three paired seeds) reports the paired loss difference with a $t$ interval on two degrees of freedom and says "no conclusion" if it contains zero; the measured leak (attention mass on foreign documents per layer) is positive in every layer. Throughput of four pipelines (padded unpacked, packed unmasked, SDPA mask, FlexAttention mask) at $T \in \{1024, 4096\}$ as a committed table `lm-pdp-throughput`.
5. Reused: `d2l.Qwen3Attention` and `d2l.Qwen3GPT` (1.2) as the parents; `d2l.TokenShards` (3.6); `d2l.repro` seeds. The reduced configuration is the section itself (T1 with a `--tokens` flag).
**Figures:** `fig_lm_packing_mask` (a 32-token window holding documents of 12, 8 and 12 tokens with its block-diagonal causal mask); `fig_lm_data_cursor` (the manifest, the seeded permutation, the per-rank slice and the consumed-count cursor); `fig_lm_pdp_throughput` (plot: tokens per second of the four pipelines at two context lengths).
**Restate / forward:** `sec_attention-scoring-functions` (a packed sequence's mask is the conjunction of the same-document test and the causal test); `sec_windowed-linear-attention` (FlexAttention compiles a Python mask predicate into a block-sparse kernel, so packed documents run at fused speed); `sec_repro` (seeds fix the data order; kernel nondeterminism remains). Forward: none.
**Exercises:** 5 to 7. Derive: token-weighted against sequence-weighted gradients ($(\textrm{softmax} - \textrm{onehot})/N$ per token, and what per-sequence averaging does to long documents); the cut fraction under best-fit packing. Measure: the mask's cost against document length at $T = 256$ and $4096$; the leak at those lengths. Open: a Feistel-permutation shuffle with $O(1)$ state; a five-source mixture that stays exact (answered in 4.4 by `d2l.mix_streams`).
**Writer notes:**
- Silent breakages: a re-seeded shuffle at restart replays data; `DataLoader` workers duplicate RNG state; a literal `<|endoftext|>` in text becomes a real EOS unless the tokenizer encodes with specials disallowed; masking attention but not the loss at document boundaries (the target at the first token of a document is masked with `-100`); per-document position resets matter only for learned tables, not RoPE (say so, it is the A4 written question); DDP without a global valid-token count; `uint16` wraps above 65,535; FlexAttention retraces the block mask when $B$ or $T$ change (fix them).
- Tolerances: packed-versus-alone `1e-5` in fp32 on CPU; `flex` against `sdpa` `2e-2` in bf16 (to be tightened after measurement); `chunked_lm_loss` equals `token_weighted_loss` to `1e-6` in value and gradient.
- Do not re-teach: document masks (`sec_attention-scoring-functions`, attention-scoring.md "packed sequences"), FlexAttention's `mask_mod` and `create_block_mask` (`sec_windowed-linear-attention`), seeds and determinism (`sec_repro`), the BPE tokenizer (`sec_text-sequence`). New here: shards and manifests, token-weighted loss, resumable cursors, position reset.
- Hardware fallback: if FlexAttention fails to compile on the build box, `sdpa` is the test default and the throughput table prints "flex unavailable".
- Blueprint: ch04 §4.1 holds the exact signatures, the three derivations (D1 packed equals unpacked with the induction over layers, D2 cut fraction, D3 token weighting), the headline protocol and `tab_lm_pdp_exactness` (three exactness levels across schemes). This section keeps all of it and adds the one-sentence restatements above.
- Open before writing: whether 3.6's `.idx` includes the EOS in each document (the boundary derivation changes if not); FineWeb-Edu's ODC-By attribution text on the derived shards (Chapter 3 decided it ships with attribution, 2026-10-29).
**Sources:**
- Papers: [The Llama 3 Herd of Models](https://arxiv.org/abs/2407.21783) (document masks and packing in a production pretraining recipe); [The FineWeb Datasets](https://arxiv.org/abs/2406.17557) (the corpus the shards are cut from, with its licence).
- Code and documentation: [FlexAttention API](https://docs.pytorch.org/docs/stable/nn.attention.flex_attention.html) (block masks for packed documents); [PyTorch data loading](https://docs.pytorch.org/docs/stable/data.html) (worker behaviour and iterable datasets); [Hugging Face `datasets` streaming](https://huggingface.co/docs/datasets/stream) and [torchdata](https://github.com/pytorch/data) (the production streaming and stateful-loader equivalents); [datatrove](https://github.com/huggingface/datatrove) (pipeline metadata and manifests at scale); [TorchTitan](https://github.com/pytorch/torchtitan) (a production data path and checkpoint contract to compare with).
- Courses, tutorials and blogs: [FlexAttention blog](https://pytorch.org/blog/flexattention/) (why a Python mask can run at fused speed); [CS336 assignment 1](https://github.com/stanford-cs336/assignment1-basics) (the test discipline for the data path); [build-nanogpt](https://github.com/karpathy/build-nanogpt) (the incremental data-loader construction).
-->

<!-- Placeholder (stage 2, infra step): the cells below are the section's
shared library, written before the section; the prose around them comes with
the section. Cell ids follow the section specification (S/spec/21.1.md). -->

```{.python .input #data-path-imports}
%%tab pytorch
import math
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from d2l import torch as d2l
```

```{.python .input #data-path-the-pretraining-data-path-1}
%%tab pytorch
import collections
```

<!-- Placeholder (cell 5): document ids and positions from the EOS that ends each document. -->

```{.python .input #data-path-doc-ids}
%%tab pytorch
#@save
def doc_ids_from_eos(tokens, eos_id=0):
    """Document index and position within the document of every token of a
    packed window (last axis); an EOS belongs to the document it ends."""
    is_eos = (tokens == eos_id).int()
    doc_ids = torch.cumsum(is_eos, -1) - is_eos   # EOS strictly before t
    t = torch.arange(tokens.shape[-1], device=tokens.device).expand_as(tokens)
    starts = torch.where(doc_ids != F.pad(doc_ids, (1, 0))[..., :-1], t, 0)
    first = torch.cummax(starts, -1).values       # where t's document starts
    return doc_ids.to(tokens.dtype), (t - first).to(tokens.dtype)
```

<!-- Placeholder (cell 7): attention over packed windows; the mask is the
conjunction of the same-document test and the causal test. -->

```{.python .input #data-path-packed-attention}
%%tab pytorch
#@save
PackCtx = collections.namedtuple('PackCtx', ['doc_ids', 'pos_ids'])

#@save
def doc_mask_mod(doc_ids):
    """FlexAttention's mask predicate for packed windows."""
    return lambda b, h, q, k: (doc_ids[b, q] == doc_ids[b, k]) & (k <= q)

#@save
class PackedAttention(d2l.Qwen3Attention):
    """Qwen3Attention for packed windows: given ctx (a PackCtx), a query sees
    the earlier keys of its own document only. Backends: 'sdpa' (boolean
    mask), 'flex' (FlexAttention block mask, CUDA), 'math' (explicit scores,
    needed for an attention-logit softcap); 'auto' picks flex on CUDA."""
    def __init__(self, num_hiddens, num_heads, num_kv_heads, *, head_dim=None,
                 qk_norm=False, rope_theta=10000.0, norm_eps=1e-6, bias=False,
                 rope=True, rope_inv_freq=None, rope_mscale=1.0, softcap=None,
                 backend='auto'):
        super().__init__(num_hiddens, num_heads, num_kv_heads,
                         head_dim=head_dim, qk_norm=qk_norm,
                         rope_theta=rope_theta, norm_eps=norm_eps, bias=bias,
                         rope=rope)
        assert backend in ('auto', 'sdpa', 'flex', 'math'), backend
        self.softcap, self.backend = softcap, backend
        self.set_rope(rope_inv_freq, rope_mscale)

    def set_rope(self, inv_freq=None, mscale=1.0):
        """RoPE frequencies (None: base rope_theta) and YaRN's scale m."""
        self.rope_inv_freq = (None if inv_freq is None else
                              torch.as_tensor(inv_freq, dtype=torch.float32))
        self.rope_mscale = mscale

    def qkv(self, X, offset=0, tables=None):
        if self.rope and tables is None:  # positions offset, offset + 1, ...
            tables = d2l._rope_tables(
                torch.arange(X.shape[1], device=X.device) + offset,
                self.head_dim, self.rope_theta, self.rope_inv_freq,
                self.rope_mscale)
        return super().qkv(X, offset, tables)

    def forward(self, X, keys=None, values=None, mask=None, tables=None,
                ctx=None):
        backend = d2l._attn_backend(self, X, ctx)
        if backend == 'causal':  # one document per row: Qwen3Attention
            return super().forward(X, keys, values, mask, tables)
        if mask is None:  # PackedGPT builds the mask once for all blocks
            mask = d2l._doc_mask(ctx, backend, X.shape[1], X.device)
        q, k, v = self.qkv(X, tables=tables)
        gqa = self.num_kv_heads < self.num_heads
        if backend == 'flex':
            Y = d2l._flex_attention()(q, k, v, block_mask=mask,
                                      enable_gqa=gqa)
        elif backend == 'sdpa':
            Y = F.scaled_dot_product_attention(q, k, v, attn_mask=mask,
                                               enable_gqa=gqa)
        else:
            Y = d2l._math_attention(q, k, v, mask, self.softcap)
        return self.W_o(Y.transpose(1, 2).flatten(2))
```

<!-- Placeholder (cell 8): the packed model; RoPE angles from per-document
positions, conversions to and from the Qwen3 port, scaled RoPE for 21.5. -->

```{.python .input #data-path-packed-gpt}
%%tab pytorch
#@save
class PackedGPT(d2l.Qwen3GPT):
    """Qwen3GPT for pretraining on packed windows: forward(X, ctx) with
    per-document positions and masks, the final hidden states for a chunked
    loss, scaled RoPE (set_rope) and a cached step that uses it."""
    def __init__(self, vocab_size, num_hiddens, num_heads, num_kv_heads,
                 num_blks, *, head_dim=None, qk_norm=False, rope_theta=1e4,
                 ffn_hidden=None, norm_eps=1e-6, tie_embeddings=True,
                 max_len=4096, bias=False, pos='rope', rope_inv_freq=None,
                 rope_mscale=1.0, softcap=None, backend='auto'):
        assert pos in ('rope', 'learned'), pos
        self.packed_options = dict(rope=pos == 'rope', softcap=softcap,
                                   backend=backend)
        super().__init__(vocab_size, num_hiddens, num_heads, num_kv_heads,
                         num_blks, head_dim=head_dim, qk_norm=qk_norm,
                         rope_theta=rope_theta, ffn_hidden=ffn_hidden,
                         norm_eps=norm_eps, tie_embeddings=tie_embeddings,
                         max_len=max_len, bias=bias)
        self.pos, self.trained_len = pos, max_len
        if pos == 'learned':  # an absolute table, indexed by pos_ids
            self.pos_emb = nn.Embedding(max_len, num_hiddens)
            nn.init.normal_(self.pos_emb.weight, std=0.02)
        self.set_rope(rope_inv_freq, rope_mscale)

    def CausalAttention(self, d, H, *, bias, rope, num_kv_heads):
        return PackedAttention(d, H, num_kv_heads, bias=bias,
                               **self.attn_options, **self.packed_options)

    def forward(self, X, ctx=None, return_hidden=False, logits_to_keep=None):
        pos = (ctx.pos_ids if ctx is not None
               else torch.arange(X.shape[1], device=X.device))
        H, tables = self.token_emb(X), None
        if self.pos == 'learned':
            H = H + self.pos_emb(pos.long())
        else:  # angles of each token's position within its document
            tables = d2l._rope_tables(pos, self.head_width, self.rope_theta,
                                      self.rope_inv_freq, self.rope_mscale)
        mask = d2l._packed_mask(self, ctx, X)  # once for every block
        for blk in self.blks:
            H = blk(H, mask, tables=tables, ctx=ctx)
        if logits_to_keep is not None:
            H = H[:, -logits_to_keep:]
        H = self.norm(H)
        W = self.lm_head.weight if hasattr(self, 'lm_head') else \
            self.token_emb.weight
        return (H, W) if return_hidden else F.linear(H, W)

    def set_rope(self, inv_freq=None, mscale=1.0, scheme='none', scale=1.0):
        """Use RoPE frequencies inv_freq (None: base) and YaRN's scale m in
        every block; the model then accepts scale times its trained length."""
        self.rope_inv_freq = (None if inv_freq is None else
                              torch.as_tensor(inv_freq, dtype=torch.float32))
        self.rope_mscale, self.rope_scheme, self.rope_scale = (mscale, scheme,
                                                               scale)
        self.max_len = round(scale * self.trained_len)
        for blk in self.blks:
            blk.attention.set_rope(inv_freq, mscale)

    @torch.inference_mode()
    def forward_cached(self, X, ks, vs, t, all_logits=False):
        """Qwen3GPT.forward_cached with this model's (possibly scaled) RoPE;
        all_logits=True returns the logits of every new position."""
        tables = d2l._rope_tables(
            torch.arange(t, t + X.shape[1], device=X.device), self.head_width,
            self.rope_theta, self.rope_inv_freq, self.rope_mscale)
        H = self.token_emb(X)
        for i, blk in enumerate(self.blks):
            H = H + blk.attention.forward_step(blk.norm1(H),
                                               (ks[i], vs[i], t), tables)
            H = H + blk.ffn(blk.norm2(H))
        H = self.norm(H if all_logits else H[:, -1])
        W = self.lm_head.weight if hasattr(self, 'lm_head') else \
            self.token_emb.weight
        return F.linear(H, W), ks, vs

    def config(self):
        """Every constructor argument and the RoPE scaling, as JSON values."""
        return d2l._packed_config(self)

    @classmethod
    def from_arch(cls, arch, **kw):
        """A model of the shape d2l.r2_config(...).arch gives."""
        kw = dict(dict(head_dim=arch.d_h, ffn_hidden=arch.f,
                       qk_norm=arch.qk_norm, rope_theta=arch.rope_base,
                       tie_embeddings=arch.tie, max_len=arch.T), **kw)
        return cls(arch.vocab, arch.d, arch.H, arch.H_kv, arch.L, **kw)

    @classmethod
    def adopt(cls, model, backend='auto'):
        """A PackedGPT holding the weights of a d2l.Qwen3GPT (no copy)."""
        with torch.device('meta'):
            new = cls(**d2l._qwen3_shape(model), backend=backend)
        d2l._check_same_tensors(new.state_dict(), model.state_dict())
        new.load_state_dict(model.state_dict(), assign=True)
        return new
```

<!-- Placeholder (cell 11): the token-weighted loss, and the chunked form
that never holds the full (tokens x vocabulary) logits. -->

```{.python .input #data-path-losses}
%%tab pytorch
#@save
def token_weighted_loss(logits, labels, ignore_index=-100):
    """Summed next-token NLL over the valid targets (fp32) and their count."""
    logits = logits.flatten(0, -2)
    sum_nll = F.cross_entropy(logits.to(torch.promote_types(
        logits.dtype, torch.float32)), labels.flatten(),
        ignore_index=ignore_index, reduction='sum')
    return sum_nll, (labels != ignore_index).sum()

#@save
def chunked_lm_loss(hidden, weight, labels, chunk=8192, z_loss=0.0,
                    ignore_index=-100, *, return_nll=False):
    """token_weighted_loss(F.linear(hidden, weight), labels), plus z_loss *
    logsumexp^2 per valid target, computed chunk positions at a time; each
    chunk's logits are recomputed in the backward pass instead of stored.
    return_nll=True also returns the summed NLL without the z-loss term."""
    H, y = hidden.flatten(0, -2), labels.flatten()

    def piece(h, W, y):  # (loss, nll) of one chunk, from fp32 logits
        logits = F.linear(h, W)
        logits = logits.to(torch.promote_types(logits.dtype, torch.float32))
        lse, valid = torch.logsumexp(logits, -1), y != ignore_index
        nll = lse - logits.gather(-1, y.clamp(min=0)[:, None])[:, 0]
        nll = torch.where(valid, nll, 0.0).sum()
        z = torch.where(valid, lse ** 2, 0.0).sum()
        return torch.stack([nll + z_loss * z, nll])

    total = sum(torch.utils.checkpoint.checkpoint(
        piece, H[i:i + chunk], weight, y[i:i + chunk], use_reentrant=False)
        for i in range(0, len(y), chunk))
    n_valid = (y != ignore_index).sum()
    return (total[0], n_valid, total[1]) if return_nll else (total[0],
                                                             n_valid)
```

<!-- Placeholder (cell 14): the resumable stream; the batch at step s is a
pure function of the manifest, the seed and s. -->

```{.python .input #data-path-packed-stream}
%%tab pytorch
#@save
class PackedStream:
    """Windows of seq_len + 1 tokens at offsets i * seq_len of a shard set.
    Step s takes slice s of a permutation seeded by (seed, epoch); rank r
    takes every world-th window of it. Targets after an EOS are -100."""
    def __init__(self, shards, seq_len, global_batch, seed=0, rank=0, world=1,
                 epoch=0, skip=()):
        if global_batch % world:
            raise ValueError(f'global_batch {global_batch} is not a multiple'
                             f' of world {world}')
        self.shards, self.seq_len, self.global_batch = (shards, seq_len,
                                                        global_batch)
        self.seed, self.rank, self.world, self.epoch = seed, rank, world, epoch
        self.skipped = tuple(sorted(set(skip)))
        self.n_seq = (shards.n_tokens - 1) // seq_len
        self.steps_per_epoch = self.n_seq // global_batch
        self.layout_hash, self.step = d2l._layout_hash(shards), 0

    def batch(self, step):
        idx = d2l._window_indices(self, self.physical_index(step))
        W = torch.from_numpy(d2l._windows(self, idx[self.rank::self.world]))
        X, Y = W[:, :-1].contiguous(), W[:, 1:].clone()
        Y[X == self.shards.eos_id] = -100  # the next token starts a document
        doc_ids, pos_ids = doc_ids_from_eos(X.int(), self.shards.eos_id)
        return dict(input_ids=X, labels=Y, doc_ids=doc_ids, pos_ids=pos_ids)

    def __iter__(self):
        while True:
            self.step += 1
            yield self.batch(self.step - 1)

    def physical_index(self, step):
        """The step of the unskipped schedule that logical step uses."""
        for k in self.skipped:
            step += k <= step
        return step

    def skip(self, step):
        """Drop physical step `step` (a bad batch); later steps move up."""
        self.skipped = tuple(sorted(set(self.skipped) | {step}))

    def state_dict(self):
        return dict(version=1, seed=self.seed, epoch=self.epoch,
                    step=self.step, skip=list(self.skipped),
                    layout_hash=self.layout_hash, seq_len=self.seq_len,
                    global_batch=self.global_batch)

    def load_state_dict(self, state):
        for key in ('layout_hash', 'seq_len', 'global_batch'):
            if state[key] != getattr(self, key):
                raise ValueError(f'{key} {getattr(self, key)} != '
                                 f'{state[key]}')
        self.seed, self.epoch, self.step = (state['seed'], state['epoch'],
                                            state['step'])
        self.skipped = tuple(state['skip'])
```
