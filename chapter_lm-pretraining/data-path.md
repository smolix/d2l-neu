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
<!-- Placeholder (prose): opening: the afternoon run of sec_gpt sliced one text file; two silent failures of a real data path, one number each. -->

```{.python .input #data-path-opening-leak}
%%tab pytorch
torch.manual_seed(0)
vocab, lengths = 128, torch.tensor([5, 4, 3])  # three documents, one window
T = int(lengths.sum())
model = d2l.PackedGPT(vocab, num_hiddens=64, num_heads=4, num_kv_heads=2,
                      num_blks=2, max_len=T, backend='sdpa').eval()
X = torch.randint(1, vocab, (1, T))
X[0, lengths.cumsum(0) - 1] = 0  # each document ends in EOS (id 0)
doc = torch.repeat_interleave(torch.arange(len(lengths)), lengths)
same = doc[:, None] == doc[None, :]  # query and key in the same document
causal = torch.ones(T, T, dtype=torch.bool).tril()
print('\n'.join(''.join('1' if m else '.' for m in row)
                for row in same & causal))
pos = torch.cat([torch.arange(n) for n in lengths.tolist()])
doc2 = slice(int(lengths[0]), int(lengths[:2].sum()))  # second document
with torch.no_grad():
    alone = model(X[:, doc2])
    masked = model(X, d2l.PackCtx(doc[None], pos[None]))[:, doc2]
    unmasked = model(X)[:, doc2]  # causal attention across the window
for name, logits in (('masked:  ', masked), ('unmasked:', unmasked)):
    print(f'max |logit change| of doc 2, {name} '
          f'{(logits - alone).abs().max():.1e}')
```

<!-- Placeholder (prose): the second failure: a restart that re-seeds its shuffle. -->

```{.python .input #data-path-opening-restart}
%%tab pytorch
r = d2l.reseeded_restart_overlap(n_seq=10_000, batch=64, crash_step=40,
                                 steps=50)
print(f're-seeded restart: {r.naive_match} of {r.steps} later batches as '
      f'planned, {r.naive_seen} of {r.windows} windows repeated')
print(f'stateless cursor: {r.stateless_match} of {r.steps} later batches as '
      f'planned, {r.stateless_seen} repeated')
```

<!-- Placeholder (prose): the contract: batch(step) is a pure function of (manifest, seed, epoch, step, skip); exactness levels L1-L3 (tab_lm_pdp_exactness); dataset reproducibility is not run reproducibility (sec_repro). -->

## The Corpus as Shards

<!-- Placeholder (prose): the d2l-corpus/1 layout of 20.6 (flat uint16 tokens, uint32 document offsets, meta.parquet, manifest with SHA-256 per file); uint16 wraps above 65,535; HF datasets streaming and torchdata named once. -->

```{.python .input #data-path-shards}
%%tab pytorch
train, val = d2l.reference_shards('100m'), d2l.reference_shards('val')
for name, shards in (('train', train), ('val', val)):
    print(f'{name}: {shards.n_tokens:,} tokens in {shards.n_docs:,} '
          f'documents ({shards.n_shards} shard, {shards.dtype.name})')
ell = np.diff(train.offsets(0))  # document lengths, EOS included
print(f'mean document length: {ell.mean():.0f} tokens '
      f'(median {np.median(ell):.0f})')
```

## Packing with Document Masks

### Document Ids and Positions

<!-- Placeholder (prose): conventions (tab_lm_pdp_conventions): the EOS ends and belongs to its document and is a predicted target; doc_ids count the EOS strictly before t; pos_ids restart at each document start and at a fragment start; a literal <|endoftext|> in text is encoded as text by the R2 tokenizer (20.6). -->

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
<!-- Placeholder (prose): the fixture of the plan. -->

```{.python .input #data-path-doc-ids-fixture}
%%tab pytorch
doc_ids, pos_ids = doc_ids_from_eos(torch.tensor([5, 6, 0, 7, 0, 0, 8]))
print('doc_ids:', doc_ids.tolist())
print('pos_ids:', pos_ids.tolist())
```

### Packed Attention

<!-- Placeholder (prose): restate in one sentence each: the mask as the conjunction of same-document and causal (sec_attention-scoring-functions); FlexAttention's mask_mod and create_block_mask (sec_windowed-linear-attention), with the SDPA boolean mask as the CPU path; fig_lm_packing_mask. FlexAttention retraces the block mask when B or T change (the library caches it per (B, T, device)). -->

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
    def __init__(self, num_hiddens, num_heads, num_kv_heads, *,
                 backend='auto', softcap=None, **kwargs):
        super().__init__(num_hiddens, num_heads, num_kv_heads, **kwargs)
        self.backend, self.softcap = backend, softcap

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
<!-- Placeholder (prose): the packed model: RoPE angles from per-document positions, a learned table indexed by pos_ids for the reset check, the mask built once per forward; adopt/from_arch/set_rope/forward_cached named in a sentence each. -->

```{.python .input #data-path-packed-gpt}
%%tab pytorch
#@save
class PackedGPT(d2l.Qwen3GPT):
    """Qwen3GPT for pretraining on packed windows: forward(X, ctx) with
    per-document positions and the document mask, and the final hidden
    states for a chunked loss (return_hidden=True)."""
    def __init__(self, vocab_size, num_hiddens, num_heads, num_kv_heads,
                 num_blks, *, pos='rope', backend='auto', softcap=None,
                 qk_norm=False, rope_theta=1e4, max_len=4096, **kwargs):
        self.packed_options = dict(rope=pos == 'rope', backend=backend,
                                   softcap=softcap)
        super().__init__(vocab_size, num_hiddens, num_heads, num_kv_heads,
                         num_blks, qk_norm=qk_norm, rope_theta=rope_theta,
                         max_len=max_len, **kwargs)
        self.pos = pos
        if pos == 'learned':  # an absolute table, indexed by pos_ids
            self.pos_emb = nn.Embedding(max_len, num_hiddens)
            nn.init.normal_(self.pos_emb.weight, std=0.02)

    def CausalAttention(self, d, H, *, bias, rope, num_kv_heads):
        return PackedAttention(d, H, num_kv_heads, bias=bias,
                               **self.attn_options, **self.packed_options)

    def forward(self, X, ctx=None, return_hidden=False, logits_to_keep=None):
        pos = (ctx.pos_ids if ctx is not None
               else torch.arange(X.shape[1], device=X.device))
        H, tables = self.token_emb(X), None
        if self.pos == 'learned':
            H = H + self.pos_emb(pos.long())
        else:  # RoPE angles of each token's position within its document
            tables = d2l._packed_rope_tables(self, pos)
        mask = d2l._packed_mask(self, ctx, X)  # once for every block
        for blk in self.blks:
            H = blk(H, mask, tables=tables, ctx=ctx)
        if logits_to_keep is not None:
            H = H[:, -logits_to_keep:]
        H = self.norm(H)
        W = self.lm_head.weight if hasattr(self, 'lm_head') else \
            self.token_emb.weight
        return (H, W) if return_hidden else F.linear(H, W)
```
<!-- Placeholder (prose): predict before running: with RoPE, does resetting positions change the logits within a document? Derivation D1 (induction over layers) follows the check. -->

```{.python .input #data-path-packed-equals-alone}
%%tab pytorch
vocab, lengths = 128, torch.tensor([5, 9])  # two documents, one window
T, n_1 = int(lengths.sum()), int(lengths[0])
torch.manual_seed(0)
X = torch.randint(1, vocab, (1, T))
X[0, lengths.cumsum(0) - 1] = 0  # EOS ends each document
doc_ids = torch.repeat_interleave(torch.arange(2), lengths)[None]
reset = torch.cat([torch.arange(n) for n in lengths.tolist()])[None]
no_reset = torch.arange(T)[None]
settings = {  # name: (position scheme, pos_ids, document mask)
    'RoPE, reset': ('rope', reset, True),
    'RoPE, no reset': ('rope', no_reset, True),
    'RoPE, unmasked': ('rope', no_reset, False),
    'learned, reset': ('learned', reset, True),
    'learned, no reset': ('learned', no_reset, True)}
delta = {}
for name, (pos, pos_ids, masked) in settings.items():
    torch.manual_seed(0)
    net = PackedGPT(vocab, num_hiddens=64, num_heads=4, num_kv_heads=2,
                    num_blks=2, max_len=T, pos=pos, backend='sdpa').eval()
    one_doc = torch.zeros_like(doc_ids)  # unmasked: one document
    ctx = PackCtx(doc_ids if masked else one_doc, pos_ids)
    with torch.no_grad():
        alone = torch.cat([net(X[:, :n_1]), net(X[:, n_1:])], 1)
        delta[name] = (net(X, ctx) - alone).abs().max().item()
print(f'{"setting":<18} max |Δ|')
for name, value in delta.items():
    print(f'{name:<18} {value:.1e}')
exact = ('RoPE, reset', 'RoPE, no reset', 'learned, reset')
assert max(delta[k] for k in exact) <= 1e-5
assert min(delta['RoPE, unmasked'], delta['learned, no reset']) >= 1e-3
```

### How Many Documents a Window Cuts

<!-- Placeholder (prose): derivation D2: a document of length l is cut with probability min(1, (l-1)/T), so f_cut(T) = E[l min(1, (l-1)/T)] / E[l]; predicted from the histogram alone, measured from the real offsets of the training shard (the stream the run packs; on the 8.4M-token validation shard one layout at T = 8,192 misses by about 2%, which is sampling noise over window origins, not a flaw of the formula); best-fit packing as the alternative (an exercise). -->

```{.python .input #data-path-window-cuts}
%%tab pytorch
offsets = train.offsets(0).astype(np.int64)  # the stream the run packs
ell, starts = np.diff(offsets), offsets[:-1]  # document lengths and starts
print(f'{"T":>5} {"predicted":>10} {"measured":>9}')
for T in (256, 512, 1024, 2048, 4096, 8192):
    predicted = (ell * np.minimum(1, (ell - 1) / T)).sum() / ell.sum()
    cut = starts // T != (starts + ell - 1) // T  # a window boundary inside
    print(f'{T:>5} {predicted:>10.3f} {ell[cut].sum() / ell.sum():>9.3f}')
edges = np.logspace(np.log10(ell.min()), np.log10(ell.max()), 40)
counts, _ = np.histogram(ell, edges)
d2l.plot(np.sqrt(edges[:-1] * edges[1:]), [counts],
         'document length (tokens)', 'documents', xscale='log')
```

## The Token-Weighted Loss

<!-- Placeholder (prose): the loss is a sum over valid targets divided by the global count; masking attention but not the loss predicts the first token of a document from the previous one's EOS (the -100 rule); chunked_lm_loss keeps the [tokens, vocabulary] logits out of memory. -->

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
def lm_loss_terms(h, W, y, ignore_index=-100):
    """Summed NLL and summed logsumexp^2 over the valid targets y, from the
    fp32 logits of hidden states h under output weights W."""
    logits = F.linear(h, W)
    logits = logits.to(torch.promote_types(logits.dtype, torch.float32))
    lse, valid = torch.logsumexp(logits, -1), y != ignore_index
    nll = lse - logits.gather(-1, y.clamp(min=0)[:, None])[:, 0]
    return torch.stack([torch.where(valid, nll, 0.0).sum(),
                        torch.where(valid, lse ** 2, 0.0).sum()])

#@save
def chunked_lm_loss(hidden, weight, labels, chunk=8192, z_loss=0.0,
                    ignore_index=-100):
    """token_weighted_loss(F.linear(hidden, weight), labels), chunk
    positions at a time: each chunk's logits are recomputed in the backward
    pass instead of stored. z_loss adds z_loss * logsumexp^2 per valid
    target (the z-loss of a pretraining run)."""
    H, y = hidden.flatten(0, -2), labels.flatten()
    nll, z = sum(torch.utils.checkpoint.checkpoint(
        lm_loss_terms, H[i:i + chunk], weight, y[i:i + chunk], ignore_index,
        use_reentrant=False) for i in range(0, len(y), chunk))
    return nll + z_loss * z, (y != ignore_index).sum()
```
<!-- Placeholder (prose): derivation D3: mean of micro-batch means minus the token-weighted loss is -Cov(n_k / n-bar, l-bar_k); the two-micro-batch example, then the eight micro-batches of the plan. -->

```{.python .input #data-path-token-weighting}
%%tab pytorch
n_k, ell_k = torch.tensor([1024., 300.]), torch.tensor([3.0, 4.0])
print(f'two batches: token-weighted {(n_k * ell_k).sum() / n_k.sum():.2f}, '
      f'mean of means {ell_k.mean():.2f}')
vocab = 128
torch.manual_seed(0)
counts = [1024, 900, 512, 1024, 300, 1000, 700, 1024]  # valid targets
batches = [(torch.randn(n, vocab, dtype=torch.float64),  # fp64: exact checks
            torch.randint(0, vocab, (n,))) for n in counts]
N = sum(counts)
sums = [token_weighted_loss(Z, y)[0] for Z, y in batches]
single = token_weighted_loss(torch.cat([Z for Z, _ in batches]),
                             torch.cat([y for _, y in batches]))[0] / N
accumulated = sum(sums) / N
ell_k = torch.stack(sums) / torch.tensor(counts)  # mean loss per batch
w = torch.tensor(counts, dtype=torch.float64) / (N / len(counts))  # n/n̄
cov = ((w - 1) * (ell_k - ell_k.mean())).mean()
print(f'eight batches: accumulated − single pass = {accumulated - single:.1e}')
print(f'mean of means − single pass = {ell_k.mean() - single:.4f}')
print(f'predicted −Cov(n/n̄, l̄) = {-cov:.4f}')
assert abs(accumulated - single) <= 1e-6
assert abs(ell_k.mean() - single + cov) <= 1e-6
```

<!-- Placeholder (prose): what chunking saves, at the t1 widths (the t1 preset of the next section): one window of 8 x 1,024 positions against a 32,768-token vocabulary. -->

```{.python .input #data-path-chunked-memory}
%%tab pytorch
B, T, d, vocab = 8, 1024, 384, 32768  # t1 widths, the R2 vocabulary
if torch.cuda.is_available():
    torch.manual_seed(0)
    h = torch.randn(B, T, d, device='cuda', requires_grad=True)
    W = (0.02 * torch.randn(vocab, d, device='cuda')).requires_grad_()
    y = torch.randint(0, vocab, (B, T), device='cuda')
    losses = {  # the mean NLL, computed both ways
        'full logits': lambda: F.cross_entropy(F.linear(h, W).flatten(0, 1),
                                               y.flatten()),
        'chunked (1,024 positions)': lambda: chunked_lm_loss(
            h, W, y, chunk=1024)[0] / y.numel()}
    kept = {}

    def forward_backward(name):  # keeps the loss and gradients to compare
        loss = losses[name]()
        kept[name] = (loss.detach(), torch.autograd.grad(loss, (h, W)))
    for name in losses:  # memory above the inputs, forward and backward
        gib = d2l.peak_memory(forward_backward, name) / 2**30
        print(f'{name}: {gib:.1f} GiB peak')
    (l_full, g_full), (l_chunk, g_chunk) = kept.values()
    grad_gap = max((a - b).abs().max() for a, b in zip(g_full, g_chunk))
    print(f'max |Δ| loss / grad: {(l_full - l_chunk).abs():.1e} / '
          f'{grad_gap:.1e}')
else:
    print(f'GPU not available; the full logits alone take '
          f'{B * T * vocab * 4 / 2**30:.1f} GiB (B·T·V·4 bytes)')
```

## A Stream That Names Its Batches

<!-- Placeholder (prose): stateless addressing: perm = default_rng([seed, epoch]).permutation(n_seq), step s takes perm[sG:(s+1)G], rank r takes every world-th window; state_dict is the consumed count plus the layout hash; fig_lm_data_cursor. DataLoader workers that fork one RNG duplicate batches; the book's d2l.DataModule is the stack this replaces for pretraining. -->

```{.python .input #data-path-packed-stream}
%%tab pytorch
#@save
class PackedStream:
    """Windows of seq_len + 1 tokens at offsets i * seq_len of a shard set.
    Step s takes slice s of a permutation seeded by (seed, epoch), with the
    steps in skip removed; rank r takes every world-th window of it.
    Targets after an EOS are -100."""
    def __init__(self, shards, seq_len, global_batch, seed=0, rank=0, world=1,
                 epoch=0, skip=()):
        self.shards, self.seq_len, self.global_batch = (shards, seq_len,
                                                        global_batch)
        self.seed, self.rank, self.world, self.epoch = seed, rank, world, epoch
        self.skipped = tuple(sorted(set(skip)))
        self.n_seq = (shards.n_tokens - 1) // seq_len
        self.steps_per_epoch = self.n_seq // global_batch
        self.layout_hash, self.step = d2l._layout_hash(shards), 0

    def batch(self, step):
        idx = d2l._window_indices(self, step)  # step's permutation slice
        W = torch.from_numpy(d2l._windows(self, idx[self.rank::self.world]))
        X, Y = W[:, :-1].contiguous(), W[:, 1:].clone()
        Y[X == self.shards.eos_id] = -100  # the next token starts a document
        doc_ids, pos_ids = doc_ids_from_eos(X.int(), self.shards.eos_id)
        return dict(input_ids=X, labels=Y, doc_ids=doc_ids, pos_ids=pos_ids)

    def __iter__(self):  # the cursor: step counts the batches consumed
        while True:
            self.step += 1
            yield self.batch(self.step - 1)

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
<!-- Placeholder (prose): resume, elasticity, skip and the layout check, on the training shard at the t1 window and batch. -->

```{.python .input #data-path-resume}
%%tab pytorch
T, G = 1024, 64  # the t1 window and global batch (sequences)
stream = PackedStream(train, T, G, seed=0)
stream.step = 37  # the run dies here; a fresh process resumes
resumed = PackedStream(train, T, G, seed=1)  # the state restores seed 0
resumed.load_state_dict(stream.state_dict())
identical = sum(torch.equal(stream.batch(37 + s)['input_ids'],
                            b['input_ids'])
                for s, b in zip(range(50), resumed))
print(f'resume at 37: {identical} of 50 batches identical')

def windows(world):  # the windows of steps 0-9 on all ranks, sorted
    ranks = [PackedStream(train, T, G, rank=r, world=world)
             for r in range(world)]
    return sorted(row.numpy().tobytes() for s in range(10) for rank in ranks
                  for row in rank.batch(s)['input_ids'])
one, two, four = windows(1), windows(2), windows(4)
print(f'world sizes 1, 2, 4: {"same" if one == two == four else "different"}'
      f' {len(one)} windows at 10 steps')
skipped = PackedStream(train, T, G, skip=(3,))
physical = next(p for p in range(10) if torch.equal(
    skipped.batch(3)['input_ids'], stream.batch(p)['input_ids']))
print(f'skip=(3,): logical step 3 = physical step {physical}')
try:
    stream.load_state_dict(PackedStream(train, 2 * T, G).state_dict())
except ValueError as e:
    print(f'layout check: ValueError: {e}')
assert identical == 50 and one == two == four and physical == 4
```

### Data Parallelism

<!-- Placeholder (prose): the DDP form: the step's valid-target count is all-reduced before the backward pass, and each rank scales its summed loss by world / N; two simulated ranks with unequal counts. -->

```{.python .input #data-path-ddp}
%%tab pytorch
def ddp_loss(sum_loss, n_valid_global, world):
    """One rank's loss: DDP averages the ranks' gradients, so scaling by
    world over the global count of valid targets yields the gradient of
    the global token-weighted mean."""
    return sum_loss * world / n_valid_global

vocab = 128
torch.manual_seed(0)
W = torch.randn(vocab, 16, requires_grad=True)  # logits h W^T
ranks = [(torch.randn(n_r, 16) + r, torch.randint(0, vocab, (n_r,)))
         for r, n_r in enumerate((1000, 300))]  # unequal valid counts
N, world = sum(len(y) for _, y in ranks), len(ranks)

def ddp_grad(loss_of):  # DDP: the mean of the ranks' gradients
    return torch.stack([torch.autograd.grad(loss_of(
        *token_weighted_loss(h @ W.T, y)), W)[0]
        for h, y in ranks]).mean(0)
sum_all, n_all = token_weighted_loss(torch.cat([h for h, _ in ranks]) @ W.T,
                                     torch.cat([y for _, y in ranks]))
single = torch.autograd.grad(sum_all / n_all, W)[0]
glob = ddp_grad(lambda s, n: ddp_loss(s, N, world))
local = ddp_grad(lambda s, n: s / n)  # each rank divides by its own count
print(f'global count: max |Δ grad| = {(glob - single).abs().max():.1e}')
print(f'local count: max |Δ grad| = {(local - single).abs().max():.4f}')
assert (glob - single).abs().max() <= 1e-6
assert (local - single).abs().max() >= 1e-3
```

## Measurements

### Four Pipelines

<!-- Placeholder (prose): useful tokens per second of the t1 training step under four input pipelines (padded documents, packed without mask, SDPA mask, FlexAttention mask) at T = 1,024 and 4,096, the same tokens per step; a difference is stated only where the IQRs do not overlap; flash_attn_varlen_func with cu_seqlens named as the other production route. -->

```{.python .input #data-path-throughput}
%%tab pytorch
tp = d2l.packing_throughput(
    d2l.r2_config('t1'), pipelines=('padded', 'packed', 'sdpa-mask',
                                    'flex-mask'),
    seq_lens=(1024, 4096), steps=50)
d2l.plot_intervals(
    tp.pipeline.unique(),
    {f'T = {T}': (g.tok_s / 1e3, g.tok_s_q1 / 1e3, g.tok_s_q3 / 1e3)
     for T, g in tp.groupby('T')},
    'pipeline', 'useful tokens/s (thousands)')
print(f'{"pipeline":<10} {"T":>5} {"useful tok/s":>13} {"IQR":>10} '
      f'{"padding":>8} {"peak GiB":>9}')
for r in tp.itertuples():  # tokens/s in thousands: median, quartiles
    iqr = f'{r.tok_s_q1 / 1e3:.0f}-{r.tok_s_q3 / 1e3:.0f}k'
    row = (f'{r.tok_s / 1e3:>12.0f}k {iqr:>10} {r.padding:>8.0%} '
           f'{r.peak_gib:>9.1f}' if r.status == 'ok' else f'{r.status:>13}')
    print(f'{r.pipeline:<10} {r.T:>5} {row}')
```

### Data Wait

<!-- Placeholder (prose): input starvation measured as the time to build one batch against the step it feeds (flex-mask, T = 1,024, from the table above); the cost, if any, is reading and the doc-id cumsum, not packing. -->

```{.python .input #data-path-data-wait}
%%tab pytorch
batches = iter(PackedStream(train, 1024, 64))  # the t1 window, batch
t_batch = d2l.Benchmark(lambda: next(batches), warmup=5, repeats=50).time
t_step = tp.set_index(['pipeline', 'T']).step_ms[('flex-mask', 1024)] / 1e3
print(f'stream: {1e3 * t_batch:.1f} ms per batch (64 × 1,024 tokens)')
print(f't1 training step ({tp.gpu.iloc[0]}): {1e3 * t_step:.0f} ms; '
      f'data wait {t_batch / t_step:.1%} of a step')
```

### Does the Mask Change the Model?

<!-- Placeholder (prose): six runs of the t1 preset for 25M tokens: masked and unmasked, seeds 0-2, a pair sharing its batches; both validated document by document (with the mask), as at inference; paired differences with a t interval on two degrees of freedom, and how weak that is. -->

```{.python .input #data-path-mask-runs}
%%tab pytorch
runs = d2l.packing_comparison(d2l.r2_config('t1'), tokens=25_000_000,
                              seeds=(0, 1, 2))
for r in runs.table.itertuples():
    print(f'seed {r.seed}: masked {r.masked:.3f}, unmasked {r.unmasked:.3f}')
diff = runs.table.unmasked - runs.table.masked
mean, half = d2l.t_interval(diff)
print(f'paired difference (unmasked − masked): {mean:.3f} ± {half:.3f} '
      f'(95% t, {len(diff) - 1} dof)')
print('no conclusion' if abs(mean) <= half else
      f'unmasked {"higher" if mean > 0 else "lower"}: the interval excludes 0')
```

<!-- Placeholder (prose): the leak: attention mass a query of the unmasked model puts on earlier keys of other documents, per block, on 64 validation windows (seed 0); the dashed line is the share uniform attention would give. A softmax gives every visible key some mass, so the leak is positive in any model; what training changes is whether a block puts less than the uniform share there (it learns to look away from other documents) or not. -->

```{.python .input #data-path-leak}
%%tab pytorch
leak = runs.leak
axes = d2l.plot_intervals(leak.layer + 1,
                          {'unmasked, seed 0': (leak.leak, leak.lo, leak.hi)},
                          'block', 'attention mass on other documents',
                          reference=leak.uniform.iloc[0])  # uniform attention
axes.set_ylim(bottom=0)
uniform = leak.uniform.iloc[0]
below = (leak.layer[leak.hi < uniform] + 1).tolist()  # 95% interval below
print(f'uniform-attention share {uniform:.2f}; leak below it in '
      f'{len(below)} of {len(leak)} blocks {below}')
print('leak / uniform share, block 1 to {}: {}'.format(
    len(leak), ' '.join(f'{x:.2f}' for x in leak.leak / uniform)))
```

## Summary

<!-- Placeholder (prose): 3-6 sentences (W1). -->

## Exercises

<!-- Placeholder (prose): 5-7 exercises per S/spec/21.1.md part H. -->
