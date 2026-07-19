# Ch. 9-10 modernization: frozen `#@save` API stubs (Phase 1)

*2026-07-12. The four shared library components the handoff (§5.6) requires,
frozen here so Phase 2-3 section writers code against stable signatures.
Namespace checked against d2l/torch.py + d2l/jax.py: no collisions (`bleu`,
`Vocab`, `tokenize` exist and are untouched; nothing named `BPETokenizer`,
`chrf`, `associative_scan`, `sample_next`, `generate`, `beam_search`).*

Conventions: pure-Python components live in **untagged** `#@save` cells (land in
all four `d2l/<fw>.py`); the scan is **pytorch-tagged** (lands only in
d2l/torch.py; JAX uses `jax.lax.associative_scan` natively in-section). After any
`#@save` change: `make lib`, and remember library rebuilds can affect notebooks
outside the edited file.

## 1. `BPETokenizer` — host: text-sequence.md (9.2); consumers: 9.3, 9.5, 9.7, 10.2, 10.4

```python
class BPETokenizer:  #@save
    """Byte-level BPE tokenizer trained by iterated most-frequent-pair merges.

    Token ids 0-255 are raw bytes; learned merges get ids 256..vocab_size-1;
    special tokens sit above those and are never produced by BPE itself.
    `pattern` is an optional pre-tokenization regex (e.g. GPT-2's): text is
    split into chunks and merges never cross chunk boundaries. Encoding applies
    learned merges greedily by rank, so train and inference agree by
    construction."""

    GPT2_PATTERN = r"..."   # GPT-2's actual pre-tokenization regex, annotated in 9.2

    def __init__(self, vocab_size=1024, pattern=None,
                 specials=('<pad>', '<bos>', '<eos>')): ...
    def train(self, text): ...
        # learns self.merges: dict[(int, int) -> int] (pair -> merged id; rank = id order)
        # and self.vocab: dict[int -> bytes]
    def encode(self, text, allow_special=False) -> "list[int]": ...
    def decode(self, ids) -> str: ...
    def __len__(self): ...              # vocab size including specials
    @property
    def pad(self) -> int: ...           # likewise bos, eos (10.2 batching needs these)
    @classmethod
    def from_tiktoken(cls, name) -> "BPETokenizer": ...
        # load a published bytes->rank table (e.g. 'gpt2') into OUR encoder;
        # the 9.2 verification moment reproduces tiktoken output token-for-token
```

The legacy `Vocab` class survives (9.3 uses word tokens for n-gram readability);
do not merge the two. `BPETokenizer` state must be picklable so 10.2/10.4 can
retrain-or-reuse cheaply.

## 2. Decoding helpers — host: decoding.md (9.7); consumers: 9.5 (preview), 10.2, 10.4

Framework-agnostic: strategies operate on **numpy logits**; each framework tab
wraps its model in a one-line `step_fn`.

```python
def sample_next(logits, strategy='greedy', temperature=1.0,
                k=None, p=None, min_p=None, rng=None) -> int:  #@save
    """Choose the next token id from a 1-D numpy logits array.
    strategy: 'greedy' | 'sample' (with optional top-k / top-p / min-p
    truncation applied to the temperature-scaled distribution)."""

def generate(step_fn, prefix, num_tokens, eos_id=None, **strategy) -> "list[int]":  #@save
    """Autoregressive generation. step_fn(ids: list[int]) -> numpy logits for
    the next token. Returns prefix + continuation (stops early on eos_id)."""

def beam_search(step_fn, prefix, num_tokens, beam_size=4, alpha=0.75,
                eos_id=None) -> "list[tuple[float, list[int]]]":  #@save
    """Length-normalized beam search (score = log P / len^alpha).
    Returns candidates sorted best-first; [0][1] is the argmax sequence."""
```

`rng` is a `numpy.random.Generator` (callers seed explicitly; keeps captures
reproducible). The 9.7 overlay figure uses `sample_next`'s truncation logic
directly, so implement truncation as a pure function of the distribution.

## 3. `associative_scan` (pytorch) — host: ssm.md (10.3); consumers: 10.3, 10.4

```python
def associative_scan(a, b, dim=0):  #@save  (%%tab pytorch cell)
    """Parallel prefix scan for the diagonal affine recurrence
    h_t = a_t * h_{t-1} + b_t  (elementwise), with h_{-1} = 0.
    a, b: same-shaped tensors with time along `dim`. Log-depth Blelloch
    combine ((a2, b2) o (a1, b1) = (a1*a2, a2*b1 + b2)); differentiable;
    returns all h_t. Teaching implementation: O(T log T) work, log-depth."""
```

Must be numerically verified against the sequential loop in-section (10.3 does
this explicitly). JAX sections use `jax.lax.associative_scan` with the same
combine; TF (if kept) implements the same algorithm in-section, not in d2l.

## 4. `chrf` — host: seq2seq.md (10.2); consumers: 10.2 (+ exercises)

```python
def chrf(pred, label, n=6, beta=2):  #@save
    """chrF (Popovic 2015): character n-gram F-score, the WMT-recommended
    lexical MT metric. Whitespace-stripped character n-grams of orders 1..n;
    per-order precision/recall, macro-averaged, combined with beta=2 (recall-
    weighted). ~10 lines, pure Python. Existing d2l.bleu stays (demoted in
    prose to a remark)."""
```

## Sequencing reminder (handoff §9)

9.2 must land and `make lib` run before 9.3/9.5/9.7 code that imports
`BPETokenizer`; 9.7 before 10.2/10.4 use of `generate`/`beam_search`; 10.3's
scan before 10.4. Only one agent edits d2l.bib / _quarto.yml /
CHAPTER_NUMBERING at a time (orchestrator does those serially).
