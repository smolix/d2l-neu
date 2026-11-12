#!/usr/bin/env python3
"""Generate the illustrative figures of ``chapter_lm-pretrained`` (Chapter 18,
"Working with Pretrained Language Models") in the shared house style.

Nine figures (spec part 3.2), all written to ``img/mdl-lm-pretrained-<id>.svg``:

  hub-repo                What a model repository holds at one commit
                          (Qwen/Qwen3.5-2B at 15852e8c) and which loader reads
                          each file; the two range requests of the header read.
  layer-layout            Layer types and per-layer cache of Qwen3.5-2B and
                          Gemma 4 E2B (from the committed configs); Gemma's
                          KV-shared layers 15-34.
  weight-map              Qwen3-0.6B tensor names and shapes against
                          d2l.Qwen3GPT names; the RoPE permutation for d_h = 8.
  chat-template           The tokens of a chat prompt (Qwen3-1.7B at 70d244cc)
                          and of the plain layout; the label token in each.
  scoring-conventions     One multiple-choice item under cloze, letter and
                          generative scoring (Qwen3-0.6B-Base token splits).
  benchmark-lifecycle     Schematic lifecycle of a benchmark (figstyle.mpl).
  patching                The five intervention sites of a pre-norm block and
                          noising one cell of the clean run from the corrupted one.
  sae-splice              A top-k sparse autoencoder spliced into the residual
                          stream; reconstruct, ablate, steer.
  api-lost                What each kind of endpoint lets a client measure.

Block diagrams use the deterministic SVG composer ``figstyle.svg`` (figure-style
guide section 7.2); the lifecycle curve schematic uses ``figstyle.mpl``
(section 7.3).  Literal data the figures hardcode (token ids, tensor names and
shapes, layer types) is checked: ``check_data()`` compares the layer types and
shapes with the committed configs in ``tools/lm/fixtures/hf_models/`` on every
run; ``tools/test_lm_prompting.py`` (``test_chat_figure_tokens``) re-renders
the chat-template conversation with the pinned tokenizer and compares its ids
with ``CHAT_ROWS``; ``tools/test_lm_figures.py`` re-tokenizes the scoring and
patching strings.  The hub-repo file names are those of the Hub API's file
listing of Qwen/Qwen3.5-2B at 15852e8c (checked 2026-11-12: no
generation_config.json at that commit, one shard).  The endpoint matrix of
``fig_api_lost`` states what each kind of endpoint can offer (``API_ROWS``),
not what one probe found.

Run with the repo's pytorch venv (fontTools and matplotlib):

    .venv-pytorch/bin/python tools/gen_mdl_lm_pretrained_figures.py

Byte-idempotent: no timestamps, no randomness, fixed svg.hashsalt.
"""

from __future__ import annotations

import json
import math
import os
import sys
from xml.sax.saxutils import escape as xml_escape

TOOLS = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, TOOLS)

from figstyle import tokens as T  # noqa: E402
from figstyle.svg import Figure, Span, _f  # noqa: E402
from figstyle.textmetrics import measure  # noqa: E402

PREFIX = "mdl-lm-pretrained"
FIXTURES = os.path.join(TOOLS, "lm", "fixtures", "hf_models")
WRITTEN: list[str] = []

STREAM, ATTN, FFN = T.ROLE["stream"], T.ROLE["attention"], T.ROLE["ffn"]
EMBED, STATE, HIGH = T.ROLE["embed"], T.ROLE["state"], T.ROLE["highlight"]

# --------------------------------------------------------------------------- #
# Literal data                                                                #
# --------------------------------------------------------------------------- #

# Qwen3-1.7B at revision 70d244cc86ccca08cf5af4e1e306ecf908b1ad5e:
# apply_chat_template([system, user], add_generation_prompt=True,
# enable_thinking=False), split into the figure's three rows.  Asserted by
# tools/test_lm_prompting.py::test_chat_figure_tokens.
CHAT_SYSTEM = "Name the topic of the headline."
CHAT_USER = "Headline: Local team wins cup final"
CHAT_REVISION = "70d244cc86ccca08cf5af4e1e306ecf908b1ad5e"
CHAT_ROWS = [
    [("<|im_start|>", 151644), ("system", 8948), ("\n", 198),
     ("Name", 675), (" the", 279), (" topic", 8544), (" of", 315),
     (" the", 279), (" headline", 31163), (".", 13), ("<|im_end|>", 151645),
     ("\n", 198)],
    [("<|im_start|>", 151644), ("user", 872), ("\n", 198), ("Head", 12346),
     ("line", 1056), (":", 25), (" Local", 8774), (" team", 2083),
     (" wins", 14816), (" cup", 10525), (" final", 1590),
     ("<|im_end|>", 151645), ("\n", 198)],
    [("<|im_start|>", 151644), ("assistant", 77091), ("\n", 198),
     ("<think>", 151667), ("\n\n", 271), ("</think>", 151668),
     ("\n\n", 271)],
]
# The four label words, as the first token after a newline (chat layout) ...
CHAT_LABELS = [("World", 10134), ("Sports", 40979), ("Business", 22727),
               ("Tech", 34097)]
# ... and the plain layout 'Headline: ...\nTopic:' with its next token.
PLAIN_TAIL = [(" final", 1590), ("\n", 198), ("Topic", 26406), (":", 25)]
PLAIN_LABEL = (" Sports", 13180)
SPECIAL_IDS = {151644, 151645, 151667, 151668}
ROLE_IDS = {8948, 872, 77091}

# Qwen3-0.6B-Base token splits of the scoring figure's continuations
# (printed by cell scoring-format-print of sec_lm_scoring).
SCORE_QUESTION = "Which gas do plants take in from the air to make their food?"
SCORE_OPTIONS = ["oxygen", "carbon dioxide", "nitrogen", "water vapor"]
SCORE_CLOZE = [[" oxygen"], [" carbon", " dioxide"], [" nitrogen"],
               [" water", " vapor"]]

# Qwen3-0.6B tokens of the first recall pair (lm-interp-recall-v1, row 1).
PATCH_CLEAN = ["The", " E", "iff", "el", " Tower", " is", " located", " in",
               " the", " city", " of"]
PATCH_CORRUPT = ["The", " Col", "os", "se", "um", " is", " located", " in",
                 " the", " city", " of"]
PATCH_SUBJECT_LAST = 4

# Qwen3-0.6B: Transformers tensor names and shapes against d2l.Qwen3GPT names.
QWEN3_06B = dict(V=151936, d=1024, H=16, H_kv=8, d_h=128, f=3072, L=28)


def _load_config(repo):
    with open(os.path.join(FIXTURES, repo, "config.json")) as fh:
        cfg = json.load(fh)
    return cfg.get("text_config", cfg)


def qwen35_layer_types():
    return _load_config("Qwen/Qwen3.5-2B")["layer_types"]


def gemma_layout():
    cfg = _load_config("google/gemma-4-E2B-it")
    return (cfg["layer_types"], cfg["num_kv_shared_layers"],
            cfg["sliding_window"])


def check_data():
    """Compare the literals above with the committed configs."""
    q = _load_config("Qwen/Qwen3-0.6B")
    assert (q["vocab_size"], q["hidden_size"], q["num_attention_heads"],
            q["num_key_value_heads"], q["head_dim"], q["intermediate_size"],
            q["num_hidden_layers"]) == tuple(QWEN3_06B.values()), q
    assert q["tie_word_embeddings"] is True
    types = qwen35_layer_types()
    assert len(types) == 24 and types.count("full_attention") == 6
    g_types, shared, window = gemma_layout()
    assert len(g_types) == 35 and shared == 20 and window == 512
    assert len(PATCH_CLEAN) == len(PATCH_CORRUPT)
    assert sum(len(r) for r in CHAT_ROWS) == 32


# --------------------------------------------------------------------------- #
# Helpers                                                                     #
# --------------------------------------------------------------------------- #

def mono(s, color=None):
    return Span(s, "m", color=color)


def tw(text, size, kind="m"):
    fam, sty = {"r": ("sans", "regular"), "b": ("sans", "bold"),
                "i": ("sans", "italic"), "m": ("mono", "regular")}[kind]
    return measure(text, size, fam, sty).width


def show_token(s):
    """Visible form of a token string: newline as ⏎, leading space as ␣."""
    if s and set(s) == {"\n"}:
        return [mono("⏎" * len(s))]
    if s.startswith(" "):
        return [mono("␣", T.FAINT), mono(s[1:])]
    return [mono(s)]


def spans_width(spans, size):
    return sum(tw(sp.text, size, sp.kind) for sp in spans)


def rot_text(f, x, y, spans, *, size=T.FS_SMALL, color=T.INK, angle=-90,
             anchor="end"):
    """Text rotated by ``angle`` degrees about (x, y)."""
    g = Figure(pad=0)
    x0, y0, x1, y1 = g.text(0, 0, spans, size=size, color=color,
                            anchor=anchor, va="middle")
    a = math.radians(angle)
    ca, sa = math.cos(a), math.sin(a)
    pts = [(x + u * ca - v * sa, y + u * sa + v * ca)
           for u in (x0, x1) for v in (y0, y1)]
    xs, ys = [p[0] for p in pts], [p[1] for p in pts]
    f.raw(f'<g transform="translate({_f(x)},{_f(y)}) rotate({_f(angle)})">'
          + "".join(g._parts) + "</g>", min(xs), min(ys), max(xs), max(ys))


def brace(f, x0, x1, y, *, down=True, color=T.INK, tick=6.0, sw=T.SW_HAIR):
    """A square bracket from x0 to x1 whose ends point up (down=True means
    the bracket hangs below what it gathers)."""
    s = -1 if down else 1
    f.polyline([(x0, y + s * tick), (x0, y), (x1, y), (x1, y + s * tick)],
               stroke=color, sw=sw)


def underlay(f, draw):
    """Run ``draw(f)`` and move what it drew beneath everything drawn so far
    (panels whose size depends on their content)."""
    n = len(f._parts)
    draw(f)
    new = f._parts[n:]
    del f._parts[n:]
    f._parts[0:0] = new


def save(f, name, desc):
    path = f.save(f"{PREFIX}-{name}", desc=xml_escape(desc))
    WRITTEN.append(path)


# --------------------------------------------------------------------------- #
# 18.1  fig_lm_hub_repo                                                       #
# --------------------------------------------------------------------------- #

def fig_hub_repo():
    """POINT: a repository is one commit of files; each file has one reader;
    the modeling code lives in the library; a shard's header is read alone."""
    f = Figure()
    fs = T.FS_SMALL
    files = [
        ("config.json", "cfg"),
        ("tokenizer.json", "tok"),
        ("tokenizer_config.json", "tok"),
        ("chat_template.jinja", "tok"),
        ("generation_config.json", "gen"),
        ("model.safetensors.index.json", "load"),
    ]
    fw = tw("model.safetensors.index.json", fs) + 2 * T.PAD_BLOCK_X
    x_files = 18.0
    pitch = 44.0
    y0 = 62.0
    rects = []
    for i, (name, _) in enumerate(files):
        dash = T.DASH_SOFT if name == "generation_config.json" else None
        rects.append(f.block(x_files + fw / 2, y0 + i * pitch, [mono(name)],
                             size=fs, min_w=fw, dash=dash))
    # the shard: its file name, then a header strip and a long tensor region
    shard_name = "model.safetensors-00001-of-00001.safetensors"
    y_sn = y0 + len(files) * pitch + 4
    f.text(x_files, y_sn, [mono(shard_name)], size=T.FS_MIN, anchor="start")
    sh_y, sh_h, hdr_w = y_sn + 14, 34.0, 78.0
    sh_w = max(fw, tw(shard_name, T.FS_MIN))
    f.rect(x_files, sh_y, hdr_w, sh_h, fill=HIGH.tint, stroke=HIGH.base,
           sw=T.SW_BOX, r=4)
    f.text(x_files + hdr_w / 2, sh_y + sh_h / 2, "header", size=fs)
    f.rect(x_files + hdr_w, sh_y, sh_w - hdr_w, sh_h, fill=T.PAPER,
           stroke=T.INK, sw=T.SW_BOX, r=4)
    f.text(x_files + hdr_w + (sh_w - hdr_w) / 2, sh_y + sh_h / 2,
           "tensor data", size=fs)
    shard_cy = sh_y + sh_h / 2
    # repository panel and its name with the commit as a tag
    pw = sh_w + 2 * x_files
    ph = sh_y + sh_h + 16
    underlay(f, lambda g: g.rect(0, 0, pw, ph, fill=T.PANEL, stroke="none",
                                 sw=0, r=T.R_PANEL))
    f.text(x_files, 24, [mono("Qwen/Qwen3.5-2B")], size=T.FS_LABEL,
           anchor="start")
    sha = "@ 15852e8c"
    sx = x_files + tw("Qwen/Qwen3.5-2B", T.FS_LABEL) + 12
    sw_ = tw(sha, fs) + 16
    f.rect(sx, 24 - 11, sw_, 22, fill=T.PAPER, stroke=T.MUTED, sw=T.SW_HAIR,
           r=11)
    f.text(sx + sw_ / 2, 24, [mono(sha)], size=fs, color=T.INK)

    # readers, inside the library panel; the model code below from_pretrained
    xm = pw + 26                       # merge column of the connectors
    lx0 = pw + 52
    cw = tw("from_pretrained", fs) + 2 * T.PAD_BLOCK_X + 8
    mod_name = "modeling_qwen3_5.py"
    cw = max(cw, tw(mod_name, fs) + 2 * T.PAD_BLOCK_X)
    ccx = lx0 + 18 + cw / 2

    def cy_of(i):
        return rects[i][1] + rects[i][3] / 2

    readers = {"cfg": ("AutoConfig", cy_of(0)),
               "tok": ("AutoTokenizer", cy_of(2)),
               "gen": ("generate", cy_of(4)),
               "load": ("from_pretrained", cy_of(5))}
    crect = {k: f.block(ccx, cy, [mono(name)], size=fs, min_w=cw)
             for k, (name, cy) in readers.items()}
    mr = f.block(ccx, shard_cy, [mono(mod_name)], size=fs, min_w=cw,
                 accent=T.SLATE)
    lr = crect["load"]
    f.arrow(ccx, mr[1], ccx, lr[1] + lr[3], stroke=T.INK)
    lib_x1 = lx0 + 36 + cw
    underlay(f, lambda g: g.rect(lx0, 0, lib_x1 - lx0, ph, fill="none",
                                 stroke=T.FAINT, sw=T.SW_HAIR, r=T.R_PANEL,
                                 dash=T.DASH_SOFT))
    f.text(lx0 + 18, 24, "LIBRARY", size=T.FS_TINY, color=T.INK,
           anchor="start", tracking=T.LETTERSPACE_CAPS)

    # file -> reader connectors (rectilinear, merged per reader)
    xr = x_files + fw
    for (name, key), r in zip(files, rects):
        cy = r[1] + r[3] / 2
        ty = crect[key][1] + crect[key][3] / 2
        dash = T.DASH_SOFT if key == "gen" else None
        if abs(cy - ty) < 0.5:
            f.arrow(xr, cy, crect[key][0], ty, stroke=T.INK, dash=dash)
        else:
            f.polyline([(xr, cy), (xm, cy), (xm, ty)], stroke=T.INK)
    ty = crect["tok"][1] + crect["tok"][3] / 2
    ty_load = crect["load"][1] + crect["load"][3] / 2
    f.polyline([(x_files + sh_w, shard_cy), (xm, shard_cy), (xm, ty_load)],
               stroke=T.INK)

    # the header read: two range requests, the emphasized path
    hx = x_files + hdr_w / 2
    hy = ph + 46
    hb = f.block(ccx, hy, [mono("d2l.safetensors_header")], size=fs,
                 accent=HIGH)
    f.ortho_arrow([(hx, sh_y + sh_h), (hx, hy), (hb[0], hy)],
                  stroke=HIGH.base, sw=T.SW_HEAVY)
    f.text(hx + 16, ph + 4, "2 range requests: the 8-byte length,",
           size=fs, anchor="start", va="top")
    f.text(hx + 16, ph + 22, "then the JSON header; no tensor data",
           size=fs, anchor="start", va="top")
    save(f, "hub-repo",
         "The files of the Hugging Face repository Qwen/Qwen3.5-2B at commit "
         "15852e8c and the loader that reads each: config.json to AutoConfig; "
         "tokenizer.json, tokenizer_config.json and chat_template.jinja to "
         "AutoTokenizer; the optional generation_config.json (absent at this "
         "commit) to generate; the index and the single safetensors shard to "
         "from_pretrained, which takes the model code from modeling_qwen3_5.py "
         "in the Transformers library. Two range requests read only the "
         "shard's JSON header.")


# --------------------------------------------------------------------------- #
# 18.1  fig_lm_hf_layer_layout                                                #
# --------------------------------------------------------------------------- #

SQ, SQ_GAP = 15.0, 4.0


def _layer_square(f, x, y, kind, shared=False):
    if kind == "full_attention":
        fill, edge = ATTN.base, ATTN.base
    elif kind == "sliding_attention":
        fill, edge = ATTN.tint, ATTN.base
    else:
        fill, edge = STATE.tint, STATE.dark
    f.rect(x, y, SQ, SQ, fill=fill, stroke=edge, sw=T.SW_HAIR, r=2)
    if shared:   # two diagonal hatch strokes
        c = T.PAPER if kind == "full_attention" else ATTN.dark
        for a in (0.42, 0.95):
            u = a * SQ
            f.line(x + 2, y + u, x + u, y + 2, stroke=c, sw=1.3)
        f.line(x + 0.45 * SQ, y + SQ - 2, x + SQ - 2, y + 0.45 * SQ,
               stroke=c, sw=1.3)


def _cache_glyph(f, x, y, kind):
    """Memory a layer keeps per sequence: a fixed state (teal square), a
    growing K/V cache (purple bar with an arrowhead), a capped cache."""
    cx = x + SQ / 2
    if kind == "linear_attention":
        f.rect(cx - 5, y, 10, 10, fill=STATE.base, stroke="none", sw=0)
    elif kind == "full_attention":
        f.line(cx, y, cx, y + 26, stroke=EMBED.base, sw=4.0)
        f._head(cx, y + 34, math.pi / 2, color=EMBED.base, length=9.0,
                width=9.0)
    else:
        f.line(cx, y, cx, y + 14, stroke=EMBED.base, sw=4.0)
        f.line(cx - 5, y + 15, cx + 5, y + 15, stroke=EMBED.base, sw=2.5)


def fig_layer_layout():
    """POINT: which layers keep a growing cache, a capped cache, a fixed state
    or nothing at all."""
    f = Figure()
    pitch = SQ + SQ_GAP
    x0 = 0.0
    fs = T.FS_SMALL
    # Row 1: Qwen3.5-2B
    q_types = qwen35_layer_types()
    y1 = 24.0
    f.text(x0, 6, [Span("Qwen3.5-2B", "b"), Span(f"   {len(q_types)} layers")],
           size=fs, anchor="start")
    for i, k in enumerate(q_types):
        _layer_square(f, x0 + i * pitch, y1, k)
        _cache_glyph(f, x0 + i * pitch, y1 + SQ + 6, k)
    # Row 2: Gemma 4 E2B
    g_types, n_shared, window = gemma_layout()
    n = len(g_types)
    first_shared = n - n_shared
    y2 = 166.0
    f.text(x0, y2 - 66, [Span("Gemma 4 E2B", "b"), Span(f"   {n} layers")],
           size=fs, anchor="start")
    # per-layer embedding table: a slim block left of the row, a bus under it
    bus_y = y2 + SQ + 6
    ple_x = x0 - 30
    f.rect(ple_x, y2 - 26, 12, bus_y - y2 + 40, fill=EMBED.tint,
           stroke=EMBED.base, sw=T.SW_HAIR, r=2)
    rot_text(f, ple_x - 10, (y2 - 26 + bus_y + 14) / 2,
             "per-layer embeddings", size=T.FS_MIN, color=T.INK,
             anchor="middle")
    f.line(ple_x + 12, bus_y, x0 + (n - 1) * pitch + SQ / 2, bus_y,
           stroke=EMBED.base, sw=T.SW_HAIR)
    for i, k in enumerate(g_types):
        sh = i >= first_shared
        x = x0 + i * pitch
        _layer_square(f, x, y2, k, shared=sh)
        f.line(x + SQ / 2, bus_y, x + SQ / 2, y2 + SQ, stroke=EMBED.base,
               sw=T.SW_HAIR)
        if not sh:
            _cache_glyph(f, x, bus_y + 6, k)
    for i in (0, first_shared - 1, first_shared, n - 1):
        f.text(x0 + i * pitch + SQ / 2, bus_y + 54, str(i), size=T.FS_MIN,
               color=T.INK)
    # KV sharing: layers 15-34 read the K/V of the last earlier layer of
    # their type (13 for sliding, 14 for full)
    last = {}
    for i in range(first_shared):
        last[g_types[i]] = i
    xs0 = x0 + first_shared * pitch
    xs1 = x0 + (n - 1) * pitch + SQ
    yb = y2 - 9
    brace(f, xs0, xs1, yb, down=False)
    stem = xs0 + 24
    yh = yb - 20
    jx_s = x0 + last["sliding_attention"] * pitch + SQ / 2
    jx_f = x0 + last["full_attention"] * pitch + SQ / 2
    f.polyline([(stem, yb), (stem, yh), (jx_s, yh)], stroke=T.INK,
               sw=T.SW_HAIR)
    for jx in (jx_s, jx_f):
        f.arrow(jx, yh - 0.75, jx, y2 - 2, stroke=T.INK, sw=T.SW_HAIR)
    f.text(stem + 12, yh, [Span("reuse the keys and values of layer "),
                           Span(str(last["sliding_attention"])),
                           Span(" or "), Span(str(last["full_attention"]))],
           size=fs, anchor="start")
    # legend: layer kinds, then memory kinds
    ly = bus_y + 84
    lx = x0
    for k, txt in (("full_attention", "full attention"),
                   ("sliding_attention", f"sliding window ({window} tokens)"),
                   ("linear_attention", "linear attention")):
        _layer_square(f, lx, ly - SQ / 2, k)
        lx = f.text(lx + SQ + 7, ly, txt, size=fs, anchor="start")[2] + 24
    ly += 34
    lx = x0
    glyphs = (("linear_attention", "fixed-size state", -5),
              ("full_attention", "cache grows with the context", -17),
              ("sliding_attention", "cache capped at the window", -8))
    for k, txt, dy in glyphs:
        _cache_glyph(f, lx, ly + dy, k)
        lx = f.text(lx + SQ + 7, ly, txt, size=fs, anchor="start")[2] + 24
    _layer_square(f, lx, ly - SQ / 2, "sliding_attention", shared=True)
    f.text(lx + SQ + 7, ly, "no cache", size=fs, anchor="start")
    save(f, "layer-layout",
         "Layer layouts of Qwen3.5-2B (24 layers: 18 linear-attention layers "
         "with a fixed-size state, 6 full-attention layers with a growing "
         "key-value cache) and Gemma 4 E2B (35 layers: sliding-window layers "
         "with a cache capped at the window of 512 tokens, full-attention "
         "layers with a growing cache, and a per-layer embedding table read "
         "by every layer; layers 15 to 34 keep no cache and reuse the keys "
         "and values of layer 13 or 14).")

# --------------------------------------------------------------------------- #
# 18.2  fig_lm_weight_map                                                     #
# --------------------------------------------------------------------------- #

def qwen3_rope_permutation(head_dim):
    """idx[2i] = i, idx[2i+1] = i + head_dim // 2 (as d2l.qwen3_rope_permutation)."""
    h = head_dim // 2
    return [i // 2 if i % 2 == 0 else i // 2 + h for i in range(head_dim)]


def fig_weight_map():
    """POINT: most tensors are renamed; four are renamed and permuted, because
    the two codes rotate different coordinate pairs."""
    q = QWEN3_06B
    V, d, f_, d_h = q["V"], q["d"], q["f"], q["d_h"]
    qw, kw = q["H"] * d_h, q["H_kv"] * d_h
    # (Transformers name, shape, book name, permuted, group)
    rows = [
        ("model.embed_tokens", (V, d), "token_emb", False, None),
        ("input_layernorm", (d,), "norm1", False, "L"),
        ("q_proj", (qw, d), "W_q", True, "A"),
        ("k_proj", (kw, d), "W_k", True, "A"),
        ("v_proj", (kw, d), "W_v", False, "A"),
        ("o_proj", (d, qw), "W_o", False, "A"),
        ("q_norm", (d_h,), "q_norm", True, "A"),
        ("k_norm", (d_h,), "k_norm", True, "A"),
        ("post_attention_layernorm", (d,), "norm2", False, "L"),
        ("gate_proj", (f_, d), "W_g", False, "F"),
        ("up_proj", (f_, d), "W_1", False, "F"),
        ("down_proj", (d, f_), "W_2", False, "F"),
        ("model.norm", (d,), "norm", False, None),
        ("lm_head", (V, d), None, False, None),
    ]
    f = Figure()
    fs = T.FS_SMALL
    pitch = 25.0
    ind = 18.0                    # indent of names inside a layer panel
    sub = 14.0                    # extra indent inside self_attn / mlp
    name_w = max(tw(n, fs) for n, *_ in rows) + ind + sub
    shp = [("(" + ", ".join(f"{s}" for s in r[1]) + ")") for r in rows]
    shape_w = max(tw(s, fs) for s in shp)
    xl = 0.0
    xshape = xl + name_w + 16 + shape_w       # right edge of the shape column
    xc0, xc1 = xshape + 14, xshape + 96       # connector span
    xr = xc1 + 10
    y_top = 46.0
    ys = []
    y = y_top
    for i, r in enumerate(rows):
        if i == 1:
            y += 20                           # gaps around the layer panel
        if i == 12:
            y += 10
        if i == 13:
            y += 6
        ys.append(y)
        y += pitch
    # layer panels (left and right) behind rows 1-11
    p_top, p_bot = ys[1] - 30, ys[11] + 15
    right_w = max(tw("attention.k_norm", fs), tw("ffn.W_g", fs)) + ind + 26
    underlay(f, lambda g: g.rect(xl - 10, p_top, xshape - xl + 20,
                                 p_bot - p_top, fill=T.PANEL, stroke="none",
                                 sw=0, r=T.R_PANEL))
    underlay(f, lambda g: g.rect(xr - 10, p_top, right_w, p_bot - p_top,
                                 fill=T.PANEL, stroke="none", sw=0,
                                 r=T.R_PANEL))
    f.text(xl, p_top + 14, [mono("model.layers.{i}.")], size=fs,
           color=T.INK, anchor="start")
    f.text(xr, p_top + 14, [mono("blks.{i}.")], size=fs, color=T.INK,
           anchor="start")
    f.text(xl, 8, "TRANSFORMERS CHECKPOINT", size=T.FS_TINY, color=T.INK,
           anchor="start", tracking=T.LETTERSPACE_CAPS)
    f.text(xr - 10, 8, [mono("d2l.Qwen3GPT", T.INK)], size=fs,
           anchor="start")
    prev_group = None
    for i, (name, _shape, book, perm, group) in enumerate(rows):
        y = ys[i]
        dropped = book is None
        col = T.INK
        x = xl + (ind if group else 0)
        lname = name
        if group == "A":
            lname = "self_attn." + name
        elif group == "F":
            lname = "mlp." + name
        if group in ("A", "F") and prev_group == group:
            f.text(x, y, [mono(" " * len(lname.split(".")[0] + "."), col),
                          mono(name, ATTN.dark if perm else col)], size=fs,
                   anchor="start")
        else:
            pre, _, last = lname.rpartition(".")
            spans = ([mono(pre + ".", col)] if pre else []) + [
                mono(last, ATTN.dark if perm else col)]
            f.text(x, y, spans, size=fs, anchor="start")
        f.text(xshape, y, [mono(shp[i], T.INK)],
               size=fs, anchor="end")
        if dropped:
            f.text(xc0 + 4, y, "copy of embed_tokens, dropped",
                   size=fs, color=T.INK, anchor="start")
        else:
            bx = xr + (ind if group else 0)
            if group == "A":
                bl = ("attention." + book) if prev_group != "A" else book
                bx_name = bx if prev_group != "A" else bx + tw("attention.", fs)
            elif group == "F":
                bl = ("ffn." + book) if prev_group != "F" else book
                bx_name = bx if prev_group != "F" else bx + tw("ffn.", fs)
            else:
                bl, bx_name = book, bx
            pre, _, last = bl.rpartition(".")
            spans = ([mono(pre + ".")] if pre else []) + [
                mono(last, ATTN.dark if perm else T.INK)]
            f.text(bx_name, y, spans, size=fs, anchor="start")
            color = ATTN.base if perm else T.FAINT
            sw = T.SW_BOX if perm else T.SW_HAIR
            f.line(xc0, y, xc1, y, stroke=color, sw=sw)
            if perm:
                cx = (xc0 + xc1) / 2
                f.circle(cx, y, 9.5, fill=ATTN.tint, stroke=ATTN.base,
                         sw=T.SW_HAIR)
                f.text(cx, y, [Span("P", "b")], size=T.FS_MIN,
                       color=ATTN.dark)
        prev_group = group

    # inset below the table: one head with d_h = 8, Transformers pairs
    # (i, i+4) on top, the book's pairs (2i, 2i+1) below, idx between them
    idx = qwen3_rope_permutation(8)
    assert idx == [0, 4, 1, 5, 2, 6, 3, 7]
    c = 24.0
    ix0 = xl + 4
    f.text(xl, ys[-1] + 40, [Span("One head with "), Span("d", "i"),
                             Span("h", "i", script=-1), Span(" = 8")],
           size=fs, anchor="start")
    iy_top = ys[-1] + 90
    iy_bot = iy_top + 82
    for j in range(8):
        for yy, lab in ((iy_top, j), (iy_bot, idx[j])):
            f.rect(ix0 + j * c, yy, c, c, fill=T.PAPER, stroke=T.INK,
                   sw=T.SW_HAIR)
            f.text(ix0 + j * c + c / 2, yy + c / 2, [mono(str(lab))],
                   size=T.FS_SMALL)
    # arcs above the top row: (i, i + 4)
    for i in range(4):
        xa, xb = ix0 + i * c + c / 2, ix0 + (i + 4) * c + c / 2
        hgt = 14 + 7 * i
        f.path(f"M {_f(xa)} {_f(iy_top - 2)} C {_f(xa)} {_f(iy_top - hgt)} "
               f"{_f(xb)} {_f(iy_top - hgt)} {_f(xb)} {_f(iy_top - 2)}",
               xa, iy_top - hgt, xb, iy_top, stroke=ATTN.base, sw=T.SW_HAIR)
    # arcs below the bottom row: (2i, 2i + 1)
    for i in range(4):
        xa, xb = ix0 + 2 * i * c + c / 2, ix0 + (2 * i + 1) * c + c / 2
        yb = iy_bot + c + 2
        f.path(f"M {_f(xa)} {_f(yb)} C {_f(xa)} {_f(yb + 14)} {_f(xb)} "
               f"{_f(yb + 14)} {_f(xb)} {_f(yb)}", xa, yb, xb, yb + 14,
               stroke=ATTN.base, sw=T.SW_HAIR)
    # crossing lines: bottom cell j takes top cell idx[j]
    for j in range(8):
        xt = ix0 + idx[j] * c + c / 2
        xb = ix0 + j * c + c / 2
        f.line(xt, iy_top + c + 2, xb, iy_bot - 2, stroke=ATTN.base,
               sw=T.SW_HAIR)
    lx = ix0 + 8 * c + 18
    f.text(lx, iy_top + c / 2, [Span("Transformers rotates pairs ("),
                                Span("i", "i"), Span(", "), Span("i", "i"),
                                Span(" + 4)")], size=fs, anchor="start")
    f.text(lx, iy_bot + c / 2, [Span("the book rotates pairs (2"),
                                Span("i", "i"), Span(", 2"), Span("i", "i"),
                                Span(" + 1)")], size=fs, anchor="start")
    f.text(lx, (iy_top + iy_bot + c) / 2, [mono("idx = [0,4,1,5,2,6,3,7]",
                                                ATTN.dark)],
           size=fs, anchor="start")
    save(f, "weight-map",
         "The weight map from the Qwen3-0.6B checkpoint to d2l.Qwen3GPT: "
         "embed_tokens to token_emb; per layer input_layernorm to norm1, "
         "q_proj, k_proj, v_proj, o_proj to W_q, W_k, W_v, W_o, q_norm and "
         "k_norm unchanged, post_attention_layernorm to norm2, gate_proj, "
         "up_proj, down_proj to W_g, W_1, W_2; the final norm; lm_head is a "
         "copy of embed_tokens and is dropped. q_proj, k_proj, q_norm and "
         "k_norm are also permuted. Inset: for one head with d_h = 8, "
         "Transformers rotates pairs (i, i+4) and the book pairs (2i, 2i+1); "
         "the permutation is idx = [0, 4, 1, 5, 2, 6, 3, 7].")


# --------------------------------------------------------------------------- #
# 18.3  fig_lm_chat_template                                                  #
# --------------------------------------------------------------------------- #

TOK_H = 26.0
TOK_PAD = 5.0
TOK_GAP = 4.0
ID_FS = T.FS_MIN


def token_box(f, x, cy, tok, tid=None, *, style="text", size=T.FS_SMALL,
              id_dy=21.0):
    """One token as a box at left edge x; its id below.  Returns the width."""
    spans = show_token(tok)
    w = max(spans_width(spans, size) + 2 * TOK_PAD,
            tw(str(tid), ID_FS) + 4 if tid is not None else 0, 14.0)
    if style == "special":
        fill, edge, sw = EMBED.tint, EMBED.base, T.SW_BOX
    elif style == "role":
        fill, edge, sw = T.PAPER, T.INK, T.SW_BOX
    elif style == "score":
        fill, edge, sw = ATTN.tint, ATTN.base, T.SW_BOX
    elif style == "label":
        fill, edge, sw = HIGH.tint, HIGH.base, T.SW_BOX
    elif style == "prompt":
        fill, edge, sw = T.PANEL, T.FAINT, T.SW_HAIR
    else:
        fill, edge, sw = T.PAPER, T.FAINT, T.SW_HAIR
    f.rect(x, cy - TOK_H / 2, w, TOK_H, fill=fill, stroke=edge, sw=sw, r=4)
    f.text(x + w / 2, cy, spans, size=size)
    if tid is not None:
        f.text(x + w / 2, cy + id_dy, [mono(str(tid))], size=ID_FS,
               color=T.INK)
    return w


def token_row(f, x, cy, toks, styles=None):
    """Lay out (token, id) pairs left to right; returns their x extents."""
    ext = []
    for k, (tok, tid) in enumerate(toks):
        st = styles[k] if styles else (
            "special" if tid in SPECIAL_IDS else
            "role" if tid in ROLE_IDS else "text")
        w = token_box(f, x, cy, tok, tid, style=st)
        ext.append((x, x + w))
        x += w + TOK_GAP
    return ext


def fig_chat_template():
    """POINT: the model sees a template's tokens; the label is read at the
    next position, and the same word is a different token after a newline
    and after a colon."""
    f = Figure()
    fs = T.FS_SMALL
    pitch = 60.0
    y = 30.0
    f.text(0, 0, "CHAT LAYOUT", size=T.FS_TINY, color=T.INK,
           anchor="start", tracking=T.LETTERSPACE_CAPS)
    exts = []
    for r, toks in enumerate(CHAT_ROWS):
        exts.append(token_row(f, 0, y + r * pitch, toks))
    # brackets under the generation prefix
    ext3 = exts[2]
    y3 = y + 2 * pitch
    yb = y3 + 34
    brace(f, ext3[0][0], ext3[2][1], yb)
    brace(f, ext3[3][0], ext3[6][1], yb)
    f.text((ext3[0][0] + ext3[2][1]) / 2, yb + 15,
           [mono("add_generation_prompt=True")], size=T.FS_MIN)
    f.text((ext3[3][0] + ext3[6][1]) / 2, yb + 15,
           [mono("enable_thinking=False")], size=T.FS_MIN)
    # the next position: the label token
    nx = ext3[-1][1] + 22
    f.arrow(ext3[-1][1] + 3, y3, nx - 2, y3, stroke=T.INK, sw=T.SW_HAIR)
    w = token_box(f, nx, y3, "Sports", 40979, style="label")
    f.text(nx + w / 2, yb + 15, "label read here", size=fs)
    # side stack of the four label ids, the chosen one level with the box
    sx = nx + w + 34
    for k, (word, tid) in enumerate(CHAT_LABELS):
        yy = y3 + (k - 1) * 19
        on = word == "Sports"
        f.text(sx, yy, [mono(word, T.INK)], size=fs,
               anchor="start")
        f.text(sx + 112, yy, [mono(str(tid), T.INK)],
               size=fs, anchor="end")
    f.leader(sx - 6, y3, nx + w + 4, y3)
    # plain layout
    y4 = yb + 76
    f.text(0, y4 - 30, "PLAIN LAYOUT", size=T.FS_TINY, color=T.INK,
           anchor="start", tracking=T.LETTERSPACE_CAPS)
    x = 0.0
    f.text(x + 10, y4, "…", size=T.FS_LABEL)
    ext4 = token_row(f, x + 26, y4, PLAIN_TAIL)
    nx4 = ext4[-1][1] + 22
    f.arrow(ext4[-1][1] + 3, y4, nx4 - 2, y4, stroke=T.INK, sw=T.SW_HAIR)
    w4 = token_box(f, nx4, y4, PLAIN_LABEL[0], PLAIN_LABEL[1], style="label")
    f.text(nx4 + w4 + 14, y4, "label read here: the same word, another token",
           size=fs, anchor="start")
    save(f, "chat-template",
         "The prompt as tokens of Qwen3-1.7B. Chat layout: <|im_start|> "
         "system, newline, 'Name the topic of the headline.', <|im_end|>, "
         "newline; <|im_start|> user, newline, 'Headline: Local team wins cup "
         "final', <|im_end|>, newline; the generation prefix <|im_start|> "
         "assistant newline (add_generation_prompt=True) and <think>, two "
         "newlines, </think>, two newlines (enable_thinking=False). The label "
         "is read at the next position: Sports, id 40979 (World 10134, Sports "
         "40979, Business 22727, Tech 34097). Plain layout: '... final', "
         "newline, 'Topic', ':' followed by ' Sports' with a leading space, "
         "id 13180.")

# --------------------------------------------------------------------------- #
# 18.4  fig_lm_scoring_conventions                                            #
# --------------------------------------------------------------------------- #

def _text_panel(f, x, y, w, h, lines, *, size=T.FS_SMALL, lh=19.0,
                fill=T.PANEL, stroke="none", top_pad=12.0):
    """A panel of monospace lines (each a list of Spans or a string)."""
    f.rect(x, y, w, h, fill=fill, stroke=stroke, sw=T.SW_HAIR, r=T.R_BLOCK)
    for k, ln in enumerate(lines):
        spans = [mono(ln)] if isinstance(ln, str) else ln
        f.text(x + 12, y + top_pad + 6 + k * lh, spans, size=size,
               anchor="start")


def fig_scoring_conventions():
    """POINT: one item, three procedures that read different numbers."""
    f = Figure()
    fs = T.FS_SMALL
    q_lines = ["Question: Which gas do plants", "take in from the air to make",
               "their food?"]
    opts = [f"{chr(65 + i)}. {o}" for i, o in enumerate(SCORE_OPTIONS)]
    cols = [
        q_lines + ["Answer:"],
        q_lines + opts + ["Answer:"],
        ["Which gas do plants take in", "from the air to make their",
         "food?"] + opts + ["Give the letter of the correct",
                            r"option as \boxed{X}."],
    ]
    lh = 19.0
    cw = max(tw(ln, fs) for c in cols for ln in c) + 24
    gap = 26.0
    xs = [k * (cw + gap) for k in range(3)]
    y_p = 26.0
    y_rs = []
    for k, (x, lines) in enumerate(zip(xs, cols)):
        ph = 12 + 6 + (len(lines) - 1) * lh + 18
        f.text(x, 6, f"({'abc'[k]})", size=T.FS_LABEL, anchor="start")
        if k == 2:
            f.text(x + 34, 6, "USER TURN", size=T.FS_TINY, color=T.INK,
                   anchor="start", tracking=T.LETTERSPACE_CAPS)
        _text_panel(f, x, y_p, cw, ph, lines, lh=lh)
        y_rs.append(y_p + ph + 26)   # first continuation row of the column
    y_r = y_rs[0]
    rp = 36.0                    # row pitch
    # (a) cloze: four continuations, scored tokens highlighted
    ext_last = None
    for i, toks in enumerate(SCORE_CLOZE):
        y = y_r + i * rp
        x = xs[0]
        w = token_box(f, x, y, "…Answer:", style="prompt")
        x += w + TOK_GAP
        ext = []
        for t in toks:
            w = token_box(f, x, y, t, style="score")
            ext.append((x, x + w))
            x += w + TOK_GAP
        ext_last = ext
    yb = y_r + 3 * rp + TOK_H / 2 + 7
    brace(f, ext_last[0][0], ext_last[-1][1], yb)
    y_r = y_rs[1]
    f.text(ext_last[0][0], yb + 15, [Span("sum or mean of log "),
                                      Span("p", "i")],
           size=fs, anchor="start")
    # (b) letter: four single-token continuations
    for i in range(4):
        y = y_r + i * rp
        x = xs[1]
        w = token_box(f, x, y, "…Answer:", style="prompt")
        x += w + TOK_GAP
        w = token_box(f, x, y, " " + chr(65 + i), style="score")
        if i == 3:
            yb = y_r + 3 * rp + TOK_H / 2 + 7
            brace(f, x, x + w, yb)
            f.text(x, yb + 15, [Span("log "), Span("p", "i"),
                                Span(" of one letter")], size=fs,
                   anchor="start")
    # (c) generative: a reply, the extracted span, the comparison
    x = xs[2]
    y_r = y_rs[2]
    f.text(x, y_r - 12, "MODEL REPLY", size=T.FS_TINY, color=T.INK,
           anchor="start", tracking=T.LETTERSPACE_CAPS)
    reply = ["Plants take in carbon", "dioxide from the air,", "so the answer is"]
    rh = 12 + 6 + len(reply) * lh + 14
    _text_panel(f, x, y_r, cw, rh, reply, fill=T.PAPER, stroke=T.INK,
                top_pad=8)
    bx = x + 12
    by = y_r + 8 + 6 + len(reply) * lh
    bw = tw(r"\boxed{B}", fs)
    f.text(bx, by, [mono(r"\boxed{B}")], size=fs, anchor="start")
    f.rect(bx - 4, by - 12, bw + 8, 24, fill="none", stroke=ATTN.base,
           sw=T.SW_BOX, r=4)
    f.arrow(bx + bw + 6, by, bx + bw + 40, by, stroke=T.INK, sw=T.SW_HAIR)
    f.text(bx + bw + 46, by, "B", size=fs, anchor="start")
    f.text(x, y_r + rh + 22, "parse, compare with key", size=fs,
           anchor="start")
    save(f, "scoring-conventions",
         "One multiple-choice item, 'Which gas do plants take in from the air "
         "to make their food?' with options oxygen, carbon dioxide, nitrogen "
         "and water vapor, under three scoring conventions. (a) Cloze: after "
         "the prompt ending in 'Answer:', each option's tokens (one for "
         "oxygen and nitrogen, two for carbon dioxide and water vapor) are "
         "scored by the sum or mean of their log-probabilities. (b) Letter: "
         "the options are listed with letters and the single tokens A to D "
         "are scored. (c) Generative: the model writes a reply ending in "
         r"\boxed{B}, which is parsed and compared with the answer key.")


# --------------------------------------------------------------------------- #
# 18.5  fig_lm_benchmark_lifecycle  (figstyle.mpl)                            #
# --------------------------------------------------------------------------- #

def fig_benchmark_lifecycle():
    """POINT: a benchmark separates models, is climbed, saturates below a
    label-error ceiling and is replaced; a fresh test set follows at an
    offset (shift), and overfitting shows only on an i.i.d. private split."""
    import matplotlib
    matplotlib.use("svg")
    import matplotlib.pyplot as plt
    import numpy as np
    from figstyle.mpl import save as mpl_save, use_style

    use_style(hashsalt="mdl-lm-pretrained")

    def s_curve(t, lo, hi, t0, w):
        return lo + (hi - lo) / (1 + np.exp(-(t - t0) / w))

    t = np.linspace(0, 10, 400)
    ceiling, human, shift = 95.0, 70.0, 14.0
    public = s_curve(t, 25, 91, 3.2, 0.55)
    fresh = public - shift
    t_rep = t[t >= 5.6]
    repl = s_curve(t_rep, 14, 86, 8.6, 0.6)
    t_of = t[t >= 3.0]
    overfit = (s_curve(t_of, 25, 91, 3.2, 0.55) - shift
               - 19 * (1 - np.exp(-(t_of - 3.0) / 1.2)))
    t_hum = t[t <= 6.4]                   # the public benchmark's era

    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    top = 114.0
    # stage bands along the top
    stages = [(0, 2.2, "introduced"), (2.2, 4.4, "climbed"),
              (4.4, 6.4, "saturated"), (6.4, 10, "replaced")]
    for k, (a, b, name) in enumerate(stages):
        ax.axvspan(a, b, ymin=101 / top, ymax=1.0,
                   color=T.PANEL if k % 2 == 0 else T.HAIRLINE, lw=0)
        ax.text((a + b) / 2, (101 + top) / 2, name, ha="center",
                va="center", fontsize=T.FS_TINY, color=T.INK)
    ax.axhline(ceiling, color=T.INK, lw=T.SW_HAIR, ls=(0, (4, 3)))
    ax.plot(t_hum, np.full_like(t_hum, human), color=T.MUTED, lw=T.SW_HAIR,
            ls=(0, (1.5, 2.5)))
    ax.plot(t, public, color=T.BLUE.base, lw=T.SW_HEAVY)
    ax.plot(t, fresh, color=T.BLUE.base, lw=T.SW_LINE, alpha=0.45)
    ax.plot(t_of, overfit, color=T.RED.base, lw=T.SW_BOX, ls=(0, (4, 3)))
    ax.plot(t_rep, repl, color=T.ORANGE.base, lw=T.SW_HEAVY)
    # labels, offset from the curves
    kw = dict(fontsize=T.FS_TINY, linespacing=1.15)
    ax.text(0.1, ceiling - 1.5, "ceiling: label errors", ha="left",
            va="top", color=T.INK, **kw)
    ax.text(0.1, human + 1.5, "human baseline", ha="left", va="bottom",
            color=T.INK, **kw)
    ax.text(0.15, 36, "public test set", ha="left", va="bottom",
            color=T.BLUE.dark, **kw)
    ax.text(0.15, 9.5, "fresh test set", ha="left", va="top",
            color=T.BLUE.dark, **kw)
    i5 = np.searchsorted(t, 5.2)
    ax.annotate("", xy=(5.2, public[i5] - 1.2), xytext=(5.2, fresh[i5] + 1.0),
                arrowprops=dict(arrowstyle="<|-|>", color=T.INK,
                                lw=T.SW_HAIR, shrinkA=0, shrinkB=0,
                                mutation_scale=8))
    ax.text(5.35, (public[i5] + fresh[i5]) / 2, "distribution shift",
            ha="left", va="center", color=T.INK, **kw)
    ax.text(4.35, 52, "overfitting, visible only\non an i.i.d. private split",
            ha="left", va="top", color=T.RED.dark, **kw)
    ax.text(8.15, 22, "replacement\nbenchmark", ha="left", va="top",
            color=T.ORANGE.dark, **kw)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, top)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_yticklabels(["0%", "25%", "50%", "75%", "100%"])
    ax.set_xticks([])
    ax.set_xlabel("time since release")
    ax.set_ylabel("best published score")
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color(T.INK)
    ax.tick_params(axis="y", colors=T.INK)
    path = mpl_save(fig, f"{PREFIX}-benchmark-lifecycle")
    WRITTEN.append(path)


# --------------------------------------------------------------------------- #
# 18.6  fig_lm_patching                                                       #
# --------------------------------------------------------------------------- #

def _site(f, x, y, name, *, side="right", color=T.INK):
    f.circle(x, y, 5.5, fill=HIGH.base, stroke=T.PAPER, sw=1.5)
    dx = 11 if side == "right" else -11
    f.text(x + dx, y, [mono(name)], size=T.FS_SMALL, color=color,
           anchor="start" if side == "right" else "end")


def fig_patching():
    """POINT: five sites of a pre-norm block; noising copies one site from
    the corrupted run into the clean run and reads LD at the end."""
    f = Figure()
    fs = T.FS_SMALL
    # ---- left: one pre-norm block, bottom to top --------------------------
    SX, BC = 0.0, 112.0
    NW, BW = 64.0, 150.0
    y_in, y_out = 410.0, 24.0
    f.arrow(SX, y_in, SX, y_out, stroke=STREAM.base, sw=T.SW_HEAVY)
    f.text(SX, y_in + 16, "block input", size=fs, color=T.INK)

    def sublayer(y_read, boxes, y_add, accent):
        _, ny, _, _ = f.block(BC, y_read, "norm", min_w=NW, size=fs)
        f.line(SX, y_read, BC - NW / 2, y_read, stroke=accent.base)
        prev_top = ny
        rects = []
        for yb, label in boxes:
            r = f.block(BC, yb, label, accent=accent, min_w=BW, size=fs)
            f.arrow(BC, prev_top, BC, r[1] + r[3], stroke=accent.base)
            prev_top = r[1]
            rects.append(r)
        f.polyline([(BC, prev_top), (BC, y_add), (SX + 13, y_add)],
                   stroke=accent.base)
        f.pill_op(SX, y_add, "+", r=11)
        return rects

    ra = sublayer(384, [(334, "attention heads"),
                        (272, [Span("W", "b"), Span("o", "i", script=-1)])],
                  226, ATTN)
    rf = sublayer(180, [(130, "MLP")], 82, FFN)
    _site(f, BC, (ra[0][1] + ra[1][1] + ra[1][3]) / 2, "head_z")
    _site(f, BC, (ra[1][1] + 226) / 2, "attn_out")
    _site(f, SX, 203, "resid_mid", side="left")
    _site(f, BC, (rf[0][1] + 82) / 2, "mlp_out")
    _site(f, SX, 58, "resid_post", side="left")

    # ---- right: block-by-token grids ---------------------------------------
    n_tok, n_rows = len(PATCH_CLEAN), 8
    c = 17.0
    gx_cor = BC + BW / 2 + 80
    gx_cln = gx_cor + n_tok * c + 46
    gy = 136.0                                   # top of the grids
    g_bot = gy + n_rows * c
    row_patch = 4                                # from the top: a middle block
    col = PATCH_SUBJECT_LAST

    def grid(gx, toks, title, answer, clean):
        for r in range(n_rows):
            for k in range(n_tok):
                fill = T.PAPER
                if clean and k >= col and r < row_patch:
                    fill = STREAM.tint           # cells the copy can change
                if r == row_patch and k == col:
                    fill = HIGH.tint
                f.rect(gx + k * c, gy + r * c, c, c, fill=fill,
                       stroke=T.HAIRLINE, sw=1.0)
        f.rect(gx, gy, n_tok * c, n_rows * c, fill="none", stroke=T.INK,
               sw=T.SW_HAIR)
        f.rect(gx + col * c, gy + row_patch * c, c, c, fill="none",
               stroke=HIGH.base, sw=T.SW_BOX)
        for k, tok in enumerate(toks):
            rot_text(f, gx + k * c + c / 2, g_bot + 6, [mono(tok.strip())],
                     size=T.FS_MIN, anchor="end")
        yt = g_bot + 70
        f.text(gx, yt, [Span(title, "b")], size=fs, anchor="start")
        f.text(gx, yt + 19, [Span("answer "), mono(answer.strip())],
               size=fs, anchor="start")
    grid(gx_cor, PATCH_CORRUPT, "corrupted run", " Rome", False)
    grid(gx_cln, PATCH_CLEAN, "clean run", " Paris", True)
    # block axis left of the corrupted grid
    f.text(gx_cor - 7, gy + c / 2, "27", size=T.FS_MIN, color=T.INK,
           anchor="end")
    f.text(gx_cor - 7, g_bot - c / 2, "0", size=T.FS_MIN, color=T.INK,
           anchor="end")
    rot_text(f, gx_cor - 30, gy + n_rows * c / 2, "block", size=fs,
             anchor="middle")
    # the copy: corrupted cell -> clean cell, arcing over the grids
    x_a = gx_cor + col * c + c / 2
    x_b = gx_cln + col * c + c / 2
    y_c = gy + row_patch * c
    ctl = gy - 62
    d = (f"M {_f(x_a)} {_f(y_c - 2)} C {_f(x_a)} {_f(ctl)} "
         f"{_f(x_b)} {_f(ctl)} {_f(x_b)} {_f(y_c - 12)}")
    f.path(d, x_a, ctl, x_b, y_c, stroke=T.PAPER, sw=6)
    f.path(d, x_a, ctl, x_b, y_c, stroke=HIGH.base, sw=T.SW_BOX)
    f._head(x_b, y_c - 1, math.pi / 2, color=HIGH.base)
    peak = 0.25 * (y_c - 2) + 0.75 * ctl
    f.text((x_a + x_b) / 2, peak - 13, "copy one site", size=fs)
    # the metric, read at the final position after the last block
    x_ld = gx_cln + (n_tok - 1) * c + c / 2
    f.arrow(x_ld, gy - 2, x_ld, gy - 30, stroke=T.INK, sw=T.SW_HAIR)
    xr = x_ld + 10
    f.text(xr, gy - 62, [Span("LD", "i"), Span(" = logit "), mono("Paris")],
           size=fs, anchor="end")
    f.text(xr, gy - 42, [Span("− logit "), mono("Rome")], size=fs,
           anchor="end")
    save(f, "patching",
         "Activation patching. Left: a pre-norm transformer block (norm, "
         "attention heads, W_o, residual add, norm, MLP, residual add) with "
         "the five sites head_z, attn_out, resid_mid, mlp_out and resid_post. "
         "Right: grids of blocks by tokens for the corrupted run 'The "
         "Colosseum is located in the city of' (answer Rome) and the clean "
         "run 'The Eiffel Tower is located in the city of' (answer Paris); "
         "one cell, the last subject token at a middle block, is copied from "
         "the corrupted run into the clean run, where it can change the cells "
         "above it and to its right; the logit difference LD between Paris "
         "and Rome is read at the final position after the last block.")


# --------------------------------------------------------------------------- #
# 18.6  fig_lm_sae_splice                                                     #
# --------------------------------------------------------------------------- #

def math_text(f, x, y, parts, *, size=T.FS_SMALL, color=T.INK,
              anchor="start"):
    """Spans plus the token "XHAT" (bold x with a hat, which the bundled
    fonts cannot compose)."""
    widths = []
    for p in parts:
        if p == "XHAT":
            widths.append(tw("x", size, "b"))
        else:
            widths.append(spans_width([p], size) if p.kind != "b" else
                          tw(p.text, size * (0.72 if p.script else 1), "b"))
    total = sum(widths)
    x0 = {"middle": x - total / 2, "end": x - total}.get(anchor, x)
    for p, w in zip(parts, widths):
        if p == "XHAT":
            f.text(x0, y, [Span("x", "b")], size=size, color=color,
                   anchor="start")
            top = y - measure("x", size).x_height / 2 - 0.13 * size
            f.polyline([(x0 + 0.16 * w, top), (x0 + 0.5 * w, top - 0.2 * size),
                        (x0 + 0.84 * w, top)], stroke=color, sw=1.3)
        else:
            f.text(x0, y, [p], size=size, color=color, anchor="start")
        x0 += w


def fig_sae_splice():
    """POINT: encode to many sparse features, decode back into the stream;
    the same decoder columns reconstruct, ablate and steer."""
    f = Figure()
    fs = T.FS_SMALL
    SX = 0.0
    y_bot, y_tap, y_add, y_top = 300.0, 256.0, 70.0, 10.0
    f.line(SX, y_bot, SX, y_tap, stroke=STREAM.base, sw=T.SW_HEAVY)
    f.arrow(SX, y_add - 11, SX, y_top, stroke=STREAM.base, sw=T.SW_HEAVY)
    f.text(SX, y_bot + 16, "residual stream, block 14", size=fs,
           color=T.INK)
    f.circle(SX, y_tap, 4.5, fill=STREAM.base, stroke="none", sw=0)
    f.text(SX - 12, y_tap, [Span("x", "b")], size=T.FS_LABEL, anchor="end")
    # the error term: dashed side branch from the tap to the splice
    f.line(SX, y_tap, SX, y_add + 11, stroke=T.MUTED, sw=T.SW_BOX,
           dash=T.DASH_SOFT)
    ym = (y_tap + y_add) / 2
    math_text(f, SX - 12, ym - 10, [Span("error "), Span("x", "b"),
                                    Span(" − "), "XHAT"], anchor="end")
    f.text(SX - 12, ym + 10, "(kept or dropped)", size=fs, color=T.INK,
           anchor="end")
    f.pill_op(SX, y_add, "+", r=11)
    # encoder -> features -> decoder, to the right of the stream
    n, c = 32, 11.0
    fx0 = 110.0
    fy = 164.0
    lit = (3, 9, 17, 22, 28)
    j = 22
    for k in range(n):
        on = k in lit
        f.rect(fx0 + k * c, fy - c / 2, c, c,
               fill=FFN.base if on else T.PAPER,
               stroke=FFN.base if on else T.FAINT, sw=1.0)
    fx1 = fx0 + n * c
    fmid = (fx0 + fx1) / 2
    f.text(fx1 + 12, fy - 9, [Span("z", "b"), Span(": 32,768 features,")],
           size=fs, anchor="start")
    f.text(fx1 + 12, fy + 10, [Span("the "), Span("k", "i"),
                               Span(" largest kept")], size=fs,
           anchor="start")
    er = f.block(fmid, y_tap, [Span("encoder, top-"), Span("k", "i")],
                 accent=FFN, size=fs)
    f.arrow(SX + 5, y_tap, er[0], y_tap, stroke=T.INK)
    f.arrow(fmid, er[1], fmid, fy + c / 2 + 2, stroke=T.INK)
    y_dec = y_add + 32
    dr = f.block(fmid, y_dec, [Span("decoder "), Span("W", "b"),
                               Span("dec", "r", script=-1)],
                 accent=FFN, size=fs)
    f.arrow(fmid, fy - c / 2 - 2, fmid, dr[1] + dr[3], stroke=T.INK)
    f.ortho_arrow([(dr[0], y_dec), (72, y_dec), (72, y_add),
                   (SX + 11, y_add)], stroke=T.INK)
    math_text(f, 82, y_dec - 13, ["XHAT"], size=T.FS_LABEL)
    # steering: a second input into the splice
    f.arrow(fx0 - 20, y_add - 40, SX + 9, y_add - 7, stroke=HIGH.base,
            sw=T.SW_BOX, dash=T.DASH_SOFT)
    # annotations
    f.text(fx0 - 12, y_add - 44, [Span("steer: add "), Span("α", "i"),
                                  Span("w", "b"), Span("j", "i", script=-1)],
           size=fs, anchor="start")
    math_text(f, 84, y_dec + 22, [Span("reconstruct: splice "), "XHAT"],
              anchor="start")
    xj = fx0 + j * c + c / 2
    f.leader(xj + 18, fy + 38, xj + 1, fy + c / 2 + 3)
    f.text(xj + 22, fy + 40, [Span("ablate: clamp "), Span("z", "i"),
                              Span("j", "i", script=-1), Span(" to 0")],
           size=fs, anchor="start")
    save(f, "sae-splice",
         "A sparse autoencoder on the residual stream at block 14: the "
         "encoder maps x to 32,768 feature activations z and keeps the k "
         "largest; the decoder W_dec writes their sum of directions back as "
         "x-hat, which is spliced into the stream, with the error x minus "
         "x-hat kept or dropped as a dashed side branch. Annotations: "
         "reconstruct (splice x-hat), ablate (clamp feature z_j to 0), steer "
         "(add alpha times the decoder column w_j).")
# --------------------------------------------------------------------------- #
# 18.7  fig_lm_api_lost                                                       #
# --------------------------------------------------------------------------- #

# 2 available, 1 depends on the provider or server, 0 withheld by the protocol.
# Columns: weights in the book's code, local server, open weights via a
# router, closed model via a router.  These state what each kind of endpoint
# can offer, not what one probe found.
API_ROWS = [
    ("weights and activations", (2, 0, 0, 0)),
    ("prompt log-probabilities", (2, 1, 0, 0)),
    ("generated-token log-probabilities", (2, 1, 1, 1)),
    ("rendered prompt and chat template", (2, 2, 0, 0)),
    ("sampler beyond temperature and top-p", (2, 1, 1, 0)),
    ("seed and batching control", (2, 1, 0, 0)),
    ("token counts from usage", (2, 2, 2, 2)),
    ("structured output", (2, 1, 1, 1)),
    ("many samples", (2, 2, 2, 2)),
]
API_COLS = [("weights in the", "book\u2019s code"), ("local", "server"),
            ("open weights", "via a router"), ("closed model", "via a router")]


def _avail(f, cx, cy, level, r=8.0):
    if level == 2:
        f.circle(cx, cy, r, fill=T.INK, stroke=T.INK, sw=T.SW_HAIR)
    elif level == 0:
        f.circle(cx, cy, r, fill=T.PAPER, stroke=T.INK, sw=T.SW_HAIR)
    else:
        f.circle(cx, cy, r, fill=T.PAPER, stroke=T.INK, sw=T.SW_HAIR)
        d = (f"M {_f(cx)} {_f(cy - r)} A {_f(r)} {_f(r)} 0 0 0 {_f(cx)} "
             f"{_f(cy + r)} Z")
        f.path(d, cx - r, cy - r, cx, cy + r, fill=T.INK, stroke="none",
               sw=0)


def fig_api_lost():
    """POINT: which of the chapter's instruments survive each endpoint kind."""
    f = Figure()
    fs = T.FS_SMALL
    lab_w = max(tw(name, fs, "r") for name, _ in API_ROWS)
    col_w = 108.0
    x_c0 = lab_w + 24 + col_w / 2
    pitch = 30.0
    y0 = 60.0
    for k, (a, b) in enumerate(API_COLS):
        cx = x_c0 + k * col_w
        f.text(cx, 10, a, size=fs)
        f.text(cx, 29, b, size=fs)
    for i, (name, levels) in enumerate(API_ROWS):
        y = y0 + i * pitch
        if i % 2 == 0:
            f.rect(-8, y - pitch / 2, x_c0 + 3.5 * col_w + 8 + 8, pitch,
                   fill=T.PANEL, stroke="none", sw=0)
        f.text(0, y, name, size=fs, anchor="start")
        for k, lev in enumerate(levels):
            _avail(f, x_c0 + k * col_w, y, lev)
    ly = y0 + len(API_ROWS) * pitch + 12
    lx = 0.0
    for lev, txt in ((2, "available"), (1, "depends on the provider or server"),
                     (0, "withheld by the protocol")):
        _avail(f, lx + 8, ly, lev)
        lx = f.text(lx + 22, ly, txt, size=fs, anchor="start")[2] + 26
    save(f, "api-lost",
         "What each kind of endpoint lets a client measure. Columns: weights "
         "in the book's code, a local server, open weights via a router, a "
         "closed model via a router. Rows: weights and activations; prompt "
         "log-probabilities; generated-token log-probabilities; rendered "
         "prompt and chat template; sampler beyond temperature and top-p; "
         "seed and batching control; token counts from usage; structured "
         "output; many samples. Filled circles mark available, half circles "
         "depends on the provider or server, open circles withheld by the "
         "protocol.")

# --------------------------------------------------------------------------- #
# Registry                                                                    #
# --------------------------------------------------------------------------- #

FIGURES = [fig_hub_repo, fig_layer_layout, fig_weight_map, fig_chat_template,
           fig_scoring_conventions, fig_benchmark_lifecycle, fig_patching,
           fig_sae_splice, fig_api_lost]


def main():
    check_data()
    only = set(sys.argv[1:])
    for fn in FIGURES:
        if not only or fn.__name__ in only:
            fn()
    print(f"Wrote {len(WRITTEN)} figures:")
    for p in WRITTEN:
        size = os.path.getsize(p)
        assert size > 0, f"empty: {p}"
        print(f"  {os.path.basename(p):48s} {size:>8,d} bytes")


if __name__ == "__main__":
    main()
