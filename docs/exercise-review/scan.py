#!/usr/bin/env python3
"""Mechanical scan of ## Exercises sections in d2l-neu .md sources.

Emits one TSV row per file: counts that anchor/cross-check the agent reviews.
"""
import re
import sys
from pathlib import Path

ROOT = Path("/Users/smola/Repositories/github/d2l-neu")

CHAPTERS = """chapter_preface chapter_introduction chapter_preliminaries
chapter_linear-regression chapter_linear-classification
chapter_multilayer-perceptrons chapter_builders-guide
chapter_convolutional-neural-networks chapter_convolutional-modern
chapter_recurrent-neural-networks chapter_recurrent-modern chapter_attention
chapter_transformers chapter_optimization chapter_computational-performance
chapter_computer-vision chapter_natural-language-processing-pretraining
chapter_natural-language-processing-applications
chapter_reinforcement-learning chapter_deep-reinforcement-learning
chapter_gaussian-processes chapter_hyperparameter-optimization
chapter_generative-adversarial-networks chapter_recommender-systems
chapter_mdl-linear-algebra chapter_mdl-calculus chapter_mdl-optimization
chapter_mdl-probability-statistics chapter_mdl-information-theory
chapter_mdl-dynamics chapter_appendix-tools-for-deep-learning""".split()

ITEM = re.compile(r"^(\d+)\.\s")          # top-level list item (0-indent)
SUBITEM = re.compile(r"^(\s+)(\d+)\.\s")  # indented numbered item
TAG = re.compile(r"^\d+\.\s+\[([a-zA-Z-]+)\]")
ITALNAME = re.compile(r"^\d+\.\s+(?:\[[a-zA-Z-]+\]\s+)?\*([^*]+?)[.!?]?\*")
BOLDNAME = re.compile(r"^\d+\.\s+(?:\[[a-zA-Z-]+\]\s+)?\*\*([^*]+?)\*\*")
INLINE_LETTERS = re.compile(r"(?:^|\s)\(?[a-e]\)\s+\S.*(?:^|\s)\(?[b-f]\)\s", re.S)
CITE = re.compile(r":cite[t]?:")
XREF = re.compile(r":(?:numref|eqref|ref):")

rows = []
for ch in CHAPTERS:
    for f in sorted((ROOT / ch).glob("*.md")):
        text = f.read_text(encoding="utf-8")
        m = re.search(r"^## Exercises\s*$", text, re.M)
        if not m:
            continue
        sec = text[m.end():]
        nxt = re.search(r"^## ", sec, re.M)
        if nxt:
            sec = sec[: nxt.start()]
        lines = sec.split("\n")
        n_items = 0
        literal_nums = []
        tags = {}
        ital = bold = 0
        nested = 0
        for ln in lines:
            im = ITEM.match(ln)
            if im:
                n_items += 1
                literal_nums.append(int(im.group(1)))
                tm = TAG.match(ln)
                if tm:
                    tags[tm.group(1)] = tags.get(tm.group(1), 0) + 1
                if BOLDNAME.match(ln):
                    bold += 1
                elif ITALNAME.match(ln):
                    ital += 1
            elif SUBITEM.match(ln):
                nested += 1
        # numbering style
        if not literal_nums:
            numbering = "none"
        elif all(n == 1 for n in literal_nums):
            numbering = "repeated-1"
        elif literal_nums == list(range(1, len(literal_nums) + 1)):
            numbering = "sequential"
        else:
            numbering = "mixed"
        # inline letter subproblems, crude: a) ... b) appearing in one item body
        inline_letters = len(re.findall(r"(?<![\w$])[a-e]\)\s", sec))
        rows.append(
            (
                str(f.relative_to(ROOT)),
                n_items,
                numbering,
                ital,
                bold,
                ",".join(f"{k}:{v}" for k, v in sorted(tags.items())) or "-",
                nested,
                inline_letters,
                len(CITE.findall(sec)),
                len(XREF.findall(sec)),
            )
        )

print("file\tn_ex\tnumbering\tital_names\tbold_names\ttags\tnested_subitems\tinline_letter_marks\tcites\txrefs")
for r in rows:
    print("\t".join(str(x) for x in r))
print(f"# files={len(rows)} total_exercises={sum(r[1] for r in rows)}", file=sys.stderr)
