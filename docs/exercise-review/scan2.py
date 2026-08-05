#!/usr/bin/env python3
"""Second scan pass: discussions-state + tab-aware exercise counts."""
import re
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

ITEM = re.compile(r"^(?:\d+\.|\*)\s")
BULLET = re.compile(r"^\*\s")

rows = []
for ch in CHAPTERS:
    for f in sorted((ROOT / ch).glob("*.md")):
        text = f.read_text(encoding="utf-8")
        m = re.search(r"^## Exercises\s*$", text, re.M)
        if not m:
            continue
        sec = text[m.end():]
        nxt = re.search(r"^## ", sec, re.M)
        tail_after = sec[nxt.start():] if nxt else ""
        if nxt:
            sec = sec[: nxt.start()]

        # tab-aware item count: items outside tabs + max-per-tab-group inside
        lines = sec.split("\n")
        outside = 0
        bullets = 0
        tab_groups = []       # list of dicts tabname->count for consecutive tab runs
        cur_group = None
        in_tab = False
        cur_tab_count = 0
        for ln in lines:
            bt = re.match(r"^:begin_tab:`([^`]+)`", ln)
            if bt:
                in_tab = True
                cur_tab_count = 0
                if cur_group is None:
                    cur_group = []
                continue
            if ln.startswith(":end_tab:"):
                in_tab = False
                cur_group.append(cur_tab_count)
                continue
            if ITEM.match(ln):
                if in_tab:
                    cur_tab_count += 1
                else:
                    outside += 1
                    # close a tab group when a top-level item follows it
                    if cur_group:
                        tab_groups.append(cur_group)
                        cur_group = None
                if BULLET.match(ln):
                    bullets += 1
        if cur_group:
            tab_groups.append(cur_group)
        in_tab_dedup = sum(max(g) for g in tab_groups if g)
        raw_in_tab = sum(sum(g) for g in tab_groups if g)
        n_real = outside + in_tab_dedup
        n_raw = outside + raw_in_tab

        # discussions state (search sec + a bit of what follows? links live in sec)
        zone = sec
        tabs = re.findall(r":begin_tab:`([^`]+)`\s*\n\[Discussions\]\((https?://[^)]*)\)", zone)
        single = re.findall(r"^\[Discussions\]\((https?://[^)]*)\)", zone, re.M)
        prose_disc = bool(re.search(r"^## Discussions", tail_after, re.M))
        def is_placeholder(u):
            return re.fullmatch(r"https://d2l\.discourse\.group/?", u) is not None
        if tabs:
            urls = [u for _, u in tabs]
            if all(is_placeholder(u) for u in urls):
                state = f"tabbed({len(tabs)})-placeholder"
            elif len(set(urls)) == 1 and len(urls) > 1:
                state = f"tabbed({len(tabs)})-identical"
            else:
                state = f"tabbed({len(tabs)})"
            state += ":" + ",".join(t for t, _ in tabs)
        elif single:
            state = "single-placeholder" if is_placeholder(single[0]) else "single-link"
        elif prose_disc:
            state = "prose-##Discussions-after"
        else:
            state = "none"

        slides = "yes" if re.search(r"<!--\s*slides", tail_after) or re.search(r"<!--\s*slides", sec) else "no"
        rows.append((str(f.relative_to(ROOT)), n_real, n_raw, bullets, state, slides))

print("file\tn_real\tn_raw\tbullet_items\tdiscussions\tslides_after")
for r in rows:
    print("\t".join(str(x) for x in r))
tot_real = sum(r[1] for r in rows)
tot_raw = sum(r[2] for r in rows)
print(f"# files={len(rows)} real={tot_real} raw={tot_raw}")
from collections import Counter
c = Counter(r[4].split(":")[0] for r in rows)
print("# discussions states:", dict(c))
