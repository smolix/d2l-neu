"""Extract review candidates where Taima's Arabic PDF pairs Arabic and English.

Writes candidates to /tmp only. The PDF text layer has bidi/extraction errors, so
these are never automatically promoted to verified glossary entries.
"""

import json
from pathlib import Path
import re
import unicodedata


BASE = Path(__file__).resolve().parent
ROOT = Path("/tmp/d2l-glossary-corpora")


def clean(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    return "".join(ch for ch in text if unicodedata.category(ch) != "Cf")


def main() -> None:
    concepts = [json.loads(line) for line in (BASE / "concepts.jsonl").read_text().splitlines() if line.strip()]
    pages = []
    for part in (1, 2, 3):
        text = (ROOT / f"d2l-ar-taima-part{part}.txt").read_text(errors="replace")
        pages.extend((part, pageno, clean(page)) for pageno, page in enumerate(text.split("\f"), 1) if pageno >= 20)
    output = []
    for concept in concepts:
        hits = []
        for alias in [concept["en"], *concept.get("aliases_en", [])]:
            if len(alias) < 5:
                continue
            pattern = re.compile(r"(?i)(?<![a-z])" + r"\s+".join(map(re.escape, alias.split())) + r"(?![a-z])")
            for part, pageno, page in pages:
                for match in pattern.finditer(page):
                    left = page[max(0, match.start() - 160):match.start()].split("\n")[-1]
                    if re.search(r"[A-Za-z][A-Za-z-]*\s+$", left):
                        continue  # English modifier, e.g. regression in linear regression.
                    near = re.search(r"[\u0600-\u06FF][^\u0600-\u06FF]*$", left)
                    if not near or len(near.group()) > 22:
                        continue
                    match_words = re.search(r"([\u0600-\u06FF]+(?:[\s،\-]+[\u0600-\u06FF]+){0,8})\s*[\.,،()]*\s*$", left)
                    if not match_words:
                        continue
                    words = re.findall(r"[\u0600-\u06FF]+", match_words.group(1))
                    if not words:
                        continue
                    n = min(len(words), max(1, len(alias.split()) + 1), 5)
                    phrase = " ".join(words[-n:])
                    trigger = bool(re.search(r"(?:يسمى|تسمى|يُعرف|نعني|يدعى|يسمي|يطلق|يسمى بـ)\s+", left[-80:]))
                    hits.append({"part": part, "page": pageno, "alias": alias, "candidate": phrase,
                                 "context": re.sub(r"\s+", " ", left[-100:] + match.group() + page[match.end():match.end()+40]),
                                 "score": (2 if trigger else 0) + (1 if len(near.group()) < 8 else 0)})
            if hits:
                break
        hits.sort(key=lambda x: (-x["score"], x["part"], x["page"]))
        if hits:
            output.append({"concept_id": concept["id"], "en": concept["en"], "hits": hits[:4]})
    path = ROOT / "ar-pairs.jsonl"
    path.write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in output))
    print(path, len(output), "concepts with adjacent Arabic-English candidate")


if __name__ == "__main__":
    main()
