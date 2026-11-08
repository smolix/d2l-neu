#!/usr/bin/env python3
"""Fetch localized Google ML glossary headings and extract D2L text candidates.

Everything emitted here is a *candidate*, not verified terminology. Full source
HTML is cached under /tmp; the checked-in Google JSONL contains headings only.
The script uses only the Python standard library.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from hashlib import sha256
from html.parser import HTMLParser
import json
from pathlib import Path
import re
import sys
import time
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


REPO = Path(__file__).resolve().parents[3]
CORPORA = Path("/tmp/d2l-glossary-corpora")
GOOGLE_CACHE = Path("/tmp/d2l-glossary-google")
D2L_CACHE = Path("/tmp/d2l-glossary-candidates")
GOOGLE_OUT = REPO / "docs/multilingual/candidates"
GOOGLE_LOCALES = {
    "en": "en", "zh-Hans": "zh-cn", "ja": "ja", "ko": "ko",
    "es": "es", "ar": "ar", "pt-BR": "pt-br", "vi": "vi",
    "fr": "fr", "hi": "hi", "tr": "tr", "de": "de",
}
REPO_LOCALES = {"d2l-en": "en", "d2l-zh": "zh", "d2l-ja": "ja",
                "d2l-ko": "ko", "d2l-pt": "pt", "d2l-vi": "vi",
                "d2l-tr": "tr", "d2l-de": "de"}
USER_AGENT = "Mozilla/5.0 (compatible; D2LGlossaryCandidateExtractor/1.0)"


class HeadingParser(HTMLParser):
    """Collect h2/h3 text associated with a stable DOM id."""
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.active = False
        self.level = 0
        self.anchor = ""
        self.parts: list[str] = []
        self.rows: list[tuple[str, str, int]] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attrs_dict = dict(attrs)
        if tag in ("h2", "h3"):
            self.active = True
            self.level = int(tag[1])
            self.anchor = attrs_dict.get("id") or ""
            self.parts = []
        elif self.active and tag == "a" and not self.anchor:
            # Some rendered heading IDs are attached to a permalink child.
            self.anchor = dict(attrs).get("id") or ""

    def handle_data(self, data: str) -> None:
        if self.active:
            self.parts.append(data)

    def handle_endtag(self, tag: str) -> None:
        if self.active and tag in ("h2", "h3"):
            term = re.sub(r"\s+", " ", " ".join(self.parts)).strip()
            # Google uses non-heading visual controls in heading markup; discard
            # empty and unanchored structural headings.
            if self.anchor and term:
                self.rows.append((self.anchor, term, self.level))
            self.active = False


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def fetch_one(locale: str, hl: str) -> dict:
    url = f"https://developers.google.com/machine-learning/glossary?hl={hl}"
    target = GOOGLE_CACHE / f"glossary-{locale}.html"
    target.parent.mkdir(parents=True, exist_ok=True)
    err = None
    for attempt in range(1, 5):
        try:
            req = Request(url, headers={"User-Agent": USER_AGENT})
            with urlopen(req, timeout=35) as response:
                body = response.read()
                final_url = response.geturl()
                status = getattr(response, "status", 200)
            if status != 200:
                raise RuntimeError(f"HTTP status {status}")
            target.write_bytes(body)
            return {"locale": locale, "requested_url": url, "url": final_url,
                    "retrieved": utc_now(), "sha256": sha256(body).hexdigest(),
                    "bytes": len(body), "cache_path": str(target), "status": status,
                    "error": None}
        except (HTTPError, URLError, TimeoutError, OSError, RuntimeError) as exc:
            err = f"{type(exc).__name__}: {exc}"
            if attempt < 4:
                time.sleep(attempt * 1.5)
    return {"locale": locale, "requested_url": url, "url": url,
            "retrieved": utc_now(), "sha256": None, "bytes": 0,
            "cache_path": str(target), "status": None, "error": err}


def parse_google(meta: dict) -> list[dict]:
    if not meta.get("sha256"):
        return []
    raw = Path(meta["cache_path"]).read_bytes()
    parser = HeadingParser()
    parser.feed(raw.decode("utf-8", "replace"))
    rows = []
    for anchor, term, level in parser.rows:
        rows.append({"anchor": anchor, "term": term, "level": level})
    return rows


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def google_extract() -> tuple[int, int]:
    GOOGLE_CACHE.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = [pool.submit(fetch_one, locale, hl) for locale, hl in GOOGLE_LOCALES.items()]
        metas = [f.result() for f in as_completed(futures)]
    metas.sort(key=lambda x: x["locale"])
    parsed = {m["locale"]: parse_google(m) for m in metas}
    en_by_anchor = {r["anchor"]: r["term"] for r in parsed.get("en", [])}
    candidates: list[dict] = []
    for locale, rows in parsed.items():
        if locale == "en":
            continue
        for row in rows:
            en = en_by_anchor.get(row["anchor"])
            if en:
                candidates.append({"en": en, "anchor": row["anchor"], "locale": locale,
                                   "term": row["term"],
                                   "same_as_english": row["term"].casefold() == en.casefold(),
                                   "url": f"{next(m['url'] for m in metas if m['locale'] == locale)}#{row['anchor']}"})
    candidates.sort(key=lambda r: (r["anchor"], r["locale"]))
    sources = [{"source_id": f"google-ml-glossary-{m['locale']}",
                "title": "Google for Developers Machine Learning Glossary headings",
                "locale": m["locale"], "kind": "localized_teaching_glossary_heading_candidates",
                "url": m["url"], "requested_url": m["requested_url"],
                "retrieved": m["retrieved"], "sha256": m["sha256"],
                "bytes": m["bytes"], "http_status": m["status"],
                "cache_path": m["cache_path"], "error": m["error"],
                "candidate_only": True} for m in metas]
    for source in sources:
        matched = [c for c in candidates if c['locale'] == source['locale']]
        fraction = sum(c['same_as_english'] for c in matched) / len(matched) if matched else None
        source['fraction_headings_identical_to_english'] = fraction
        source['body_language_warning'] = (
            'Likely English-body fallback: over 90% of aligned headings are unchanged. A localized title is not translated evidence.'
            if fraction is not None and fraction > .9 else None)
        if source['sha256']:
            raw = Path(source['cache_path']).read_text(encoding='utf-8')
            lang = re.search(r'<html[^>]*\blang=[\"\x27]([^\"\x27]+)', raw)
            source['html_lang'] = lang.group(1) if lang else None
            source['machine_translation_disclosed'] = bool(lang and '-x-mtfrom-' in lang.group(1))
    write_jsonl(GOOGLE_OUT / "google-headings.jsonl", candidates)
    write_jsonl(GOOGLE_OUT / "google-sources.jsonl", sources)
    failed = sum(not m["sha256"] for m in metas)
    print(f"Google headings: {len(candidates)} aligned rows; {failed} failed locale downloads")
    if failed:
        for m in metas:
            if m["error"]:
                print(f"  {m['locale']}: {m['error']}", file=sys.stderr)
    return len(candidates), failed


def visible_markdown_lines(path: Path):
    """Yield (line number, content) outside fenced/inline code and figure blocks."""
    fence_char = None
    fence_len = 0
    in_figure = False
    for number, raw in enumerate(path.read_text(encoding="utf-8", errors="replace").splitlines(), 1):
        line = raw
        fence = re.match(r"^\s*(`{3,}|~{3,})", line)
        if fence:
            marker = fence.group(1)
            if fence_char is None:
                fence_char, fence_len = marker[0], len(marker)
            elif marker[0] == fence_char and len(marker) >= fence_len:
                fence_char, fence_len = None, 0
            continue
        if fence_char is not None:
            continue
        if re.search(r"<figure\b", line, re.I):
            in_figure = True
        if in_figure:
            if re.search(r"</figure\s*>", line, re.I):
                in_figure = False
            continue
        # Figure captions/alt text are outside this glossary's prose scope.
        if re.match(r"^\s*!\[", line) or re.match(r"^\s*<img\b", line, re.I):
            continue
        line = re.sub(r"!\[[^\]]*\]\([^)]*\)(?:\{[^}]*\})?", " ", line)
        if re.match(r"^\s*```", line):
            continue
        # Remove inline code and HTML code/pre blocks where they occur on one line.
        line = re.sub(r"`+[^`]*`+", " ", line)
        line = re.sub(r"<(code|pre)\b[^>]*>.*?</\1\s*>", " ", line, flags=re.I)
        yield number, line


EMPHASIS = re.compile(r"(?<!\\)(\*\*|__|\*|_)(.+?)(?<!\\)\1")
PAREN = re.compile(r"([^()]{2,100}?)\s*\(([^()]{2,100})\)")


def clean_term(s: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"\[([^\]]+)\]\([^)]*\)", r"\1", s)).strip(" \t*_`.,;:—–-")


def d2l_extract() -> int:
    out = []
    D2L_CACHE.mkdir(parents=True, exist_ok=True)
    for repo, lang in REPO_LOCALES.items():
        base = CORPORA / repo
        if not base.is_dir():
            continue
        for path in sorted(base.rglob("*.md")):
            if ".git" in path.parts or path.name.endswith("_origin.md"):
                continue
            rel = path.relative_to(base).as_posix()
            for line_no, text in visible_markdown_lines(path):
                context = re.sub(r"\s+", " ", text).strip()
                if not context:
                    continue
                terms = [(clean_term(m.group(2)), "emphasis") for m in EMPHASIS.finditer(text)]
                for m in PAREN.finditer(text):
                    left, right = clean_term(m.group(1)), clean_term(m.group(2))
                    # Keep compact term-like pairs; reject sentence fragments,
                    # links, and long explanatory parentheses as obvious noise.
                    termish = (left and right and len(left) <= 80 and len(right) <= 80
                               and not re.search(r"[.!?\[\]/]", left + right))
                    if termish and left.casefold() != right.casefold():
                        terms.append((f"{left} ({right})", "bilingual_parentheses"))
                for term, extraction in terms:
                    if len(term) < 2 or len(term) > 180:
                        continue
                    out.append({"candidate": True, "lang": lang, "path": rel,
                                "line": line_no, "term": term, "extraction": extraction,
                                "context": context[:360], "repository": repo})
    out.sort(key=lambda r: (r["lang"], r["path"], r["line"], r["term"]))
    write_jsonl(D2L_CACHE / "d2l-emphasis.jsonl", out)
    print(f"D2L text candidates: {len(out)} rows at {D2L_CACHE / 'd2l-emphasis.jsonl'}")
    return len(out)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--google-only", action="store_true")
    ap.add_argument("--d2l-only", action="store_true")
    args = ap.parse_args()
    if not args.d2l_only:
        _, failed = google_extract()
    else:
        failed = 0
    if not args.google_only:
        d2l_extract()
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
