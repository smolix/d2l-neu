#!/usr/bin/env python3
"""Incrementally fetch university logos for entries in universities.json
where `logo` is null.

For each missing entry: query Wikipedia REST `page/summary` for the university,
take the `originalimage`/`thumbnail` URL, download to
`static/landing/universities/<slug>.<ext>`, and mark the entry's `logo` field
in universities.json. The JSON is rewritten after every successful download
so progress is durable across restarts and rate-limit interruptions.

Usage:
    python tools/download_logos.py            # process all missing
    python tools/download_logos.py --limit 5  # stop after 5 successful downloads
    python tools/download_logos.py --dry-run  # report only

A small `_logo_skip.txt` may list slugs to permanently skip (e.g. organizations
that are not real universities or for which Wikipedia has no image).
"""
import argparse
import json
import re
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

REPO = Path("/home/smola/d2l/d2l-neu")
JSON_PATH = REPO / "tools" / "universities.json"
LOGOS_DIR = REPO / "static" / "landing" / "universities"
SKIP_FILE = REPO / "tools" / "_logo_skip.txt"

WIKI_SUMMARY = "https://en.wikipedia.org/api/rest_v1/page/summary/{title}"
USER_AGENT = "d2l-ai-logo-collector/1.0 (https://d2l.ai)"


def load_skip() -> set:
    if SKIP_FILE.exists():
        return {l.strip() for l in SKIP_FILE.read_text().splitlines() if l.strip() and not l.startswith("#")}
    return set()


def http_get(url: str, timeout=10) -> tuple[int, bytes]:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT, "Accept": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, r.read()
    except urllib.error.HTTPError as e:
        return e.code, b""
    except Exception as e:
        print(f"  ! request failed: {e}", file=sys.stderr)
        return 0, b""


def http_get_binary(url: str, timeout=20) -> tuple[int, bytes, str]:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            ct = r.headers.get("Content-Type", "")
            return r.status, r.read(), ct
    except urllib.error.HTTPError as e:
        return e.code, b"", ""
    except Exception as e:
        print(f"  ! download failed: {e}", file=sys.stderr)
        return 0, b"", ""


def title_candidates(name: str):
    """Generate Wikipedia title candidates for a university name."""
    # Strip leading number, common suffixes
    n = re.sub(r"^\s*\d+\.\s*", "", name).strip()
    # Strip "(Country)" / "— Course" suffixes
    n = re.sub(r"\s*[—–]\s*.*$", "", n)
    n = re.sub(r"\s*\([^()]+\)\s*$", "", n).strip()
    yield n
    # Try without comma-separated tails ("BITS Pilani, Hyderabad" → "BITS Pilani")
    if "," in n:
        yield n.split(",")[0].strip()
    # Common abbreviation fixes
    if "Universitat" in n:
        yield n.replace("Universitat", "University of").strip()
    if "Universität" in n:
        yield n.replace("Universität", "University of").strip()
    if "Université" in n:
        yield n.replace("Université", "University of").strip()
    if "Universidad" in n:
        yield n.replace("Universidad", "University of").strip()


def fetch_logo_url(name: str) -> tuple[str, str] | None:
    """Returns (image_url, page_title) or None."""
    for title in title_candidates(name):
        url = WIKI_SUMMARY.format(title=urllib.parse.quote(title.replace(" ", "_")))
        status, body = http_get(url)
        if status == 429:
            print("  ! 429 rate-limited; sleeping 60s")
            time.sleep(60)
            status, body = http_get(url)
        if status != 200 or not body:
            continue
        try:
            data = json.loads(body)
        except json.JSONDecodeError:
            continue
        # Skip disambiguation pages
        if data.get("type") == "disambiguation":
            continue
        # Prefer originalimage (full size), fall back to thumbnail
        for key in ("originalimage", "thumbnail"):
            obj = data.get(key)
            if obj and obj.get("source"):
                return obj["source"], data.get("title", title)
    return None


def ext_for_image(url: str, content_type: str) -> str:
    # Trust URL extension first
    m = re.search(r"\.(svg|png|jpe?g|webp)(\?.*)?$", url, re.I)
    if m:
        return m.group(1).lower().replace("jpeg", "jpg")
    if "svg" in content_type:
        return "svg"
    if "png" in content_type:
        return "png"
    if "webp" in content_type:
        return "webp"
    return "jpg"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="stop after N successful downloads")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--sleep", type=float, default=0.6, help="sleep between requests (sec)")
    args = ap.parse_args()

    skip = load_skip()
    entries = json.loads(JSON_PATH.read_text(encoding="utf-8"))

    pending = [e for e in entries if not e.get("logo") and e["slug"] not in skip]
    print(f"Pending logo downloads: {len(pending)} (of {len(entries)} total)")
    if args.dry_run:
        for e in pending[:20]:
            print(f"  - {e['slug']:<60} ({e['name']})")
        if len(pending) > 20:
            print(f"  ... and {len(pending) - 20} more")
        return

    succeeded = 0
    failed = 0
    for e in pending:
        if args.limit and succeeded >= args.limit:
            break
        print(f"[{succeeded + failed + 1}/{len(pending)}] {e['name']}")
        result = fetch_logo_url(e["name"])
        if not result:
            print("  - no Wikipedia image found; skip")
            failed += 1
            time.sleep(args.sleep)
            continue
        img_url, wiki_title = result
        print(f"  page: {wiki_title}")
        print(f"  image: {img_url}")
        status, body, ct = http_get_binary(img_url)
        if status != 200 or not body:
            print(f"  ! download failed (status {status})")
            failed += 1
            time.sleep(args.sleep)
            continue
        ext = ext_for_image(img_url, ct)
        filename = f"{e['slug']}.{ext}"
        out_path = LOGOS_DIR / filename
        out_path.write_bytes(body)
        e["logo"] = filename
        # Save canonical JSON immediately for resume safety
        JSON_PATH.write_text(json.dumps(entries, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"  saved {filename} ({len(body)} bytes)")
        succeeded += 1
        time.sleep(args.sleep)

    print(f"\nDone. succeeded={succeeded}, failed={failed}, remaining={len(pending) - succeeded - failed}")


if __name__ == "__main__":
    main()
