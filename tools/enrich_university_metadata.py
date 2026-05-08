#!/usr/bin/env python3
"""Fill university logo and location metadata.

Data sources, in order:
  * Wikidata entity search and claims for coordinates (P625), logo image (P154),
    and official website (P856).
  * Clearbit logo by official website domain when Wikidata has no logo.
  * A generated text SVG fallback for entries where no external logo is found.

The fallback is intentionally marked in the entry's metadata so it can be
replaced later with an official logo without ambiguity.
"""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import mimetypes
import re
import sys
import time
import unicodedata
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

REPO = Path("/home/smola/d2l/d2l-neu")
UNIVERSITIES = REPO / "tools" / "universities.json"
LOGOS_DIR = REPO / "static" / "landing" / "universities"

USER_AGENT = "d2l-ai-university-metadata/1.0 (https://d2l.ai)"


def slugify(name: str) -> str:
    slug = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode()
    slug = re.sub(r"[^A-Za-z0-9]+", "-", slug).strip("-")
    return slug or "University"


def norm(s: str) -> str:
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode().lower()
    return re.sub(r"[^a-z0-9]+", " ", s).strip()


def http_json(url: str, timeout: int = 15) -> dict | None:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT, "Accept": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read().decode("utf-8"))
    except Exception as exc:
        print(f"  ! json request failed: {exc}", file=sys.stderr)
        return None


def http_bytes(url: str, timeout: int = 25) -> tuple[bytes, str] | None:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.read(), r.headers.get("Content-Type", "")
    except urllib.error.HTTPError as exc:
        print(f"  ! download failed {exc.code}: {url}", file=sys.stderr)
    except Exception as exc:
        print(f"  ! download failed: {exc}", file=sys.stderr)
    return None


def search_wikidata(name: str) -> str | None:
    params = urllib.parse.urlencode(
        {
            "action": "wbsearchentities",
            "search": name,
            "language": "en",
            "format": "json",
            "limit": 5,
        }
    )
    data = http_json(f"https://www.wikidata.org/w/api.php?{params}")
    if not data:
        return None
    wanted = norm(name)
    best = None
    for item in data.get("search", []):
        label = norm(item.get("label", ""))
        desc = (item.get("description") or "").lower()
        if label == wanted:
            return item["id"]
        if best is None and any(word in desc for word in ("university", "college", "institute", "school")):
            best = item["id"]
    return best


def entity(qid: str) -> dict | None:
    data = http_json(f"https://www.wikidata.org/wiki/Special:EntityData/{qid}.json")
    if not data:
        return None
    return data.get("entities", {}).get(qid)


def first_claim(entity_obj: dict, prop: str):
    claims = entity_obj.get("claims", {}).get(prop) or []
    if not claims:
        return None
    return claims[0].get("mainsnak", {}).get("datavalue", {}).get("value")


def coordinate(entity_obj: dict) -> dict | None:
    value = first_claim(entity_obj, "P625")
    if not value:
        return None
    return {
        "lat": round(float(value["latitude"]), 6),
        "lon": round(float(value["longitude"]), 6),
        "source": "wikidata",
    }


def website(entity_obj: dict) -> str | None:
    value = first_claim(entity_obj, "P856")
    return value if isinstance(value, str) else None


def commons_file_url(filename: str) -> str:
    quoted = urllib.parse.quote(filename.replace(" ", "_"))
    return f"https://commons.wikimedia.org/wiki/Special:Redirect/file/{quoted}"


def ext_for(url: str, content_type: str) -> str:
    m = re.search(r"\.(svg|png|jpe?g|webp)(?:[?#].*)?$", url, re.I)
    if m:
        return m.group(1).lower().replace("jpeg", "jpg")
    guessed = mimetypes.guess_extension(content_type.split(";")[0].strip())
    if guessed:
        return guessed.lstrip(".").replace("jpeg", "jpg")
    if "svg" in content_type:
        return "svg"
    if "png" in content_type:
        return "png"
    return "jpg"


def logo_candidates(entity_obj: dict, site: str | None) -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []
    value = first_claim(entity_obj, "P154")
    if isinstance(value, str):
        out.append((commons_file_url(value), "wikidata-logo"))
    if site:
        domain = urllib.parse.urlparse(site if "://" in site else f"https://{site}").netloc
        domain = domain.removeprefix("www.")
        if domain:
            out.append((f"https://logo.clearbit.com/{domain}", "clearbit"))
    return out


def initials(name: str) -> str:
    words = re.findall(r"[A-Za-z0-9]+", unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode())
    skip = {"of", "the", "and", "for", "at", "in"}
    letters = [w[0].upper() for w in words if w.lower() not in skip]
    return "".join(letters[:4]) or "U"


def fallback_svg(name: str) -> bytes:
    digest = hashlib.sha256(name.encode("utf-8")).hexdigest()
    hue = int(digest[:2], 16)
    bg = f"hsl({hue}, 55%, 30%)"
    fg = "#ffffff"
    text = html.escape(initials(name))
    title = html.escape(name)
    return f'''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 256 256" role="img" aria-label="{title}">
  <rect width="256" height="256" rx="28" fill="{bg}"/>
  <text x="128" y="142" text-anchor="middle" dominant-baseline="middle" font-family="Arial, Helvetica, sans-serif" font-size="72" font-weight="700" fill="{fg}">{text}</text>
</svg>
'''.encode("utf-8")


def save_logo(entry: dict, url: str, source: str) -> bool:
    downloaded = http_bytes(url)
    if not downloaded:
        return False
    body, content_type = downloaded
    if len(body) < 200:
        return False
    ext = ext_for(url, content_type)
    if ext not in {"svg", "png", "jpg", "jpeg", "webp"}:
        ext = "png"
    filename = f"{entry['slug']}.{ext.replace('jpeg', 'jpg')}"
    (LOGOS_DIR / filename).write_bytes(body)
    entry["logo"] = filename
    entry["logo_source"] = source
    return True


def ensure_logo(entry: dict, entity_obj: dict | None) -> None:
    if entry.get("logo"):
        return
    site = website(entity_obj) if entity_obj else None
    for url, source in logo_candidates(entity_obj or {}, site):
        print(f"  logo try {source}: {url}")
        if save_logo(entry, url, source):
            return
    filename = f"{entry['slug']}.svg"
    (LOGOS_DIR / filename).write_bytes(fallback_svg(entry["name"]))
    entry["logo"] = filename
    entry["logo_source"] = "generated-fallback"


def enrich_entry(entry: dict, force: bool = False) -> bool:
    needs_location = force or not entry.get("location")
    needs_logo = force or not entry.get("logo")
    if not needs_location and not needs_logo:
        return False

    qid = entry.get("wikidata_id") or search_wikidata(entry["name"])
    entity_obj = entity(qid) if qid else None
    if qid:
        entry["wikidata_id"] = qid

    if needs_location and entity_obj:
        loc = coordinate(entity_obj)
        if loc:
            loc["wikidata_id"] = qid
            entry["location"] = loc

    if needs_logo:
        ensure_logo(entry, entity_obj)

    return True


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--sleep", type=float, default=0.25)
    parser.add_argument("--missing-only", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    entries = json.loads(UNIVERSITIES.read_text(encoding="utf-8"))
    changed = 0
    processed = 0
    for entry in entries:
        if args.missing_only and entry.get("logo") and entry.get("location"):
            continue
        if args.limit and processed >= args.limit:
            break
        processed += 1
        print(f"[{processed}] {entry['name']}")
        if enrich_entry(entry, force=args.force):
            changed += 1
            UNIVERSITIES.write_text(json.dumps(entries, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        time.sleep(args.sleep)

    missing_logo = sum(1 for e in entries if not e.get("logo"))
    missing_location = sum(1 for e in entries if not e.get("location"))
    print(f"processed={processed} changed={changed} missing_logo={missing_logo} missing_location={missing_location}")


if __name__ == "__main__":
    main()
