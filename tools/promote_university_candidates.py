#!/usr/bin/env python3
"""Promote verified adoption candidates into tools/universities.json.

The candidate file is a holding area for universities that have verified D2L
course evidence but were not yet renderable because they had no logo asset.
This script adds them to the canonical university list with a stable slug. A
later metadata pass fills in logos and locations.
"""

import json
import re
import unicodedata
from pathlib import Path

REPO = Path("/home/smola/d2l/d2l-neu")
UNIVERSITIES = REPO / "tools" / "universities.json"
CANDIDATES = REPO / "tools" / "university_adoption_candidates.json"


def norm(s: str) -> str:
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode().lower()
    return re.sub(r"[^a-z0-9]+", " ", s).strip()


def slugify(name: str) -> str:
    slug = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode()
    slug = re.sub(r"[^A-Za-z0-9]+", "-", slug).strip("-")
    return slug or "University"


def main() -> None:
    entries = json.loads(UNIVERSITIES.read_text(encoding="utf-8"))
    candidates = json.loads(CANDIDATES.read_text(encoding="utf-8"))

    by_name = {norm(e["name"]): e for e in entries}
    slugs = {e["slug"].lower() for e in entries}
    added = 0

    for candidate in candidates:
        key = norm(candidate["name"])
        if key in by_name:
            entry = by_name[key]
            existing_urls = {ev.get("url", "") for ev in entry.get("evidence") or []}
            for ev in candidate.get("evidence") or []:
                if ev.get("url") and ev["url"] not in existing_urls:
                    entry.setdefault("evidence", []).append(ev)
            if candidate.get("country") and not entry.get("country"):
                entry["country"] = candidate["country"]
            continue

        base = slugify(candidate["name"])
        slug = base
        i = 2
        while slug.lower() in slugs:
            slug = f"{base}-{i}"
            i += 1
        slugs.add(slug.lower())

        entries.append(
            {
                "slug": slug,
                "name": candidate["name"],
                "country": candidate.get("country", ""),
                "logo": None,
                "location": None,
                "evidence": candidate.get("evidence") or [],
            }
        )
        by_name[key] = entries[-1]
        added += 1

    entries.sort(key=lambda e: e["slug"].lower())
    UNIVERSITIES.write_text(json.dumps(entries, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"promoted {added} candidates; total entries {len(entries)}")


if __name__ == "__main__":
    main()
