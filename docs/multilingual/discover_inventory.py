"""Rebuild the public D2L language endpoint/source inventory.

The DNS sweep is a candidate check, not an exhaustive zone transfer.
"""

from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import urllib.error
import urllib.request


HERE = Path(__file__).resolve().parent
CORPORA = Path("/tmp/d2l-glossary-corpora")
CODES = "en zh ja ko es ar pt vi fr hi tr de id ru it fa bn th uk nl pl ta te ml ur he sw ms sv no da fi cs el hu ro sk sr bg hr ca eu gl zh-v1 zh-v2 classic preview".split()
OFFICIAL = {"en", "zh", "ja", "ko", "pt", "vi", "tr", "de"}
FORKS = {"es": "acca3003/d2l-es", "fr": "julien-blanchon/d2l-fr", "ar": "mohamed-ali/d2l-ar"}


def dig(name: str, rtype: str) -> dict:
    proc = subprocess.run(["dig", "+time=2", "+tries=1", "+noall", "+comments", "+answer", name, rtype], capture_output=True, text=True)
    status = "UNKNOWN"
    for line in proc.stdout.splitlines():
        if "status:" in line:
            status = line.split("status:", 1)[1].split(",", 1)[0].strip()
            break
    answers = [line.strip() for line in proc.stdout.splitlines() if line and not line.startswith(";")]
    return {"status": status, "answers": answers}


def http_url(url: str) -> dict:
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (D2L translation inventory)"})
        with urllib.request.urlopen(req, timeout=8) as res:
            body = res.read(4096).decode("utf-8", "replace")
            import re
            match = re.search(r"<title>(.*?)</title>", body, re.I | re.S)
            return {"status": res.status, "url": res.geturl(), "title": match.group(1).strip() if match else None}
    except Exception as exc:
        return {"error": str(exc)[:160]}


def http(code: str) -> dict:
    return http_url(f"https://{code}.d2l.ai/")


def repo(name: str, provenance: str) -> dict:
    target = CORPORA / name.split("/")[-1]
    if not (target / ".git").exists():
        return {"repository": name, "provenance": provenance, "cache": None}
    sha = subprocess.check_output(["git", "-C", str(target), "rev-parse", "HEAD"], text=True).strip()
    branch = subprocess.check_output(["git", "-C", str(target), "symbolic-ref", "--short", "HEAD"], text=True).strip()
    files = [p for p in target.glob("chapter_*/*.md") if not p.name.endswith("_origin.md")]
    origins = list(target.glob("chapter_*/*_origin.md"))
    licenses = [p.name for p in target.glob("LICENSE*")]
    return {
        "repository": name,
        "provenance": provenance,
        "url": f"https://github.com/{name}",
        "branch": branch,
        "commit": sha,
        "archive": f"https://github.com/{name}/archive/{sha}.zip",
        "raw_url_template": f"https://raw.githubusercontent.com/{name}/{sha}/{{path}}",
        "chapter_markdown_files": len(files),
        "origin_files_to_exclude": len(origins),
        "license_files": licenses,
        "cache": str(target),
    }


def main() -> None:
    checked = datetime.now(timezone.utc).isoformat()
    names = [f"{c}.d2l.ai" for c in CODES] + ["random-nonexistent-language-20261107.d2l.ai"]
    with ThreadPoolExecutor(max_workers=12) as pool:
        dns = {name: result for name, result in zip(names, pool.map(lambda name: dig(name, "A"), names))}
    live = [code for code in CODES if dns[f"{code}.d2l.ai"]["status"] == "NOERROR" and dns[f"{code}.d2l.ai"]["answers"]]
    with ThreadPoolExecutor(max_workers=12) as pool:
        web = {code: result for code, result in zip(live, pool.map(http, live))}
    sources = [repo(f"d2l-ai/d2l-{c}", "official d2l-ai organization") for c in ["en", "zh", "ja", "ko", "pt", "vi", "tr", "de"]]
    sources.extend(repo(v, "third-party fork") for v in FORKS.values())
    data = {
        "checked_utc": checked,
        "method": "Public DNS candidate sweep, HTTP GET, official D2L homepage links, GitHub organization listing and sparse repository snapshots. DNS is not exhaustive; no authorized zone inventory was available.",
        "authoritative_ns": dig("d2l.ai", "NS"),
        "soa": dig("d2l.ai", "SOA"),
        "wildcard_probe": "random-nonexistent-language-20261107.d2l.ai",
        "dns": dns,
        "http": web,
        "external_vietnamese_homepage_link_http": http_url("https://d2l.aivivn.com/"),
        "homepage_translation_links": {
            "zh": "https://zh.d2l.ai",
            "pt": "https://pt.d2l.ai/",
            "tr": "https://tr.d2l.ai/",
            "vi": "https://d2l.aivivn.com/",
            "ko": "https://ko.d2l.ai/",
            "ja": "https://ja.d2l.ai/",
        },
        "source_repositories": sources,
        "additional_sources": [
            {"language": "ar", "kind": "independent Arabic translation by Dr. Alaa Taima", "url": "https://dlarabic.com/كتاب-التعمق-في-التعلم-العميقالجزء-الا/", "part1_pdf": "https://drive.google.com/file/d/1SrmT_r8dNK42IqyS0gwXtbLCZbk5G8eu/view", "part2_pdf": "https://drive.google.com/file/d/1UqEu0amRfAvJD0L1HosIn3UJi0FkNemU/view", "part3_pdf": "https://drive.google.com/file/d/1-HlY2IqKIIEKfFrxebBu51z9oAPftLfA/view", "text_cache": "/tmp/d2l-glossary-corpora/d2l-ar-taima-part{1,2,3}.txt", "note": "Three-part Arabic translation. Author-hosted Google Drive PDF parts 2 and 3 are SHA-256 identical to the linguist.page mirrors used for text extraction; check reuse rights separately from D2L GitHub source license."},
            {"language": "hi", "kind": "no D2L Hindi translation found", "note": "No hi.d2l.ai DNS answer or relevant GitHub d2l-hi repository found in candidate searches."},
        ],
    }
    path = HERE / "translation-inventory.json"
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n")
    print(path, "live", live)


if __name__ == "__main__":
    main()
