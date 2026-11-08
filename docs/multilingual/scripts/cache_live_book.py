#!/usr/bin/env python3
"""Cache published English prose for terminology audits (requires beautifulsoup4).

HTML and extracted text stay in /tmp. Only retrieval hashes/URLs are written to
the repository. Code, equations and figures are removed before extraction.
"""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from urllib.parse import urljoin, urlparse
from urllib.request import Request, urlopen
import re
import time

from bs4 import BeautifulSoup

BASE = Path(__file__).resolve().parents[1]
CACHE = Path('/tmp/d2l-live-book')
HOME = 'https://d2l.smola.org/'


def get(url):
    error = None
    for attempt in range(3):
        try:
            with urlopen(Request(url, headers={'User-Agent': 'D2LGlossaryAudit/1.0'}), timeout=30) as res:
                return res.read()
        except Exception as exc:
            error = exc
            time.sleep(attempt + 1)
    raise error


def fetch(url):
    name = urlparse(url).path.strip('/') or 'index.html'
    path = CACHE / name
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        if path.exists():
            data = path.read_bytes()
        else:
            data = get(url)
            path.write_bytes(data)
        soup = BeautifulSoup(data, 'html.parser')
        main = soup.find('main') or soup
        for tag in main.select('pre, code, figure, img, .sourceCode, .cell-output, .math, nav, script, style, .references, #refs'):
            tag.decompose()
        headings = [{'text': x.get_text(' ', strip=True), 'anchor': x.get('id') or (x.parent.get('id') if x.parent else ''), 'level': x.name} for x in main.find_all(re.compile('^h[1-6]$'))]
        emphasis = []
        for x in main.find_all(['em', 'strong', 'dfn']):
            term = x.get_text(' ', strip=True)
            if not term or len(term) > 160:
                continue
            section = x.find_parent('section')
            emphasis.append({'term': term, 'anchor': section.get('id', '') if section else '', 'context': x.parent.get_text(' ', strip=True)[:800]})
        text = main.get_text('\n', strip=True)
        path.with_suffix('.txt').write_text(text)
        path.with_suffix('.json').write_text(json.dumps({'url': url, 'headings': headings, 'emphasis': emphasis, 'text': text}, ensure_ascii=False))
        return {'url': url, 'retrieved_utc': datetime.fromtimestamp(path.stat().st_mtime, timezone.utc).isoformat(), 'sha256': hashlib.sha256(data).hexdigest(), 'html_bytes': len(data), 'text_chars': len(text), 'headings': len(headings), 'emphasized_candidates': len(emphasis), 'cache_path': str(path), 'status': 'retrieved'}
    except Exception as exc:
        return {'url': url, 'status': 'failed', 'error': str(exc)}


def main():
    CACHE.mkdir(parents=True, exist_ok=True)
    soup = BeautifulSoup(get(HOME), 'html.parser')
    urls = set()
    for a in soup.select('a[href]'):
        u = urlparse(urljoin(HOME, a['href']))
        if u.netloc == 'd2l.smola.org' and u.path.startswith('/chapter_') and u.path.endswith('.html'):
            urls.add(u._replace(query='', fragment='').geturl())
    with ThreadPoolExecutor(max_workers=10) as pool:
        records = list(pool.map(fetch, sorted(urls)))
    manifest = {'extracted_utc': datetime.now(timezone.utc).isoformat(), 'index_url': HOME, 'exclusions': ['code', 'figures', 'equations', 'cell output', 'bibliography'], 'pages': records}
    (BASE / 'live-book-manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n')
    print(f"Cached {sum(x['status'] == 'retrieved' for x in records)}/{len(records)} live pages at {CACHE}")


if __name__ == '__main__':
    main()
