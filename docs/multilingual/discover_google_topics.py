"""Cache locale-specific Google ML glossary topic pages for evidence review."""

from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import urllib.request


ROOT = Path("/tmp/d2l-glossary-corpora/google-topics")
TOPICS = "fundamentals agent clustering df generative googlecloud metrics responsible-ai tensorflow".split()
LOCALES = "en ar de tr hi".split()


def fetch(topic: str, locale: str) -> str:
    url = f"https://developers.google.com/machine-learning/glossary/{topic}?hl={locale}"
    path = ROOT / f"{topic}-{locale}.html"
    if path.exists() and path.stat().st_size > 10000:
        return f"cached {topic} {locale} {path.stat().st_size}"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=20) as res:
            data = res.read()
        path.write_bytes(data)
        return f"fetched {topic} {locale} {len(data)}"
    except Exception as exc:
        return f"ERROR {topic} {locale} {exc}"


def main() -> None:
    ROOT.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(max_workers=10) as pool:
        for fut in as_completed([pool.submit(fetch, topic, locale) for topic in TOPICS for locale in LOCALES]):
            print(fut.result(), flush=True)


if __name__ == "__main__":
    main()
