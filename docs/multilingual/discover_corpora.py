"""Fetch sparse, text-only D2L translation corpora for glossary research.

This is a discovery helper, not part of the site's build. Run it from any cwd.
It clones public GitHub repositories into /tmp/d2l-glossary-corpora.
"""

from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import subprocess
import sys


ROOT = Path("/tmp/d2l-glossary-corpora")
REPOS = ["d2l-en", "d2l-zh", "d2l-ja", "d2l-ko", "d2l-pt", "d2l-vi", "d2l-tr", "d2l-de"]
PATTERNS = ["/*.md", "/*.qmd", "/chapter_*/*.md", "/chapter_*/*.qmd", "/LICENSE*", "/config.ini"]


def run(*args: str) -> str:
    return subprocess.check_output(args, text=True, stderr=subprocess.STDOUT).strip()


def fetch(repo: str) -> str:
    target = ROOT / repo
    try:
        if not (target / ".git").exists():
            run("git", "clone", "--depth=1", "--filter=blob:none", "--sparse", f"https://github.com/d2l-ai/{repo}.git", str(target))
        run("git", "-C", str(target), "sparse-checkout", "set", "--no-cone", *PATTERNS)
        sha = run("git", "-C", str(target), "rev-parse", "HEAD")
        count = sum(1 for p in target.rglob("*") if p.is_file() and p.suffix in {".md", ".qmd"} and ".git" not in p.parts)
        return f"{repo} {sha} {count} text files"
    except subprocess.CalledProcessError as exc:
        return f"{repo} ERROR {exc.output[-500:]}"


def main() -> None:
    ROOT.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(max_workers=3) as pool:
        for future in as_completed([pool.submit(fetch, repo) for repo in REPOS]):
            print(future.result(), flush=True)


if __name__ == "__main__":
    main()
