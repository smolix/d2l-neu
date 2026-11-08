# D2L translation source discovery

Checked 2026-11-08 UTC. The machine-readable evidence and exact pinned commits are in [translation-inventory.json](translation-inventory.json). Rebuild it with `python3 docs/multilingual/discover_inventory.py`; fetch sparse text-only GitHub corpora with `python3 docs/multilingual/discover_corpora.py`.

## Live D2L language sites and repository sources

| Language | `*.d2l.ai` DNS / homepage | Source text | Assessment for glossary work |
|---|---|---|---|
| English | [en.d2l.ai](https://en.d2l.ai/) 200; [d2l.ai](https://d2l.ai/) 200 | [d2l-ai/d2l-en](https://github.com/d2l-ai/d2l-en), `master` | Baseline English source. |
| Chinese | [zh.d2l.ai](https://zh.d2l.ai/) 200 | [d2l-ai/d2l-zh](https://github.com/d2l-ai/d2l-zh), `master` | Strong edited translation; 140 chapter Markdown files. Historic `zh-v1` and `zh-v2` hosts also resolve and return 200. |
| Japanese | [ja.d2l.ai](https://ja.d2l.ai/) 200 | [d2l-ai/d2l-ja](https://github.com/d2l-ai/d2l-ja), `master` | 167 chapter files; `config.ini` says `translator = aws`; inspect terms for machine translation artifacts. |
| Korean | [ko.d2l.ai](https://ko.d2l.ai/) 200 | [d2l-ai/d2l-ko](https://github.com/d2l-ai/d2l-ko), `master` | 167 chapter files; `translator = aws`; inspect terms. |
| Portuguese | [pt.d2l.ai](https://pt.d2l.ai/) 200 | [d2l-ai/d2l-pt](https://github.com/d2l-ai/d2l-pt), `master` | 168 chapter files, but English passages remain, including the opening of `linear-regression.md`; verify each target phrase. |
| Vietnamese | [vi.d2l.ai](https://vi.d2l.ai/) 200 | [d2l-ai/d2l-vi](https://github.com/d2l-ai/d2l-vi), `main` | 166 chapter files. The English homepage instead links to `d2l.aivivn.com`, which currently serves unrelated gambling content; use the official repository or `vi.d2l.ai`. |
| Turkish | [tr.d2l.ai](https://tr.d2l.ai/) 200 | [d2l-ai/d2l-tr](https://github.com/d2l-ai/d2l-tr), `master` | Substantial Turkish body text in 167 chapter files, with machine translation caveats. Include Turkish in glossary work. |
| German | `de.d2l.ai` NXDOMAIN | [d2l-ai/d2l-de](https://github.com/d2l-ai/d2l-de), `master` | Only 31 chapter files at 2020 snapshot, but their prose is genuinely German. Useful for early chapter terms; supplement elsewhere. |
| Arabic | `ar.d2l.ai` NXDOMAIN | [mohamed-ali/d2l-ar](https://github.com/mohamed-ali/d2l-ar) third-party translation and [Dr. Alaa Taima's three-part edition](https://dlarabic.com/%D9%83%D8%AA%D8%A7%D8%A8-%D8%A7%D9%84%D8%AA%D8%B9%D9%85%D9%82-%D9%81%D9%8A-%D8%A7%D9%84%D8%AA%D8%B9%D9%84%D9%85-%D8%A7%D9%84%D8%B9%D9%85%D9%8A%D9%82%D8%A7%D9%84%D8%AC%D8%B2%D8%A1-%D8%A7%D9%84%D8%A7/) | GitHub fork has 24 chapter files but only about six with much Arabic prose. Taima's three PDFs are richer and locally cached as text in `/tmp/d2l-glossary-corpora/d2l-ar-taima-part{1,2,3}.txt`; distinguish this independent edition and its reuse rights. |
| Spanish | `es.d2l.ai` NXDOMAIN | [acca3003/d2l-es](https://github.com/acca3003/d2l-es) third-party fork | Mostly unmodified English chapters. Spanish README, preface and introduction exist; not a general Spanish parallel corpus. |
| French | `fr.d2l.ai` NXDOMAIN | [julien-blanchon/d2l-fr](https://github.com/julien-blanchon/d2l-fr) third-party fork | Despite name and French description, chapter prose is essentially English. Not a French terminology authority. |
| Hindi | `hi.d2l.ai` NXDOMAIN | No relevant `d2l-hi` GitHub repository found in public repository search | No D2L Hindi parallel corpus found; build Hindi terms from independent Hindi technical teaching sources and label proposals honestly. |

The current [D2L English homepage](https://d2l.ai/) links Chinese, Portuguese, Turkish, Vietnamese, Korean, and Japanese translations. It does not link German, Arabic, Spanish, French, or Hindi. The official [d2l-ai organization repository list](https://github.com/orgs/d2l-ai/repositories) includes source repos for English, Chinese, Japanese, Korean, Portuguese, Vietnamese, Turkish, and German. Its automatically generated notebook mirror repositories are separate from chapter source.

The linked Vietnamese hostname `d2l.aivivn.com` returned HTTP 200 on this check, but its page title was unrelated: “SAWER138 : Situs Slot Gacor PG Soft Terpercaya & Resmi Gampang Menang Hari Ini | Foot Locker ID.” This observation establishes that it is currently unsuitable as a D2L translation source; it does not establish why the content changed.

Dr. Taima's author-hosted PDF links are [part 1](https://drive.google.com/file/d/1SrmT_r8dNK42IqyS0gwXtbLCZbk5G8eu/view), [part 2](https://drive.google.com/file/d/1UqEu0amRfAvJD0L1HosIn3UJi0FkNemU/view), and [part 3](https://drive.google.com/file/d/1-HlY2IqKIIEKfFrxebBu51z9oAPftLfA/view). The downloaded author-hosted parts 2 and 3 were byte-identical (SHA-256) to the `linguist.page` mirrors used for local text extraction. This supports using the author links as the cited edition; it does not establish reuse rights.

## DNS scope and limitations

`dig` queried A records for 47 candidate subdomains: language codes `en zh ja ko es ar pt vi fr hi tr de id ru it fa bn th uk nl pl ta te ml ur he sw ms sv no da fi cs el hu ro sk sr bg hr ca eu gl`, plus `zh-v1`, `zh-v2`, `classic`, and `preview`. Positive language answers were `en zh ja ko pt vi tr`; historical/non-language hosts `zh-v1 zh-v2 classic preview` also answered. A randomized nonexistent name returned NXDOMAIN, so this is not a broad A-record wildcard. The `d2l.ai` zone advertises four AWS Route 53 authoritative nameservers; no zone transfer or provider inventory was available. DNS candidate probing **cannot prove that no other language subdomain exists**. The GitHub organization list and live homepage links are independent cross-checks, and the candidate set covers all languages requested here.

All seven positive language hosts returned HTTP 200 with language-specific or book-specific titles. A DNS answer alone would not establish translation content. The `preview` host had an HTTPS handshake timeout during this check, and it is not treated as a language edition. Raw DNS responses, TTLs, HTTP URLs and status are retained in the JSON.

## Corpus use and licensing

The sparse clones under `/tmp/d2l-glossary-corpora/` contain chapter `*.md` and `*.qmd`, top-level Markdown, config and license files, without large image assets. Use `chapter_*/<slug>.md` from pinned commits as locators. Files ending `_origin.md` are English originals inside several translations; exclude them from target-language evidence. The source repositories' `config.ini` files mostly declare book prose CC BY-SA 4.0 and sample code MIT-0; check each repository's exact license files before redistributing text. The Chinese repository's root `LICENSE` is Apache 2.0 while its `config.ini` names CC BY-SA 4.0 and MIT-0, so this mismatch needs explicit resolution for any redistribution. The German and Arabic GitHub repositories lack a root `LICENSE` file in the sparse snapshot, although their config files include copyright statements. Dr. Taima's PDF edition is independently published; its license is not established by the D2L GitHub license.

The local corpus cache is an analysis aid, not a vendored project dependency. The JSON contains each exact commit and an archive URL; regenerate the cache when source versions change.
