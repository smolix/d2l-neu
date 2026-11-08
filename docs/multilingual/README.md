# Multilingual D2L terminology

This directory contains a reusable, source-traceable glossary for translating **prose** in [the current D2L book](https://d2l.smola.org/). It covers English → Simplified Chinese (`zh-Hans`), Japanese (`ja`), Korean (`ko`), Spanish (`es`), Arabic (`ar`), Brazilian Portuguese (`pt-BR`), Vietnamese (`vi`), French (`fr`), and Hindi (`hi`). Turkish (`tr`) and German (`de`) are included because source discovery found earlier D2L translations in those languages.

**Read [COVERAGE.md](COVERAGE.md) before using the data.** A populated cell can be an unverified proposal. This is a terminology resource with an explicit review queue, not a claim of complete linguistic or native-speaker certification. Coverage of the English book and evidence coverage in each target language are separate measurements.

The final catalog has **1,184 distinct English concepts** and **13,024 target entries**. Of those target entries, **2,145 are source-attested** (including deliberately retained terms) and **10,879 are proposals**. All entries have a preferred expression, but the proposals still need terminology review. The live-book inventory covers 275 navigation-linked pages; glossary completeness is not implied by that inventory.

## Files to use

| File | Purpose |
| --- | --- |
| [concepts.jsonl](concepts.jsonl) | Canonical English concepts, senses, aliases, and book references. |
| [concept-redirects.json](concept-redirects.json) | Retired synonym IDs mapped directly to retained concepts; see [merge decisions](notes/deduplication.md). |
| `terms/<locale>.jsonl` | Canonical target expressions, alternatives, evidence, and decision notes. One record per concept. |
| [sources.jsonl](sources.jsonl) | Generated source registry; editable source records live in `sources/`. |
| [exports/glossary-wide.csv](exports/glossary-wide.csv) | Review table with a separate status column for every language. UTF-8 with BOM. |
| `exports/<locale>.md` | Readable glossary with source locators. |
| `exports/automation/<locale>.jsonl` | Source-attested subset for contextual translation assistance. Excludes proposals and unresolved entries. |
| [COVERAGE.md](COVERAGE.md), [coverage.json](coverage.json) | Counts by language and topic, and the remaining review queue. |
| [ENGLISH_COVERAGE.md](ENGLISH_COVERAGE.md), [english-coverage.json](english-coverage.json) | English chapter inventory and source-reference coverage. A count is not proof of exhaustive review. |
| [live-book-manifest.json](live-book-manifest.json) | Retrieval hashes for the 275 pages linked from the live book's navigation. |
| [DISCOVERY.md](DISCOVERY.md), [translation-inventory.json](translation-inventory.json) | D2L edition discovery, DNS/HTTP observations, repository commits, and source limitations. |
| [PLAN.md](PLAN.md) | Astra's plan and the subsequent Hindi and expanded-coverage requirements. |
| [QA.md](QA.md), [evidence-audit.json](evidence-audit.json) | Final checks, known limitations, and reproducible citation audit results. |
| `notes/` | Language-specific source choices, conflicts, search limits, and audits. |
| `candidates/` | Mechanical extraction and machine suggestions. **Never use directly as approved terminology.** |

## Statuses and evidence

`verified_d2l` means an inspected earlier D2L source attests the expression in the intended sense. `verified_external` means an inspected external teaching source does. These labels describe source attestation; some published sources themselves use machine translation, and their provenance is recorded. `invariant` means an attested name or loanword is deliberately retained, with a reason. `proposed` is a translation suggestion lacking sufficient supporting evidence. `unresolved` has no defensible preferred expression yet.

Some proposals are machine-translation suggestions for public English terms. They are labeled as such in the candidate data and remain unverified unless a separate teaching source and sense check support them. A machine translation service is not treated as a teaching source.

The canonical fields are:

- Concept: `id`, `en`, `aliases_en`, `topic`, `sense`, `book_refs` (local `path` or live `url`, plus `locator`).
- Target: `concept_id`, `preferred`, `alternatives`, `status`, `source_ids`, `evidence`, `notes`.
- Evidence: `source_id`, `locator`, `source_term`, `context`. Context generally summarizes why the occurrence supports the meaning; it is not necessarily a quotation.
- Source: `id`, `title`, `locale`, `kind`, `url`, `accessed`, `publisher`, with version/commit/license information when available.

English concepts also carry `source_review`: `sense_spot_checked` identifies the focused English audit; `prose_occurrence_checked` records an occurrence that passed the artifact screen but has not received individual semantic certification. For merged synonyms, this is the strongest existing review level, not certification of every merged reference. See [notes/english-audit.md](notes/english-audit.md). Do not interpret a large catalog as proof that every English source reference has been independently reviewed in context.

Do not infer that an alternative is verified merely because the preferred term is. Notes distinguish variants, rejected literal translations, and proposed alternatives. English aliases refer to one sense; identical English words in different senses must remain distinguishable through concept IDs and definitions. Spelling normalization must not collapse statistical bias into a model's additive bias, a covariance kernel into a convolution kernel, or an RL return into a programming statement.

## Applying the glossary

Supply the relevant concept definitions and source-attested target expressions to a translator together with the paragraph being translated. Match longer phrases before shorter ones, but let the paragraph's meaning decide which sense applies. Permit target-language inflection and word order. A glossary lemma is not a command to substitute a literal string everywhere.

Preserve code blocks, inline program identifiers, variable names, mathematical symbols, URLs, link destinations, citation keys, cross-reference anchors, framework directives, and figure assets/text. Figure captions and image alt text are excluded from this glossary's extraction scope as a conservative interpretation of the requested text-only translation. Ordinary prose about a tensor, gradient, kernel, or model remains in scope even when related code uses those words as identifiers.

For Chinese this resource chooses Simplified Chinese; for Portuguese it chooses Brazilian usage. Do not silently treat those as covering Traditional Chinese or European Portuguese. Arabic strings are stored in logical Unicode order, without decorative bidi controls. Display direction belongs to the consuming application.

## Refreshing and validating

Validate and regenerate exports without network access:

```sh
python docs/multilingual/scripts/validate_export.py
python docs/multilingual/scripts/report_english_coverage.py
```

The validator checks concept/locale completeness, duplicate IDs, source references, evidence fields, statuses, Unicode normalization, and export eligibility. It stops without writing exports when structural errors occur. It does **not** prove semantic equivalence, natural usage, or source licensing. Those require the language audits and source inspection.

With the recorded corpus snapshots cached locally, `python docs/multilingual/scripts/audit_evidence.py` checks cited source expressions against original file line numbers. It explicitly reports web/PDF references and unavailable snapshots as untested. A matching string is evidence of occurrence, not a substitute for checking its meaning.

Discovery/extraction helpers use caches under `/tmp/d2l-glossary-corpora`, `/tmp/d2l-glossary-google`, and `/tmp/d2l-live-book`. These caches are not required to consume the final glossary. Python 3 is required; `scripts/cache_live_book.py` additionally uses `beautifulsoup4`; repository/DNS discovery uses `git` and `dig`. PDF inspection may require PyMuPDF or Poppler. Helpers that download or regenerate candidates must be followed by review; do not overwrite reviewed canonical records with unreviewed candidates.

The `build_*` and language-specific scripts are construction aids, not the normal refresh command: rerunning them can replace later editorial decisions. Use the checked-in canonical JSONL for consumption and the validator above for exports. `scripts/reconcile_redirects.py` applies reviewed concept merges to existing shards and archives all pre-merge variants in `candidates/merged-variant-terms.jsonl`; it does not replay old archived choices over later corrections. Optional `merged_from` fields identify merged target records. Archived variants do not inherit the retained preferred term's status.

```sh
python docs/multilingual/discover_corpora.py
python docs/multilingual/discover_inventory.py
python docs/multilingual/scripts/cache_live_book.py
python docs/multilingual/scripts/extract_candidates.py
```

The live-book cacher reuses existing cached HTML; use a fresh cache for a new snapshot. Repository discovery may fetch newer heads, whereas existing evidence URLs are pinned to the recorded commits. Retain old snapshots when reproducing old evidence. Google’s large all-topics glossary sometimes returns an English body under a localized title; topic pages were inspected separately. HTTP success and a language selector alone do not establish translated content.

## Attribution and execution

D2L authors and translation contributors are identified through the linked edition repositories. External authors/publishers and document locators are recorded in the source registry. This directory stores technical terms, independently written definitions/decision notes, and brief evidence rather than redistributing the source books. See [DISCOVERY.md](DISCOVERY.md) and source metadata for license details, including differences among independent translations. The source registry is not a blanket license for republishing source passages.

Astra produced the plan. Terra constructed and expanded the English catalog and worked on Romance-language sources. Sol agents performed source discovery and language review; Luna handled bounded mechanical extraction. Deterministic scripts handled downloads, joins, validation, and exports. No dollar-cost estimate is supplied because billing totals are not available in this workspace.

See [English source portability](notes/english-source-portability.md) for the final fresh-checkout validation changes and the 137 excerpt-backed citations still needing current-book review.
