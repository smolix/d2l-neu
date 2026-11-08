# Validation and review handoff

The final catalog contains **1,184 distinct English concepts**, with **13,024 preferred target expressions across 11 locales**. The requested nine languages include Hindi; discovery also found Turkish and German editions. There are **2,145 source-attested target entries** (1,347 D2L, 794 external, and 4 attested invariants) and **10,879 editorial proposals**. There are no blank or unresolved entries. The proposal count is a review queue, not a count of verified translations.

The working catalog grew from 445 to 1,205 entries as newer chapters and definitions were inspected. A final semantic deduplication merged 21 equivalent names, retaining their aliases, distinct source references, and translation variants. See [the decisions](notes/deduplication.md), [ID redirects](concept-redirects.json), and the archived `candidates/merged-variant-terms.jsonl`. This catalog is extensive but does not claim that every possible technical phrase in the book has been captured.

## Checks completed

| Check | Result and boundary |
| --- | --- |
| Canonical structure and exports | Validator passes for 1,184 concepts, 13,024 target records, and 424 source records. Checks IDs, redirects, locale completeness, evidence linkage, status, Unicode normalization, and automation eligibility. Source records include individual repository files and pages, not 424 independent publications. |
| Repository evidence | 1,651 evidence expressions match the cited original-file line or the following two lines, with zero failures. Snapshot commits are checked. Occurrence alone does not prove meaning. |
| Other evidence | 499 evidence items are not tested by the repository-line script, principally web and PDF references. Source/page inspection and selected rendered-page checks are described in the language notes; this is not a claim of automated verification of all 499. |
| Live-book inventory | 275 navigation-linked pages cached with retrieval metadata and hashes. The 269 same-stem local files still match their recorded snapshot hashes. Six deployed diffusion pages have no same-stem local file and were inspected from the live snapshot. A filename match does not imply live/local text identity. |
| English sense review | 42 retained concepts carry the stronger semantic spot-check flag; 1,142 carry the weaker prose-occurrence flag. Merged records retain the strongest prior flag, not certification of every merged reference. |
| Target sense review | Language notes record bounded samples across source families and ambiguous senses. Hindi received an additional 121 whole-phrase editorial corrections. These are agent reviews, not native-speaker certification of the complete matrix. |
| Automation files | Contain only `verified_d2l`, `verified_external`, and attested `invariant` entries. Every row retains the English sense and review level. Proposals remain available in the review CSV and canonical shards. |
| Repository changes | Work is confined to `docs/multilingual`; downloaded source corpora and PDFs are temporary analysis caches. Book prose, code, identifiers, and figures were not translated or edited. |

Examples caught during review include a convolution kernel confused with a kernel-method similarity function, statistical versus additive bias, an RL return confused with ordinary/programming usage, Japanese batch normalization losing its “batch” modifier, anatomical “joint” probability in German, spatial translation rendered as language translation in Hindi, and a general “Vocabulary” heading mistaken for an NLP token inventory. These corrections demonstrate why a parallel-heading match is only a candidate.

## Evidence limitations and next use

Earlier D2L editions vary substantially in completeness and editorial quality. Some use machine translation, and Google's localized teaching pages explicitly disclose machine translation. The registry records that provenance; source attestation is not an assertion of human authorship or natural usage. Arabic PDF text sometimes required spelling normalization, and Hindi CBSE evidence required visual transcription because the text layer was garbled. Only the documented sample PDF pages received visual inspection.

DNS probing, official homepage links, and repository discovery cross-check the language inventory. DNS probing cannot enumerate the full zone. No Hindi D2L corpus was found. The historical Vietnamese hostname linked from the English homepage currently serves unrelated content, so it was excluded in favor of the official repository and `vi.d2l.ai`.

Use the source-attested subset as **contextual translation assistance**, with paragraph meaning and target grammar taking precedence over literal replacement. Review proposals before making them mandatory terminology. Prioritize frequent foundational terms, remaining sense collisions, and modern generative/architecture terminology. Preserve code blocks and program identifiers, mathematical symbols, reference keys, link destinations, and figures; figure captions and alt text were conservatively excluded from this extraction scope.

Canonical JSONL and recorded evidence are sufficient to consume this work later; `/tmp` caches are not runtime dependencies. Regenerate exports with `scripts/validate_export.py`. Rerun `scripts/audit_evidence.py` with the pinned corpora present to reproduce repository occurrence checks; absent caches are reported as untested. Language construction scripts can overwrite later editorial decisions and are not a substitute for reviewing an update.

## Model allocation

Astra produced the plan. Terra built the English catalog, handled the Romance-language group, and reviewed synonym merges. Sol handled source discovery and language terminology work, including Hindi. Luna handled bounded mechanical extraction. Downloading, joining, Unicode checks, citation occurrence checks, and exporting used scripts. Billing totals are unavailable, so no dollar-cost claim is made.

Detailed review records: [English](notes/english-audit.md), [Chinese/Japanese/Korean/Vietnamese](notes/east-asian.md), [Arabic/Turkish/German](notes/ar-tr-de.md), [Spanish/French/Portuguese](notes/es-fr-pt.md), and [Hindi](notes/hi.md).

See [English source portability](notes/english-source-portability.md) for the final fresh-checkout validation changes and the 137 excerpt-backed citations still needing current-book review.
