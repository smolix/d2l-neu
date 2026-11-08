# Multilingual terminology execution plan

## Execution updates requested by the user

Hindi is required. The initial 445-concept catalog was a seed, not adequate coverage of the current book. Expand by inspecting every chapter published at https://d2l.smola.org, including terminology defined in prose as well as headings. Expect at least 1,200–1,500 concepts and continue where chapter coverage requires it; do not pad the count with synonymous spellings or stop at a numerical quota. Compare the live table of contents with local sources because the deployed diffusion chapter and the checkout may differ. Preserve existing concept IDs, document merges, and regenerate all locale files against the final catalog.

The smaller allocations below record Astra's initial plan and are superseded by this chapter-based expansion. Language work proceeds in reusable batches so this expansion does not discard source evidence already collected.

This plan covers a prose glossary for translating the new English D2L book into Chinese, Japanese, Korean, Spanish, Arabic, Portuguese, Vietnamese, French, and Hindi (added by the user during execution). It is a working plan, not a claim that the translations or sources described below have already been verified.

## Scope and corpus

Aim for 450–550 distinct concepts, allowing 400–600 where evidence and coverage warrant it. Count concepts, not spelling variants. Every concept must occur in the current book's prose, headings, or mathematical explanations. Exclude program identifiers, code comments, API names used only as code, figure text, image metadata, and figure captions. Keep mathematical concepts such as “gradient” and “transition kernel,” but do not translate mathematical symbols.

Use `_quarto.yml` to identify the published chapters and resolve its generated `.qmd` paths to the corresponding source `.md` files. Do not indiscriminately scan reviews, slides, archived source duplicates, `_outline`, or generated output. Chapters actually included under the “Attic” part remain in scope. Parse fenced blocks before collecting headings: a simple heading regex also captures Python comments. Extract definitions, emphasized terminology, headings, repeated technical noun phrases, and high-risk polysemous words from the remaining prose. Record a local source path and section label for every concept.

The local book includes substantial newer material beyond earlier D2L translations: Muon, attention at scale, KV caches, mixture of experts, state space models, Mamba, DeltaNet, computational performance, deep reinforcement learning, adversarial objectives, diffusion and flow matching, and extended mathematics. These need explicit coverage rather than a glossary copied entirely from an earlier edition.

Suggested coverage allocation (530 concepts, adjustable):

| Area | Target | Important coverage |
| --- | ---: | --- |
| Linear algebra, calculus, automatic differentiation | 60 | spaces, matrix decompositions, norms, Jacobian/Hessian, forward/reverse modes |
| Probability, statistics, information theory | 60 | conditioning, inference, estimation, entropy, divergences, latent variables |
| Learning foundations, optimization, generalization | 65 | losses, regularization, distribution shift, training, optimizers, constraints |
| Convolutional networks and computer vision | 45 | convolution, pooling, normalization, detection, segmentation |
| Sequences, attention, transformers, modern recurrence | 45 | attention variants, position encoding, caches, state space models, experts |
| Natural language processing and adaptation | 40 | tokenization, embeddings, decoding, pretraining, fine-tuning, downstream tasks |
| Implementation concepts and computational performance | 35 | model state, devices, precision, memory, parallelism, throughput |
| Reinforcement and deep reinforcement learning | 45 | policy, return, advantage, occupancy, Bellman operators, off-policy/offline learning |
| Generative models, diffusion, dynamics | 50 | adversarial learning, score matching, Langevin dynamics, flows, ODEs/SDEs |
| Gaussian processes, hyperparameter search, recommendation | 25 | kernels, acquisition, successive halving, factorization, ranking |

Resolve senses before translation. Separate statistical bias from a neural network bias, a probability density score from an evaluation score, an RL return from a function return, and a covariance kernel from a convolution kernel. English spelling variants and abbreviations belong in aliases unless they change the meaning.

## Canonical data

Use UTF-8 JSONL as the editable source and derive readable tables and CSV exports. Do not edit generated tables independently.

`concepts.jsonl`: one record per concept with `id`, `en`, `aliases_en`, `topic`, `sense`, and `book_refs`. IDs are stable ASCII slugs, with a sense suffix where needed. `book_refs` contain source `path` and section `locator`; `sense` is a short English definition that constrains translation.

`terms/<locale>.jsonl`: one record for every concept in each requested locale, including unresolved ones. Fields are `concept_id`, `preferred`, `alternatives`, `status`, `source_ids`, `evidence`, and `notes`. Each evidence item contains `source_id`, `locator`, `source_term`, and a concise `context` explaining how the source establishes this concept. Alternatives may be strings, but notes must distinguish attested alternatives, regional variants, and rejected renderings. An alternative must never silently inherit the preferred term's verification.

`sources.jsonl`: one record per source with `id`, `title`, `locale`, `kind`, `url`, `accessed`, `publisher`, and optional `repository`, `commit`, `path`, `edition`, `license`, and `notes`. Use exact document URLs or repository files, not search results. Include provenance for an external official translation. Pin repository sources to commits where feasible. A source registry entry alone is not term-level evidence.

Required locales: `zh-Hans`, `ja`, `ko`, `es`, `ar`, `pt-BR`, `vi`, `fr`, and `hi`. The coordinator's discovery also identifies Turkish and German D2L repositories; add `tr` and `de` once actual translated prose is confirmed, documenting the discovery and its coverage. Apply the same schema and gates to these added languages. Simplified Chinese and Brazilian Portuguese are practical defaults; document them in the README rather than implying that these choices cover every regional variety. Preserve Traditional Chinese or European Portuguese evidence as explicitly labeled variants; do not silently relabel it. Arabic should use standard written Arabic and preserve source spelling. Store logical Unicode order without decorative bidirectional controls.

Example structure (illustrative placeholders, not real evidence):

```json
{"concept_id":"score-density","preferred":null,"alternatives":[],"status":"unresolved","source_ids":[],"evidence":[],"notes":"Density-score sense; source search pending."}
```

The schema can gain fields needed by implementation, but all workers must agree on the canonical fields before generating language shards.

## Evidence and status policy

| Status | Meaning | Eligible for automatic terminology constraints? |
| --- | --- | --- |
| `verified_d2l` | Inspected earlier D2L translation attests the target expression in the same technical sense. | Yes, subject to conflict review. |
| `verified_external` | Inspected published teaching material attests the target expression in the same technical sense. | Yes, subject to conflict review. |
| `proposed` | Plausible editorial or model-generated rendering without sufficient attestation. | No. |
| `unresolved` | No defensible preferred rendering yet, conflicting evidence unresolved, or insufficient contextual match. | No. |
| `invariant` | An explicitly reviewed term is intentionally retained in English or as an acronym; document usage evidence and reason. | Yes, only after that review. |

Use `preferred: null` for unresolved entries. A proposed term may populate `preferred`, but must remain visibly proposed in every export. Do not create numeric confidence scores that imply nonexistent calibration. Verification is per language and sense, never inherited from another language or from the English concept's existence.

Prefer earlier D2L terminology when it is technically correct and coherent. A published external source may supersede an awkward or obsolete D2L rendering if the reason is documented. Verify the actual term and its context; a source's title, a search snippet, or a general link to a relevant course does not verify a translation. A localized navigation shell around English teaching prose does not count as localized evidence. English-only publications can clarify a concept but cannot attest a non-English term.

For common terms, one clear same-sense educational attestation is sufficient. For novel, ambiguous, or contested terms, seek two independent teaching sources where available; otherwise retain a documented provisional decision or unresolved status. Avoid manufacturing compounds from separately attested component words. Keep evidence excerpts short, preferably the term and a locator, and summarize context instead of reproducing textbook passages.

## Source discovery and systematic matching

1. Inventory earlier D2L translation sites and repositories from the current official site, translation listings, repository organizations, public DNS results, and public certificate records. Resolve likely language subdomains using A/AAAA/CNAME and inspect resulting sites. Record candidate, DNS response, redirect destination, actual content language, repository, edition, access date, and accessibility. DNS success does not establish a complete translation; DNS failure does not establish that no translation exists. Public DNS queries cannot enumerate every subdomain, so describe the result as a best-effort inventory unless an authoritative zone inventory is available. Report discovered additional languages separately from the nine requested output languages.
2. Retrieve accessible translation repositories or stable site content once and cache outside deliverables where appropriate. Align translated chapters with English chapter paths, section labels, titles, and shared equations. Extract bilingual parenthetical introductions and emphasized definitions first; these yield many high-quality matches cheaply.
3. Search every concept and its English aliases across each translated corpus, then inspect aligned translated sections for concepts whose English expression is absent. A literal English search alone misses fully translated terms. Track each concept-language pair as matched, reviewed without match, inaccessible, or pending.
4. Supplement missing concepts in batches by topic using published native-language teaching material. Candidate families include localized official machine-learning course glossaries, university course notes and open textbooks, and educational material from recognized research organizations. The coordinator has located Google's localized machine-learning glossary for the requested languages and Turkish/German; its shared heading IDs offer an efficient alignment route. Inspect each localized definition as well as its heading, because an awkward localized heading can mislead. Describe it as published localized teaching material; do not claim human translation or native-speaker review without evidence. Record machine-translation provenance only when a publisher discloses it.
5. Search unusual modern terms explicitly in the target language, using English acronyms and the concept's technical context. Inspect full teaching pages or PDF sections. Do not equate widespread use of an English loanword with a missing translation: an attested loanword may be the correct preference.
6. Stop a low-yield pair after inspecting the relevant D2L section and two plausible external source families or targeted searches. Record the gap and searched sources, then prioritize untouched high-frequency concepts. This is a search budget, not a waiver of unresolved status. Return to high-impact unresolved terms after baseline coverage is complete.

Maintain a per-locale search log with sources and topics searched, retrieval failures, and unresolved counts. Reuse sources across dozens of concepts, but retain individual evidence locators. Automated substring matching can propose evidence; it cannot by itself establish sense equivalence. Each locale must receive the same concept inventory and a complete accounting of all entries.

## Cost-conscious division of work

Respect the user's model roles and the available agent slots. Astra produces this plan only. The coordinating agent controls schema, assignments, integration, and final reporting. Avoid asking a high-cost model to generate thousands of unsourced translations.

| Worker | Main responsibility | Deliverable ownership |
| --- | --- | --- |
| Terra | Parse the current book; construct and deduplicate the English concept set; implement validators and deterministic exports. | Concepts, extraction/validation scripts, generated coverage data. |
| Sol | Discover and inspect D2L/external teaching sources; align terminology; resolve ambiguous meanings and conflicting translations. | Source inventory and assigned language evidence shards. |
| Luna | Perform bounded extraction from supplied retrieved passages; normalize source metadata; enumerate gaps and prepare candidate matches. | Assigned language shards or temporary extraction batches, never files concurrently owned by another worker. |
| Coordinator | Freeze schema and concept IDs; assign disjoint locale or concept batches; review evidence and merge; handle final gaps and documentation. | README, integration, audit report, final user summary. |

Run at most the allowed concurrent agents. Reuse the planning slot for evidence work once this plan is delivered. Do not split concepts and languages across uncontrolled overlapping writers. After concept IDs stabilize, locale ownership is the simplest partition; a worker may cover several languages in sequential batches. Give extraction workers 40–80 concepts with relevant source passages and exact URLs, rather than the entire book or open-ended research. Use code for normalization, joins, counts, duplicate checks, and export generation. Escalate only ambiguous senses and contradictory sources to deeper review.

A practical sequence is: freeze concept inventory; collect reusable source corpora; create all nine locale skeletons; complete D2L extraction; fill common external gaps; review modern/ambiguous terms; validate and generate exports. These are stages of the requested work, not reasons to stop after delivering a plan or a candidate-only glossary.

## Quality gates and deliverables

Structural gates:

- Unique concept IDs and exactly one entry per concept per locale; all cross-references resolve.
- Every concept has an actual local prose occurrence and an unambiguous sense.
- Every verified preferred term has an inspected, locale-appropriate source, exact source expression, and resolvable section/page/file locator.
- Unresolved terms have no preferred rendering; proposals and unresolved entries are excluded from the automation export.
- Normalize to Unicode NFC; reject accidental replacement characters, hidden control characters, code identifiers masquerading as prose concepts, and inconsistent locale labels.
- Flag identical English/target text for review, without rejecting legitimate loanwords, names, or acronyms automatically.
- Report verified/proposed/unresolved/invariant counts by locale and topic; distinguish source absence from retrieval failure and pending review.

Semantic gates:

- Review all polysemous concepts, conflicting sources, retained English terms, and unsupported-looking compounds.
- Audit at least 20 verified entries per locale, stratified across D2L and external sources and across topics; include rare/new terms. Verify the supporting passage again. If a systematic problem appears, repair the affected batch and expand the sample rather than declaring the whole glossary reviewed.
- Check common confusion pairs: loss/objective/cost; likelihood/probability; normalization/standardization; parameter/hyperparameter; embedding/encoding; state/action/policy; reward/return; density score/evaluation score; precision/accuracy; supervised/self-supervised/unsupervised.
- Inspect representative multilingual tables, Arabic logical order, CJK text, Vietnamese diacritics, and CSV quoting. Do not claim native-speaker review unless it occurred.

The final directory should contain a concise README explaining scope and safe automation use, canonical concepts and locale files, the sources and discovery inventory, readable glossary tables, a coverage/gaps report, and a reproducible validator/exporter. A wide CSV is useful for review; a filtered JSON export is preferable for automatic translation. The translation consumer must match whole phrases in prose context, prefer longer phrases, and use sense constraints. It must preserve code, links, formulas, cross-reference labels, and figure material rather than run blind global substitutions.

Completion means all nine languages are represented over the same extensive concept set, the available prior translations and teaching sources have been systematically searched, evidence and gaps are explicit, and the data passes structural and semantic checks. Do not substitute a high filled-cell count for evidence. Report actual attested coverage prominently and make residual proposals or gaps visible instead of calling a mostly proposed glossary translation-ready.
