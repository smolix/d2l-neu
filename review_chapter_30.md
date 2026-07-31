# Review of Chapter 30: Tools for Deep Learning

## Scope

Reviewed the only tracked Markdown source in `chapter_tools-for-deep-learning/`,
`index.md`, in the context of its role as the part landing page for the eleven
sections listed under it in `_quarto.yml`. The generated `index.qmd` was not
treated as an authoring source. The audit covers the landing page's prose,
navigation, scope, structure, terminology, and relationship to the substantive
tools appendix.

## Executive assessment

The page is clean and restrained, but too generic to orient a reader. It names
broad subjects without stating the practical decisions the sections help make,
and it does not expose the dependency path from running one notebook to choosing
hardware, scaling training, serving a model, or contributing to the book. The
last paragraph also refers vaguely to “final sections” and “conventions” even
though the part contains distinct teaching sections and generated API references.
There are no technical errors in the thirteen-line source; the main defect is
that it behaves like catalogue copy rather than a useful entry point.

Scores (0–10): **writing quality 7.2**, **explanation/pedagogy 5.8**,
**technical/logical quality 8.4**.

## Architecture and logical order

As a part landing page, this source should not duplicate the long appendix
introduction. It should answer three questions quickly:

```text
What practical problem brought the reader here?
→ Which section answers it?
→ Which quantities or constraints should the reader estimate first?
```

The actual section order supports a useful progression: local/hosted execution
→ cloud and hardware choice → ecosystem → distributed training → serving →
contribution and API reference. The revised page should make this route visible
and also tell experienced readers that the sections can be used independently.

## Detailed issues

| ID | Severity | Location | Problem and violated guide rule | Concrete revision direction |
|---|---|---|---|---|
| C30-01 | Major | `chapter_tools-for-deep-learning/index.md:3-8` | The opening begins “This part is a practical reference” and immediately lists topics. It gives no concrete task, constraint, or discrepancy, contrary to the guide's problem-first opening rule. | Begin with a reader trying to run, scale, or serve a model. State that the right tool depends first on memory fit, data movement, reproducibility, and cost per result. |
| C30-02 | Major | `index.md:3-13` | The page has no visible navigation despite serving as the part landing page for eleven configured sections. Readers cannot map a need to a destination from the prose. | Add a compact task-to-section table using existing cross-references: run/debug, hosted notebooks, cloud rental, hardware, ecosystem, distributed training, serving, contributing, API reference. |
| C30-03 | Moderate | `index.md:4-6` | “Choose hardware by estimating memory use and throughput” is useful but abstract. It does not name the concrete comparison: memory fit before peak FLOP/s, bandwidth and communication limits, and cost per completed experiment. | State these quantities explicitly as examples of the part's decision method. |
| C30-04 | Moderate | `index.md:5-7` | Training and serving are joined in one sentence, although they optimize different workloads and metrics. Training balances compute, memory, and accelerator communication; serving adds latency, throughput, batching, and KV-cache constraints. | Separate the two questions and name their principal metrics. |
| C30-05 | Moderate | `index.md:6-8` | “Software ecosystem for models and datasets” is too vague to tell readers whether the section covers package installation, model hubs, dataset hubs, papers, benchmarks, or reproducibility. | Describe the section as finding and evaluating models, datasets, implementations, and benchmark evidence. |
| C30-06 | Major | `index.md:10-13` | “The final sections” is an ambiguous demonstrative. It conflates the contributor guide with generated utility and `d2l` API pages and does not explain which are tutorial material versus searchable reference. | Name the contributor guide and generated API references separately. State that the API pages are lookup material rather than part of the teaching sequence. |
| C30-07 | Moderate | `index.md:12-13` | “The sections are largely independent” contradicts the implicit progression without explaining what may be skipped and what assumptions recur. | Give both reading modes: follow the progression when building a system, or jump directly to a task from the table. |
| C30-08 | Minor | Whole source | There are no labels, figures, captions, slides, code, exercises, or summary. That is acceptable for a short part page, but the absence of a task map leaves the page with no pedagogical structure at all. | Keep the page short; use one problem-driven opening, one navigation table, and one paragraph on reading modes and time-sensitive details. |

## Math and notation

The source contains no mathematics. The revision should introduce only the
quantities needed for practical orientation—memory capacity, throughput,
latency, bandwidth, communication volume, and total cost—without equations or
new symbols. These terms should be presented as measurements, not as a taxonomy.

## Figures, captions, and slides

There are no figures, captions, or slides, and none is needed. A small navigation
table conveys the required relationships more efficiently than a diagram or
deck. The page should remain a landing page rather than become a miniature
chapter.

## Code and experiment pedagogy

There is no code. The revised prose should explain that the later sections use
quantitative estimates and measurements to make tool choices. It should not
promise commands, APIs, or hardware recommendations on the landing page, since
those details are time-sensitive and belong in the linked sections.

## Recurring artifacts

- Generic “This part explains ...” catalogue prose.
- Broad noun lists without a decision or consequence.
- Ambiguous structural references such as “the final sections.”
- Tutorial material and generated reference documentation treated as one kind
  of content.

## Strengths to preserve

- Concise, calm prose without promotional language.
- Accurate high-level scope.
- Explicit statement that sections can be consulted independently.
- Separation of authoring source from generated API documentation.

## Prioritized revision plan

1. Replace the catalogue opening with a concrete run/scale/serve problem.
2. Add a task-to-section navigation table using existing section labels.
3. Separate training, serving, contribution, and API-reference purposes.
4. State the two reading modes and flag time-sensitive operational details.
5. Keep the landing page compact and free of unnecessary code, figures, or
   survey material.
