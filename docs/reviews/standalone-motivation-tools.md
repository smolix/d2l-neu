# Standalone Motivation Review: Tools for Deep Learning

## Executive assessment

This part is substantially stronger than a conventional tools appendix. Most
notebooks begin from an operational decision or failure mode, distinguish facts
that will age from principles that will not, and organize products by workload
rather than popularity. The part-level index, notebook workflow, hosted
runtimes, cloud computing, hardware, distributed training, and serving pages
already satisfy the standalone-motivation requirement.

The main problems are narrower:

1. The chapter index is a topic itinerary, while the stronger part index already
   provides a problem-oriented entry point.
2. The ecosystem and distributed-training pages place long product catalogues
   before the decision aids that govern those catalogues.
3. The utility page is a raw implementation dump. It does not explain its
   reference status, navigation, framework coverage, or relation to the places
   where the utilities are taught.

The intended chapter progression is sound:

```text
reproducible local notebook
-> temporary hosted runtime
-> rented compute
-> hardware purchase
-> external models, data, and evidence
-> distributed training
-> model serving
-> contribution workflow
-> generated reference pages
```

The revision should preserve that progression and avoid turning every page into
another hardware overview. A direct reader needs the local decision that the
page resolves; an in-order reader needs only one sentence connecting that
decision to the preceding scale.

## Part index: `chapter_tools-for-deep-learning/index.md`

**Assessment: preserve.** The opening begins with a laptop workload that stops
fitting and immediately separates memory capacity, compute, data movement, and
cost ([source](../../chapter_tools-for-deep-learning/index.md#L3)).
The task-to-section table gives direct readers a problem-oriented navigation
aid ([source](../../chapter_tools-for-deep-learning/index.md#L15)),
and the closing paragraph explicitly distinguishes teaching pages from API
references ([source](../../chapter_tools-for-deep-learning/index.md#L27)).

No additional motivation is needed. Do not replace the table with a prose list
of section names. If the chapter index below remains separate, reuse this same
problem-oriented structure rather than maintaining two different roadmaps.

## Chapter index: `chapter_appendix-tools-for-deep-learning/index.md`

**Medium priority.** The chapter index lists the sections in reading order
([source](../../chapter_appendix-tools-for-deep-learning/index.md#L4))
but does not expose the decisions connecting them. A direct reader already gets
a better entry point from the part index. The two pages also share the same
title, which makes their different purposes unclear.

**Revision specification.** Replace the itinerary with a compact version of the
part index's task table. Precede it with one sentence explaining the scale
transition: first make one notebook reproducible, then decide where it should
run, then decide how training or serving should scale. Retain the warning that
prices and products are dated ([source](../../chapter_appendix-tools-for-deep-learning/index.md#L16))
and the explicit exclusion of the generated reference pages from the teaching
sequence ([source](../../chapter_appendix-tools-for-deep-learning/index.md#L35)).

## Notebooks: `interactive-development.md`

**Assessment: strong; low-priority bridge only.** The notebook starts from the
book's executable format and gives readers an immediate reason to edit and rerun
it ([source](../../chapter_appendix-tools-for-deep-learning/interactive-development.md#L4)).
The document/kernel distinction creates the need for restart-and-run-all before
the operational instructions begin ([source](../../chapter_appendix-tools-for-deep-learning/interactive-development.md#L21)).
The remote-machine section also establishes the split between local editor and
remote accelerator before showing SSH commands, and it includes the necessary
security boundary ([source](../../chapter_appendix-tools-for-deep-learning/interactive-development.md#L171)).

**Revision specification.** Preserve the opening example and figure. Add only a
short sentence before “Running the Book Locally” distinguishing the two reader
paths: use a generated notebook archive to experiment with the published book;
clone the source repository only when editing or contributing to the book. This
prevents a direct reader from choosing the more complex workflow without a
reason. Do not add a general explanation of Python environments; the current
kernel-identity check is more useful.

## Hosted notebooks: `hosted-notebooks.md`

**Assessment: preserve.** The opening states both the capability and the
principal constraint: the provider owns a temporary runtime, while only
explicitly persisted artifacts survive ([source](../../chapter_appendix-tools-for-deep-learning/hosted-notebooks.md#L4)).
The lifecycle figure makes later Colab and Kaggle differences answer a visible
question. “Launching a Book Notebook” then explains the repository-to-provider
path before giving platform instructions ([source](../../chapter_appendix-tools-for-deep-learning/hosted-notebooks.md#L28)).

No new introductory material is needed. Preserve the contrast between Colab's
repository pointer and Kaggle's imported copy. When updating provider details,
keep them under the existing stable axes: runtime ownership, persistence,
hardware guarantees, quotas, and reproducibility. Avoid adding another list of
features unless each feature changes one of those axes.

## Cloud computing: `cloud-instances.md`

**Assessment: strong opening; medium-priority reordering.** The page gives three
concrete reasons to leave hosted notebooks and states that rented compute and
its local storage are disposable ([source](../../chapter_appendix-tools-for-deep-learning/cloud-instances.md#L4)).
The provider tiers are organized by integration, price, governance, and
operational responsibility rather than by brand alone
([source](../../chapter_appendix-tools-for-deep-learning/cloud-instances.md#L20)).

The decision principle “cost per completed result” does not arrive until after
the provider and price catalogues. A direct reader may therefore compare hourly
prices before learning which costs matter.

**Revision specification.** Move the conceptual core of “Cost per Result, Not
per Hour” (currently beginning at line 83) ahead of the dated price snapshot.
Introduce a five-field workload card—required accelerator memory, expected
runtime, number/interconnect of accelerators, data location, and checkpoint or
preemption tolerance. Use that card to choose a provider tier, then present the
price table as an example. Preserve the dated snapshot, but ensure no selection
recommendation depends on it without the workload card.

## Hardware: `hardware.md`

**Assessment: preserve with one navigation improvement.** The page states that
capacity determines whether a workload runs, bandwidth often determines decode
speed, and compute matters differently for prefill and training
([source](../../chapter_appendix-tools-for-deep-learning/hardware.md#L4)).
It asks “What Are You Buying For?” and calculates memory fit before naming
specific products ([source](../../chapter_appendix-tools-for-deep-learning/hardware.md#L33)).
This is the desired problem-to-constraint-to-example order.

**Low-priority revision.** Add a short decision table immediately before the
product sections: training/fine-tuning, local interactive inference, batch
inference, and multi-GPU experimentation; for each, identify the first binding
constraint and the relevant machine class. This will let direct readers skip
irrelevant SKU discussions. Keep the specific products beginning at
“Training Boxes” as dated reference points, not universal recommendations.

## Ecosystem: `software-ecosystem.md`

**Medium priority.** The opening correctly defines three tasks—find artifacts,
judge relevance, and record provenance
([source](../../chapter_appendix-tools-for-deep-learning/software-ecosystem.md#L4)).
The notebook then begins with lists of model, dataset, paper, and code sources
([source](../../chapter_appendix-tools-for-deep-learning/software-ecosystem.md#L11)).
The stronger selection rule—build a target-task evaluation and treat leaderboard
rank as indirect evidence—arrives only later. “Staying Current” is similarly a
list of feeds ([source](../../chapter_appendix-tools-for-deep-learning/software-ecosystem.md#L94))
before the artifact-identity workflow begins.

**Revision specification.** Introduce one running task before the source lists,
for example selecting a checkpoint and dataset for a domain classifier. State
the selection record the reader should produce: task and license constraints,
candidate artifact and immutable revision, preprocessing contract, target-task
evaluation, and known limitations. Then make each source answer where one field
of that record can be found. Move “Pin the Identity” or a condensed version of
it ahead of “Staying Current.” Keep the source lists, but make them subordinate
to the reproducible selection workflow.

## Distributed model training: `training-systems.md`

**Assessment: strong core; medium-priority catalogue repair.** The opening names
the two reasons to add devices—memory fit and elapsed time—and explicitly says to
start from a measured one-device baseline
([source](../../chapter_appendix-tools-for-deep-learning/training-systems.md#L4)).
The scaling ladder then introduces each parallelism form because the preceding
rung no longer satisfies a constraint
([source](../../chapter_appendix-tools-for-deep-learning/training-systems.md#L15)).

“The Library Landscape” becomes a dense catalogue of PyTorch, Hugging Face,
DeepSpeed, Megatron, TorchTitan, fine-tuning frontends, JAX stacks, and RL
systems before the reader sees the selection table
([source](../../chapter_appendix-tools-for-deep-learning/training-systems.md#L115)).

**Revision specification.** Move “What to Use at Which Scale” and its table
ahead of the library descriptions. Add a second axis for the constraint being
removed: replicated throughput, sharded model state, activation memory, or
multi-axis model parallelism. Rewrite each library paragraph as evidence for a
row in that table: what constraint it handles, what scale justifies it, and what
complexity it adds. Remove model-release anecdotes that do not change that
choice. Preserve the final long-run reliability section, which supplies a
different and necessary problem after throughput has been addressed.

## Model serving: `model-serving.md`

**Assessment: preserve.** The page distinguishes serving from training before
naming tools and gives the relevant objectives—latency distributions,
concurrency, capacity, and operational control
([source](../../chapter_appendix-tools-for-deep-learning/model-serving.md#L4)).
The workload table makes product recommendations conditional on use case
([source](../../chapter_appendix-tools-for-deep-learning/model-serving.md#L23)).
Only then does the page introduce local and server engines.

No additional motivation is needed. Preserve TTFT, TPOT, throughput, and
goodput as the governing quantities. Future engines should be added only by
placing them in an existing workload row or by establishing a genuinely new
serving constraint. Do not expand the engine list independently of the four
mechanisms developed later.

## Contributor guide: `developers-guide.md`

**Assessment: preserve.** The opening treats the book as an executable software
project and identifies the build, Git workflow, optional agent use, and patch
verification as the local tasks
([source](../../chapter_appendix-tools-for-deep-learning/developers-guide.md#L4)).
“One Source, Many Artifacts” explains why the author must edit Markdown before
listing commands ([source](../../chapter_appendix-tools-for-deep-learning/developers-guide.md#L13)).
The subsequent Git and agent sections therefore operate on an established
artifact model.

No motivation expansion is required. Preserve the consequence-first bullets
about generated files, framework tabs, and stable cell identity. When build
commands change, update the artifact diagram and verification contract together;
otherwise a direct contributor receives conflicting guidance.

## Utility functions and classes: `utils.md`

**High priority.** The file executes a framework-tab selection block before its
title and then supplies only one sentence before a long sequence of imports and
implementations ([source](../../chapter_appendix-tools-for-deep-learning/utils.md#L1)).
A direct reader cannot tell whether this is a tutorial, generated API reference,
library source, or a notebook intended to be executed. Headings such as
“Hyperparameter Utilities” and “Progress and Plotting” name categories but do
not state what later chapters require from them
([source](../../chapter_appendix-tools-for-deep-learning/utils.md#L52)).

This page should not receive a long tutorial introduction; it should become an
explicit reference page.

**Revision specification.** Put the title and reference contract before any
executable cell. State that the page collects `#@save` implementations extracted
into framework-specific `d2l` modules, that most functions are taught where
first used, and that readers should use the API page to search alphabetically.
Add a compact contents table with category, principal classes/functions, and
the chapter where each group is explained. Add one or two sentences before each
major category stating the shared role of the utilities and linking to the
teaching notebook. Do not explain every function line by line. If the file is
generated, generate these category notes from stable metadata rather than
allowing them to drift from the code.

Also resolve the apparent conflict between the chapter index, which says the
utility page is generated and excluded from the teaching sequence, and this
source file, which currently looks like an ordinary executable section. The
rendered page must make its reference status unmistakable.

## `d2l` API reference: `d2l.md`

**Assessment: appropriate for its role.** The opening states that the page is
alphabetical and links entries to their definitions and explanations
([source](../../chapter_appendix-tools-for-deep-learning/d2l.md#L4)).
That is sufficient motivation for a generated reference page. Do not add a
chapter-style conceptual opening. Ensure the generated class and function lists
are populated in rendered output and add a link back to the utility-category
contents table proposed above.

## Implementation sequence

1. **Repair `utils.md`.** Its ambiguous role is the only high-priority
   standalone failure and affects navigation to utilities throughout the book.
2. **Unify the two indexes.** Make the chapter index use the part index's
   problem-oriented table and preserve one canonical roadmap.
3. **Reorder the distributed-training landscape.** Put the scale/constraint
   decision table before library names and rewrite the catalogue around it.
4. **Reorder the ecosystem page.** Establish an artifact-selection record and
   target-task evaluation before listing discovery sources.
5. **Move cloud cost reasoning before price comparison.** Make the workload card
   govern provider and SKU choices.
6. **Add small navigation aids to hardware and local notebooks.** These are
   low-risk changes and should not expand already effective openings.
7. **Leave hosted notebooks, serving, and the contributor guide structurally
   intact.** Update dated facts without weakening their existing problem-led
   organization.
