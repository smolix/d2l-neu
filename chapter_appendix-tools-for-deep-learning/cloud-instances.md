# Cloud Computing
:label:`sec_cloud_instances`

Sooner or later a workload outgrows the free tiers of
:numref:`sec_hosted_notebooks`: the dataset takes a day to preprocess, the
model needs 80 GB of accelerator memory, or you want eight GPUs for an
afternoon. Renting is the natural next step, and it has never been cheaper —
the same H100 that rented for around \$8 per hour during the 2023 shortage
goes for \$2–4 on specialist clouds in mid-2026, and a consumer RTX 4090
can be had for well under a dollar an hour. For calibration: the notebooks
in this book are kept below 8 GB of GPU memory, so *any* rentable GPU with
16 GB runs everything here with room to spare. It is working with large
language models — fine-tuning beyond LoRA, serving long contexts,
pretraining anything — that pushes you up the memory ladder, and the price
ladder with it.

The hard part of cloud computing is not clicking **Create instance**. It is
knowing what the market offers, picking a machine that fits the workload
rather than the marketing page, and running it with the discipline that
disposable compute demands: results leave the machine, and the machine gets
deleted.

## The Rental Market

### Three Tiers of Provider

![Cloud options trade managed integration and governance for price dispersion and operational responsibility.](../img/tools-cloud-spectrum.svg)
:label:`fig_tools_cloud_spectrum`

* **Hyperscalers** — AWS, Google Cloud, Microsoft Azure — sell GPUs
  embedded in a full cloud: identity management, virtual networks, managed
  storage, and every adjacent service. You pay for that integration; their
  on-demand GPU prices are consistently the highest, and the largest
  instances often hide behind quota requests or sales conversations. They
  are the right answer when the data already lives there, when compliance
  matters, or when a training job is one part of a bigger system. Google
  additionally rents TPUs (a v5e chip for about \$1.20 per hour), the
  natural target for the JAX code in this book.
* **GPU specialists** — Lambda, CoreWeave, Crusoe, Nebius, Voltage Park,
  Together, and others — do one thing: accelerators with fast interconnect
  and ML-ready images. Self-serve H100s run \$2–4.30 per hour (July 2026);
  Lambda and Nebius are fully self-serve, while CoreWeave publishes prices
  but onboards through sales. This tier is the sweet spot for serious
  training runs that do not need a hyperscaler's ecosystem.
* **Marketplaces** — Vast.ai, RunPod, TensorDock, Prime Intellect, and the
  auction-style SF Compute — aggregate machines from many independent
  operators, including consumer GPUs that the big clouds do not carry. This
  is where compute is cheapest: an RTX 4090 for \$0.30–0.60 per hour, an
  H100 from about \$1.50. The catch is variance: host reliability, disk
  speed, and network quality differ per listing, and your code runs on a
  stranger's machine — fine for coursework and public data, inappropriate
  for anything sensitive unless the platform's vetted tier is used.

### What Things Cost

Prices move quickly — the long arc is downward, though 2026's capacity
crunch pushed some rates back up — so treat the following July 2026
snapshot as a calibration, not a catalog. The *ratios* between tiers,
however, have been stable for years:

:GPU rental snapshot, on-demand (July 2026)
:label:`tab_cloud_prices`

| Provider (tier) | Cheap GPU option | ≈ \$/hr | 1× H100 80 GB ≈ \$/hr |
|---|---|---|---|
| Vast.ai (marketplace) | RTX 4090 24 GB | 0.35 | 1.50–1.90 |
| RunPod (marketplace) | RTX 4090 24 GB | 0.35–0.70 | 2.90 (vetted hosts) |
| Prime Intellect (marketplace) | RTX 4090 24 GB | 0.32 | 1.49 |
| Lambda (specialist) | A100 40 GB | 1.99 | 3.99–4.29 |
| Nebius (specialist) | RTX PRO 6000 48 GB | 1.80 | 3.85 |
| Voltage Park (specialist) | H100 (only SKU) | 1.99 | 1.99 |
| AWS (hyperscaler) | L4 24 GB (g6) | 0.80 | 6.88 (p5) |
| Azure (hyperscaler) | A10 24 GB | 1.43 | 6.98 |
| Google Cloud (hyperscaler) | L4 24 GB | 0.70 | ≈ 11 (A3) |

Three practical notes. First, the free money: new accounts get trial
credits (Google Cloud \$300, Azure \$100–200 for students, AWS \$1,000
through its startup program), which comfortably covers every experiment in
this book many times over. Second, multi-GPU nodes price linearly per GPU
on most providers, but the interconnect does not: an 8×H100 machine with
NVLink is a qualitatively different tool from eight PCIe cards, and
communication-heavy training (:numref:`sec_training_systems`) will feel the
difference. Third, newer is not always cheaper per unit of work: B200-class
instances (\$6–14 per GPU-hour) only pay off when you exploit their memory
and FP8/FP4 throughput.

### Cost per Result, Not per Hour

The hourly price is the most visible term of a larger sum: setup time,
idle time while you debug, storage that keeps billing after the run, data
egress, and — the term engineers habitually forget — your own time. A
faster, pricier GPU frequently wins on cost per *completed* experiment.
The little model below is worth re-running with your own assumptions;
here it compares a cheap marketplace card against two datacenter GPUs for
the same eight-hour (on the slowest card) job:

```{.python .input #cloud-instances-cost-model}
import numpy as np

gpu = ["RTX 4090", "A100 80GB", "H100"]
gpu_per_hour = np.array([0.40, 1.50, 2.50])   # marketplace, July 2026
relative_speed = np.array([1.0, 1.6, 3.0])    # measure for your workload!
setup_hours = np.array([0.5, 0.5, 0.5])
storage_and_egress = np.array([2.0, 2.0, 2.0])
engineer_per_hour = 60.0

wall_hours = 8.0 / relative_speed
invoice = (wall_hours + setup_hours) * gpu_per_hour + storage_and_egress
complete = invoice + setup_hours * engineer_per_hour
for row in zip(gpu, np.round(wall_hours, 1), np.round(invoice, 2),
               np.round(complete, 2)):
    print(row)
```

The invoice favors the cheap card; the complete cost is nearly a tie, and
if slow iteration means one extra debugging round, the fast card wins. This
is also why an unreliable \$0.30 host can be the most expensive machine you
ever rent. For scale: a LoRA fine-tune of a 7B model is a few hours on one
consumer GPU — \$3–15 total on a marketplace — while pretraining even a
small LLM from scratch is thousands of GPU-hours. Know which regime you
are in before optimizing pennies.

Two cost traps deserve their own warnings:

* **Spot and interruptible capacity** is 50–90% off, and it is the right
  default for any job that checkpoints and resumes cleanly
  (:numref:`sec_training_systems`). It is the wrong discount for an
  interactive session or an uncheckpointed run — eviction notice can be as
  short as a few seconds on marketplace spot tiers. Test recovery *before*
  the long run, not during it.
* **Egress fees.** Hyperscalers charge roughly \$0.09–0.12 per GB to move
  data out. Re-downloading a 140 GB checkpoint daily costs more per month
  than many GPUs. Keep data and compute in the same region, and prefer
  providers with free egress (most marketplaces) when your workflow moves
  big artifacts around.

## Working on a Rented Machine

### Boot, Connect, Verify

Start from the provider's current deep-learning image — drivers, CUDA, and
container runtime preinstalled. Hand-installing CUDA on a bare OS image is
an afternoon you never get back, and reproducing it next month is another.
Connect with the SSH-tunnel pattern of :numref:`sec_interactive_development`
(the same two commands work on every provider), or use VS Code Remote SSH.
Run long jobs under `tmux` so an SSH disconnect does not kill training, and
verify the machine before trusting it with a long run:

```bash
nvidia-smi                 # driver ok? expected GPU? memory free?
df -h                      # scratch disk has room for data + checkpoints?
python -c "import torch; print(torch.cuda.get_device_name(0))"
```

A one-minute smoke test — one batch through the model, one checkpoint
written and read back — catches most of what the console's green checkmark
does not: broken drivers, full disks, read-only mounts, and datasets that
stream too slowly to keep the GPU busy.

### Compute Is Disposable, Results Are Not

![Provision compute, connect securely, checkpoint to durable storage, and delete the VM; resume from that storage on a replacement machine.](../img/tools-cloud-lifecycle.svg)
:label:`fig_tools_cloud_lifecycle`

The workflow in :numref:`fig_tools_cloud_lifecycle` separates the lifetime
of the machine from the lifetime of your work. Code lives in Git and is
cloned onto the instance; data and checkpoints sync to durable object
storage (S3, GCS, or the provider's volume product) on a schedule the job
controls; the instance itself can then be preempted, crashed, or deleted
without losing more than the last checkpoint interval. This is not merely a
safety practice — it is what makes spot pricing and marketplace hosts
usable at all.

When the experiment ends, tear down *everything that bills*: the instance,
its disks and snapshots, reserved IP addresses, and stale buckets. Set a
billing alert on day one; a forgotten idle GPU costs the same as a busy
one, and quota limits are the only thing standing between a leaked
credential and a very large invoice.

## Summary

* Every notebook in this book fits in 8 GB of GPU memory, so the cheapest
  rentable GPUs suffice; LLM-scale work is what climbs the price ladder.
* The market has three tiers — hyperscalers, GPU specialists, and
  marketplaces — with roughly a 4× price spread for the same GPU and a
  matching spread in integration, reliability, and trust.
* Compare cost per completed result, not per hour: include setup, idle
  time, storage, egress, and your own time.
* Spot capacity is the right default for checkpointed jobs and wrong for
  everything else; egress fees punish moving large artifacts.
* Treat instances as disposable: provider image, SSH tunnel, `tmux`,
  checkpoints to durable storage, then delete every billable resource.

## Exercises

1. Price a fine-tuning job you care about on three providers from
   :numref:`tab_cloud_prices` using cost per completed run. State the date
   and your speed assumptions, then check how prices have moved since this
   table was written.
1. Take whichever training notebook of this book you ran most recently and
   measure its actual GPU memory high-water mark. Which entries of
   :numref:`tab_cloud_prices` could run it?
1. Simulate an interruption: start a checkpointed training run, kill the
   process mid-epoch, and resume on a fresh machine from object storage.
   Time the recovery and identify what you forgot to save.
1. Draw the trust boundary for (a) a marketplace host processing a public
   dataset and (b) a hyperscaler processing medical records. What changes?
