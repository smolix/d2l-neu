# Cloud Computing
:label:`sec_cloud_instances`

Most readers run the notebooks of this book on a laptop, a desktop GPU, or
a hosted service such as Colab (:numref:`sec_hosted_notebooks`). Sooner or
later a project outgrows all three: preprocessing runs longer than a hosted session lasts, a model
needs 80 GB of accelerator memory, or an experiment needs several GPUs at
once. The usual answer is to rent a machine by the hour, which raises two
questions. Which machine should we rent, and what will a finished result cost?
And how do we work on a machine that can disappear at any moment?

Prices change monthly, so the numbers below are dated examples, not a
catalog. Two ideas outlast them. A machine is chosen by its cost per
completed result rather than its hourly price, and a rented instance is
treated as disposable, with everything that matters copied to durable
storage before the instance is deleted. The book's own notebooks need modest
hardware: they are executed on 24 GB RTX 4090 cards and most use less than
8 GB, so a 24 GB rental runs every single-GPU notebook. Only the multi-GPU
sections of :numref:`chap_performance` need more than one card.

## The Rental Market

### Three Tiers of Provider

GPU rental is sold by three kinds of company, which differ less in the
hardware than in what surrounds it and who operates it
(:numref:`fig_tools_cloud_spectrum`).

![Cloud options trade managed integration and governance for price dispersion and operational responsibility.](../img/tools-cloud-spectrum.svg)
:label:`fig_tools_cloud_spectrum`

* **Hyperscalers** such as AWS, Google Cloud and Microsoft Azure sell GPUs
  inside a full cloud: identity management, virtual networks, managed
  storage, and every adjacent service. Their on-demand GPU prices are the
  highest, and the largest instances often require a quota request or sales
  contact. They suit work whose data already lives in that cloud, work under
  compliance requirements, and training that is part of a larger managed
  system. Google also rents TPUs, which run the JAX code of this book; a v5e
  chip costs \$1.20 per hour as of November 2026.
* **GPU specialists** such as Lambda, Nebius, Voltage Park and CoreWeave rent
  accelerators with fast interconnect and machine-learning images, and little
  else. As of November 2026 an H100 costs about \$2–6 per GPU-hour in this
  tier, and Lambda, Nebius and Voltage Park let a new account launch one
  without contacting sales. This tier suits substantial training runs that
  do not need a hyperscaler's other services.
* **Marketplaces** such as Vast.ai and RunPod aggregate machines from many
  independent operators, including consumer GPUs that the large clouds do not
  carry. They list the lowest hourly prices: as of November 2026 an RTX 4090
  for \$0.30–0.50 per hour and an H100 from about \$1.80. Quality varies
  with the listing. Host reliability, disk speed and network quality differ
  from host to host, and the code runs on a stranger's machine. That is fine for
  coursework and public data, and inappropriate for anything sensitive unless
  the job runs on a vetted datacenter tier such as RunPod's Secure Cloud.

### A Dated Price Snapshot

Prices vary by region, availability, commitment and host.
:numref:`tab_cloud_prices` illustrates the spread between tiers; check
current quotes before buying.

:GPU rental snapshot, on-demand, per GPU (November 2026)
:label:`tab_cloud_prices`

| Provider (tier) | Cheap GPU option | ≈ \$/hr | 1× H100 80 GB ≈ \$/hr |
|---|---|---|---|
| Vast.ai (marketplace) | RTX 4090 24 GB | 0.30–0.50 | 1.80–2.40 |
| RunPod (marketplace) | RTX 4090 24 GB | 0.34–0.89 | 1.99–3.99 |
| Lambda (specialist) | A100 40 GB | 1.99 | 3.29–4.29 |
| Nebius (specialist) | RTX PRO 6000 96 GB | 1.80 | 4.50 |
| Voltage Park (specialist) | H100 80 GB (only listed price) | 1.99 | 1.99 |
| AWS (hyperscaler) | L4 24 GB (g6.xlarge) | 0.80 | 6.88 (p5) |
| Azure (hyperscaler) | T4 16 GB (NC4as T4 v3) | 0.53 | 6.98 (NC40ads H100 v5, 94 GB) |
| Google Cloud (hyperscaler) | L4 24 GB (g2-standard-4) | 0.71 | ≈ 11 (8-GPU A3) |

RunPod's ranges run from community hosts to its datacenter tier. Azure's
single-GPU H100 instance carries the 94 GB H100 NVL. Google Cloud sells the
H100 on demand only as an 8-GPU machine, quoted per GPU.

Two qualifications matter. First, multi-GPU nodes are priced roughly
linearly per GPU, but the interconnect is not: an 8×H100 machine with NVLink is a different tool from
eight PCIe cards, and communication-heavy training
(:numref:`sec_training_systems`) shows the difference. Second, newer is not
always cheaper per unit of work. B200 instances cost about \$7–14 per
GPU-hour as of November 2026, and they pay off only when a job uses their
larger memory and FP8/FP4 throughput.

### Cost per Completed Result

The hourly price is the most visible term of a larger sum that also includes
setup time, idle time while you debug, storage that keeps billing after the
run, data egress, and engineering time. A faster, more expensive GPU can
therefore reduce the cost of a *completed* experiment. The model below asks
which of three marketplace GPUs completes a job most cheaply when the job
takes eight hours on the slowest. Rerun it with your own assumptions:

```{.python .input #cloud-instances-cost-model}
import numpy as np

gpu = ["RTX 4090", "A100 80GB", "H100"]
gpu_per_hour = np.array([0.40, 1.20, 2.50])   # marketplace, November 2026
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

The invoices come to \$5.40, \$8.60 and \$9.92, so the invoice favors the
inexpensive card. Half an hour of the engineer's time brings the totals to
\$35.40, \$38.60 and \$39.92, much closer but in the same order. If the
slow card needs one more half-hour debugging round than the fast one, its
total rises above \$65 and the order reverses, so the number of rounds
belongs in the cost model as much as the hourly price. The size of the
workload matters more still: a LoRA fine-tune of a 7B model may take hours on one
consumer GPU, whereas pretraining a small language model can take thousands
of GPU-hours. Estimate the workload before comparing small differences in
hourly price.

Two costs are easy to overlook:

* **Spot and interruptible capacity** is often discounted by half or more
  and suits jobs that checkpoint and resume cleanly
  (:numref:`sec_training_systems`), not interactive sessions or
  uncheckpointed runs. AWS warns two minutes ahead, Google Cloud by
  default allows a shutdown period of at most 30 seconds, and a Vast.ai
  interruptible instance stops, killing its processes, as soon as it is
  outbid. Test recovery before the long run, not during it.
* **Egress fees.** As of November 2026 AWS and Azure charge about \$0.09 per
  GB for data sent to the internet beyond the first 100 GB a month.
  Downloading a 140 GB checkpoint daily for a month costs about \$370, more
  than running a marketplace RTX 4090 around the clock (720 hours at \$0.44,
  about \$320). Keep data and compute in the same region, and prefer
  providers that do not charge for egress, RunPod among them, when a
  workflow moves large artifacts around.

## Working on a Rented Machine

### Instance Setup and Checks

A new instance is an unknown machine, and the first minutes on it decide
whether a long run can be trusted. Start from the provider's current
deep-learning image, with drivers, CUDA and the container runtime
preinstalled; a bare OS image needs setup that must also be reproduced
later. Connect with the SSH tunnel of :numref:`sec_interactive_development`,
which works the same way on every provider, or with VS Code Remote SSH. Run
long jobs under `tmux` so that an SSH disconnect does not kill training, and
check the machine first:

```bash
nvidia-smi                 # driver ok? expected GPU? memory free?
df -h                      # scratch disk has room for data + checkpoints?
python -c "import torch; print(torch.cuda.get_device_name(0))"
```

A short smoke test, one batch through the model and one checkpoint written
and read back, detects problems that the provider's health indicators miss:
broken drivers, full disks, read-only mounts, and datasets that stream too
slowly to keep the GPU busy.

### Disposable Instances, Durable Results

A rented instance can be preempted, crash, or be deleted by mistake, and the
work on it must survive all three.

![Provision compute, connect securely, checkpoint to durable storage, and delete the VM; resume from that storage on a replacement machine.](../img/tools-cloud-lifecycle.svg)
:label:`fig_tools_cloud_lifecycle`

The workflow in :numref:`fig_tools_cloud_lifecycle` separates the lifetime
of the machine from the lifetime of the work. Code is stored in Git and
cloned onto the instance. Data and checkpoints are copied to durable object
storage (S3, GCS, or the provider's volume product) on a schedule the job
controls. Losing the instance then costs at most the work since the last
checkpoint, and a replacement machine resumes from storage.

When the experiment ends, delete everything that bills: the instance, its
disks and snapshots, reserved IP addresses, and stale buckets. Set a billing
alert on the first day. An idle GPU bills at the same rate as a busy one.
Alerts lag behind usage, so the account's GPU quota is what bounds the loss
from a leaked credential.

We can now choose a rented machine by the cost of a completed result and
keep the work safe when the machine fails. A GPU that is busy on most days
of the year eventually costs more to rent than to buy, and
:numref:`sec_hardware_buyers` asks which hardware to buy.

## Summary

* Most of the book's notebooks need less than 8 GB of GPU memory and a 24 GB
  card runs every single-GPU notebook; LLM-scale work is what raises cost.
* The market has three tiers, hyperscalers, GPU specialists and
  marketplaces, with a several-fold price spread for the same GPU and
  different levels of integration, reliability, and operational trust.
* Compare cost per completed result, not per hour: include setup, idle
  time, storage, egress, and your own time.
* Spot capacity reduces the cost of checkpointed, restartable jobs but is
  unsuitable for interactive or uncheckpointed work; egress fees make
  repeated transfers of large artifacts expensive.
* Treat instances as disposable: provider image, SSH tunnel, `tmux`,
  checkpoints to durable storage, then delete every billable resource.

## Exercises

1. [code] **Peak memory and batch size.** Most of the book's notebooks stay
   below 8 GB of GPU memory, so a 24 GB card in :numref:`tab_cloud_prices`
   leaves room for a larger batch. Run the encoder fine-tuning of
   :numref:`sec_lm_bert` on a GPU, for example in a hosted
   notebook (:numref:`sec_hosted_notebooks`), once with its batch size and
   once with half of it. Record the peak of
   each of the three views of GPU memory in :numref:`fig_bg_allocator` for
   both runs.
    1. Which of the three views decides whether a run fits on a card, and
       why?
    1. Fit the peak of that view as a linear function of the batch size, and
       predict the largest batch that fits on a 24 GB card and on an 80 GB
       card. Which part of the memory does the intercept represent?
    1. Run a batch size between your two runs, or the 24 GB prediction if
       such a card is available, and compare the measured peak with the
       linear prediction.
   :exlabel:`ex_cloud-instances-peak-memory`

1. **Trust boundary.** A fine-tuning job reads its training data from
   object storage with an access key stored on the instance, and writes
   checkpoints to the instance's local disk.
    1. The data is a public dataset, and the job runs on a marketplace host.
       Besides the dataset, what can the host operator read or change while
       the job runs? Which of these items matters although the data is
       public? Why does encrypting the instance's disk not protect them from
       the host operator?
    1. The data is a hospital's medical records, and the job moves to a
       hyperscaler. Which parties can still read the data and the
       checkpoints while the job runs? Besides technical controls, what has
       to be in place before the records may be uploaded?

1. [code] **Engineering time in the cost model.** In
   `#cloud-instances-cost-model`, the column `complete` adds
   `setup_hours * engineer_per_hour` to the invoice.
    1. Explain why this term cannot change which GPU is cheapest, whatever
       the value of `engineer_per_hour`, as long as `setup_hours` is the same
       for all three GPUs.
    1. An engineer who waits for results or checks on the run spends a
       fraction $f$ of `wall_hours` on it. Add this time at
       `engineer_per_hour` to `complete`, and find the value of $f$ at which
       the H100 and the RTX 4090 cost the same. How many minutes of attention
       during the 8-hour run on the RTX 4090 does this value correspond to?
    1. Leave out both engineering terms. Which `relative_speed` would the H100
       need to match the RTX 4090 on the invoice alone?
    1. Obtain today's on-demand prices for the three GPUs from one
       marketplace of :numref:`tab_cloud_prices`, note the date, and compute
       the break-even fraction $f$ again. Does it move by more than a factor
       of two?
   :exlabel:`ex_cloud-instances-engineering-time`

1. [code] **Smoke test script.** Write a script that a rented machine must
   pass before a long run. It trains a small model of your choice on
   `d2l.FashionMNIST` and exits with a clear message at the first failed
   check.
    1. Implement four checks: the accelerator named on the command line is
       visible; the checkpoint directory has a given number of free bytes;
       one training step runs on the accelerator; and a checkpoint written
       with `d2l.save_checkpoint` and read back with `d2l.load_checkpoint`
       reproduces every parameter and optimizer tensor exactly.
    1. Add a fifth check that compares the batches per second the data
       iterator delivers with the training steps per second the model
       sustains on a batch that is already on the device. Why must the two
       rates be measured separately?
    1. Provoke each failure on your own machine. Request more free space than
       the disk has, make the checkpoint directory read-only, and slow the
       iterator with a `time.sleep` per batch. Run the read-only test as a
       non-root user, because root ignores directory permissions. Without
       the script, when would each failure have surfaced in an 8-hour run
       that writes a checkpoint every hour? Which of them would never stop
       the run?
   :exlabel:`ex_cloud-instances-smoke-test`

1. **NVLink or PCIe.** ● Two providers offer eight H100s at the same price
   per GPU-hour. One node connects the GPUs with NVLink, the other only over
   PCIe 5.0. A data-parallel job trains a transformer with $N$ parameters
   and BF16 gradients, so that $S = 2N$ bytes enter the communication time
   of :eqref:`eq_dp_cost`. Each GPU processes $b$ tokens per step, which
   costs $6Nb$ FLOPs by :eqref:`eq_six_nd`. Take the bus bandwidth
   $1/\beta_\textrm{link}$ as the nominal per-direction rate, 450 GB/s for
   NVLink and 64 GB/s for PCIe, and let both nodes sustain 40% of the
   H100's 989 TFLOPS of dense BF16 compute. Assume that communication does
   not overlap with computation.
    1. For $N = 1.3 \times 10^9$, compute the step time on each node at
       $b = 16{,}384$ and at $b = 4{,}096$. By what fraction must the PCIe
       price drop to match the NVLink node in cost per token?
    1. The ratio of communication time to compute time does not depend on
       $N$, because both times grow in proportion to
       it (:numref:`sec_hardware`). On each node, find the number of tokens per
       GPU and step at which communication takes as long as computation.
    1. Gradient accumulation (:numref:`subsec_mp-accumulation`) communicates
       once every $N_{\textrm{accumulation}}$ microbatches. How does it
       change the threshold you found, and what does it change about
       training?

1. **Restarts on spot capacity.** ● A job needs $T$ hours of compute. On
   spot capacity, interruptions arrive as a Poisson process with mean time
   $M$ between them. After each interruption, a replacement instance takes
   $R$ hours to start, and the job begins again from scratch. Spot capacity
   costs a fraction $q$ of the on-demand price, and the restart time is
   billed too. Interruptions do not strike while a replacement starts, and
   the expected wall-clock time to finish is

   $$\mathbb{E}[T_{\textrm{wall}}] = \left(e^{T/M} - 1\right)(M + R).$$

   Problem :exref:`ex_hosted-notebooks-checkpoint-interval`b of
   :numref:`sec_hosted_notebooks` derives the corresponding formula for a
   restore that interruptions can also strike. There the factor $M + R$
   becomes $M e^{R/M}$.
    1. For $T = 8$ h, $R = 0.25$ h and $q = 0.5$, compute the expected spot
       cost as a fraction of the on-demand cost for $M = 24$, $8$ and $4$ h.
    1. Let $R \ll M$ and $q = 0.5$. Show that spot capacity is cheaper in
       expectation exactly when $T$ is below a fixed multiple of $M$, and
       compute the multiple.
    1. Now let the job write a checkpoint after every $\tau = \sqrt{2 \delta
       M}$ hours of compute, the interval of :citet:`Young.1974` from
       problem :exref:`ex_hosted-notebooks-checkpoint-interval`a, where each
       write takes $\delta$ hours. An interruption loses only the work since
       the last checkpoint. Apply the formula for
       $\mathbb{E}[T_{\textrm{wall}}]$ to one segment of length
       $\tau + \delta$, and find the largest $q$ at which
       spot capacity is cheaper in expectation for $\delta = 2$ min,
       $R = 0.25$ h and $M = 24$, $8$, $4$, $2$ and $1$ h.
   :exlabel:`ex_cloud-instances-spot-restart`

[Discussions](https://d2l.discourse.group/t/36843)

<!-- slides -->

::: {.slide}
::: {.cover}
[Dive into Deep Learning · §39.3]{.kicker}

Cloud Computing: GPU Rental Tiers, Cost per Completed Result, and Spot Resilience<br>**hyperscalers vs. specialists vs. marketplaces · cost per result · NVLink vs. PCIe · spot restarts · egress fees · disposable VMs**
:::
:::

::: {.slide title="When a Project Outgrows Local and Hosted Runtimes"}
[Motivation]{.kicker}

::: {.cols .vc}
::: {.col}
- **Three walls that force cloud rental:** (1) Training exceeds 12-hour hosted sessions/quotas; (2) full fine-tunes or long contexts need $80\text{--}96\text{ GB}$ HBM; (3) multi-GPU scaling needs $450\text{--}900\text{ GB/s}$ NVLink.
- **Two durable cloud rules:** Select instances by **total cost per completed result** (never hourly sticker price alone), and treat every rented VM as **disposable**.
:::

::: {.col .fig .big}
@fig:ch39-tools-three-cloud-walls
:::
:::
:::

::: {.slide title="Three Tiers of Cloud Accelerator Providers"}
[The Rental Market]{.kicker}

GPU rental providers differ less in the silicon itself than in the surrounding managed services, pricing dispersion, and operational trust boundary:

::: {.cols .vc}
::: {.col}
- **Hyperscalers (AWS, GCP, Azure):** Full enterprise cloud (IAM, VPCs, compliance, managed storage, GCP TPUs at $\sim\$1.20/\text{hr}$ for v5e); highest on-demand GPU prices and quota approvals.
- **GPU Specialists (Lambda, Nebius, Voltage Park, CoreWeave):** Fast multi-GPU interconnect and prebuilt ML images with self-serve launch ($\sim\$2\text{--}6/\text{hr}$ per H100).
- **Marketplaces (Vast.ai, RunPod):** Aggregate datacenter and community hosts, including consumer RTX 4090s ($\sim\$0.30\text{--}0.89/\text{hr}$) and H100s from $\sim\$1.80/\text{hr}$.
:::

::: {.col .fig}
![](../img/tools-cloud-spectrum.svg){.nostretch width=100%}
:::
:::
:::

::: {.slide title="Trust Boundaries, Compliance, and Host Isolation"}
[Security Architecture]{.kicker}

::: {.cols .vc}
::: {.col}
- **Community marketplace hosts:** Third-party operators have physical and hypervisor root access; disk encryption cannot hide keys or weights that must sit in cleartext RAM/VRAM during execution.
- **Vetted datacenters & hyperscalers:** Use vetted tiers and short-lived scoped tokens for proprietary weights, and require a signed legal **BAA** (plus VPC/KMS controls) before uploading regulated data (HIPAA).
:::

::: {.col .fig .big}
@fig:ch39-tools-trust-boundaries
:::
:::
:::

::: {.slide title="Dated On-Demand GPU Price Snapshot (Nov 2026)"}
[Market Spread]{.kicker}

::: {.cols .vc}
::: {.col}
- **$3\times\text{--}5\times$ price dispersion for identical silicon:** An 80 GB H100 rents for $\$1.80\text{--}3.99/\text{hr}$ on marketplaces, $\$1.99\text{--}4.50/\text{hr}$ on GPU specialists, and $\$6.88\text{--}11.00/\text{hr}$ on hyperscalers.
- **Entry 24 GB tier:** Marketplace RTX 4090s ($\$0.30\text{--}0.89/\text{hr}$) and hyperscaler L4s ($\$0.71\text{--}0.80/\text{hr}$) cover single-GPU fine-tuning and evaluation at minimal invoice cost.
:::

::: {.col .fig .big}
@fig:ch39-tools-price-spread
:::
:::
:::

::: {.slide title="Why Multi-GPU Interconnects Are Not Priced Linearly"}
[Interconnect Arithmetic]{.kicker}

::: {.cols .vc}
::: {.col}
$$b_{\text{balance}} = \frac{2(W-1)}{W} \cdot \frac{F \cdot \beta_{\text{link}}}{3} \quad (N\text{ cancels!})$$

- **NVLink ($450\text{ GB/s}$) vs. PCIe 5.0 ($64\text{ GB/s}$) on $8\times\text{H100}$:** At $395.6\text{ TFLOP/s}$ sustained, communication equals compute at $b_{\text{balance}} \approx 513$ tok/GPU on NVLink vs. $3{,}606$ tok/GPU on PCIe ($W = 8$).
- **Impact at $b = 4{,}096$ tok/GPU:** NVLink adds only $\sim 13\%$ overhead, whereas PCIe adds **$88\%$ overhead** unless amortized by gradient accumulation.
:::

::: {.col .fig}
@fig:ch39-tools-nvlink-vs-pcie-cloud
:::
:::
:::

::: {.slide title="Total Cost per Completed Result vs. Hourly Price"}
[Experiment Economics]{.kicker}

::: {.cols .vc}
::: {.col}
$$\text{Cost} = (T_{\text{wall}} + T_{\text{setup}})p_{\text{gpu}} + C_{\text{store}} + C_{\text{egress}} + (T_{\text{setup}} + f T_{\text{wall}})p_{\text{eng}}$$

- **Direct cloud invoice:** Includes execution wall time $T_{\text{wall}}$, setup hours $T_{\text{setup}}$, persistent storage $C_{\text{store}}$ (billed even while stopped!), and outbound egress $C_{\text{egress}}$.
- **Engineer time ($p_{\text{eng}}$):** Setup/debugging rounds plus fractional waiting attention $f \cdot T_{\text{wall}}$ often outweigh the raw GPU invoice.
:::

::: {.col .fig}
@fig:ch39-tools-completed-result-cost
:::
:::
:::

::: {.slide title="Modeling Experiment Cost Across GPU Tiers" layout="tight"}
[Executable Cost Model]{.kicker}

Compare a marketplace RTX 4090 ($\$0.40/\text{hr}$), A100 80 GB ($\$1.20/\text{hr}$), and H100 ($\$2.50/\text{hr}$) on a job that takes 8 hours on the 4090:

::: {.cols .vc}
::: {.col}
- **Invoice alone:** The 4090 wins ($\$5.40$ vs. $\$8.60$ and $\$9.92$) because its $6.25\times$ lower hourly price beats the H100's $3\times$ speedup.
- **With 0.5 hr setup at $\$60/\text{hr}$:** Total costs converge to **$\$35.40$ vs. $\$38.60$ vs. $\$39.92$**—a fixed setup cost shifts all three equally, leaving the gap under $\$5$.
:::

::: {.col .fig}
@!cloud-instances-cost-model
:::
:::
:::

::: {.slide title="Modeling Experiment Cost Across GPU Tiers (Code)" layout="tight"}
[Executable Cost Model]{.kicker}

@-cloud-instances-cost-model

:::

::: {.slide title="When a Pricier GPU Is Cheaper Overall"}
[Break-Even Analysis]{.kicker}

::: {.cols .vc}
::: {.col}
- **1. One extra 30-min debug round ($\$30+$):** Pushes an 8-hour RTX 4090 run from $\$35.40$ above $\$65$, making the 2.67-hour H100 ($\$39.92$) far cheaper.
- **2. The 7-minute attention threshold ($f \approx 1.4\%$):** Because the H100 saves $5.33\text{ hours}$ of wall time for only $\$4.52$ more on the invoice, if waiting on the 8-hour run consumes even **$6.8\text{ minutes}$** of engineer attention at $\$60/\text{hr}$, the H100 wins!
:::

::: {.col .fig .big}
@fig:ch39-tools-breakeven-attention
:::
:::
:::

::: {.slide title="Spot and Interruptible Capacity Economics"}
[Preemptible Compute]{.kicker}

::: {.cols .vc}
::: {.col}
$$\mathbb{E}[T_{\text{wall}}] = \left(e^{T/M} - 1\right)(M + R)$$

- **Exponential blowup without checkpoints:** If an $8\text{ h}$ run restarts from scratch after Poisson interruptions ($M = 4\text{ h}$, $R = 0.25\text{ h}$), expected wall time reaches $27.2\text{ h}$—costing **$1.70\times$ more** than on-demand even at a $50\%$ spot discount!
- **Never run uncheckpointed jobs or live interactive sessions on spot VMs.**
:::

::: {.col .fig}
@fig:ch39-tools-spot-blowup
:::
:::
:::

::: {.slide title="How Checkpointing Makes Spot Capacity Profitable"}
[Resilient Spot Runs]{.kicker}

When a training script saves a durable checkpoint of duration $\delta$ every $\tau^* = \sqrt{2\delta M}$ hours, each interruption loses at most $\tau^*$ of progress rather than the entire run:

::: {.cols}
::: {.col}
- **Segment wall-time expectation:** Each segment of length $\tau^* + \delta$ finishes in expected wall time:
  $$\mathbb{E}[T_{\text{seg}}] = \left(e^{(\tau^* + \delta)/M} - 1\right)(M + R)$$
- Because $(\tau^* + \delta) / M \ll 1$, the exponential blowup disappears ($e^x - 1 \approx x + x^2/2$).
:::

::: {.col}
- **Concrete impact ($T = 8\text{ h}$, $\delta = 2\text{ min}$, $R = 15\text{ min}$):**
  - Even when interruptions strike every $M = 4\text{ hours}$ on average, checkpointing every $\tau^* \approx 31\text{ minutes}$ keeps total overhead small enough that spot instances remain cheaper than on-demand for any price ratio $q \lesssim 0.80$.
:::
:::
:::

::: {.slide title="The Hidden Bill: Data Egress and Idle Storage"}
[Storage & Network Costs]{.kicker}

::: {.cols .vc}
::: {.col}
- **The egress fee shock ($\sim\$0.09/\text{GB}$):** Downloading a $140\text{ GB}$ checkpoint once a day for 30 days ($4.2\text{ TB}$) costs **$\approx\$370$ in egress fees**—more than renting a marketplace RTX 4090 24/7 for the entire month!
- **Three defensive rules:** Co-locate compute and object storage in the same cloud region, use zero-egress storage (e.g., R2 / RunPod) for cross-cloud workflows, and delete unattached NVMe volumes immediately.
:::

::: {.col .fig .big}
@fig:ch39-tools-egress-shock
:::
:::
:::

::: {.slide title="First-Minute Verification and Smoke Testing"}
[Instance Onboarding]{.kicker}

::: {.cols .vc}
::: {.col}
- **Boot & shell check:** Start inside `tmux new -s train`, run `nvidia-smi` and `df -h`, and verify framework CUDA visibility before starting long jobs.
- **60-second automated smoke test:** Verify (1) free GPU VRAM, (2) writable checkpoint mount (catches read-only disks before hour 1!), (3) 1 forward/backward step + bitwise checkpoint round-trip, and (4) data loader batch/s $>$ GPU step/s.
:::

::: {.col .fig .big}
@fig:ch39-tools-smoke-test-gates
:::
:::
:::

::: {.slide title="Disposable Compute, Durable Artifacts"}
[Operational Lifecycle]{.kicker}

Separate the lifetime of your research from the lifetime of any single virtual machine:

::: {.cols .vc}
::: {.col}
- **Provision & clone:** Boot from a standard DL image, connect via SSH tunnel, and `git clone` your repository.
- **Sync checkpoints out-of-box:** Periodically push checkpoints and logs to durable object storage (S3, GCS, R2) so losing the VM loses at most one checkpoint interval.
- **Teardown & guardrails:** Delete the VM, scratch disks, snapshots, and reserved IPs immediately upon completion. Set a **billing alert** on day one, and keep **account GPU quotas** tight so a leaked API key cannot spin up 100 GPUs overnight.
:::

::: {.col .fig}
![](../img/tools-cloud-lifecycle.svg){.nostretch width=100%}
:::
:::
:::

::: {.slide title="Concept Check: Cloud Economics and Reliability"}
[Check Your Understanding]{.kicker}

Test your cloud systems reasoning on these three problems:

1. In memory profiling of BERT fine-tuning across batch sizes, peak active memory follows $M(B) = 1.8\text{ GiB} + 0.15\text{ GiB} \cdot B$, while the caching allocator's reserved pool is $10\%$ higher. What does the $1.8\text{ GiB}$ intercept represent, and which memory metric determines whether a run OOMs on a 24 GB card?
2. Without a pre-run smoke test, when would a read-only checkpoint directory (`/mnt/ckpt`) surface during an 8-hour training run that saves every hour? What about a slow network disk that feeds 2 batches/s to a GPU capable of 10 steps/s?
3. Excluding engineer time, what `relative_speed` would the H100 ($\$2.50/\text{hr}$) need in `#cloud-instances-cost-model` to match the RTX 4090 ($\$0.40/\text{hr}$) on the cloud invoice alone?

::: {.d2l-note .fragment .fade-up}
**Answers:**

1. The **$1.8\text{ GiB}$ intercept** is the batch-independent memory (parameters, gradients, optimizer states, and CUDA context). The **allocator reserved/peak allocated memory against physical VRAM** decides OOMs.
2. The read-only directory crashes **1 hour into the run** at the first save (wasting an hour of compute). The slow data loader **never raises an error**—it silently starves the GPU at $20\%$ utilization for $5\times$ the cost!
3. Matching the 4090's $\$5.40$ invoice ($\$3.40$ of GPU compute after $\$2.00$ storage/egress) requires $(8.0 / s + 0.5) \times 2.50 = 3.40$ $\implies$ $8.0 / s = 0.86$ $\implies$ **`relative_speed` $\approx 9.3\times$**.
:::
:::

::: {.slide title="Summary: Cloud Accelerator Engineering"}
[Key Takeaways]{.kicker}

| Decision Area | Naive Approach | Systems-Grade Practice |
| :--- | :--- | :--- |
| **Provider tier selection** | Always pick the lowest $\$/\text{hr}$ listing | Match tier to trust boundary: marketplaces for public data, specialists/hyperscalers for private/regulated data |
| **Multi-GPU node selection** | Rent $8\times\text{H100}$ on PCIe at the same price as NVLink | Verify interconnect bandwidth ($450\text{ GB/s}$ NVLink vs. $64\text{ GB/s}$ PCIe) against per-step token batch $b$ |
| **Cost comparison** | Compare hourly GPU rental price $p_{\text{gpu}}$ | Compare **cost per completed result** including wall time, setup/debug rounds, storage, egress, and engineer time |
| **Spot / interruptible VMs** | Run long jobs or interactive sessions without checkpoints | Use spot only with automated checkpointing ($\tau^* = \sqrt{2\delta M}$) and tested restore scripts |
| **Data & checkpoint transfers** | Repeatedly download 100+ GB checkpoints across clouds | Co-locate compute and storage in one region or use zero-egress providers |
| **Instance lifecycle** | Store code and only checkpoints on the VM's local disk | Treat VMs as disposable: Git for code, `tmux` + smoke test on boot, sync to object storage, delete all resources |
:::
