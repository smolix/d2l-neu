# Chapter 22 campaign packages

Chapter 22 (Hardware for Training and Inference) measures what the book's
4 x RTX 4090 box can show and packages the rest for A100 hardware. The
packages are entries of `tools/lm_campaigns.yaml`, each with its hardware,
configuration, command, expected FLOP or bytes, hours and memory, stop time,
output files with their `DATA_HUB` keys, and the steps after a run:

| Entry | Hardware | Launcher | Logs (`DATA_HUB` keys) |
|---|---|---|---|
| `22-hw-c-t2` | one A100 PCIe 40 GB | `tools/hw_regen.py ... --config t2 --out OUT`, `tools/hw_clock_sweep.sh` | `lm-hw-a100-microbench-v1`, `lm-hw-a100-workloads-v1`, `lm-hw-a100-clocksweep-v1` |
| `22-hw-c-t2pair` | two A100 PCIe 40 GB with an NVLink bridge | `tools/hw_regen.py ... --config t2pair --out OUT` | `lm-hw-t2pair-moe-v1`, `lm-hw-t2pair-p2p-v1` |
| `22-hw-c-t3` | one 8 x A100 40 GB node | `node_t3.sh OUT` | `lm-hw-node-storage-v1`, `lm-hw-node-p2p-v1` |
| `22-hw-c-t4` | two such nodes with RDMA (Slurm) | `sbatch cluster_t4.sbatch OUT` (`hw_job.sh` per task, `pack_t4.py` packs) | `lm-hw-cluster-nccl-v1` |

None of them has run: each needs the author's go. The files here:

- `node_t3.sh`: C-T3's pre-checks and its two regenerations.
- `cluster_t4.sbatch`, `hw_job.sh`: C-T4's Slurm job and its per-node tasks
  (ibstat, perftest per rail, nccl-tests on 16 ranks for every forced pair).
- `pack_t4.py`: C-T4's plan (`--plan`), pair verdicts and packer; it also
  packs the committed sample logs (`--make-sample`, `--sample`) and reads a
  tar back as the pages will (`--verify`).
- `estimate.py`: the YAML's FLOP, bytes, hours and memory, from the pages'
  arithmetic and the box's measured seconds.
- `dry_run.sh OUT [--gpu]`: the packages' dry run on the box (commands,
  `bash -n`, `hw_regen.py --dry-run`, the launchers' `--dry-run`, the sample
  tar; with `--gpu`, one real one-card `box-microbench` run with `--out` and
  the `--config t2` refusal on a card that is not an A100).

After a campaign: `make upload-data FILES=... PREFIX=lm/`, one `DATA_HUB`
statement per key (the sha1 the launcher printed) in the owning section's
`-lib.md`, `make lib`, a run of the section, and a rewrite of every sentence
marked `<!-- campaign: <key> -->`.
