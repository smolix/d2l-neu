#!/usr/bin/env python3
"""One rank of campaign C-T4's hierarchical all-reduce (chapter 22, section
sec_lm_collectives, claim C5), started on every GPU of the two nodes by
tools/hw_campaign/hw_job.sh through srun (one rank per GPU; RANK, WORLD_SIZE,
LOCAL_RANK, LOCAL_WORLD_SIZE, MASTER_ADDR and MASTER_PORT set there).

    hier_t4.py SIZES_JSON ITERS WARMUP CHECK_BYTES

The groups are the cluster form of the box's 2 x 2 split: the intra groups are
the ranks of one node (NCCL over NVLink), the inter groups the ranks with the
same local rank on every node (NCCL over that GPU's own rail), all on NCCL; the
box runs its inter stage on gloo because it has no network. At each size it
times, as the median over iterations of the slowest rank, NCCL's default
all-reduce on all ranks ('flat NCCL'), the page's `d2l.hier_allreduce`
('hierarchical') and its two stages alone ('intra stage' on S bytes, 'inter
stage' on S / local ranks), and checks `hier_allreduce` once against a CPU sum
of every rank's input (rtol 1e-5). Rank 0 prints one JSON line per check and
per (variant, size), the rows of box-coll's entry `hier`; pack_t4.py reads
them. Run it twice: with every rail, and with NCCL_IB_HCA set to one rail.
"""
import json
import os
import sys
import time

import torch
import torch.distributed as dist
from d2l import torch as d2l


def main():
    sizes, iters, warmup, check = (json.loads(sys.argv[1]), int(sys.argv[2]),
                                   int(sys.argv[3]), int(sys.argv[4]))
    rank, k = int(os.environ['RANK']), int(os.environ['WORLD_SIZE'])
    local = int(os.environ['LOCAL_WORLD_SIZE'])
    nodes = k // local
    dev = torch.device('cuda', int(os.environ['LOCAL_RANK'])
                       % torch.cuda.device_count())
    torch.cuda.set_device(dev)
    dist.init_process_group('nccl', device_id=dev)
    intra, _ = dist.new_subgroups_by_enumeration(
        [list(range(n * local, (n + 1) * local)) for n in range(nodes)])
    inter, _ = dist.new_subgroups_by_enumeration(
        [[n * local + i for n in range(nodes)] for i in range(local)])
    variants = {
        'flat NCCL': lambda x: dist.all_reduce(x),
        'hierarchical': lambda x: d2l.hier_allreduce(x, intra, inter),
        'intra stage': lambda x: dist.all_reduce(x, group=intra),
        'inter stage': lambda x: dist.all_reduce(x[:x.numel() // local],
                                                 group=inter)}
    if check:
        n = check // 4
        gen = lambda r: torch.Generator().manual_seed(r)
        ref = sum(torch.randn(n, generator=gen(r)) for r in range(k))
        x = torch.randn(n, generator=gen(rank)).to(dev)
        d2l.hier_allreduce(x, intra, inter)
        torch.cuda.synchronize()
        err = float(((x.cpu() - ref).abs() / ref.abs().max()).max())
        ok = bool(torch.allclose(x.cpu(), ref, rtol=1e-5, atol=1e-6))
        if rank == 0:
            print(json.dumps(dict(check=check, max_rel_err=err, ok=ok,
                                  ranks=k, nodes=nodes)), flush=True)
    for size in sizes:
        for name, op in variants.items():
            x = torch.ones(size // 4, device=dev)
            for _ in range(warmup):
                op(x)
            ts = []
            for _ in range(iters):
                torch.cuda.synchronize()
                dist.barrier()
                t = time.perf_counter()
                op(x)
                torch.cuda.synchronize()
                ts.append(time.perf_counter() - t)
            ts = torch.tensor(ts, device=dev)
            dist.all_reduce(ts, op=dist.ReduceOp.MAX)
            ts = ts.cpu()
            med = ts.median()
            if rank == 0:
                print(json.dumps(dict(
                    variant=name, size=size, us=float(med) * 1e6,
                    mad_us=float((ts - med).abs().median()) * 1e6)),
                    flush=True)
            del x
    dist.destroy_process_group()
    return 0


if __name__ == '__main__':
    sys.exit(main())
