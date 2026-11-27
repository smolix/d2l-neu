#!/usr/bin/env python3
"""The expected work, memory and hours of chapter 22's campaign packages
(tools/lm_campaigns.yaml, entries 22-hw-c-t2, -t2pair, -t3, -t4).

    python tools/hw_campaign/estimate.py

prints one block per campaign; the YAML entries quote it. The arithmetic is
the pages' own: the configurations of the drivers (HW_WORKLOAD_CONFIGS,
_HW_MOE, _HW_P2P, _HW_COLL, _HW_CLOCK_CONFIGS), ServingModel and
TrainStepModel at the chip-table rows (datasheet rates: predictions, not
measurements), d2l.flops_per_token and nccl_busbw_factor, and the box's
measured seconds per entry of its registered results (env['seconds'],
summed over the three processes). Time = the box's seconds of the same
entries, scaled by the predicted A100/box ratio where the work is fixed
(the workload grid, the MoE forward, collectives as bytes over the link's
datasheet rate) and unscaled where a benchmark runs for a fixed time.
Series that run with NCCL_P2P_DISABLE=1 take the box's own measured rate of
that path (the registered box-p2p fit of NCCL's shared-memory transport,
about 2.2 GB/s busbw), not a PCIe datasheet rate. The two-node collectives
are timed per protocol over all eight rails (assumptions below, labelled
as such); a pair the A100 cannot run (NVLS) costs its check only. The
figures are predictions with stated assumptions, not measurements; the
stop_hours of each entry leave room for them to be wrong. The optional
C-T2pair MoE run is reported apart from that package's hours.
Needs the registered box artifacts and the pinned model configurations
(downloaded once; no weights).
"""
import importlib.util
import inspect
import os
import sys

_spec = importlib.util.spec_from_file_location(
    'pack_t4', os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            'pack_t4.py'))
pack_t4 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(pack_t4)  # its HIER configuration

DOWNLOAD_BPS = 100e6  # bytes/s assumed for weight downloads on the campaign host
REPEATS_T4 = 3  # tools/hw_campaign/pack_t4.py
# Assumptions of the time model (not measured; each entry's stop_hours covers them):
NUMA_MAX = 8  # NUMA nodes of a two-socket node in NPS4 mode (the most there can be)
NVB_CASE_S = 30  # seconds per nvbandwidth test case and NUMA node
PAIR_CLASSES = 4  # GPU-pair relation classes of a node (NV#, PIX/PXB, NODE, SYS)
RAILS_T4 = 8  # one HDR rail per GPU, eight per node
PROTO_EFF = {'Simple': 1.0, 'LL128': 120 / 128, 'default': 1.0}  # of the rails' rate
LL_RAILS = 0.5  # forced LL at large sizes: half of one rail (LL moves 4 data bytes per 8)
LAUNCH_S = 20  # seconds per srun step (MPI start-up, NCCL communicator set-up)


def lib():
    os.environ.pop('D2L_RERUN', None)
    from d2l import torch as d2l
    return d2l


def box_seconds(d2l, name):
    """{entry: seconds} of a registered box result (all processes)."""
    folder = d2l._hw_folder(name)
    return {f[:-len('.json.gz')]: d2l._hw_read(os.path.join(folder, f))[1][
        'env']['seconds'] for f in sorted(os.listdir(folder))
        if f.endswith('.json.gz')}


def default(fn, arg):
    return inspect.signature(fn).parameters[arg].default


def grid_work(d2l, config, chip_row=None):
    """(FLOP, predicted seconds of the timed calls) of the workload grid
    HW_WORKLOAD_CONFIGS[config] on a chip-table row (default: the
    configuration's chip)."""
    spec = d2l.HW_WORKLOAD_CONFIGS[config]
    chip = (d2l.chip_specs().loc[chip_row] if chip_row
            else d2l._cut_chip(spec))
    steps = default(d2l.hw_decode_bench, 'steps')
    reps = default(d2l.hw_prefill_bench, 'repeats')
    train = (default(d2l.hw_train_bench, 'repeats')
             + default(d2l.hw_train_bench, 'warmup'))
    flop = seconds = 0.0
    for model in d2l._grid_models(spec):
        cfg = d2l._hw_model_config(model)
        sm = d2l.ServingModel(cfg, chip)
        fwd = lambda T, n: d2l.flops_per_token(cfg, T=T) / 3 * n
        kept, _ = d2l._serving_points(spec, model, chip)
        eager = [(b, L, 'eager') for mid, bs, L in spec['eager']
                 if mid == model for b in bs]
        for b, L, _ in kept + eager:
            flop += steps * fwd(L, b)
            seconds += steps * sm.decode_step(b, L)
        for n in next((p for mid, p in spec['prefill'] if mid == model), ()):
            flop += reps * d2l.flops_per_token(cfg, T=n, causal_half=True) \
                / 3 * n
            seconds += reps * sm.prefill_time(n)
        for mid, layers, shapes in spec['train']:
            for n_layers in layers if mid == model else ():
                tm = d2l.TrainStepModel(d2l._hw_model_config(model, n_layers),
                                        chip)
                for b, s in shapes:
                    flop += train * tm.flops_per_token(s) * b * s
                    seconds += train * tm.step_time(b, s)
    return flop, seconds


def nccl_work(d2l, series, g, max_bytes, iters, warmup, min_bytes=8,
              runs=1):
    """Bytes each GPU's link carries in nccl-tests sweeps (8 B to max_bytes
    in factors of 2, warm-up and timed iterations; series: (op, ranks)
    pairs), over all runs."""
    sizes = d2l._sizes(min_bytes, max_bytes)
    return runs * (iters + warmup) * sum(
        d2l.nccl_busbw_factor(op, k) * sum(sizes) for op, k in series)


def p2p_series(d2l, config, n_gpus):
    """(op, ranks, env) of _HW_P2P[config]'s series: pairs run one pair (2
    ranks), 'all' n_gpus, an integer that many."""
    out = []
    for op, place, algo, env in d2l._HW_P2P[config]['series']:
        k = 2 if place == 'pairs' else n_gpus if place == 'all' else place
        out.append((op, k, env))
    return out


def estimates(d2l):
    specs, ch = d2l.chip_specs(), {}
    a100, box = specs.loc['A100 PCIe 40 GB'], specs.loc['RTX 4090']
    # C-T2: one A100 PCIe 40 GB
    micro = sum(box_seconds(d2l, 'box-microbench').values())
    work = box_seconds(d2l, 'box-workloads')
    _, box_s = grid_work(d2l, 'box')  # predicted, both models
    flop_t2, a100_s = grid_work(d2l, 't2')
    grid_s = work['grid-box'] * a100_s / box_s  # measured x predicted ratio
    sm = lambda chip: d2l.ServingModel(
        d2l._hw_model_config('Qwen/Qwen3-8B'), chip).decode_step(1, 2048)
    control_s = work['control-Qwen3-8B'] * sm(a100) / sm(box)
    clocks = {c: len(d2l._HW_CLOCK_CONFIGS[c]['clocks']) + 1
              for c in ('box', 't2')}  # + the edge lock
    sweep_s = (sum(box_seconds(d2l, 'box-clocksweep').values())
               * clocks['t2'] / clocks['box'])
    w8 = d2l.ServingModel(d2l._hw_model_config('Qwen/Qwen3-8B'),
                          a100).weight_bytes()
    ch['22-hw-c-t2'] = dict(
        flop=flop_t2, hours=(micro + grid_s + control_s + sweep_s
                             + w8 / DOWNLOAD_BPS) / 3600,
        memory_gib=a100.mem_gb * 0.9 * 1e9 / 2**30,
        parts=dict(microbench=micro, workloads=grid_s + control_s,
                   clocksweep=sweep_s, download=w8 / DOWNLOAD_BPS))
    # C-T2pair: two A100 PCIe 40 GB with an NVLink bridge
    moe = d2l.HW_WORKLOAD_CONFIGS['t2pair']['moe']
    cfg = d2l._hw_model_config(moe['model'])
    sm = d2l.ServingModel(cfg, a100)
    tokens = (moe['n_prose'] + moe['n_code']) * moe['n_tokens']
    # the split model computes one card at a time on both hosts: one card's rate
    moe_s = (box_seconds(d2l, 'box-moe')['coverage-box']
             * box.p16_tflops / a100.p16_tflops)
    shm = d2l.box_alpha_beta(g=4)['beta']  # the box's NCCL shared-memory path
    spec = d2l._HW_P2P['t2pair']
    series = p2p_series(d2l, 't2pair', 2)
    bridge = a100.scaleup_gbs / 2 * 1e9  # per direction
    p2p_bytes = 3 * sum(nccl_work(d2l, [(op, k)], k, spec['max_bytes'],
                                  spec['iters'], spec['warmup'])
                        for op, k, _ in series)
    p2p_s = 3 * sum(nccl_work(d2l, [(op, k)], k, spec['max_bytes'],
                              spec['iters'], spec['warmup'])
                    / (shm if 'P2P_DISABLE' in env else bridge)
                    for op, k, env in series)
    caps = box_seconds(d2l, 'box-p2p')
    moe_opt = moe_s + sm.weight_bytes() / DOWNLOAD_BPS
    ch['22-hw-c-t2pair'] = dict(
        bytes=p2p_bytes,
        hours=(p2p_s + caps['captures'] + caps['gpu-matrix']) / 3600,
        memory_gib=2 * spec['max_bytes'] / 2**30,
        optional=dict(flop=2 * sm.n_matmul() * tokens, hours=moe_opt / 3600,
                      memory_gib=sm.weight_bytes() / 2 / 2**30),
        parts=dict(p2p=p2p_s, optional_moe=moe_s,
                   optional_download=sm.weight_bytes() / DOWNLOAD_BPS))
    # C-T3: one 8 x A100 40 GB node
    sxm = specs.loc['A100 SXM 80 GB']
    storage = box_seconds(d2l, 'box-storage')
    offload_gib = d2l._hw_read(os.path.join(d2l._hw_folder(
        'box-storage'), 'offload.json.gz'))[0].attrs['peak_gib']  # the toy
    storage_s = sum(storage.values()) + storage['concurrent']  # 8 cards
    raw = d2l._HW_NODE_STORAGE['t3']['raw']
    raw_s = (NUMA_MAX * len(raw['nvbandwidth']) * NVB_CASE_S
             + 3 * 4 * 2**30 * raw['gdsio']['threads'] / 3e9
             + raw['shared_fio']['runtime'] + raw['shared_fio']['ramp'])
    spec = d2l._HW_P2P['t3']
    places = [place for _, place, _, _ in spec['series']]
    series = p2p_series(d2l, 't3', 8)
    nvs = sxm.scaleup_gbs / 2 * 1e9
    runs = lambda place: PAIR_CLASSES if place == 'pairs' else 1
    p2p_bytes = 3 * sum(runs(place) * nccl_work(
        d2l, [(op, k)], k, spec['max_bytes'], spec['iters'], spec['warmup'])
        for (op, k, _), place in zip(series, places))
    p2p_s = 3 * sum(runs(place) * nccl_work(
        d2l, [(op, k)], k, spec['max_bytes'], spec['iters'], spec['warmup'])
        / (shm if 'P2P_DISABLE' in env else nvs)
        for (op, k, env), place in zip(series, places))
    ch['22-hw-c-t3'] = dict(
        bytes=p2p_bytes, hours=(storage_s + raw_s + p2p_s + caps['captures']
                                + caps['gpu-matrix'] * 4) / 3600,
        memory_gib=max(2 * spec['max_bytes'] / 2**30, offload_gib),
        parts=dict(storage=storage_s, raw=raw_s, p2p=p2p_s))
    # C-T4: two such nodes, 16 ranks
    spec, g = d2l._HW_COLL['t4'], d2l._HW_COLL['t4']['g']
    rail = d2l.systems_table().loc['8 x A100 SXM 40 GB',
                                   'scaleout_gbs_per_chip'] * 1e9
    rails = min(RAILS_T4 * rail, nvs)  # busbw over all rails, NVLink-bound
    per = nccl_work(d2l, [('all_reduce', g)], g, spec['max_bytes'],
                    spec['iters'], spec['warmup'])
    check = nccl_work(d2l, [('all_reduce', g)], g, spec['max_bytes'], 1, 1)
    rate = lambda algo, proto: (LL_RAILS * rail if proto == 'LL'
                                else PROTO_EFF[proto] * rails)
    t4_bytes = nccl_s = 0.0
    launches = 0
    for algo, proto in spec['pairs']:
        timed = 0 if algo == 'NVLS' else REPEATS_T4  # no NVLS on the A100
        t4_bytes += check + timed * per
        nccl_s += (check + timed * per) / rate(algo, proto)
        launches += 1 + timed
    t4_bytes += per  # the default pair on one rail
    nccl_s += per / rail
    launches += 1
    # the hierarchical runs: every rail, then one rail
    sizes = sum(pack_t4.HIER['sizes'])
    n_it = pack_t4.HIER['iters'] + pack_t4.HIER['warmup']
    local = g // 2
    phi = d2l.nccl_busbw_factor('all_reduce', g)
    intra = 2 * (local - 1) / local * sizes  # reduce-scatter + all-gather
    inter = sizes / local  # a 2-rank all-reduce of each shard
    hier_bytes = 2 * n_it * (phi * sizes + 2 * (intra + inter))
    hier_s = n_it * (phi * sizes / rails + 2 * (intra / nvs + inter / rail)
                     + phi * sizes / rail + 2 * (intra / nvs
                                                 + local * inter / rail))
    launches += 2
    perf_s = RAILS_T4 * 2 * 60  # one minute per rail and kind
    t4_bytes += hier_bytes
    ch['22-hw-c-t4'] = dict(
        bytes=t4_bytes, hours=(nccl_s + hier_s + perf_s
                               + launches * LAUNCH_S) / 3600,
        memory_gib=2 * spec['max_bytes'] / 2**30,
        parts=dict(nccl=nccl_s, hier=hier_s, perftest=perf_s,
                   launches=launches * LAUNCH_S))
    return ch


def main():
    d2l = lib()
    for cid, e in estimates(d2l).items():
        work = ', '.join(f'{k} {e[k]:.2e}' for k in ('flop', 'bytes')
                         if k in e)
        parts = ', '.join(f'{k} {v / 3600:.2f} h' for k, v in e['parts']
                          .items())
        print(f"{cid}: {work}; {e['hours']:.2f} h; "
              f"{e['memory_gib']:.0f} GiB per GPU\n  {parts}")
        if 'optional' in e:
            o = e['optional']
            print(f"  optional run: flop {o['flop']:.2e}; {o['hours']:.2f} h; "
                  f"{o['memory_gib']:.0f} GiB per GPU")
    return 0


if __name__ == '__main__':
    sys.exit(main())
