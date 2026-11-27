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
Needs the registered box artifacts and the pinned model configurations
(downloaded once; no weights).
"""
import inspect
import os
import sys

DOWNLOAD_BPS = 100e6  # bytes/s assumed for weight downloads on the campaign host
REPEATS_T4 = 3  # tools/hw_campaign/pack_t4.py


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
    pcie4 = 16 * d2l._PCIE_GTS[4] * 128 / 130 / 8 * 1e9  # bytes/s per dir
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
    moe_s = (box_seconds(d2l, 'box-moe')['coverage-box']
             * 4 * box.p16_tflops / (2 * a100.p16_tflops))
    spec = d2l._HW_P2P['t2pair']
    series = p2p_series(d2l, 't2pair', 2)
    bridge = a100.scaleup_gbs / 2 * 1e9  # per direction
    p2p_bytes = 3 * sum(nccl_work(d2l, [(op, k)], k, spec['max_bytes'],
                                  spec['iters'], spec['warmup'])
                        for op, k, _ in series)
    p2p_s = 3 * sum(nccl_work(d2l, [(op, k)], k, spec['max_bytes'],
                              spec['iters'], spec['warmup'])
                    / (pcie4 / 2 if 'P2P_DISABLE' in env else bridge)
                    for op, k, env in series)
    caps = box_seconds(d2l, 'box-p2p')
    ch['22-hw-c-t2pair'] = dict(
        flop=2 * sm.n_matmul() * tokens, bytes=p2p_bytes,
        hours=(moe_s + p2p_s + caps['captures'] + caps['gpu-matrix']
               + sm.weight_bytes() / DOWNLOAD_BPS) / 3600,
        memory_gib=sm.weight_bytes() / 2 / 2**30,
        parts=dict(moe=moe_s, p2p=p2p_s, download=sm.weight_bytes()
                   / DOWNLOAD_BPS))
    # C-T3: one 8 x A100 40 GB node
    sxm = specs.loc['A100 SXM 80 GB']
    storage = box_seconds(d2l, 'box-storage')
    offload_gib = d2l._hw_read(os.path.join(d2l._hw_folder(
        'box-storage'), 'offload.json.gz'))[0].attrs['peak_gib']  # the toy
    storage_s = sum(storage.values()) + storage['concurrent']  # 8 cards
    raw = d2l._HW_NODE_STORAGE['t3']['raw']
    raw_s = (2 * 60 + 3 * 4 * 2**30 * raw['gdsio']['threads'] / 3e9
             + raw['shared_fio']['runtime'] + raw['shared_fio']['ramp'])
    spec = d2l._HW_P2P['t3']
    series = p2p_series(d2l, 't3', 8)
    nvs = sxm.scaleup_gbs / 2 * 1e9
    p2p_bytes = 3 * sum(nccl_work(d2l, [(op, k)], k, spec['max_bytes'],
                                  spec['iters'], spec['warmup'])
                        for op, k, _ in series)
    p2p_s = 3 * sum(nccl_work(d2l, [(op, k)], k, spec['max_bytes'],
                              spec['iters'], spec['warmup'])
                    / (pcie4 / 2 if 'P2P_DISABLE' in env else nvs)
                    for op, k, env in series)
    ch['22-hw-c-t3'] = dict(
        bytes=p2p_bytes, hours=(storage_s + raw_s + p2p_s + caps['captures']
                                + caps['gpu-matrix'] * 4) / 3600,
        memory_gib=max(2 * spec['max_bytes'] / 2**30, offload_gib),
        parts=dict(storage=storage_s, raw=raw_s, p2p=p2p_s))
    # C-T4: two such nodes, 16 ranks
    spec = d2l._HW_COLL['t4']
    rail = d2l.systems_table().loc['8 x A100 SXM 40 GB',
                                   'scaleout_gbs_per_chip'] * 1e9
    per = nccl_work(d2l, [('all_reduce', spec['g'])], spec['g'],
                    spec['max_bytes'], spec['iters'], spec['warmup'])
    check = nccl_work(d2l, [('all_reduce', spec['g'])], spec['g'],
                      spec['max_bytes'], 1, 0)
    n_pairs = len(spec['pairs'])
    t4_bytes = n_pairs * (check + REPEATS_T4 * per) + per  # + one NIC
    ch['22-hw-c-t4'] = dict(
        bytes=t4_bytes, hours=(t4_bytes / rail + 8 * 2 * 60) / 3600,
        memory_gib=2 * spec['max_bytes'] / 2**30,
        parts=dict(nccl=t4_bytes / rail, perftest=8 * 2 * 60))
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
    return 0


if __name__ == '__main__':
    sys.exit(main())
