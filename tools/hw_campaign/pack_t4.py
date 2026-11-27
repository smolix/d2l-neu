#!/usr/bin/env python3
"""Campaign C-T4 of chapter 22 (two 8 x A100 nodes with RDMA): the plan the
job script runs and the packer of its raw logs.

    python tools/hw_campaign/pack_t4.py --plan
    python tools/hw_campaign/pack_t4.py --verdict OUT ALGO PROTO
    python tools/hw_campaign/pack_t4.py OUT [--replace] [--sample]
    python tools/hw_campaign/pack_t4.py --make-sample OUT
    python tools/hw_campaign/pack_t4.py --verify TAR

`--plan` prints, as shell assignments, the configuration the job script
(cluster_t4.sbatch) runs: `d2l._HW_COLL['t4']` (ranks, sizes, iterations,
timeout, the NCCL_ALGO x NCCL_PROTO pairs), the number of timed repeats and
the hierarchical run's sizes, iterations, check size and timeout (HIER).
`--verdict` exits 0 when a pair's check run (OUT/raw/nccl/<pair>-check.*)
keeps the pair, else 1 with the reason: the rules of the box's
`d2l._pair_check` (refused, timed out, wrong results, an INFO choice other
than the forced pair, or no choice at all for a forced pair; NCCL_ALGO=
CollNet allows NCCL's CollNetDirect and CollNetChain).

`pack_t4.py OUT` reads OUT/raw (written by the job script: node names,
ibstat per node, ib_write_bw and ib_write_lat per rail, the nccl-tests
check and timed runs per pair, the one-NIC run, the hierarchical runs)
and writes
OUT/<key>/ and OUT/<key>.tar, the campaign log `cluster-nccl`
(`d2l.HW_CAMPAIGNS`; key lm-hw-cluster-nccl-v1 while it is unregistered):
entries `pairs` (the forced-pair table of the box's box-coll, with
config 't4', every repeat's times in attrs['proc_us']), `info` (NCCL's INFO
digest of the default pair), `perftest` ({rail: {'bw': text, 'lat':
text}}, read by `d2l.fit_perftest`), `ibstat`, `onenic` (the default pair
with NCCL_IB_HCA set to one rail), `hier` (box-coll's `hier` table for 16
ranks: variant, size, us, mad_us, plus `nics` 'all' or 'one'; the checks
and any failed run in attrs), `alpha_beta.json` (one record per kept
pair and per rail, scope inter_node) and the scrubbed raw files (node
names become <node0>, <node1>). Every rank of every nccl-tests run must be
an A100 40 GB (the node campaign's GPUs, `d2l._HW_CAMPAIGN_GPUS['t4']`),
and a real pack needs 16 ranks. Then it prints the size, the sha1 and the
DATA_HUB statement to register.

`--make-sample OUT` writes OUT/raw from the committed sample logs
(tools/lm_data/hw/samples: the box's nccl-tests run and INFO lines on 4
GPUs stand in for every pair, hand-written perftest and ibstat samples), and
`pack_t4.py OUT --sample` packs it as `<key>-sample.tar` (meta sample=True,
ranks as recorded): a test of the packer, never a campaign result.
`--verify TAR` reads a tar as `d2l.hw_campaign_log('cluster-nccl')` would
once registered and prints the fits the two pages show (`fit_perftest`,
`fit_pairs`); exit 0 when both have rows.
"""
import argparse
import glob
import hashlib
import json
import os
import re
import shutil
import sys
import tempfile
from unittest import mock

REPO = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))
SAMPLES = os.path.join(REPO, 'tools', 'lm_data', 'hw', 'samples')
CONFIG, NAME, REPEATS = 't4', 'cluster-nccl', 3
HIER = dict(sizes=[2**s for s in range(20, 31, 2)], iters=10, warmup=3,
            check_bytes=2**20, timeout_s=900)  # 1 MiB to 1 GiB
ALIASES = {'CollNet': ('CollNetDirect', 'CollNetChain')}  # NCCL_ALGO=CollNet


def lib():
    os.environ.pop('D2L_RERUN', None)
    from d2l import torch as d2l
    return d2l


def stem(algo, proto):
    """File stem of a pair: 'default' or 'Ring-LL'."""
    return 'default' if algo == 'default' else f'{algo}-{proto}'


def read(path):
    with open(path, errors='replace') as f:
        return f.read()


def plan(d2l):
    s = d2l._HW_COLL[CONFIG]
    pairs = ' '.join(f'{a}:{p}' for a, p in s['pairs'])
    print(f"G={s['g']} MIN_BYTES={s['min_bytes']} MAX_BYTES={s['max_bytes']} "
          f"ITERS={s['iters']} WARMUP={s['warmup']} "
          f"TIMEOUT_S={s['timeout_s']} REPEATS={REPEATS}")
    print(f'PAIRS="{pairs}"')
    print(f"HIER_SIZES='{json.dumps(HIER['sizes'], separators=(',', ':'))}' "
          f"HIER_ITERS={HIER['iters']} HIER_WARMUP={HIER['warmup']} "
          f"HIER_CHECK={HIER['check_bytes']} "
          f"HIER_TIMEOUT_S={HIER['timeout_s']}")
    return 0


def status(base):
    """'' for a run that exited 0, else 'timed out' or 'refused' (from the
    job script's <base>.rc: the exit status, 124 or 137 after timeout)."""
    if not os.path.exists(base + '.txt') or not os.path.exists(base + '.rc'):
        return 'not run'
    rc = read(base + '.rc').strip()
    if rc == '0':
        return ''
    return 'timed out' if rc in ('124', '137') else 'refused'


def verdict(d2l, raw, algo, proto):
    """(reason or '', {bytes: pair NCCL ran}, the tuner's table) of a pair's
    check run: the rules of d2l._pair_check applied to the files."""
    base = os.path.join(raw, 'nccl', stem(algo, proto) + '-check')
    forced = algo != 'default'
    why = status(base)
    if why:
        return why, {}, {}
    table = d2l.parse_nccl_tests(read(base + '.txt'))
    if (table[['oop_wrong', 'ip_wrong']].fillna(0) > 0).any().any():
        return 'wrong results', {}, {}
    text = read(base + '.info') if os.path.exists(base + '.info') else ''
    used, tuner = d2l._choices(text), d2l._tuner_table(text)
    want = {f'{a}/{proto}'.lower() for a in ALIASES.get(algo, (algo,))}
    other = sorted({u for u in used.values()
                    if forced and u.lower() not in want})
    if other:
        return f"ignored: ran {', '.join(other)}", used, tuner
    if forced and not used:
        return 'unverified', used, tuner
    return '', used, tuner


def gpu_names(text):
    """The GPU name of every rank from nccl-tests' '#  Rank' lines."""
    return [r.strip() for r in re.findall(
        r'^#\s+Rank\s+\d+ .*? device\s+\d+ \[[^\]]*\] (.+)$', text, re.M)]


def ranks(text):
    """(number of ranks, GPU name) from nccl-tests' '#  Rank' lines."""
    rows = gpu_names(text)
    return len(rows), (rows[0] if rows else None)


def gpu_ok(d2l, name):
    """True when a device name is one of the campaign's GPU models
    (d2l._HW_CAMPAIGN_GPUS['t4']: an A100 40 GB, PCIe or SXM4); nccl-tests
    prints no memory size, so the name's capacity is compared."""
    key = lambda x: re.sub(r'[^a-z0-9]', '', x.lower())
    for model in d2l._HW_CAMPAIGN_GPUS[CONFIG][0]:
        *words, gb, _ = model.split()
        if key(''.join(words)) in key(name) and f'{gb}gb' in key(name):
            return True
    return False


def wrong_gpus(d2l, raw):
    """{device name: file} of nccl-tests ranks that are not the campaign's
    GPUs (empty when every rank of every run is one)."""
    bad = {}
    for path in sorted(glob.glob(os.path.join(raw, 'nccl', '*.txt'))):
        for name in gpu_names(read(path)):
            if not gpu_ok(d2l, name):
                bad.setdefault(name, os.path.relpath(path, raw))
    return bad


def hier_table(raw):
    """The hierarchical runs (raw/hier/all.txt and one.txt, rank 0's JSON
    lines from hier_t4.py) as box-coll's `hier` table with a column nics,
    or None when neither ran; attrs: the checks and the failed runs."""
    import pandas as pd
    frames, checks, failed = [], {}, {}
    for nics in ('all', 'one'):
        base = os.path.join(raw, 'hier', nics)
        why = status(base)
        recs = [json.loads(line) for line in (
            read(base + '.txt').splitlines() if not why else ())
            if line.startswith('{')]
        rows = [r for r in recs if 'variant' in r]
        if why or not rows:
            failed[nics] = why or 'no rows'
            continue
        checks[nics] = next((r for r in recs if 'check' in r), None)
        frames.append(pd.DataFrame(rows).assign(nics=nics))
    if not frames:
        return None
    out = pd.concat(frames, ignore_index=True)
    out.attrs.update(check=checks, failed=failed,
                     intra='the 8 ranks of a node (NCCL)',
                     inter='the ranks of one local rank across the nodes '
                           '(NCCL, one rail each)')
    return out


def nccl_version(info, text):
    """NCCL's version: from the INFO digest, else nccl-tests' header."""
    if info.get('version'):
        return info['version']
    m = re.search(r'nccl-library=(\d+)', text)
    if not m:
        return None
    v = int(m.group(1))
    return f'{v // 10000}.{v // 100 % 100}.{v % 100}'


def pairs_table(d2l, raw, spec, sample):
    """The long forced-pair table (the box's _pairs_compute layout), the
    per-repeat oop_us lists, the default pair's INFO text and the first
    timed run's text."""
    import numpy as np
    import pandas as pd
    frames, times, dropped, tuner, first = [], [], {}, {}, None
    sizes = d2l._sizes(spec['min_bytes'], spec['max_bytes'])
    info_text = ''
    for algo, proto in spec['pairs']:
        name, s = d2l._pair_name(algo, proto), stem(algo, proto)
        reason, used, table = verdict(d2l, raw, algo, proto)
        if algo == 'default':
            tuner = {f'{a}/{p}': v for (a, p), v in table.items()}
            path = os.path.join(raw, 'nccl', 'default-check.info')
            info_text = read(path) if os.path.exists(path) else ''
        runs = []
        for base in sorted(glob.glob(os.path.join(raw, 'nccl',
                                                  f'{s}-run*.txt'))):
            why = status(base[:-4]) if not reason else reason
            if why:
                reason = why
                break
            runs.append(d2l.parse_nccl_tests(read(base)))
            first = first or read(base)
        if not reason and not runs:
            reason = 'not run'
        if sample and not reason:
            sizes = sorted(runs[0]['size'])
        if reason:
            dropped[name] = reason
            runs = []
        df = (d2l._hw_combine(runs) if runs
              else pd.DataFrame({'size': sizes}))
        df = df.set_index('size').reindex(sizes).reset_index()
        df['used'] = [used.get(x, '') for x in sizes]
        frames.append(df.assign(op='all_reduce', g=0, path='all', algo=name,
                                proto=proto, env='', status=reason))
        times.append([r.set_index('size').reindex(sizes).oop_us.tolist()
                      for r in runs])  # [] for a dropped pair
    n = max(map(len, times), default=0)  # repeats of the kept pairs
    proc_us = [sum((t[i] if i < len(t) else [np.nan] * len(sizes)
                    for t in times), []) for i in range(n)]
    return frames, proc_us, dropped, tuner, info_text, first


def scrub(d2l, text, nodes):
    for i, node in enumerate(nodes):
        for name in {node, node.split('.')[0]} - {''}:
            text = re.sub(rf'(?<![\w.-]){re.escape(name)}(?![\w.-])',
                          f'<node{i}>', text)
    return d2l._hw_scrub(text)


def link_layer(ibstat):
    """'infiniband' or 'roce' from ibstat's 'Link layer' lines."""
    layers = set(re.findall(r'Link layer:\s*(\w+)', ' '.join(ibstat)))
    return ('roce' if layers == {'Ethernet'} else 'infiniband'
            if layers == {'InfiniBand'} else 'rdma')


def pack(d2l, out, replace=False, sample=False):
    import pandas as pd
    raw = os.path.join(out, 'raw')
    spec = d2l._HW_COLL[CONFIG]
    key = d2l.HW_CAMPAIGNS[NAME][0]
    nodes_file = os.path.join(raw, 'nodes.txt')
    nodes = read(nodes_file).split() if os.path.exists(nodes_file) else []
    frames, proc_us, dropped, tuner, info_text, first = pairs_table(
        d2l, raw, spec, sample)
    if first is None:
        print(f'{raw}: no timed nccl-tests run of any pair', file=sys.stderr)
        return 1
    g, gpu = ranks(first)
    if g != spec['g'] and not sample:
        print(f"{g} ranks recorded, the configuration has {spec['g']}",
              file=sys.stderr)
        return 1
    bad = wrong_gpus(d2l, raw)
    if bad and not sample:
        name, path = next(iter(bad.items()))
        print(f'{path}: a rank ran on {name}, not one of '
              f"{', '.join(d2l._HW_CAMPAIGN_GPUS[CONFIG][0])}",
              file=sys.stderr)
        return 1
    pairs = pd.concat(frames, ignore_index=True)
    pairs['g'] = g
    info = d2l._parse_info(info_text)
    info['tuner'] = {f'{a}/{p}': v for (a, p), v in info['tuner'].items()}
    version = nccl_version(info, first)
    pairs.attrs.update(dropped=dropped, tuner=tuner,
                       alpha_below=spec['alpha_below'],
                       plateau_from=spec['plateau_from'], nccl=version,
                       tool='nccl-tests', proc_us=proc_us,
                       n_proc=len(proc_us))
    perf = {}
    for bw in sorted(glob.glob(os.path.join(raw, 'perftest', '*-bw.txt'))):
        rail = os.path.basename(bw)[:-len('-bw.txt')]
        lat = bw[:-len('-bw.txt')] + '-lat.txt'
        if os.path.exists(lat):
            perf[rail] = dict(bw=read(bw), lat=read(lat))
    ibstat = {f'node{i}': scrub(d2l, read(p), nodes) for i, p in enumerate(
        sorted(glob.glob(os.path.join(raw, 'ibstat-[0-9]*.txt'))))}
    stamp = key + ('-sample' if sample else '')
    folder = os.path.join(out, stamp if sample else d2l._next_key(key))
    tar = folder + '.tar'
    if os.path.exists(folder) or os.path.exists(tar):
        if not replace:
            print(f'{folder} exists; pass --replace', file=sys.stderr)
            return 2
        shutil.rmtree(folder, ignore_errors=True)
        if os.path.exists(tar):
            os.remove(tar)
    config = dict(config=CONFIG, g=spec['g'],
                  pairs=[list(p) for p in spec['pairs']],
                  min_bytes=spec['min_bytes'], max_bytes=spec['max_bytes'],
                  iters=spec['iters'], warmup=spec['warmup'])
    tool = re.search(r'nccl-tests version (\S+)', first)
    env = dict(tier='T4', gpu=gpu, n_gpu=g, nodes=len(nodes) or None,
               nccl=version, nccl_tests=tool.group(1) if tool else None,
               contended=False, neighbours_busy=False, seconds=None,
               n_proc=len(proc_us))
    meta = dict(config=config, env=env, label=NAME, sample=sample)
    entries = dict(pairs=pairs, info=info, perftest=perf, ibstat=ibstat)
    onenic = os.path.join(raw, 'nccl', 'onenic-run0')
    if not status(onenic):
        entries['onenic'] = d2l.parse_nccl_tests(read(onenic + '.txt'))
    hier = hier_table(raw)
    if hier is not None:
        entries['hier'] = hier
    for entry, result in entries.items():
        d2l._hw_write(os.path.join(folder, f'{entry}.json.gz'), result, meta)
    layer = link_layer(list(ibstat.values()))
    recs = d2l._coll_records(d2l.fit_pairs(pairs), info)
    for r in recs:
        r.update(scope='inter_node', link=layer)
    fits = d2l.fit_perftest(dict(perftest=perf)) if perf else None
    for (op, _, rail), r in (fits.iterrows() if fits is not None else ()):
        recs.append(dict(scope='inter_node', link=layer, op=op, g=1,
                         algo=None, proto=None, transport=None, path=rail,
                         env=None, alpha_s=float(r.alpha),
                         beta_Bps=float(r.beta), alpha_ci95=[None, None],
                         beta_ci95=[None, None],
                         n_half_bytes=float(r.n_half), resid_rel=None,
                         n_points=int(r.n_points)))
    with open(os.path.join(folder, 'alpha_beta.json'), 'w') as f:
        json.dump(recs, f, indent=1)
    for path in sorted(glob.glob(os.path.join(raw, '**', '*'),
                                 recursive=True)):
        if os.path.isfile(path) and not path.endswith('nodes.txt'):
            dest = os.path.join(folder, 'raw', os.path.relpath(path, raw))
            os.makedirs(os.path.dirname(dest), exist_ok=True)
            with open(dest, 'w') as f:
                f.write(scrub(d2l, read(path), nodes))
    d2l._hw_pack(folder, 'tar', tar)
    with open(tar, 'rb') as f:
        sha1 = hashlib.sha1(f.read()).hexdigest()
    kept = [n for n in pairs.algo.unique() if n not in dropped]
    print(f'{NAME}: {g} ranks, {len(proc_us)} repeats, pairs kept '
          f"{', '.join(kept) or 'none'}; {len(dropped)} dropped; "
          f'{len(perf)} rail(s); hierarchical runs: '
          + (', '.join(sorted(set(hier.nics))) if hier is not None
             else 'none'))
    print(f'wrote {tar}\n  size {os.path.getsize(tar) / 1e6:.2f} MB, '
          f'sha1 {sha1}')
    if sample:
        print('  sample logs: a test of the packer, never uploaded')
    else:
        print(f'register: d2l.DATA_HUB[{os.path.basename(folder)!r}] = (\n'
              f'    d2l.LM_DATA_URL + {os.path.basename(tar)!r}, {sha1!r})')
    return 0


def make_sample(out):
    """OUT/raw from the committed samples (see the module docstring)."""
    d2l = lib()
    raw = os.path.join(out, 'raw')
    for sub in ('nccl', 'perftest'):
        os.makedirs(os.path.join(raw, sub), exist_ok=True)
    sample = lambda name: read(os.path.join(SAMPLES, name))

    def write(rel, text):
        with open(os.path.join(raw, rel), 'w') as f:
            f.write(text)
    write('nodes.txt', 'sample-node-a\nsample-node-b\n')
    for i in range(2):
        write(f'ibstat-{i}.txt', sample('doc_ibstat.txt'))
    write('perftest/mlx5_0-bw.txt', sample('doc_ib_write_bw.txt'))
    write('perftest/mlx5_0-lat.txt', sample('doc_ib_write_lat.txt'))
    perf, info = (sample('box_all_reduce_perf_g4.txt'),
                  sample('box_nccl_info_g4.txt'))
    for algo, proto in d2l._HW_COLL[CONFIG]['pairs']:
        s = stem(algo, proto)
        for run in ['check'] + [f'run{i}' for i in range(REPEATS)]:
            write(f'nccl/{s}-{run}.txt', perf)
            write(f'nccl/{s}-{run}.rc', '0\n')
        write(f'nccl/{s}-check.info', info)
    write('nccl/onenic-run0.txt', perf)
    write('nccl/onenic-run0.rc', '0\n')
    print(f'{raw}: sample logs (box nccl-tests on 4 GPUs for every pair; '
          'hand-written perftest and ibstat)')
    return 0


def verify(tar):
    """Read tar as hw_campaign_log('cluster-nccl') will and show the fits."""
    d2l = lib()
    key = d2l.HW_CAMPAIGNS[NAME][0]
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, os.path.basename(tar))
        shutil.copy(tar, path)
        with mock.patch.dict(d2l.DATA_HUB, {key: ('file://' + path, None)}), \
                mock.patch.object(d2l, 'download_keyed', return_value=path):
            log = d2l.hw_campaign_log(NAME)
        rails, pairs = d2l.fit_perftest(log), d2l.fit_pairs(log)
        d2l.show_fits(rails)
        d2l.show_pair_fits(pairs)
    return 0 if len(rails) and len(pairs) else 1


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('out', nargs='?')
    p.add_argument('--plan', action='store_true')
    p.add_argument('--verdict', nargs=3, metavar=('OUT', 'ALGO', 'PROTO'))
    p.add_argument('--make-sample', metavar='OUT')
    p.add_argument('--verify', metavar='TAR')
    p.add_argument('--sample', action='store_true')
    p.add_argument('--replace', action='store_true')
    a = p.parse_args()
    if a.plan:
        return plan(lib())
    if a.verdict:
        out, algo, proto = a.verdict
        reason = verdict(lib(), os.path.join(out, 'raw'), algo, proto)[0]
        print(reason or 'kept')
        return 1 if reason else 0
    if a.make_sample:
        return make_sample(a.make_sample)
    if a.verify:
        return verify(a.verify)
    if not a.out:
        p.error('give OUT, or one of --plan, --verdict, --make-sample, '
                '--verify')
    return pack(lib(), a.out, a.replace, a.sample)


if __name__ == '__main__':
    sys.exit(main())
