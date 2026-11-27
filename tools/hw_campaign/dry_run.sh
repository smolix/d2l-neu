#!/usr/bin/env bash
# Dry run of chapter 22's campaign packages (C-T2, C-T2pair, C-T3, C-T4) on the book's box.
# It proves the launchers, not a campaign: nothing it writes is uploaded.
#
#   bash tools/hw_campaign/dry_run.sh OUT [--gpu]
#
# Without --gpu (CPU only): every package's commands from tools/lm_campaigns.yaml; bash -n on
# the shell scripts; tools/hw_regen.py --dry-run for every regeneration of the packages (name
# and configuration checked, the folder and key it would write printed); node_t3.sh and
# cluster_t4.sbatch with --dry-run (OUT/t3, OUT/t4); pack_t4.py on the committed sample logs
# (OUT/t4-sample) and the tar read back as hw_campaign_log('cluster-nccl') will read it.
# With --gpu (one card, under the GPU-lock helper; CUDA_VISIBLE_DEVICES names it): one real
# tools/hw_regen.py box-microbench --out OUT/box --processes 1 (the campaign launcher's --out
# path on the box configuration), then box-microbench --config t2 on this GPU, which must
# refuse (it is not the A100 PCIe 40 GB). Exit 0 when every step passed.
set -u
OUT=${1:?usage: dry_run.sh OUT [--gpu]}
GPU=; [ "${2:-}" = --gpu ] && GPU=1
REPO=$(cd "$(dirname "$0")/../.." && pwd)
PY=${D2L_HW_PYTHON:-$REPO/.venv-lm/bin/python}
export PYTHONPATH=$REPO${PYTHONPATH:+:$PYTHONPATH}
mkdir -p "$OUT"; OUT=$(cd "$OUT" && pwd)
cd "$REPO" || exit 2
failed=0
ts() { date -u +%FT%TZ; }
check() {  # check LABEL CMD...: run, report, count failures
  local label=$1; shift
  echo "== $(ts) $label"
  if "$@"; then echo "ok: $label"; else echo "FAILED ($?): $label"; failed=$((failed + 1)); fi
}

check 'package commands (tools/lm_campaigns.yaml)' "$PY" - <<'EOF'
import yaml
doc = yaml.safe_load(open('tools/lm_campaigns.yaml'))
for c in doc['campaigns']:
    if c['id'].startswith('22-hw-'):
        print(f"{c['id']}: {c['hardware']}\n  {c['command'].strip()}")
EOF
for f in tools/hw_campaign/node_t3.sh tools/hw_campaign/cluster_t4.sbatch \
         tools/hw_campaign/hw_job.sh tools/hw_campaign/dry_run.sh tools/hw_clock_sweep.sh; do
  check "bash -n $f" bash -n "$f"
done
regens=$("$PY" - <<'EOF'
import yaml
for c in yaml.safe_load(open('tools/lm_campaigns.yaml'))['campaigns']:
    for r in c.get('regenerations', []) if c['id'].startswith('22-hw-') else []:
        print(r['name'], r['config'])
EOF
)
while read -r name config; do
  check "hw_regen.py $name --config $config --dry-run" \
    "$PY" tools/hw_regen.py "$name" --config "$config" --out "$OUT/$config" --dry-run
done <<< "$regens"
check 'node_t3.sh --dry-run (pre-checks print, then hw_regen --dry-run)' \
  env D2L_HW_NVME="$OUT/nvme" bash tools/hw_campaign/node_t3.sh "$OUT/t3" --dry-run
check 'cluster_t4.sbatch --dry-run' bash tools/hw_campaign/cluster_t4.sbatch --dry-run "$OUT/t4"
rm -rf "$OUT/t4-sample"
check 'pack_t4.py on the sample logs' bash -c "'$PY' tools/hw_campaign/pack_t4.py \
  --make-sample '$OUT/t4-sample' && '$PY' tools/hw_campaign/pack_t4.py '$OUT/t4-sample' --sample"
check 'the sample tar read as hw_campaign_log(cluster-nccl)' \
  "$PY" tools/hw_campaign/pack_t4.py --verify "$OUT/t4-sample/lm-hw-cluster-nccl-v1-sample.tar"

if [ -n "$GPU" ]; then
  rm -rf "$OUT/box" "$OUT/t2-refusal"
  check 'hw_regen.py box-microbench --out OUT/box --processes 1 (real, one card)' \
    "$PY" tools/hw_regen.py box-microbench --out "$OUT/box" --processes 1
  echo "== $(ts) hw_regen.py box-microbench --config t2 on $(nvidia-smi --query-gpu=name \
    --format=csv,noheader -i "${CUDA_VISIBLE_DEVICES%%,*}" 2>/dev/null): must refuse"
  "$PY" tools/hw_regen.py box-microbench --config t2 --out "$OUT/t2-refusal" --processes 1 \
    > "$OUT/t2-refusal.log" 2>&1
  rc=$?
  if [ "$rc" != 0 ] && grep -q 'measures the A100 PCIe 40 GB' "$OUT/t2-refusal.log"; then
    echo "ok: refused (exit $rc): $(grep -o 'config t2 measures.*' "$OUT/t2-refusal.log" | head -n 1)"
  else
    echo "FAILED: --config t2 did not refuse (exit $rc)"; failed=$((failed + 1))
  fi
fi
echo "== $(ts) dry run: $failed step(s) failed"
[ "$failed" = 0 ]
