#!/usr/bin/env bash
# Campaign C-T3 of chapter 22 (Hardware for Training and Inference) on one 8 x A100 40 GB node:
# sections 22.4 (memory and storage tiers) and 22.5 (data paths inside a server).
# Packaged, not run: it needs the author's go (tools/lm_campaigns.yaml, entry 22-hw-c-t3).
#
#   bash tools/hw_campaign/node_t3.sh OUT [--dry-run]
#
# D2L_HW_NVME (required): a folder on the node's local NVMe drive; fio's 16 GiB file and
#   gdsio's files go there (data/hw-scratch is linked to $D2L_HW_NVME/hw-scratch).
# D2L_HW_SHARED_FS (optional): a folder on the cluster's shared file system for one fio read;
#   without it that run is recorded as skipped.
# Pre-checks (any failure stops before a measurement): exactly 8 GPUs, all A100 40 GB, no
# compute process on them; 60 GB free on D2L_HW_NVME and 2 GB in OUT; the tools (fio,
# nvbandwidth, all_reduce_perf, p2pBandwidthLatencyTest from tools/build_hw_tools.sh, numactl);
# gdsio is optional (GPUDirect Storage; without it the GDS runs are recorded as skipped).
# Then, with every GPU of the node:
#   tools/hw_regen.py box-storage --config t3 --out OUT   -> OUT/lm-hw-node-storage-v1.tar
#   tools/hw_regen.py box-p2p --config t3 --out OUT       -> OUT/lm-hw-node-p2p-v1.tar
# and a summary of the tars (size, sha1). --dry-run prints the checks and runs hw_regen.py
# --dry-run only. Exit: 0 done, 2 usage, 3 a pre-check failed, else the failing run's status.
set -u
OUT=${1:-}; DRY=
[ "${2:-}" = --dry-run ] && DRY=1
[ -n "$OUT" ] && [ "$OUT" != --dry-run ] || { sed -n '2,20p' "$0" >&2; exit 2; }
REPO=$(cd "$(dirname "$0")/../.." && pwd)
PY=$REPO/.venv-lm/bin/python
BIN=${D2L_HW_BIN:-$REPO/data/hw-tools/bin}
ts() { date -u +%FT%TZ; }
fail() { echo "$(ts) pre-check: $*" >&2; [ -n "$DRY" ] || exit 3; }
mkdir -p "$OUT"
OUT=$(cd "$OUT" && pwd)

gpus=$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null)
total=$(grep -c . <<< "$gpus") n=$(grep -c 'A100.*40[0-9][0-9][0-9] MiB' <<< "$gpus")
[ "$total" -eq 8 ] && [ "$n" -eq 8 ] \
  || fail "$total GPU(s), $n of them A100 40 GB; the node campaign needs exactly 8 A100 40 GB"
busy=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -c .)
[ "$busy" = 0 ] || fail "$busy compute process(es) on the GPUs"
[ -n "${D2L_HW_NVME:-}" ] && [ -d "$D2L_HW_NVME" ] || fail 'D2L_HW_NVME is not a folder'
free_gb() { df -P -B1G "$1" 2>/dev/null | awk 'NR == 2 {print $4}'; }
[ "$(free_gb "${D2L_HW_NVME:-/nonexistent}")" -ge 60 ] 2>/dev/null || fail 'under 60 GB free on D2L_HW_NVME'
[ "$(free_gb "$OUT")" -ge 2 ] 2>/dev/null || fail "under 2 GB free in $OUT"
for t in fio nvbandwidth all_reduce_perf p2pBandwidthLatencyTest; do
  [ -x "$BIN/$t" ] || command -v "$t" > /dev/null || fail "no $t (bash tools/build_hw_tools.sh)"
done
command -v numactl > /dev/null || fail 'no numactl'
[ -x "$BIN/gdsio" ] || command -v gdsio > /dev/null \
  || echo "$(ts) note: no gdsio; the GPUDirect Storage runs are recorded as skipped"
[ -x "$PY" ] || fail "no $PY (make venv-lm)"
SCRATCH=$REPO/data/hw-scratch
if [ -n "${D2L_HW_NVME:-}" ] && [ "$(readlink -f "$SCRATCH")" != "$(readlink -f "$D2L_HW_NVME/hw-scratch")" ]; then
  if [ -e "$SCRATCH" ] && [ ! -L "$SCRATCH" ]; then
    fail "$SCRATCH exists and is not a link to D2L_HW_NVME"
  elif [ -n "$DRY" ]; then
    echo "+ ln -sfn $D2L_HW_NVME/hw-scratch $SCRATCH"
  else
    mkdir -p "$D2L_HW_NVME/hw-scratch" && ln -sfn "$D2L_HW_NVME/hw-scratch" "$SCRATCH"
  fi
fi

cd "$REPO" || exit 2
for name in box-storage box-p2p; do
  echo "$(ts) $name --config t3"
  if [ -n "$DRY" ]; then
    "$PY" tools/hw_regen.py "$name" --config t3 --out "$OUT" --dry-run || exit $?
    continue
  fi
  "$PY" tools/hw_regen.py "$name" --config t3 --out "$OUT" || exit $?
done
[ -n "$DRY" ] && exit 0
for tar in "$OUT"/lm-hw-node-*.tar; do
  echo "$(basename "$tar"): $(du -h "$tar" | cut -f1), sha1 $(sha1sum "$tar" | cut -d' ' -f1)"
done
