#!/usr/bin/env bash
# Chapter 22's clock sweep (regeneration box-clocksweep, section 22.8) on one card.
#
#   bash tools/hw_clock_sweep.sh <gpu> [--processes N] [--config C] [--out DIR]
#
# <gpu> is the nvidia-smi index of the card (PCI bus order). Defaults: --processes 3 --config box.
# Run it under the GPU-lock helper that holds this card's lock (it waits while another user's
# process is on the card); the script itself refuses to start while any compute process is on it.
#
# Ruling H1 (2026-11-25): clocks may be locked with `sudo -n nvidia-smi -lgc`, only inside a job
# that holds the card's lock with no other user's process on it, and always reset with -rgc on
# exit. So: the job runs as the user; only nvidia-smi goes through `sudo -n` (the library adds it
# when D2L_HW_SUDO=1, which only this script sets); the Python job runs in the background, so the
# traps below run at once on INT, TERM or HUP: reset the card, pass the signal on, wait, exit 143.
# The EXIT trap resets again and records the card's clocks (and power, with the cap arm) in the
# log; it also covers a `sudo nvidia-smi -lgc` still in flight when the job stopped. A kill -9 of
# this script skips its traps, and nothing here or in the GPU-lock helper resets the card then:
# whoever ran it issues one `sudo -n nvidia-smi -rgc -i <uuid>` by hand once the job's PID is gone.
#
# D2L_HW_GPU_LOCK_DIR=<dir> (the GPU-lock helper's lock folder): refuse to start unless
# <dir>/gpu<gpu>.lock is held (by the helper that runs this script), as ruling H1 requires.
#
# The power-cap arm (nvidia-smi -pl) runs only when the configuration's caps are not empty (none
# is, until the author allows -pl); then the default limit is read first and restored by every
# reset.
#
# D2L_HW_DRYRUN=1 prints every sudo command instead of running it (here and in the library).
# Test hooks: D2L_HW_PYTHON (the interpreter), D2L_HW_NVIDIA_SMI (nvidia-smi for the queries).
set -u

usage() { echo "usage: hw_clock_sweep.sh <gpu> [--processes N] [--config C] [--out DIR]" >&2; exit 2; }
[ $# -ge 1 ] || usage
GPU=$1; shift
case "$GPU" in ''|*[!0-9]*) usage ;; esac
PROCESSES=3; CONFIG=box; OUT=
while [ $# -gt 0 ]; do
  case "$1" in
    --processes) [ $# -ge 2 ] || usage; PROCESSES=$2; shift 2 ;;
    --config) [ $# -ge 2 ] || usage; CONFIG=$2; shift 2 ;;
    --out) [ $# -ge 2 ] || usage; OUT=$2; shift 2 ;;
    *) usage ;;
  esac
done

REPO=$(cd "$(dirname "$0")/.." && pwd)
PY=${D2L_HW_PYTHON:-$REPO/.venv-lm/bin/python}
SMI=${D2L_HW_NVIDIA_SMI:-nvidia-smi}
DRY=${D2L_HW_DRYRUN:-}
ts() { date -u +%FT%TZ; }
priv() {  # a privileged nvidia-smi call: printed in a dry run, else through sudo -n
  if [ "$DRY" = 1 ]; then echo "+ sudo -n nvidia-smi $*"; else sudo -n nvidia-smi "$@"; fi
}

UUID=$("$SMI" --query-gpu=uuid --format=csv,noheader -i "$GPU" 2>/dev/null | head -n 1)
[ -n "$UUID" ] || { echo "$(ts) no GPU $GPU" >&2; exit 2; }
if [ -n "${D2L_HW_GPU_LOCK_DIR:-}" ]; then
  LOCK=$D2L_HW_GPU_LOCK_DIR/gpu$GPU.lock
  if [ ! -e "$LOCK" ] || flock -n "$LOCK" true; then
    echo "$(ts) $LOCK is not held: run this under the GPU-lock helper; refused" >&2; exit 3
  fi
fi
if [ -n "${CUDA_VISIBLE_DEVICES:-}" ] && [ "$CUDA_VISIBLE_DEVICES" != "$GPU" ] \
   && [ "$CUDA_VISIBLE_DEVICES" != "$UUID" ]; then
  echo "$(ts) CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES is not GPU $GPU: refused" >&2; exit 2
fi
APPS=$("$SMI" --query-compute-apps=gpu_uuid,pid --format=csv,noheader 2>/dev/null) \
  || { echo "$(ts) cannot list the compute processes: refused" >&2; exit 3; }
BUSY=$(printf '%s\n' "$APPS" | grep -c "^$UUID")
if [ "$BUSY" != 0 ]; then
  echo "$(ts) $BUSY compute process(es) on GPU $GPU ($UUID): refused" >&2; exit 3
fi
CAPS=$("$PY" - "$CONFIG" <<'EOF'
import sys
from d2l import torch as d2l
print(len(d2l._HW_CLOCK_CONFIGS[sys.argv[1]]['caps']))
EOF
) || { echo "$(ts) no clock-sweep configuration $CONFIG" >&2; exit 2; }
DEFAULT_PL=
if [ "$CAPS" != 0 ]; then
  DEFAULT_PL=$("$SMI" --query-gpu=power.default_limit --format=csv,noheader,nounits -i "$GPU" | head -n 1)
  [ -n "$DEFAULT_PL" ] || { echo "$(ts) default power limit unknown: refused" >&2; exit 2; }
fi

reset() {
  priv -rgc -i "$UUID"
  [ -z "$DEFAULT_PL" ] || priv -pl "$DEFAULT_PL" -i "$UUID"
}
child=
on_exit() {
  reset
  echo "$(ts) card $GPU after the job:"
  "$SMI" -q -d CLOCK -i "$UUID"
  [ -z "$DEFAULT_PL" ] || "$SMI" -q -d POWER -i "$UUID"
}
on_signal() {
  echo "$(ts) signal: resetting GPU $GPU, stopping the job"
  reset
  if [ -n "$child" ]; then kill -TERM "$child" 2>/dev/null; wait "$child"; fi
  exit 143
}
trap on_exit EXIT
trap on_signal INT TERM HUP

export D2L_HW_SUDO=1 CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES="$UUID"
ARGS=(tools/hw_regen.py box-clocksweep --processes "$PROCESSES" --config "$CONFIG")
[ -z "$OUT" ] || ARGS+=(--out "$OUT")
echo "$(ts) GPU $GPU ($UUID), caps arm: $([ "$CAPS" = 0 ] && echo off || echo "on, default $DEFAULT_PL W")"
echo "$(ts) running: $PY ${ARGS[*]}"
cd "$REPO" || exit 2
"$PY" "${ARGS[@]}" &
child=$!
wait "$child"
rc=$?
child=
echo "$(ts) job exit $rc"
exit "$rc"
