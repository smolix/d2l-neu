#!/bin/bash
#SBATCH --job-name=d2l-hw-c-t4
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=8
#SBATCH --gpus-per-node=8
#SBATCH --exclusive
#SBATCH --time=02:00:00
#SBATCH --output=d2l-hw-c-t4-%j.log
# Campaign C-T4 of chapter 22 (Hardware for Training and Inference): one allocation of two
# 8 x A100 nodes with RDMA, shared by sections 22.6 (networks) and 22.7 (collectives).
# Packaged, not run: it needs the author's go (tools/lm_campaigns.yaml, entry 22-hw-c-t4).
#
#   cd <repo> && sbatch tools/hw_campaign/cluster_t4.sbatch OUT
#   bash tools/hw_campaign/cluster_t4.sbatch --dry-run OUT     (prints every step; no Slurm)
#
# Steps, raw outputs under OUT/raw (the layout tools/hw_campaign/pack_t4.py reads):
#   1. node names (nodes.txt; scrubbed from the tar) and ibstat on each node;
#   2. perftest per rail pair: ib_write_bw and ib_write_lat from node 1's rail i to node 0's
#      rail i (perftest/<hca>-bw.txt, -lat.txt);
#   3. nccl-tests all_reduce_perf on 16 ranks (srun, one rank per GPU) for NCCL's default and
#      every NCCL_ALGO x NCCL_PROTO pair of d2l._HW_COLL['t4'] (pack_t4.py --plan): a check
#      run per pair (-c 1, one iteration, NCCL_DEBUG=INFO to files), and when pack_t4.py
#      --verdict keeps the pair, REPEATS timed runs (-c 0); each run's exit status in .rc
#      (124 after its timeout);
#   4. the default pair once more with NCCL_IB_HCA set to the first rail (the one-NIC run);
#   5. pack_t4.py OUT: lm-hw-cluster-nccl-v1.tar, with its size, sha1 and DATA_HUB line.
# Not packaged: the hand-written hierarchical all-reduce across nodes (it runs on one node on
# the box) and an NVLS/SHARP fabric test beyond the forced pairs (a pair the fabric lacks is
# dropped with its reason).
#
# Environment: D2L_REPO (default: the submit directory), D2L_HW_SRUN_MPI (srun --mpi=, default
# pmix), D2L_HW_GID_INDEX (RoCE), D2L_HW_NODES (dry run only: two names, default "node-a
# node-b"); see tools/hw_campaign/hw_job.sh for the tools' locations.
set -u
DRY=
if [ "${1:-}" = --dry-run ]; then DRY=1; shift; fi
OUT=${1:?usage: cluster_t4.sbatch [--dry-run] OUT}
if [ -n "$DRY" ]; then
  REPO=${D2L_REPO:-$(cd "$(dirname "$0")/../.." && pwd)}
else
  REPO=${D2L_REPO:-${SLURM_SUBMIT_DIR:?run under sbatch from the repository, or set D2L_REPO}}
fi
export D2L_REPO=$REPO
PY=$REPO/.venv-lm/bin/python
JOB="bash $REPO/tools/hw_campaign/hw_job.sh"
PACK="$PY $REPO/tools/hw_campaign/pack_t4.py"
MPI=--mpi=${D2L_HW_SRUN_MPI:-pmix}
eval "$($PACK --plan)" || { echo "pack_t4.py --plan failed" >&2; exit 2; }
RAW=$OUT/raw
mkdir -p "$RAW/nccl" "$RAW/perftest"
ts() { date -u +%FT%TZ; }
say() { echo "$(ts) $*"; }

step() {  # step NAME TIMEOUT_S CMD...: stdout to NAME.txt, stderr to NAME.err, status to NAME.rc
  local name=$1 t=$2; shift 2
  if [ -n "$DRY" ]; then echo "+ $* > $name.txt"; echo 0 > "$name.rc"; return 0; fi
  timeout --signal=TERM --kill-after=30 "$t" "$@" > "$name.txt" 2> "$name.err"
  local rc=$?
  echo "$rc" > "$name.rc"
  return "$rc"
}

if [ -n "$DRY" ]; then
  read -r N0 N1 <<< "${D2L_HW_NODES:-node-a node-b}"
  HCAS=mlx5_0
else
  mapfile -t NODES < <(scontrol show hostnames "$SLURM_JOB_NODELIST")
  N0=${NODES[0]} N1=${NODES[1]}
  HCAS=$(srun -N1 -n1 -w "$N0" ibstat -l | tr '\n' ' ')
fi
printf '%s\n' "$N0" "$N1" > "$RAW/nodes.txt"
say "C-T4 on $N0 and $N1; $G ranks; rails: $HCAS; out $OUT"

# 1. ibstat on each node
step "$RAW/ibstat-step" 120 srun -N2 --ntasks-per-node=1 $JOB ibstat "$RAW"

# 2. perftest per rail pair (node 1 -> node 0)
for hca in $HCAS; do
  for kind in bw lat; do
    if [ -n "$DRY" ]; then
      echo "+ srun -N1 -n1 -w $N0 $JOB perftest-server $kind $hca &"
      step "$RAW/perftest/$hca-$kind" 300 srun -N1 -n1 -w "$N1" $JOB perftest-client $kind "$hca" "$N0"
      continue
    fi
    srun -N1 -n1 -w "$N0" --overlap $JOB perftest-server $kind "$hca" > "$RAW/perftest/$hca-$kind.server" 2>&1 &
    server=$!
    sleep 5
    step "$RAW/perftest/$hca-$kind" 300 srun -N1 -n1 -w "$N1" --overlap $JOB perftest-client $kind "$hca" "$N0"
    kill "$server" 2>/dev/null; wait "$server" 2>/dev/null
  done
done

# 3. nccl-tests per pair: check, then the timed repeats of a kept pair
SIZES=(-b "$MIN_BYTES" -e "$MAX_BYTES" -f 2)
RANKS=(srun -N2 --ntasks-per-node=$((G / 2)) "$MPI")
for pair in $PAIRS; do
  algo=${pair%%:*} proto=${pair##*:}
  stem=$([ "$algo" = default ] && echo default || echo "$algo-$proto")
  base=$RAW/nccl/$stem
  step "$base-check" "$TIMEOUT_S" "${RANKS[@]}" $JOB nccl check "$algo" "$proto" "$base-check.d" \
    "${SIZES[@]}" -w 1 -n 1 -c 1
  [ -n "$DRY" ] || cat "$base-check.d"/nccl.*.log > "$base-check.info" 2>/dev/null
  rm -rf "$base-check.d"
  if [ -z "$DRY" ] && ! verdict=$($PACK --verdict "$OUT" "$algo" "$proto"); then
    say "$algo/$proto dropped: $verdict"
    continue
  fi
  for ((i = 0; i < REPEATS; i++)); do
    step "$base-run$i" "$TIMEOUT_S" "${RANKS[@]}" $JOB nccl run "$algo" "$proto" - \
      "${SIZES[@]}" -w "$WARMUP" -n "$ITERS" -c 0 || { say "$algo/$proto run $i failed"; break; }
  done
done

# 4. one rail only
first=${HCAS%% *}
step "$RAW/nccl/onenic-run0" "$TIMEOUT_S" env NCCL_IB_HCA="$first" "${RANKS[@]}" $JOB nccl run \
  default default - "${SIZES[@]}" -w "$WARMUP" -n "$ITERS" -c 0

# 5. pack
if [ -n "$DRY" ]; then
  echo "+ $PACK $OUT"
else
  $PACK "$OUT"
fi
say "C-T4 done"
