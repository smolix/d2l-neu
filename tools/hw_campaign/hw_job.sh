#!/usr/bin/env bash
# One task of campaign C-T4 (chapter 22, two 8 x A100 nodes with RDMA), started by
# tools/hw_campaign/cluster_t4.sbatch through srun on a compute node:
#
#   hw_job.sh ibstat DIR                       ibstat of this node -> DIR/ibstat-$SLURM_NODEID.txt
#   hw_job.sh perftest-server KIND HCA         ib_write_KIND server on rail HCA (KIND bw or lat)
#   hw_job.sh perftest-client KIND HCA SERVER  its client; the result table on stdout
#   hw_job.sh nccl MODE ALGO PROTO INFO_DIR [all_reduce_perf arguments ...]
#       one rank of nccl-tests' all_reduce_perf (-g 1: one GPU per rank; nccl-tests picks the
#       GPU by local rank). ALGO PROTO 'default default' leaves NCCL's choice; otherwise
#       NCCL_ALGO and NCCL_PROTO are set. MODE check adds NCCL_DEBUG=INFO written to
#       INFO_DIR/nccl.<host>.<pid>.log (not into the table); MODE run times without it.
#
# Environment: D2L_HW_BIN (the nccl-tests binaries; default <repo>/data/hw-tools/bin, built by
# tools/build_hw_tools.sh with NCCL_TESTS_MPI_HOME set, so all_reduce_perf runs under MPI),
# D2L_HW_GID_INDEX (RoCE: perftest's -x), D2L_HW_PERF_PORT (perftest's TCP port, default 18515),
# NCCL_IB_HCA and the cluster's own NCCL variables pass through.
set -u
REPO=${D2L_REPO:-$(cd "$(dirname "$0")/../.." && pwd)}
BIN=${D2L_HW_BIN:-$REPO/data/hw-tools/bin}
usage() { sed -n '2,13p' "$0" >&2; exit 2; }
[ $# -ge 1 ] || usage
cmd=$1; shift
perf_args() {  # KIND HCA: the perftest arguments the parser expects
  local args=(-d "$2" -a -F -p "${D2L_HW_PERF_PORT:-18515}")
  [ "$1" = bw ] && args+=(--report_gbits)
  [ -n "${D2L_HW_GID_INDEX:-}" ] && args+=(-x "$D2L_HW_GID_INDEX")
  echo "${args[@]}"
}
case "$cmd" in
  ibstat)
    [ $# -eq 1 ] || usage
    ibstat > "$1/ibstat-${SLURM_NODEID:-0}.txt" ;;
  perftest-server)
    [ $# -eq 2 ] || usage
    # shellcheck disable=SC2046
    exec "ib_write_$1" $(perf_args "$1" "$2") ;;
  perftest-client)
    [ $# -eq 3 ] || usage
    # shellcheck disable=SC2046
    exec "ib_write_$1" $(perf_args "$1" "$2") "$3" ;;
  nccl)
    [ $# -ge 4 ] || usage
    mode=$1 algo=$2 proto=$3 info=$4; shift 4
    [ -f "$BIN/nccl-env.sh" ] && . "$BIN/nccl-env.sh"
    if [ "$algo" != default ]; then export NCCL_ALGO=$algo NCCL_PROTO=$proto; fi
    if [ "$mode" = check ]; then
      mkdir -p "$info"
      export NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT,ENV,GRAPH,TUNING \
             NCCL_DEBUG_FILE="$info/nccl.%h.%p.log"
    elif [ "$mode" != run ]; then
      usage
    fi
    exec "$BIN/all_reduce_perf" -g 1 "$@" ;;
  *) usage ;;
esac
