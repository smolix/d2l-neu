#!/bin/bash
# Build the benchmark tools of chapter 22 (Hardware for Training and Inference)
# from source, without root and without touching the system:
#
#   fio            (storage: sequential and random reads, 22.4)
#   nvbandwidth    (host-device and device-device copy bandwidth, 22.4, 22.5)
#   nccl-tests     (all_reduce_perf, all_gather_perf, ..., 22.5-22.7)
#   p2pBandwidthLatencyTest from cuda-samples (the GPU-to-GPU matrix, 22.5)
#   stream         (McCalpin's STREAM 5.10: host memory bandwidth, 22.4)
#
# Usage:  bash tools/build_hw_tools.sh [PREFIX [TOOL ...]]
#   PREFIX defaults to $D2L_HW_PREFIX, else data/hw-tools (relative to the
#   repository root). Binaries land in PREFIX/bin, which d2l.hw_bin() reads
#   (D2L_HW_BIN overrides it); PREFIX/versions.json records what was built
#   (a build of some tools updates only their entries).
#   TOOLs, default all: fio nvbandwidth nccl-tests p2p stream. STREAM alone
#   needs only a C compiler with OpenMP (no CUDA toolkit, no GPU):
#     bash tools/build_hw_tools.sh data/hw-tools stream
# Requires: git, make, a C compiler, cmake >= 3.20, a CUDA toolkit (nvcc;
#   CUDA_HOME, default /usr/local/cuda), Boost program_options headers (for
#   nvbandwidth), and the Python environment whose torch wheel bundles NCCL
#   (PYTHON, default .venv-lm/bin/python): nccl-tests is built against that
#   wheel's nccl.h and libnccl.so.2, the library the book's PyTorch runs.
# GENCODE: the GPU architecture to compile for (default: the first visible
#   GPU's compute capability, e.g. 89 for the RTX 4090).
# NCCL_TESTS_MPI_HOME: build nccl-tests with MPI (MPI=1 MPI_HOME=...), one rank
#   per GPU across nodes under srun or mpirun (chapter 22's two-node campaign,
#   tools/hw_campaign/cluster_t4.sbatch); unset (the default) builds the
#   single-process tests the box uses.
# Each tool is pinned to a release tag and its commit (STREAM, which has no
# repository, to its file's SHA-256); a mismatch stops the build. STREAM is
# compiled with -O3 -fopenmp and changes only the compile-time parameters its
# licence lets a run change: STREAM_ARRAY_SIZE = 4 x (L2 + L3) in doubles,
# rounded up to a million (STREAM_ELEMENTS overrides it), and NTIMES = 20.
# The notebooks run without these tools (they fall back to torch sweeps or
# the committed logs), so a reader needs no build.
set -euo pipefail
REPO=$(cd "$(dirname "$0")/.." && pwd)
PREFIX=${1:-${D2L_HW_PREFIX:-$REPO/data/hw-tools}}
[ $# -gt 0 ] && shift
WANT=" ${*:-fio nvbandwidth nccl-tests p2p stream} "
want() { [[ $WANT == *" $1 "* ]]; }
mkdir -p "$PREFIX"
PREFIX=$(cd "$PREFIX" && pwd)
CUDA_HOME=${CUDA_HOME:-/usr/local/cuda}
PYTHON=${PYTHON:-$REPO/.venv-lm/bin/python}
JOBS=${JOBS:-16}
SRC=$PREFIX/src; BIN=$PREFIX/bin
mkdir -p "$SRC" "$BIN"

# tool tag commit (release tags at least two weeks old on 2026-11-25)
ALL_PINS=(
  "fio https://github.com/axboe/fio.git fio-3.43 d2dcd7e053d1902d2420f151c83990e6acdd9f8c"
  "nvbandwidth https://github.com/NVIDIA/nvbandwidth.git v0.11.0 201eaf7f25eda954d08a7c4fed9d8325e3c32ddf"
  "nccl-tests https://github.com/NVIDIA/nccl-tests.git v2.21.1 afd59abab774a6a4b492eb344bd0c845a0b7bb4d"
  "cuda-samples https://github.com/NVIDIA/cuda-samples.git v13.3 b7c5481c556c3fe98db060207ecaa41a4b9a9abc"
)
# STREAM: no repository and no release tag; its file, revision and SHA-256
STREAM_URL=https://www.cs.virginia.edu/stream/FTP/Code/stream.c
STREAM_REV=5.10
STREAM_SHA=a52bae5e175bea3f7832112af9c085adab47117f7d2ce219165379849231692b

PINS=()  # the pinned repositories this build needs
for p in "${ALL_PINS[@]}"; do
  name=${p%% *}
  [ "$name" = cuda-samples ] && name=p2p
  if want "$name"; then PINS+=("$p"); fi
done
BUILT=()  # name url tag commit of each tool built, for versions.json

NVCC_GENCODE=; NVCC_VERSION=; NCCL_VERSION=
if want nvbandwidth || want nccl-tests || want p2p; then
  if [ -z "${GENCODE:-}" ]; then
    GENCODE=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | tr -d '.')
  fi
  NVCC_GENCODE="-gencode=arch=compute_${GENCODE},code=sm_${GENCODE}"
  NVCC_VERSION=$("$CUDA_HOME/bin/nvcc" --version | sed -n 's/.*release \([0-9.]*\).*/\1/p')
fi

fetch() {  # name url tag commit
  local dir=$SRC/$1
  if [ ! -d "$dir/.git" ]; then
    git -c advice.detachedHead=false clone --quiet --depth 1 --branch "$3" "$2" "$dir"
  fi
  local got
  got=$(git -C "$dir" rev-parse HEAD)
  [ "$got" = "$4" ] || { echo "$1: $3 is commit $got, pinned $4" >&2; exit 1; }
}
for p in "${PINS[@]}"; do fetch $p; BUILT+=("$p"); done

if want fio; then
  echo "== fio"
  ( cd "$SRC/fio" && ./configure --prefix="$PREFIX" --disable-native > "$PREFIX/fio-configure.log" \
    && make -j"$JOBS" > "$PREFIX/fio-make.log" 2>&1 )
  cp "$SRC/fio/fio" "$BIN/fio"
fi

if want nvbandwidth; then
  echo "== nvbandwidth"
  # Logical endpoints (the multi-node NVLink fabric API) are switched off: a
  # toolkit newer than the driver declares cuLogicalEndpoint* symbols that the
  # driver's libcuda does not export (CUDA 13.3 headers, a 13.2 driver), and no
  # PCIe box or single node uses them.
  rm -rf "$SRC/nvbandwidth/build"
  ( cd "$SRC/nvbandwidth" && cmake -S . -B build -DCMAKE_BUILD_TYPE=Release \
      -DNVBW_HAVE_LE_API=0 -DNVBW_HAVE_LE_DEVICE_ATTRS=0 \
      -DCMAKE_CUDA_COMPILER="$CUDA_HOME/bin/nvcc" -DCMAKE_CUDA_ARCHITECTURES="$GENCODE" \
      > "$PREFIX/nvbandwidth-cmake.log" && cmake --build build -j"$JOBS" > "$PREFIX/nvbandwidth-make.log" 2>&1 )
  cp "$SRC/nvbandwidth/build/nvbandwidth" "$BIN/nvbandwidth"
fi

if want nccl-tests; then
  echo "== nccl-tests (against the torch wheel's NCCL)"
  NCCL_PKG=$("$PYTHON" -c 'import nvidia.nccl, os; print(os.path.dirname(nvidia.nccl.__file__) if nvidia.nccl.__file__ else list(nvidia.nccl.__path__)[0])')
  NCCL_HOME=$PREFIX/nccl-wheel
  mkdir -p "$NCCL_HOME/lib"
  ln -sfn "$NCCL_PKG/include" "$NCCL_HOME/include"
  ln -sf "$NCCL_PKG/lib/libnccl.so.2" "$NCCL_HOME/lib/libnccl.so.2"
  ln -sf libnccl.so.2 "$NCCL_HOME/lib/libnccl.so"
  NCCL_VERSION=$("$PYTHON" -c 'import torch; print(".".join(map(str, torch.cuda.nccl.version())))')
  MPI_ARGS=(MPI=0)
  [ -n "${NCCL_TESTS_MPI_HOME:-}" ] && MPI_ARGS=(MPI=1 MPI_HOME="$NCCL_TESTS_MPI_HOME")
  ( cd "$SRC/nccl-tests" && { make clean > /dev/null 2>&1 || true; } && make -j"$JOBS" "${MPI_ARGS[@]}" \
      CUDA_HOME="$CUDA_HOME" NCCL_HOME="$NCCL_HOME" \
      NVCC_GENCODE="$NVCC_GENCODE" > "$PREFIX/nccl-tests-make.log" 2>&1 )
  for t in "$SRC"/nccl-tests/build/*_perf; do
    # the RUNPATH finds the wheel's libnccl.so.2 next to the binaries' link dir
    cp "$t" "$BIN/"
  done
  cat > "$BIN/nccl-env.sh" <<EOF
# source before running the nccl-tests binaries: the torch wheel's NCCL
export LD_LIBRARY_PATH=$NCCL_HOME/lib\${LD_LIBRARY_PATH:+:\$LD_LIBRARY_PATH}
EOF
fi

if want p2p; then
  echo "== p2pBandwidthLatencyTest (cuda-samples)"
  P2P=$SRC/cuda-samples/cpp/5_Domain_Specific/p2pBandwidthLatencyTest  # Samples/ before v13.2
  ( cd "$P2P" && cmake -S . -B build -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_CUDA_COMPILER="$CUDA_HOME/bin/nvcc" -DCMAKE_CUDA_ARCHITECTURES="$GENCODE" \
      > "$PREFIX/p2p-cmake.log" && cmake --build build -j"$JOBS" > "$PREFIX/p2p-make.log" 2>&1 )
  cp "$P2P/build/p2pBandwidthLatencyTest" "$BIN/"
fi

if want stream; then
  echo "== stream"
  mkdir -p "$SRC/stream"
  [ -f "$SRC/stream/stream.c" ] || curl -sSfL -o "$SRC/stream/stream.c" "$STREAM_URL"
  got=$(sha256sum "$SRC/stream/stream.c" | cut -d' ' -f1)
  [ "$got" = "$STREAM_SHA" ] || { echo "stream.c: sha256 $got, pinned $STREAM_SHA" >&2; exit 1; }
  cache_bytes() {  # one cache level's total: "L3 cache: 134217728 (8 instances)"
    lscpu -B | awk -v l="$1 cache:" 'index($0, l) == 1 {print $3}'
  }
  if [ -z "${STREAM_ELEMENTS:-}" ]; then
    STREAM_ELEMENTS=$(( (4 * ($(cache_bytes L2) + $(cache_bytes L3)) / 8 + 999999) / 1000000 * 1000000 ))
  fi
  "${CC:-gcc}" -O3 -fopenmp -mcmodel=medium -DSTREAM_ARRAY_SIZE="$STREAM_ELEMENTS" -DNTIMES=20 \
    "$SRC/stream/stream.c" -o "$BIN/stream" > "$PREFIX/stream-make.log" 2>&1
  BUILT+=("stream $STREAM_URL $STREAM_REV sha256-$STREAM_SHA")
  echo "{\"elements\": $STREAM_ELEMENTS, \"ntimes\": 20, \"cc\": \"$("${CC:-gcc}" --version | head -1)\"}" \
    > "$PREFIX/stream-build.json"
fi

python3 - "$PREFIX/versions.json" "$NVCC_VERSION" "$NVCC_GENCODE" "$NCCL_VERSION" "${BUILT[@]}" <<'EOF'
import json, os, sys
out, nvcc, gencode, nccl, *pins = sys.argv[1:]
doc = json.load(open(out)) if os.path.exists(out) else dict(tools={})
for p in pins:
    name, url, tag, commit = p.split()
    doc['tools'][name] = dict(tag=tag, commit=commit, url=url)
doc.update({k: v for k, v in dict(nvcc=nvcc, gencode=gencode, nccl=nccl).items() if v})
json.dump(doc, open(out, 'w'), indent=1)
EOF
echo "built into $BIN:"; ls "$BIN"
