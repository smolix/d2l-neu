#!/usr/bin/env python3
"""Validate (and where possible repair) native runtime deps for framework venvs.

Two checks today:
  - mxnet: the custom MXNet wheel links against system OpenCV 4.6, which the
    Python package metadata cannot express. This branch reports missing libs
    and points at the right `apt-get` packages.
  - tensorflow: TF 2.21's `libtensorflow_framework.so.2` has a packaging bug —
    its RUNPATH lists nvidia/cublas/lib, nvidia/cudnn/lib, etc. but NOT
    nvidia/cusolver/lib. The cusolver wheel is installed, so TF can't dlopen
    libcusolver.so.11 and silently falls back to CPU. We symlink the cusolver
    libs into nvidia/cusparse/lib/ (which IS on the runpath). The symlinks are
    re-created on every venv sync because `uv sync` wipes them.
"""
import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent


UBUNTU_PACKAGES = {
    "libopencv_core.so.406": "libopencv-core406t64",
    "libopencv_imgproc.so.406": "libopencv-imgproc406t64",
    "libopencv_imgcodecs.so.406": "libopencv-imgcodecs406t64",
}


def nvidia_lib_path(venv_root):
    dirs = sorted(
        str(p)
        for p in (venv_root / "lib").glob("python*/site-packages/nvidia/*/lib")
        if p.is_dir()
    )
    return ":".join(dirs)


def missing_libraries(binary, ld_library_path):
    env = os.environ.copy()
    if ld_library_path:
        env["LD_LIBRARY_PATH"] = (
            ld_library_path
            + (":" + env["LD_LIBRARY_PATH"] if env.get("LD_LIBRARY_PATH") else "")
        )
    result = subprocess.run(
        ["ldd", str(binary)],
        text=True,
        capture_output=True,
        env=env,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or result.stdout.strip())
    missing = []
    for line in result.stdout.splitlines():
        if "=> not found" in line:
            missing.append(line.split("=>", 1)[0].strip())
    return missing


def check_mxnet():
    venv = ROOT / ".venv-mxnet"
    libmxnet = venv / "lib/python3.12/site-packages/mxnet/libmxnet.so"
    if not libmxnet.is_file():
        sys.exit(f"MXNet runtime check failed: {libmxnet} not found")
    if shutil.which("ldd") is None:
        sys.exit("MXNet runtime check failed: ldd not found")

    missing = missing_libraries(libmxnet, nvidia_lib_path(venv))
    if not missing:
        return

    print("MXNet runtime check failed: missing shared libraries:", file=sys.stderr)
    for lib in missing:
        print(f"  - {lib}", file=sys.stderr)

    packages = [UBUNTU_PACKAGES[lib] for lib in missing if lib in UBUNTU_PACKAGES]
    if packages:
        joined = " ".join(dict.fromkeys(packages))
        print("\nOn Ubuntu 24.04, install:", file=sys.stderr)
        print(f"  sudo apt-get install -y {joined}", file=sys.stderr)
    sys.exit(1)


def fix_tensorflow():
    """Symlink libcusolver.so.{11,Mg.11} from nvidia/cusolver/lib into
    nvidia/cusparse/lib so TF's RUNPATH can find them. Idempotent."""
    venv = ROOT / ".venv-tensorflow"
    site = venv / "lib/python3.12/site-packages/nvidia"
    cusolver_dir = site / "cusolver/lib"
    cusparse_dir = site / "cusparse/lib"

    if not cusolver_dir.is_dir():
        sys.exit(
            f"TensorFlow runtime fix: {cusolver_dir} not found "
            "(nvidia-cusolver-cu12 wheel missing?)"
        )
    if not cusparse_dir.is_dir():
        sys.exit(
            f"TensorFlow runtime fix: {cusparse_dir} not found "
            "(nvidia-cusparse-cu12 wheel missing?)"
        )

    for soname in ("libcusolver.so.11", "libcusolverMg.so.11"):
        src = cusolver_dir / soname
        dst = cusparse_dir / soname
        if not src.is_file():
            sys.exit(f"TensorFlow runtime fix: source {src} missing")
        if dst.is_symlink() or dst.exists():
            if dst.is_symlink() and dst.resolve() == src.resolve():
                continue
            dst.unlink()
        dst.symlink_to(src)
        print(f"  symlinked {dst} -> {src}")

    # Smoke-test: ask TF for the GPU list. If empty, the symlink fix didn't
    # actually unblock dlopen — bail loudly so the failure isn't silent.
    py = venv / "bin/python"
    if not py.is_file():
        return
    result = subprocess.run(
        [
            str(py),
            "-c",
            "import os; os.environ['TF_CPP_MIN_LOG_LEVEL']='2';"
            "import tensorflow as tf;"
            "gpus = tf.config.list_physical_devices('GPU');"
            "print('GPUS:', len(gpus))",
        ],
        text=True,
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        print(
            "TensorFlow runtime fix: TF import failed:",
            result.stderr.strip(),
            file=sys.stderr,
        )
        sys.exit(1)
    last = result.stdout.strip().splitlines()[-1] if result.stdout.strip() else ""
    if not last.startswith("GPUS:"):
        print("TensorFlow runtime fix: unexpected smoke-test output:", result.stdout, file=sys.stderr)
        sys.exit(1)
    n = int(last.split(":", 1)[1].strip())
    if n == 0:
        print(
            "TensorFlow runtime fix: symlinks placed but TF still reports 0 GPUs.\n"
            "Re-run with TF_CPP_VMODULE='dso_loader=2' to find the next missing lib.",
            file=sys.stderr,
        )
        sys.exit(1)
    print(f"  TF sees {n} GPU(s)")


def check_darwin(framework):
    """macOS arm64 CPU builds carry no CUDA libs and link native code as
    .dylib bundled in the wheel — `ldd`/cusolver checks don't apply. The
    meaningful runtime check here is simply that the module imports."""
    venv = ROOT / f".venv-{framework}"
    py = venv / "bin/python"
    if not py.is_file():
        sys.exit(f"{framework} runtime check failed (darwin): {py} not found")
    result = subprocess.run([str(py), "-c", f"import {framework}"],
                            text=True, capture_output=True, check=False)
    if result.returncode != 0:
        sys.exit(f"{framework} runtime check failed (darwin): "
                 f"{result.stderr.strip()}")
    print(f"  {framework} imports OK (darwin CPU build)")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("framework", choices=["mxnet", "tensorflow"])
    args = parser.parse_args()

    if sys.platform == "darwin":
        check_darwin(args.framework)
        return

    if args.framework == "mxnet":
        check_mxnet()
    elif args.framework == "tensorflow":
        fix_tensorflow()


if __name__ == "__main__":
    main()
