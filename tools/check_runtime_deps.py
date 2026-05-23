#!/usr/bin/env python3
"""Validate native runtime dependencies for framework venvs.

This catches shared-library dependencies that Python package metadata cannot
express. Today that mainly means the custom MXNet wheel, which links against
system OpenCV 4.6 in addition to the CUDA/NCCL/cuDNN libraries installed in
the venv.
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("framework", choices=["mxnet"])
    args = parser.parse_args()

    if args.framework == "mxnet":
        check_mxnet()


if __name__ == "__main__":
    main()
