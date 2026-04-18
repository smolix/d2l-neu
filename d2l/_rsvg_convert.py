"""Minimal rsvg-convert shim backed by CairoSVG.

Installed as the `rsvg-convert` console script via pyproject.toml. Provides
just the flags that Quarto's main.lua and build.sh use (-f, -o, stdin/stdout,
--version). Remaining rsvg-convert flags are accepted and ignored so callers
that pass width/height/dpi do not error out.
"""
import argparse
import sys

import cairosvg

_CONVERT = {
    "pdf": cairosvg.svg2pdf,
    "png": cairosvg.svg2png,
    "ps":  cairosvg.svg2ps,
    "svg": cairosvg.svg2svg,
}


def main() -> int:
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("-f", "--format", default="png")
    p.add_argument("-o", "--output", default=None)
    p.add_argument("-v", "--version", action="store_true")
    p.add_argument("-w", "--width", type=float, default=None)
    p.add_argument("-h", "--height", type=float, default=None)
    p.add_argument("-d", "--dpi-x", type=float, default=None)
    p.add_argument("-p", "--dpi-y", type=float, default=None)
    p.add_argument("-z", "--zoom", type=float, default=None)
    p.add_argument("-x", "--x-zoom", type=float, default=None)
    p.add_argument("-y", "--y-zoom", type=float, default=None)
    p.add_argument("-a", "--keep-aspect-ratio", action="store_true")
    p.add_argument("-b", "--background-color", default=None)
    p.add_argument("-u", "--keep-image-data", action="store_true")
    p.add_argument("--no-keep-image-data", action="store_true")
    p.add_argument("--stylesheet", default=None)
    p.add_argument("--help", action="help")
    p.add_argument("input", nargs="?")
    args = p.parse_args()

    if args.version:
        print("rsvg-convert shim 0.1 (cairosvg backend)")
        return 0

    fmt = args.format.lower()
    convert = _CONVERT.get(fmt)
    if convert is None:
        print(f"rsvg-convert shim: unsupported format {fmt!r}", file=sys.stderr)
        return 2

    kwargs = {}
    if args.input and args.input != "-":
        kwargs["url"] = args.input
    else:
        kwargs["bytestring"] = sys.stdin.buffer.read()

    if args.output and args.output != "-":
        kwargs["write_to"] = args.output
        convert(**kwargs)
    else:
        data = convert(**kwargs)
        sys.stdout.buffer.write(data)
    return 0


if __name__ == "__main__":
    sys.exit(main())
