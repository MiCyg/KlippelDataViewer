from __future__ import annotations

import argparse
from pathlib import Path

from animate_membrane import main as animate_main
from sce_parser import mean_amplitude_by_frequency


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Test app for Klippel .sce parsing + visualization.")
    p.add_argument("--sce", type=Path, required=True, help="Path to .sce file")
    g = p.add_mutually_exclusive_group(required=False)
    g.add_argument("--freq", type=float, help="Target frequency in Hz (nearest match by default)")
    g.add_argument("--j", type=int, help="Frequency index j (1-based)")
    p.add_argument("--exact", action="store_true", help="Require exact frequency match (no nearest)")
    p.add_argument("--mean-only", action="store_true", help="Only print mean amplitude table head")
    p.add_argument("--smooth", type=float, default=0.0, help="Spatial smoothing strength in [0..1]")
    p.add_argument("--smooth-iters", type=int, default=0, help="Number of smoothing iterations")
    p.add_argument("--smooth-method", choices=["mean", "median", "trimmed", "gaussian"], default="mean", help="Smoothing method")
    p.add_argument("--smooth-trim", type=float, default=0.1, help="Trim fraction for trimmed smoothing")
    p.add_argument("--smooth-sigma", type=float, default=5.0, help="Sigma [mm] for gaussian smoothing")
    return p


def main() -> None:
    args = build_parser().parse_args()

    if args.mean_only:
        df = mean_amplitude_by_frequency(args.sce)
        print(df.head(10).to_string(index=False))
        return

    # Delegate to the animation CLI for consistency.
    # We re-run its argument parser by calling it as a script-style entrypoint.
    import sys

    cmd = ["animate_membrane.py", str(args.sce)]
    if args.j is not None:
        cmd += ["--j", str(args.j)]
    else:
        cmd += ["--freq", str(args.freq if args.freq is not None else 100.0)]
    if args.exact:
        cmd += ["--exact"]
    if hasattr(args, "smooth") and args.smooth and args.smooth != 0.0:
        cmd += ["--smooth", str(args.smooth)]
    if hasattr(args, "smooth_iters") and args.smooth_iters and args.smooth_iters != 0:
        cmd += ["--smooth-iters", str(args.smooth_iters)]
    if hasattr(args, "smooth_method") and args.smooth_method and args.smooth_method != "mean":
        cmd += ["--smooth-method", str(args.smooth_method)]
    if hasattr(args, "smooth_trim") and args.smooth_trim is not None and float(args.smooth_trim) != 0.1:
        cmd += ["--smooth-trim", str(args.smooth_trim)]
    if hasattr(args, "smooth_sigma") and args.smooth_sigma is not None and float(args.smooth_sigma) != 5.0:
        cmd += ["--smooth-sigma", str(args.smooth_sigma)]

    sys.argv = cmd
    animate_main()


if __name__ == "__main__":
    main()
