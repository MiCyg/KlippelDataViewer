from __future__ import annotations

import argparse
from pathlib import Path

from animate_membrane import MembraneAnimator
from sce_parser import SceDataContainer


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Test app for Klippel .sce parsing + visualization.")
    p.add_argument("--sce", type=Path, required=True, help="Path to .sce file")
    g = p.add_mutually_exclusive_group(required=False)
    g.add_argument("--freq", type=float, help="Target frequency in Hz (nearest match by default)")
    g.add_argument("--j", type=int, help="Frequency index j (1-based)")
    p.add_argument("--exact", action="store_true", help="Require exact frequency match (no nearest)")
    p.add_argument("--mean-only", action="store_true", help="Only print AAL table head")
    p.add_argument(
        "--silent-db",
        type=float,
        default=SceDataContainer.SILENT_LEVEL_DB,
        help="Silent level threshold in dB (excluded from means)",
    )
    p.add_argument("--fps", type=float, default=30.0, help="Animation FPS")
    p.add_argument("--cycles", type=float, default=1.0, help="How many cycles per animation loop")
    p.add_argument("--scale", type=float, default=1.0, help="Displacement scale factor (visual only)")
    return p


def main() -> None:
    args = build_parser().parse_args()

    animator = MembraneAnimator(
        sce_path=args.sce,
        freq=args.freq,
        j=args.j,
        exact=args.exact,
        silent_db=args.silent_db,
        fps=args.fps,
        cycles=args.cycles,
        scale=args.scale,
        aal_only=args.mean_only,
    )
    animator.run()


if __name__ == "__main__":
    main()
