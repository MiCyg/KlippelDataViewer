from __future__ import annotations

import argparse
from pathlib import Path

from sce_parser import SceDataContainer
from animate_membrane import MembraneAnimator


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Klippel .sce visualizer"
    )

    p.add_argument(
        "--sce",
        type=Path,
        required=True,
        help="Path to .sce file"
    )

    g = p.add_mutually_exclusive_group(required=False)
    g.add_argument(
        "--j",
        type=int,
        help="Frequency index j (1-based)"
    )

    p.add_argument(
        "--aal-only",
        action="store_true",
        help="Only print AAL table head"
    )

    p.add_argument(
        "--silent-db",
        type=float,
        default=SceDataContainer.SILENT_LEVEL_DB,
        help="Silent level threshold in dB"
    )

    p.add_argument("--fps", type=float, default=30.0)
    p.add_argument("--cycles", type=float, default=1.0)
    p.add_argument("--scale", type=float, default=1.0)

    return p


def main() -> None:
    args = build_parser().parse_args()

    container = SceDataContainer(str(args.sce))

    if args.aal_only:
        print(container.get_aal().to_string(index=False))
        return

    animator = MembraneAnimator(
        sce_path=args.sce,
        j=args.j,
        silent_db=args.silent_db,
        fps=args.fps,
        cycles=args.cycles,
        scale=args.scale,
    )

    animator.run()


if __name__ == "__main__":
    main()
