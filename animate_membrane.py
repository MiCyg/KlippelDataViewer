from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib import gridspec
import matplotlib.tri as mtri
from matplotlib.colors import TwoSlopeNorm

from sce_parser import SceDataContainer


def _build_vertex_neighbors(triangles: np.ndarray, n_vertices: int) -> list[np.ndarray]:
    neigh: list[set[int]] = [set() for _ in range(n_vertices)]
    for a, b, c in triangles:
        a = int(a)
        b = int(b)
        c = int(c)
        neigh[a].update((b, c))
        neigh[b].update((a, c))
        neigh[c].update((a, b))
    return [np.fromiter(s, dtype=int) if s else np.empty(0, dtype=int) for s in neigh]



def _smooth_field(
    values: np.ndarray,
    neighbors: list[np.ndarray],
    *,
    alpha: float,
    iters: int,
    method: str = "mean",
    weights: list[np.ndarray] | None = None,
    trim: float = 0.1,
    valid: np.ndarray | None = None,
) -> np.ndarray:
    if iters <= 0 or alpha <= 0:
        return values
    if not (0.0 <= alpha <= 1.0):
        raise ValueError("smooth alpha must be in [0, 1]")
    if method not in ("mean", "median", "gaussian", "trimmed"):
        raise ValueError("smooth method must be one of: mean, median, gaussian, trimmed")
    if method == "gaussian" and weights is None:
        raise ValueError("gaussian smoothing requires precomputed weights")
    if method == "trimmed" and not (0.0 <= float(trim) < 0.5):
        raise ValueError("trim must be in [0, 0.5)")

    out = values.astype(float, copy=True)
    if valid is None:
        valid = np.ones(out.shape[0], dtype=bool)
    else:
        valid = valid.astype(bool, copy=False)

    # Iterative neighbor smoothing.
    for _ in range(int(iters)):
        new = out.copy()
        for i, nbrs in enumerate(neighbors):
            if not valid[i] or nbrs.size == 0:
                continue
            nbr_valid = nbrs[valid[nbrs]]
            if nbr_valid.size == 0:
                continue
            vals = out[nbr_valid]
            if method == "mean":
                stat = float(np.mean(vals))
            elif method == "median":
                stat = float(np.median(vals))
            elif method == "trimmed":
                if vals.size < 3:
                    stat = float(np.mean(vals))
                else:
                    k = int(np.floor(float(trim) * vals.size))
                    if k == 0:
                        stat = float(np.mean(vals))
                    else:
                        v = np.sort(vals)
                        stat = float(np.mean(v[k:-k])) if (vals.size - 2 * k) > 0 else float(np.mean(v))
            else:  # gaussian
                w = weights[i]
                wv = w[valid[nbrs]]
                sw = float(np.sum(wv))
                if sw <= 0:
                    stat = float(np.mean(vals))
                else:
                    stat = float(np.dot(wv, vals) / sw)

            new[i] = (1.0 - alpha) * out[i] + alpha * stat
        out = new
    return out


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Animate Klippel .sce membrane response at a chosen frequency.")
    p.add_argument("sce_path", type=Path, help="Path to .sce file")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--freq", type=float, help="Target frequency in Hz (nearest match by default)")
    g.add_argument("--j", type=int, help="Frequency index j (1-based as in .sce)")
    p.add_argument("--exact", action="store_true", help="Require exact frequency match (no nearest)")
    p.add_argument(
        "--silent-db",
        type=float,
        default=SceDataContainer.SILENT_LEVEL_DB,
        help="Silent level threshold in dB (excluded from means)",
    )
    p.add_argument("--fps", type=float, default=30.0, help="Animation FPS")
    p.add_argument("--cycles", type=float, default=1.0, help="How many cycles per animation loop")
    p.add_argument("--scale", type=float, default=1.0, help="Displacement scale factor (visual only)")
    p.add_argument(
        "--smooth",
        type=float,
        default=0.0,
        help="Spatial smoothing strength in [0..1] (0 disables).",
    )
    p.add_argument(
        "--smooth-iters",
        type=int,
        default=0,
        help="Number of spatial smoothing iterations (0 disables).",
    )
    p.add_argument(
        "--smooth-method",
        choices=["mean", "median", "trimmed", "gaussian"],
        default="mean",
        help="Smoothing method. 'median'/'trimmed' are more robust to outliers; 'gaussian' is distance-weighted.",
    )
    p.add_argument(
        "--smooth-trim",
        type=float,
        default=0.1,
        help="Trim fraction for --smooth-method trimmed (e.g. 0.1 trims 10%% low/high).",
    )
    p.add_argument(
        "--smooth-sigma",
        type=float,
        default=5.0,
        help="Sigma [mm] for --smooth-method gaussian (distance weighting on x-z plane).",
    )
    p.add_argument(
        "--mesh",
        action="store_true",
        help="Show triangulation mesh on top of displacement field.",
    )
    p.add_argument(
        "--surface",
        action="store_true",
        help="Show interpolated surface plot (via triangulation) instead of scatter points.",
    )
    p.add_argument(
        "--grid-points",
        type=int,
        default=100,
        help="Number of grid points per axis for surface interpolation (higher = smoother but slower).",
    )
    return p


def main() -> None:
    args = build_parser().parse_args()
    sce_path: Path = args.sce_path

    container = SceDataContainer(str(sce_path))
    freqs = container.get_frequencies()

    if args.j is None:
        if args.freq is None:
            raise SystemExit("Either --freq or --j must be provided.")
        target_f = float(args.freq)
        if args.exact:
            mask = freqs == target_f
            if not bool(mask.any()):
                raise SystemExit(f"Exact frequency {target_f} Hz not found.")
            j_selected = int(freqs[mask].index[0])
            f_selected = float(target_f)
        else:
            idx = int(np.argmin(np.abs(freqs.to_numpy(dtype=float) - target_f)))
            j_selected = int(freqs.index[idx])
            f_selected = float(freqs.iloc[idx])
    else:
        j_selected = int(args.j)
        if j_selected not in freqs.index:
            raise SystemExit(f"Frequency index j={j_selected} not found.")
        f_selected = float(freqs.loc[j_selected])

    resp_df = None
    for resp in container._iter_responses():
        if resp.j == j_selected:
            resp_df = resp.df.copy()
            break
    if resp_df is None:
        raise SystemExit(f"No response block found for j={j_selected}.")

    df = container.get_geometry().merge(resp_df, on="i", how="left")

    # prepare animation data
    if "amp_db" not in df.columns or "phase_rad" not in df.columns:
        raise SystemExit("No response columns found. Expected amp_db and phase_rad.")

    amp_db = df["amp_db"].to_numpy(dtype=float, copy=False)
    phase = df["phase_rad"].to_numpy(dtype=float, copy=False)
    amp_lin = container._db_to_lin(amp_db)  # mm/V

    # Keep only actually measured (non-silent) points.
    # Points without a measurement stay "missing" in the visualization (no interpolation).
    valid = np.isfinite(amp_db) & (amp_db > args.silent_db + 1e-9) & np.isfinite(phase)
    amp_lin = np.where(valid, amp_lin, 0.0)
    phase = np.where(valid, phase, 0.0)

    x = df["x"].to_numpy(dtype=float, copy=False)
    z = df["z"].to_numpy(dtype=float, copy=False)
    y0 = df["y"].to_numpy(dtype=float, copy=False)

    # time base: create frames for given number of cycles
    f_selected_f = float(f_selected)
    if f_selected_f <= 0:
        raise SystemExit(f"Invalid selected frequency: {f_selected_f}")

    # For a seamless loop, the total phase advance over the full animation must be 2*pi*k.
    # We enforce an integer number of cycles and sample the interval [0, T) (no endpoint),
    # so frame 0 and the "next loop" frame match exactly.
    cycles_int = int(round(float(args.cycles)))
    if cycles_int < 1:
        raise SystemExit("--cycles must be >= 1 for a seamless loop")
    omega = 2.0 * np.pi * f_selected_f
    T = cycles_int / f_selected_f
    n_frames = max(20, int(round(args.fps * T)))
    dt = T / n_frames

    # Precompute components so per-frame update is just a linear combination.
    # disp(t) = A*cos(theta+phi) = cos(theta)*(A*cos(phi)) - sin(theta)*(A*sin(phi))
    cos_phase = np.cos(phase)
    sin_phase = np.sin(phase)
    comp_c = float(args.scale) * amp_lin * cos_phase
    comp_s = float(args.scale) * amp_lin * sin_phase

    fig = plt.figure(figsize=(11, 8))
    gs = gridspec.GridSpec(2, 1, height_ratios=[3, 1], hspace=0.25)

    ax_top = fig.add_subplot(gs[0])
    axm = fig.add_subplot(gs[1])

    fig.suptitle(f"{sce_path.name} - j={j_selected}, f={f_selected:.2f} Hz")

    # mean plot
    axm.plot(container.get_frequencies(), container.aal["AAL[dB]"], "-", lw=1)
    axm.set_xlabel("Frequency [Hz]")
    axm.set_ylabel("Acumulated acceleration Level [dB]")
    axm.set_xscale("log")
    axm.grid(True, alpha=0.3)
    vline = axm.axvline(float(f_selected), color="tab:red", lw=1)

    time_text = ax_top.text(0.01, 0.99, "", transform=ax_top.transAxes, va="top")

    # 2D view: use triangulation on x-z plane and animate displacement as colormap.
    tri = mtri.Triangulation(x, z)
    neighbors = _build_vertex_neighbors(tri.triangles, x.shape[0])

    if args.smooth_iters > 0 and args.smooth > 0:
        weights = None
        if args.smooth_method == "gaussian":
            sigma = float(args.smooth_sigma)
            if sigma <= 0:
                raise SystemExit("--smooth-sigma must be > 0")
            denom = 2.0 * sigma * sigma
            weights = []
            for i, nbrs in enumerate(neighbors):
                if nbrs.size == 0:
                    weights.append(np.empty(0, dtype=float))
                    continue
                dx = x[nbrs] - x[i]
                dz = z[nbrs] - z[i]
                d2 = dx * dx + dz * dz
                weights.append(np.exp(-d2 / denom))

        comp_c = _smooth_field(
            comp_c,
            neighbors,
            alpha=float(args.smooth),
            iters=int(args.smooth_iters),
            method=str(args.smooth_method),
            weights=weights,
            trim=float(args.smooth_trim),
            valid=valid,
        )
        comp_s = _smooth_field(
            comp_s,
            neighbors,
            alpha=float(args.smooth),
            iters=int(args.smooth_iters),
            method=str(args.smooth_method),
            weights=weights,
            trim=float(args.smooth_trim),
            valid=valid,
        )

    disp0 = np.full_like(x, np.nan, dtype=float)
    disp0[valid] = 0.0

    amp_valid = np.sqrt(comp_c[valid] ** 2 + comp_s[valid] ** 2)
    if amp_valid.size:
        max_disp = float(np.nanpercentile(amp_valid, 99))
    else:
        max_disp = 0.0
    if not np.isfinite(max_disp) or max_disp <= 0:
        # Avoid a degenerate colormap (all gray) when initial data is all zeros/silent.
        max_disp = 1e-6

    norm = TwoSlopeNorm(vcenter=0.0, vmin=-max_disp, vmax=max_disp)

    # Render visualization (scatter or surface, with optional mesh).
    # Use a smaller marker for dense scans to keep the view readable.
    marker_size = 18 if x.shape[0] <= 4000 else 8
    cmap = plt.get_cmap("coolwarm").copy()
    cmap.set_bad(alpha=0.0)
    
    if args.surface:
        # Interpolated surface using tripcolor (from triangulation)
        tpc = ax_top.tripcolor(
            tri,
            disp0,
            cmap=cmap,
            norm=norm,
            edgecolors="none" if not args.mesh else "face",
            shading="flat",
        )
        if args.mesh:
            ax_top.triplot(tri, color="k", linewidth=0.3, alpha=0.2)
    else:
        # Scatter plot with optional mesh
        tpc = ax_top.scatter(
            x,
            z,
            c=disp0,
            s=marker_size,
            cmap=cmap,
            norm=norm,
            linewidths=0.0,
        )
        if args.mesh:
            ax_top.triplot(tri, color="k", linewidth=0.5, alpha=0.3)
    
    cb = fig.colorbar(tpc, ax=ax_top, pad=0.1, fraction=0.03)
    cb.set_label("displacement [mm/V] (scaled)")

    ax_top.set_aspect("equal", adjustable="box")
    ax_top.set_xlabel("x [mm]")
    ax_top.set_ylabel("z [mm]")

    ax_top.set_xlim(np.nanmin(x), np.nanmax(x))
    ax_top.set_ylim(np.nanmin(z), np.nanmax(z))

    def update(frame_idx: int):
        t = frame_idx * dt
        # Use explicit phase progress over the loop to avoid visible discontinuities.
        theta = 2.0 * np.pi * cycles_int * (frame_idx / n_frames)
        cos_wt = np.cos(theta)
        sin_wt = np.sin(theta)
        disp = cos_wt * comp_c - sin_wt * comp_s
        disp = np.where(valid, disp, np.nan)

        tpc.set_array(disp)

        time_text.set_text(f"t = {t:.4f} s")
        return (tpc, time_text, vline)

    # blit keeps the UI more responsive during animation.
    use_blit = True
    anim = FuncAnimation(
        fig,
        update,
        frames=n_frames,
        interval=1000.0 / args.fps,
        blit=use_blit,
        repeat=True,
    )

    # Key controls: Space to pause/resume (useful for pan/zoom), Esc/Q to quit.
    paused = {"v": False}

    def on_key(event):
        if event.key in (" ", "space"):
            if paused["v"]:
                anim.event_source.start()
            else:
                anim.event_source.stop()
            paused["v"] = not paused["v"]
        elif event.key in ("escape", "q"):
            plt.close(fig)

    fig.canvas.mpl_connect("key_press_event", on_key)

    plt.show()


if __name__ == "__main__":
    main()
