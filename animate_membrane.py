from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib import gridspec
import matplotlib.tri as mtri
from matplotlib.colors import TwoSlopeNorm
from matplotlib.widgets import Slider, RadioButtons

from sce_parser import (
    SILENT_LEVEL_DB,
    amp_db_to_linear_mm_per_v,
    choose_frequency_index,
    parse_frequencies,
    parse_geometry,
    parse_response_for_j,
    mean_amplitude_by_frequency,
)


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
    p.add_argument("--silent-db", type=float, default=SILENT_LEVEL_DB, help="Silent level threshold in dB (excluded from means)")
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
    return p


def main() -> None:
    args = build_parser().parse_args()
    sce_path: Path = args.sce_path

    geom = parse_geometry(sce_path)
    freqs = parse_frequencies(sce_path)
    j_selected = choose_frequency_index(
        freqs,
        f_hz=args.freq if args.freq is not None else None,
        j=args.j if args.j is not None else None,
        nearest=not args.exact,
    )
    f_selected = float(freqs.loc[int(j_selected)])

    # bottom plot: mean amplitude vs frequency
    means = mean_amplitude_by_frequency(sce_path, silent_level_db=args.silent_db, use_linear=True)

    x = geom["x"].to_numpy(dtype=float, copy=False)
    z = geom["z"].to_numpy(dtype=float, copy=False)
    y0 = geom["y"].to_numpy(dtype=float, copy=False)

    # time base: create frames for given number of cycles
    f_selected_f = float(f_selected)
    if f_selected_f <= 0:
        raise SystemExit(f"Invalid selected frequency: {f_selected_f}")

    # For a seamless loop, the total phase advance over the full animation must be 2π*k.
    # We enforce an integer number of cycles and sample the interval [0, T) (no endpoint),
    # so frame 0 and the "next loop" frame match exactly.
    cycles_int = int(round(float(args.cycles)))
    if cycles_int < 1:
        raise SystemExit("--cycles must be >= 1 for a seamless loop")
    # Keep the frame count fixed so switching frequency doesn't recreate the animation.
    n_frames = max(60, int(round(float(args.fps) * float(cycles_int))))

    # Figure layout:
    # - top-left: animated field
    # - top-right: "config panel" (placeholders for scale/smoothing)
    # - bottom: mean amplitude plot (click to switch frequency)
    fig = plt.figure(figsize=(12.5, 8.2))
    gs = gridspec.GridSpec(
        2,
        2,
        height_ratios=[3, 1],
        width_ratios=[4.6, 1.4],
        hspace=0.25,
        wspace=0.15,
    )
    ax_top = fig.add_subplot(gs[0, 0])
    ax_ctrl = fig.add_subplot(gs[0, 1])
    axm = fig.add_subplot(gs[1, :])
    ax_ctrl.set_axis_off()

    def _title(j: int, f_hz: float) -> str:
        return f"{sce_path.name} - j={int(j)}, f={float(f_hz):.2f} Hz"

    fig.suptitle(_title(int(j_selected), float(f_selected)))

    means_f = means["f_hz"].to_numpy(dtype=float, copy=False)
    means_j = means["j"].to_numpy(dtype=int, copy=False)
    means_db = means["mean_amp_db"].to_numpy(dtype=float, copy=False)

    # bottom plot
    axm.plot(means_f, means_db, "-", lw=1, color="0.25")
    axm.set_xlabel("Frequency [Hz]")
    axm.set_ylabel("Mean amplitude [dB mm/V]")
    axm.set_xscale("log")
    axm.grid(True, alpha=0.3)

    idx0 = int(np.argmin(np.abs(means_f - float(f_selected))))
    vline = axm.axvline(float(means_f[idx0]), color="tab:red", lw=1)
    sel_pt = axm.scatter([float(means_f[idx0])], [float(means_db[idx0])], s=22, color="tab:red", zorder=3)

    time_text = ax_top.text(0.01, 0.99, "", transform=ax_top.transAxes, va="top")

    tri = mtri.Triangulation(x, z)
    neighbors = _build_vertex_neighbors(tri.triangles, x.shape[0])

    state: dict[str, object] = {
        "j": int(j_selected),
        "f_hz": float(f_selected),
        "dt": float(cycles_int / float(f_selected) / n_frames),
        "scale": float(args.scale),
        "smooth_alpha": float(args.smooth),
        "smooth_iters": int(args.smooth_iters),
        "smooth_method": str(args.smooth_method),
        "smooth_trim": float(args.smooth_trim),
        "smooth_sigma": float(args.smooth_sigma),
        "valid": np.ones(x.shape[0], dtype=bool),
        "comp_c": np.zeros(x.shape[0], dtype=float),
        "comp_s": np.zeros(x.shape[0], dtype=float),
    }

    def _compute_gaussian_weights(sigma: float) -> list[np.ndarray]:
        if sigma <= 0:
            raise ValueError("sigma must be > 0")
        denom = 2.0 * sigma * sigma
        wts: list[np.ndarray] = []
        for i, nbrs in enumerate(neighbors):
            if nbrs.size == 0:
                wts.append(np.empty(0, dtype=float))
                continue
            dx = x[nbrs] - x[i]
            dz = z[nbrs] - z[i]
            d2 = dx * dx + dz * dz
            wts.append(np.exp(-d2 / denom))
        return wts

    def _load_components_for_j(j: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        resp = parse_response_for_j(sce_path, j=int(j))
        df = geom.merge(resp.df, on="i", how="left")
        amp_db = df["amp_db"].to_numpy(dtype=float, copy=False)
        phase = df["phase_rad"].to_numpy(dtype=float, copy=False)

        valid = np.isfinite(amp_db) & (amp_db > args.silent_db + 1e-9) & np.isfinite(phase)
        amp_lin = amp_db_to_linear_mm_per_v(np.where(valid, amp_db, 0.0))
        phase = np.where(valid, phase, 0.0)

        cos_phase = np.cos(phase)
        sin_phase = np.sin(phase)
        comp_c0 = amp_lin * cos_phase
        comp_s0 = amp_lin * sin_phase
        return comp_c0.astype(float, copy=False), comp_s0.astype(float, copy=False), valid

    disp0 = np.zeros_like(x, dtype=float)
    tpc = ax_top.tripcolor(tri, disp0, shading="gouraud", cmap="coolwarm")
    cb = fig.colorbar(tpc, ax=ax_top, pad=0.1, fraction=0.03)
    cb.set_label("displacement [mm/V] (scaled)")

    ax_top.set_aspect("equal", adjustable="box")
    ax_top.set_xlabel("x [mm]")
    ax_top.set_ylabel("z [mm]")
    ax_top.set_xlim(np.nanmin(x), np.nanmax(x))
    ax_top.set_ylim(np.nanmin(z), np.nanmax(z))

    def _recompute_components() -> None:
        comp_c0, comp_s0, valid = _load_components_for_j(int(state["j"]))
        state["valid"] = valid

        comp_c = float(state["scale"]) * comp_c0
        comp_s = float(state["scale"]) * comp_s0

        iters = int(state["smooth_iters"])
        alpha = float(state["smooth_alpha"])
        method = str(state["smooth_method"])
        trim = float(state["smooth_trim"])

        weights = None
        if iters > 0 and alpha > 0:
            if method == "gaussian":
                weights = _compute_gaussian_weights(float(state["smooth_sigma"]))
            comp_c = _smooth_field(
                comp_c,
                neighbors,
                alpha=alpha,
                iters=iters,
                method=method,
                weights=weights,
                trim=trim,
                valid=valid,
            )
            comp_s = _smooth_field(
                comp_s,
                neighbors,
                alpha=alpha,
                iters=iters,
                method=method,
                weights=weights,
                trim=trim,
                valid=valid,
            )

        state["comp_c"] = comp_c
        state["comp_s"] = comp_s

        f_hz = float(state["f_hz"])
        state["dt"] = float(cycles_int / f_hz / n_frames)

        amp_valid = np.sqrt(comp_c[valid] ** 2 + comp_s[valid] ** 2)
        max_disp = float(np.nanpercentile(amp_valid, 99)) if amp_valid.size else 0.0
        if not np.isfinite(max_disp) or max_disp <= 0:
            max_disp = 1e-6
        new_norm = TwoSlopeNorm(vcenter=0.0, vmin=-max_disp, vmax=max_disp)
        tpc.set_norm(new_norm)
        cb.update_normal(tpc)

    _recompute_components()

    # --- Control panel (right) ---
    ctrl_pos = ax_ctrl.get_position()
    left = ctrl_pos.x0 + 0.06 * ctrl_pos.width
    width = 0.88 * ctrl_pos.width
    y_top = ctrl_pos.y1

    ax_scale = fig.add_axes([left, y_top - 0.22 * ctrl_pos.height, width, 0.06 * ctrl_pos.height])
    s_scale = Slider(ax_scale, "Scale", 0.1, 10.0, valinit=float(state["scale"]), valstep=0.1)

    ax_salpha = fig.add_axes([left, y_top - 0.36 * ctrl_pos.height, width, 0.06 * ctrl_pos.height])
    s_salpha = Slider(ax_salpha, "Smooth", 0.0, 1.0, valinit=float(state["smooth_alpha"]), valstep=0.02)

    ax_siters = fig.add_axes([left, y_top - 0.50 * ctrl_pos.height, width, 0.06 * ctrl_pos.height])
    s_siters = Slider(ax_siters, "Iters", 0, 15, valinit=int(state["smooth_iters"]), valstep=1)

    ax_method = fig.add_axes([left, y_top - 0.92 * ctrl_pos.height, width, 0.36 * ctrl_pos.height])
    methods = ("mean", "median", "trimmed", "gaussian")
    r_method = RadioButtons(ax_method, methods, active=methods.index(str(state["smooth_method"])))
    ax_method.set_title("Method", fontsize=9)

    def _on_controls_change(_=None) -> None:
        state["scale"] = float(s_scale.val)
        state["smooth_alpha"] = float(s_salpha.val)
        state["smooth_iters"] = int(round(float(s_siters.val)))
        state["smooth_method"] = str(r_method.value_selected)
        _recompute_components()
        fig.canvas.draw_idle()

    s_scale.on_changed(_on_controls_change)
    s_salpha.on_changed(_on_controls_change)
    s_siters.on_changed(_on_controls_change)
    r_method.on_clicked(_on_controls_change)

    # --- Frequency selection: click on bottom plot ---
    def _select_nearest_frequency(f_hz: float) -> None:
        if not np.isfinite(f_hz) or f_hz <= 0:
            return
        idx = int(np.argmin(np.abs(means_f - float(f_hz))))
        j_new = int(means_j[idx])
        f_new = float(means_f[idx])
        if j_new == int(state["j"]):
            return

        state["j"] = j_new
        state["f_hz"] = f_new
        fig.suptitle(_title(j_new, f_new))
        vline.set_xdata([f_new, f_new])
        sel_pt.set_offsets(np.array([[f_new, float(means_db[idx])]]))
        _recompute_components()

    def _on_click(event) -> None:
        if event.inaxes is not axm:
            return
        if event.button != 1 or event.xdata is None:
            return
        _select_nearest_frequency(float(event.xdata))
        fig.canvas.draw_idle()

    fig.canvas.mpl_connect("button_press_event", _on_click)

    def update(frame_idx: int):
        t = frame_idx * float(state["dt"])
        theta = 2.0 * np.pi * cycles_int * (frame_idx / n_frames)
        cos_wt = np.cos(theta)
        sin_wt = np.sin(theta)
        disp = cos_wt * np.asarray(state["comp_c"]) - sin_wt * np.asarray(state["comp_s"])
        tpc.set_array(disp)
        time_text.set_text(f"t = {t:.4f} s")
        return (tpc, time_text, vline)

    use_blit = True
    anim = FuncAnimation(
        fig,
        update,
        frames=n_frames,
        interval=1000.0 / args.fps,
        blit=use_blit,
        repeat=True,
    )

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
        elif event.key in ("left", "right"):
            cur = float(state["f_hz"])
            idx = int(np.argmin(np.abs(means_f - cur)))
            idx = max(0, min(len(means_f) - 1, idx + (1 if event.key == "right" else -1)))
            _select_nearest_frequency(float(means_f[idx]))
            fig.canvas.draw_idle()

    fig.canvas.mpl_connect("key_press_event", on_key)
    plt.show()


if __name__ == "__main__":
    main()
