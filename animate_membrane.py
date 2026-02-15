from __future__ import annotations

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib import gridspec
from matplotlib.colors import TwoSlopeNorm

from sce_parser import SceDataContainer


class MembraneAnimator:
    def __init__(
        self,
        *,
        sce_path: Path,
        freq: float | None,
        j: int | None,
        exact: bool,
        silent_db: float,
        fps: float,
        cycles: float,
        scale: float,
        aal_only: bool = False,
    ) -> None:
        self.sce_path = sce_path
        self.freq = freq
        self.j = j
        self.exact = exact
        self.silent_db = silent_db
        self.fps = fps
        self.cycles = cycles
        self.scale = scale
        self.aal_only = aal_only

    def run(self) -> None:
        container = SceDataContainer(str(self.sce_path))

        if self.aal_only:
            print(container.aal.head(10).to_string(index=False))
            return

        df, f_selected, j_selected = container.get_response_raw(
            freq=self.freq,
            j=self.j,
            nearest=not self.exact,
            exact=self.exact,
        )

        if "amp_db" not in df.columns or "phase_rad" not in df.columns:
            raise SystemExit("No response columns found. Expected amp_db and phase_rad.")

        amp_db = df["amp_db"].to_numpy(dtype=float, copy=False)
        phase = df["phase_rad"].to_numpy(dtype=float, copy=False)
        amp_lin = container._db_to_lin(amp_db)

        # Keep only actually measured (non-silent) points.
        valid = np.isfinite(amp_db) & (amp_db > self.silent_db + 1e-9) & np.isfinite(phase)
        amp_lin = np.where(valid, amp_lin, 0.0)
        phase = np.where(valid, phase, 0.0)

        x = df["x"].to_numpy(dtype=float, copy=False)
        z = df["z"].to_numpy(dtype=float, copy=False)

        f_selected_f = float(f_selected)
        if f_selected_f <= 0:
            raise SystemExit(f"Invalid selected frequency: {f_selected_f}")

        cycles_int = int(round(float(self.cycles)))
        if cycles_int < 1:
            raise SystemExit("--cycles must be >= 1 for a seamless loop")
        T = cycles_int / f_selected_f
        n_frames = max(20, int(round(self.fps * T)))
        dt = T / n_frames

        cos_phase = np.cos(phase)
        sin_phase = np.sin(phase)
        comp_c = float(self.scale) * amp_lin * cos_phase
        comp_s = float(self.scale) * amp_lin * sin_phase

        fig = plt.figure(figsize=(11, 8))
        gs = gridspec.GridSpec(2, 1, height_ratios=[3, 1], hspace=0.25)

        ax_top = fig.add_subplot(gs[0])
        axm = fig.add_subplot(gs[1])

        fig.suptitle(f"{self.sce_path.name} - j={j_selected}, f={f_selected:.2f} Hz")

        axm.plot(container.aal["f[Hz]"], container.aal["AAL[dB]"], "-", lw=1)
        axm.set_xlabel("Frequency [Hz]")
        axm.set_ylabel("Acumulated acceleration Level [dB]")
        axm.set_xscale("log")
        axm.grid(True, alpha=0.3)
        vline = axm.axvline(float(f_selected), color="tab:red", lw=1)

        time_text = ax_top.text(0.01, 0.99, "", transform=ax_top.transAxes, va="top")

        disp0 = np.full_like(x, np.nan, dtype=float)
        disp0[valid] = 0.0

        amp_valid = np.sqrt(comp_c[valid] ** 2 + comp_s[valid] ** 2)
        if amp_valid.size:
            max_disp = float(np.nanpercentile(amp_valid, 99))
        else:
            max_disp = 0.0
        if not np.isfinite(max_disp) or max_disp <= 0:
            max_disp = 1e-6

        norm = TwoSlopeNorm(vcenter=0.0, vmin=-max_disp, vmax=max_disp)
        zero_eps = 1e-12

        marker_size = 18 if x.shape[0] <= 4000 else 8
        cmap = plt.get_cmap("coolwarm").copy()
        cmap.set_bad(alpha=0.0)

        tpc = ax_top.scatter(
            x,
            z,
            c=disp0,
            s=marker_size,
            cmap=cmap,
            norm=norm,
            linewidths=0.0,
        )

        cb = fig.colorbar(tpc, ax=ax_top, pad=0.1, fraction=0.03)
        cb.set_label("displacement [mm/V] (scaled)")

        ax_top.set_aspect("equal", adjustable="box")
        ax_top.set_xlabel("x [mm]")
        ax_top.set_ylabel("z [mm]")

        ax_top.set_xlim(np.nanmin(x), np.nanmax(x))
        ax_top.set_ylim(np.nanmin(z), np.nanmax(z))

        def update(frame_idx: int):
            t = frame_idx * dt
            theta = 2.0 * np.pi * cycles_int * (frame_idx / n_frames)
            cos_wt = np.cos(theta)
            sin_wt = np.sin(theta)
            disp = cos_wt * comp_c - sin_wt * comp_s
            disp = np.where(valid, disp, np.nan)

            tpc.set_array(disp)
            colors = cmap(norm(disp))
            tpc.set_facecolors(colors)

            time_text.set_text(f"t = {t:.4f} s")
            return (tpc, time_text, vline)

        use_blit = True
        anim = FuncAnimation(
            fig,
            update,
            frames=n_frames,
            interval=1000.0 / self.fps,
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

        fig.canvas.mpl_connect("key_press_event", on_key)

        plt.show()
