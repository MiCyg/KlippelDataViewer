from __future__ import annotations

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib import gridspec
from matplotlib.colors import TwoSlopeNorm

from sce_parser import SceDataContainer

class MembraneAnimator:
    def __init__(self, sce_path, j, silent_db, fps, scale, export_view_path=None, export_frequency=None):
        self.container = SceDataContainer(str(sce_path))
        self.model = MembraneModel(self.container, silent_db, scale, fps)

        self.j = j
        self.export_view_path = export_view_path
        self.export_frequency = export_frequency

    def run(self):
        f_selected, j_selected = self.model.load_frequency(self.j)

        self.fig = plt.figure(figsize=(11, 8))
        gs = gridspec.GridSpec(2, 1, height_ratios=[3, 1], hspace=0.25)

        self.ax_top = self.fig.add_subplot(gs[0])
        self.ax_bottom = self.fig.add_subplot(gs[1])

        self.fig.suptitle(f"{self.container.get_file_path()} - f={f_selected:.2f} Hz")

        # --- dolny wykres AAL
        self.ax_bottom.plot(
            self.container.get_aal()["f[Hz]"],
            self.container.get_aal()["AAL[dB]"],
            "-"
        )
        self.ax_bottom.set_xlabel("Frequency [Hz]")
        self.ax_bottom.set_ylabel("AAL [dB]")
        self.ax_bottom.set_xscale("log")
        self.ax_bottom.grid(True, alpha=0.3)

        self.vline = self.ax_bottom.axvline(f_selected, color="red")

        # --- scatter
        disp0 = np.zeros_like(self.model.x)

        self.norm = TwoSlopeNorm(
            vcenter=0.0,
            vmin=-self.model.max_disp,
            vmax=self.model.max_disp
        )

        self.tpc = self.ax_top.scatter(
            self.model.x,
            self.model.z,
            c=disp0,
            cmap="coolwarm",
            norm=self.norm,
            s=12
        )
        self.cbar = self.fig.colorbar(
            self.tpc,
            ax=self.ax_top,
            pad=0.02,
            fraction=0.04
        )
        self.cbar.set_label("Displacement [mm/V]")
        

        self.time_text = self.ax_top.text(
            0.01, 0.99, "",
            transform=self.ax_top.transAxes,
            va="top"
        )

        self.ax_top.set_aspect("equal")
        self.ax_top.set_xlabel("x [mm]")
        self.ax_top.set_ylabel("z [mm]")

        self.anim = FuncAnimation(
            self.fig,
            self.update,
            frames=self.model.n_frames,
            interval=1000.0 / self.model.fps,
            blit=True
        )

        if self.export_view_path:
            self._export_animation(self.export_view_path)

        self.fig.canvas.mpl_connect("button_press_event", self.on_click)

        plt.show()

    def _export_animation(self, export_path):
        out_path = Path(export_path)
        suffix = out_path.suffix.lower()
        fps = float(self.model.fps)

        try:
            if self.export_frequency is not None:
                f_selected, _ = self.model.load_frequency(self.export_frequency, self.j)
                self.fig.suptitle(f"{self.container.get_file_path()} - f={f_selected:.2f} Hz")
                self.norm = TwoSlopeNorm(
                    vcenter=0.0,
                    vmin=-self.model.max_disp,
                    vmax=self.model.max_disp
                )
                self.tpc.set_offsets(
                    np.column_stack([self.model.x, self.model.z])
                )
                self.tpc.set_norm(self.norm)
                self.vline.set_xdata([f_selected])

            if suffix == ".gif":
                from matplotlib.animation import PillowWriter
                writer = PillowWriter(fps=fps)
            else:
                raise SystemExit("Unsupported export extension. Use .gif")
            self.anim.save(str(out_path), writer=writer, dpi=100)
        except Exception as exc:
            raise SystemExit(f"Export failed: {exc}")

    # ---------------------------------------------------------

    def update(self, frame_idx):
        disp = self.model.displacement(frame_idx)
        self.tpc.set_array(disp)
        self.time_text.set_text(f"t = {frame_idx * 1e6 * self.model.dt:.1f} us")
        return (self.tpc, self.time_text)

    # ---------------------------------------------------------

    def on_click(self, event):
        if event.inaxes != self.ax_bottom:
            return

        new_freq = float(event.xdata)

        f_selected, _ = self.model.load_frequency(new_freq, self.j)
        self.fig.suptitle(f"{self.container.get_file_path()} - f={f_selected:.2f} Hz")
        
        self.norm = TwoSlopeNorm(
            vcenter=0.0,
            vmin=-self.model.max_disp,
            vmax=self.model.max_disp
        )

        self.tpc.set_offsets(
            np.column_stack([self.model.x, self.model.z])
        )
        self.tpc.set_norm(self.norm)

        self.vline.set_xdata([f_selected])

        # restart animacji
        self.anim.frame_seq = self.anim.new_frame_seq()

        self.fig.canvas.draw_idle()


class MembraneModel:
    def __init__(self, container, silent_db, scale, fps):
        self.container = container
        self.silent_db = silent_db
        self.scale = scale
        self.fps = fps

    def load_frequency(self, freq=None, j=None):
        if freq is None and j is None:
            freq = float(self.container.get_aal()["f[Hz]"].iloc[0])
        df, f_selected, j_selected = self.container.get_response_raw(
            freq=freq,
            j=j,
            nearest=True,
            exact=False,
        )

        amp_db = df["amp_db"].to_numpy(dtype=float)
        phase = df["phase_rad"].to_numpy(dtype=float)
        amp_lin = self.container._db_to_lin(amp_db)

        self.valid = (
            np.isfinite(amp_db)
            & (amp_db > self.silent_db + 1e-9)
            & np.isfinite(phase)
        )

        amp_lin = np.where(self.valid, amp_lin, 0.0)
        phase = np.where(self.valid, phase, 0.0)

        self.x = df["x"].to_numpy(dtype=float)
        self.z = df["z"].to_numpy(dtype=float)

        cos_phase = np.cos(phase)
        sin_phase = np.sin(phase)

        self.comp_c = self.scale * amp_lin * cos_phase
        self.comp_s = self.scale * amp_lin * sin_phase

        self.freq = float(f_selected)

        T = 1.0 / self.freq
        self.n_frames = max(20, int(round(self.fps * T)))
        self.dt = T / self.n_frames

        amp_valid = np.sqrt(
            self.comp_c[self.valid] ** 2 +
            self.comp_s[self.valid] ** 2
        )

        self.max_disp = float(np.nanpercentile(amp_valid, 99)) if amp_valid.size else 1e-6
        if self.max_disp <= 0 or not np.isfinite(self.max_disp):
            self.max_disp = 1e-6

        return f_selected, j_selected

    def displacement(self, frame_idx):
        theta = 2.0 * np.pi * (frame_idx / self.n_frames)
        cos_wt = np.cos(theta)
        sin_wt = np.sin(theta)

        disp = cos_wt * self.comp_c - sin_wt * self.comp_s
        return np.where(self.valid, disp, np.nan)

