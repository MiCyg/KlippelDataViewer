from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Iterable, Iterator

import numpy as np
import pandas as pd





@dataclass(frozen=True)
class SceResponse:
    j: int
    f_hz: float
    df: pd.DataFrame  # columns: i, amp_db, phase_rad


class SceDataContainer:
    
    SILENT_LEVEL_DB = -130.0
    rho0 = 1.21
    p0 = 20e-6
    
    
    def __init__(self, file_path: str = None):
        self._file_path = Path(file_path)
        if self._file_path:
            self.import_data()
        

    def import_data(self, path: str | None = None) -> None:
        if path is not None:
            self._file_path = Path(path)
        self._frequencies = self._parse_frequencies()
        self._geometry = self._parse_geometry()
        self._aal = self._calc_AAL()
        
    def get_file_path(self):
        return self._file_path
    
    def get_frequencies(self) -> pd.Series: 
        return self._frequencies.copy()
    
    def get_geometry(self) -> pd.DataFrame:
        return self._geometry
 
    def get_aal(self) -> pd.DataFrame:
        return self._aal
   
    def get_response(self, freq: float = None, j: int = None) -> pd.DataFrame:
        df, _, _ = self.get_response_raw(freq=freq, j=j, nearest=False, exact=True)
        disp_db = df["amp_db"].to_numpy(dtype=float, copy=False)
        disp_lin = self._db_to_lin(disp_db)

        out = pd.DataFrame(
            {
                "idx": df["i"].astype(int),
                "x": df["x"].astype(float),
                "y": df["y"].astype(float),
                "z": df["z"].astype(float),
                "disp[mm/V]": disp_lin,
                "disp[dB]": disp_db,
            }
        )
        return out


    def select_frequency(self,*,freq: float | None = None,j: int | None = None,nearest: bool = True,exact: bool = False) -> tuple[float, int]:
        if freq is None and j is None:
            raise ValueError("Either freq or j must be provided.")

        if j is not None:
            j_selected = int(j)
            if j_selected not in self._frequencies.index:
                raise ValueError(f"Frequency index j={j_selected} not found.")
            return float(self._frequencies.loc[j_selected]), j_selected

        target_f = float(freq)
        if exact or not nearest:
            mask = self._frequencies == target_f
            if not bool(mask.any()):
                raise ValueError(f"Exact frequency {target_f} Hz not found.")
            j_selected = int(self._frequencies[mask].index[0])
            return float(target_f), j_selected

        idx = int(np.argmin(np.abs(self._frequencies.to_numpy(dtype=float) - target_f)))
        j_selected = int(self._frequencies.index[idx])
        return float(self._frequencies.iloc[idx]), j_selected

    def get_response_raw(self,*,freq: float | None = None,j: int | None = None,nearest: bool = True,exact: bool = False) -> tuple[pd.DataFrame, float, int]:
        f_selected, j_selected = self.select_frequency(freq=freq, j=j, nearest=nearest, exact=exact)

        resp_df = None
        for resp in self._iter_responses():
            if resp.j == j_selected:
                resp_df = resp.df.copy()
                break
        if resp_df is None:
            raise ValueError(f"No response block found for j={j_selected}.")

        merged = self._geometry.merge(resp_df, on="i", how="left")
        return merged, f_selected, j_selected

    def _read_lines(self, *, encoding: str = "utf-8") -> list[str]:
        return Path(self._file_path).read_text(encoding=encoding).splitlines()
    
    def _iter_responses(self) -> Iterator[SceResponse]:
        """
        Iterate over response blocks in the .sce file.

        Each block has:
        frequency(j) = f;
        response = [;
            i amp_db phase_rad;
            ...
        ];

        Yields SceResponse with df columns: i, amp_db, phase_rad
        """
        lines = self._read_lines()
        re_freq = re.compile(r"^frequency\((\d+)\)\s*=\s*([0-9.+-Ee]+)\s*;")

        i = 0
        n = len(lines)
        while i < n:
            m = re_freq.match(lines[i].strip())
            if not m:
                i += 1
                continue

            j = int(m.group(1))
            f_hz = float(m.group(2))
            i += 1

            # seek 'response' start
            while i < n and not (lines[i].lstrip().startswith("response") and "[" in lines[i]):
                i += 1
            if i >= n:
                break
            i += 1  # first row after 'response = [;'

            rows: list[tuple[int, float, float]] = []
            while i < n:
                s = lines[i].strip()
                i += 1
                if not s:
                    continue
                if s == "];" or s.startswith("];"):
                    break
                s = s.split("//")[0].strip().rstrip(";")
                parts = re.split(r"\s+", s)
                if len(parts) < 3:
                    continue
                try:
                    i_pt = int(parts[0])
                    amp_db = float(parts[1])
                    phase_rad = float(parts[2])
                except ValueError:
                    continue
                rows.append((i_pt, amp_db, phase_rad))

            df = pd.DataFrame(rows, columns=["i", "amp_db", "phase_rad"])
            yield SceResponse(j=j, f_hz=f_hz, df=df)

    def _parse_geometry(self) -> pd.DataFrame:
        """
        Parse `geometry = [ ... ];` section.

        Supports two common formats (auto-detected):

        - Polar: `i r phi y`  -> `x = r*cos(phi)`, `z = r*sin(phi)`
        - Cartesian: `i x z y`

        Returns DataFrame with columns: i, y, x, z, r, phi.
        """
        lines = self._read_lines()
        inside = False
        rows: list[tuple[int, float, float, float]] = []

        for ln in lines:
            s = ln.strip()
            if not inside:
                if s.startswith("geometry"):
                    inside = True
                continue

            if s.startswith("]"):
                break
            if not s:
                continue

            s = s.split("//")[0].strip().rstrip(";")
            parts = re.split(r"\s+", s)
            if len(parts) < 4:
                continue
            try:
                i_pt = int(parts[0])
                a = float(parts[1])
                b = float(parts[2])
                c = float(parts[3])
            except ValueError:
                continue
            rows.append((i_pt, a, b, c))

        raw = pd.DataFrame(rows, columns=["i", "c1", "c2", "c3"])
        if raw.empty:
            return pd.DataFrame(columns=["i", "y", "x", "z", "r", "phi"])

        # Heuristic: if c1 contains negatives, it can't be a radius -> treat as cartesian x.
        # Otherwise, if c2 is within roughly [0..2π] (or [-π..π]) treat as polar angle.
        c1 = raw["c1"].to_numpy(dtype=float, copy=False)
        c2 = raw["c2"].to_numpy(dtype=float, copy=False)

        has_negative_c1 = bool(np.nanmin(c1) < -1e-9)
        max_abs_c2 = float(np.nanmax(np.abs(c2))) if c2.size else 0.0
        looks_like_radians = max_abs_c2 <= (2.0 * np.pi + 0.25)

        is_polar = (not has_negative_c1) and looks_like_radians

        if is_polar:
            r = raw["c1"].astype(float)
            phi = raw["c2"].astype(float)
            y = raw["c3"].astype(float)
            x = r * np.cos(phi)
            z = r * np.sin(phi)
        else:
            x = raw["c1"].astype(float)
            z = raw["c2"].astype(float)
            y = raw["c3"].astype(float)
            r = np.sqrt(x * x + z * z)
            phi = np.arctan2(z, x)

        df = pd.DataFrame({"i": raw["i"].astype(int), "y": y, "x": x, "z": z, "r": r, "phi": phi})
        return df

    def _parse_frequencies(self) -> pd.Series:
        """
        Parse all `frequency(j) = ...;` entries.

        Returns Series indexed by j (int, 1-based), values in Hz.
        """
        lines = self._read_lines()
        freqs: dict[int, float] = {}
        re_freq = re.compile(r"^frequency\((\d+)\)\s*=\s*([0-9.+-Ee]+)\s*;")
        for ln in lines:
            m = re_freq.match(ln.strip())
            if not m:
                continue
            freqs[int(m.group(1))] = float(m.group(2))
        if not freqs:
            raise ValueError("No frequency(j) entries found in .sce file.")
        return pd.Series(freqs, name="f_hz").sort_index()
    
    def _calc_AAL(self,*,silent_level_db:float = SILENT_LEVEL_DB) -> pd.DataFrame:
        """
        Compute per-frequency acumulated amplitude over points.
        - Points at silent_level_db are excluded.
        """
        rows = []
        for resp in self._iter_responses():
            omega = 2*np.pi*resp.f_hz
            level = resp.df["amp_db"].to_numpy(dtype=float, copy=False)
            
            # Displacement in lin [m/V] from level [dB]
            membrane_sensitivity = self._db_to_lin(level) / 1000.0

            # https://www.klippel.de/manuals/scanningvibrometer-partmeasurement/rma/rma.html
            # _aal = 20.0 * np.log10( self.rho0 * (omega**2) / (2.0 * np.pi * np.sqrt(2) * self.p0) * np.sum(membrane_sensitivity**2))
            _aal = 20.0 * np.log10( (omega**2) * np.sum(membrane_sensitivity))
            
            accumulated_displacement_mm = np.sum(membrane_sensitivity*1000)
            
            rows.append((resp.j, resp.f_hz, accumulated_displacement_mm, _aal))
        return pd.DataFrame(rows, columns=["j", "f[Hz]", "AAL[m/V]", "AAL[dB]"]).sort_values("f[Hz]")
    
    def _db_to_lin(self, db: float | np.ndarray, ref:float = 1.0) -> float | np.ndarray:
        return ref * np.power(10.0, np.asarray(db) / 20.0)

    def _lin_to_db(self, lin: float | np.ndarray, ref:float = 1.0) -> float | np.ndarray:
        return 20*np.log10(np.asarray(lin) / ref)



if __name__ == "__main__":
    print("START", __file__)
    import matplotlib.pyplot as plt
    
    sce_path = Path("data/MD/noew_mocowanie_sruba/perf_1mm_otwory/pomiar1.sce")
    sceData = SceDataContainer(sce_path, 1)
    print(sceData.geometry)
    
    sceData.aal.to_csv("aal.csv")
    plt.plot(sceData.aal["f[Hz]"], sceData.aal["AAL[dB]"])
    plt.xlabel("frequency[Hz]")
    plt.ylabel("AAL[dB]")
    plt.xscale("log")
    plt.grid(True)
    plt.show()
    

    
    print("END")
