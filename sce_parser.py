from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Iterable, Iterator

import numpy as np
import pandas as pd


SILENT_LEVEL_DB = -130.0


@dataclass(frozen=True)
class SceResponse:
    j: int
    f_hz: float
    df: pd.DataFrame  # columns: i, amp_db, phase_rad


def _read_lines(path: str | Path, *, encoding: str = "utf-8") -> list[str]:
    return Path(path).read_text(encoding=encoding).splitlines()


def parse_geometry(path: str | Path) -> pd.DataFrame:
    """
    Parse `geometry = [ ... ];` section.

    Returns DataFrame with columns: i, r, phi, y, x, z
    (x,z computed from polar coordinates).
    """
    lines = _read_lines(path)
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
            r = float(parts[1])
            phi = float(parts[2])
            y = float(parts[3])
        except ValueError:
            continue
        rows.append((i_pt, r, phi, y))

    df = pd.DataFrame(rows, columns=["i", "r", "phi", "y"])
    df["x"] = df["r"] * np.cos(df["phi"])
    df["z"] = df["r"] * np.sin(df["phi"])
    return df


def parse_frequencies(path: str | Path) -> pd.Series:
    """
    Parse all `frequency(j) = ...;` entries.

    Returns Series indexed by j (int, 1-based), values in Hz.
    """
    lines = _read_lines(path)
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


def choose_frequency_index(freqs: pd.Series, *, f_hz: float | None = None, j: int | None = None, nearest: bool = True) -> int:
    if j is not None:
        j = int(j)
        if j not in freqs.index:
            raise ValueError(f"Invalid j={j}. Available: 1..{int(freqs.index.max())}")
        return j
    if f_hz is None:
        raise ValueError("Provide either f_hz or j.")
    target = float(f_hz)
    if nearest:
        return int((freqs - target).abs().idxmin())
    matches = freqs[freqs == target]
    if matches.empty:
        raise ValueError(f"No exact f_hz={target} in file (use nearest=True).")
    return int(matches.index[0])


def iter_responses(path: str | Path) -> Iterator[SceResponse]:
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
    lines = _read_lines(path)
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


def parse_response_for_j(path: str | Path, *, j: int) -> SceResponse:
    for resp in iter_responses(path):
        if resp.j == int(j):
            return resp
    raise ValueError(f"Response block for j={j} not found.")


def load_sce_dataframe(
    path: str | Path,
    *,
    f_hz: float | None = None,
    j: int | None = None,
    nearest: bool = True,
    include_response: bool = True,
) -> tuple[pd.DataFrame, float | None, int | None]:
    """
    Load geometry and optionally response for selected frequency.

    Returns (df, f_selected, j_selected).
    If include_response=False, returns (geometry_df, None, None).
    """
    geom = parse_geometry(path)
    if not include_response:
        return geom, None, None

    freqs = parse_frequencies(path)
    j_sel = choose_frequency_index(freqs, f_hz=f_hz, j=j, nearest=nearest)
    resp = parse_response_for_j(path, j=j_sel)

    out = geom.merge(resp.df, on="i", how="left")
    return out, resp.f_hz, j_sel


def amp_db_to_linear_mm_per_v(amp_db: float | np.ndarray) -> float | np.ndarray:
    """
    Convert amplitude in dB (mm/V) to linear mm/V.
    Assumes 20*log10(amplitude).
    """
    return np.power(10.0, np.asarray(amp_db) / 20.0)


def mean_amplitude_by_frequency(
    path: str | Path,
    *,
    silent_level_db: float = SILENT_LEVEL_DB,
    use_linear: bool = True,
) -> pd.DataFrame:
    """
    Compute per-frequency mean amplitude over points.

    - If use_linear=True: mean of linear amplitudes (mm/V), then also provides mean_db.
    - Points at silent_level_db are excluded.
    """
    rows = []
    for resp in iter_responses(path):
        amp = resp.df["amp_db"].to_numpy(dtype=float, copy=False)
        mask = np.isfinite(amp) & (amp > silent_level_db + 1e-9)
        if not mask.any():
            mean_lin = 0.0
        else:
            if use_linear:
                mean_lin = float(np.mean(amp_db_to_linear_mm_per_v(amp[mask])))
            else:
                mean_lin = float(np.mean(amp[mask]))
        mean_db = 20.0 * np.log10(mean_lin) if mean_lin > 0 else float(silent_level_db)
        rows.append((resp.j, resp.f_hz, mean_lin, mean_db))
    return pd.DataFrame(rows, columns=["j", "f_hz", "mean_amp_lin", "mean_amp_db"]).sort_values("f_hz")

