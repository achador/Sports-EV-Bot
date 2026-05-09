"""
Adapter between the Streamlit UI and the existing Sports-EV-Bot scanners.

The repo already has CLI-driven scanners under
    src/sports/nba/scanner.py
    src/sports/tennis/scanner.py

…that load the trained XGBoost models, pull PrizePicks lines + FanDuel odds,
project each market, compute EV, and rank picks.

This module wraps those into a single `run_scan(...)` function that the
Streamlit app can call and that returns a normalized pandas DataFrame.

────────────────────────────────────────────────────────────────────────────────
IMPORTANT — wire-up step
────────────────────────────────────────────────────────────────────────────────
The function names / signatures in your scanner files may differ slightly from
what's assumed below. Search your repo for the function that already produces
the ranked pick list (look for `def scan_*` or similar in
`src/sports/nba/scanner.py` and `src/sports/tennis/scanner.py`) and update the
two import sites + call sites in `_scan_nba` / `_scan_tennis` accordingly.

Each scanner is expected to return either:
  - a list of dicts, OR
  - a pandas DataFrame

with at minimum these columns / keys:
    player, stat, side ("Over"|"Under"), line, projection,
    ev_pct (as a fraction, e.g. 0.07 for 7%),
    fd_implied_pct (as a fraction),
    book (e.g. "PrizePicks")

Optional but useful for explanations:
    opponent, is_home, rest_days, season_avg, last5_avg, last10_avg,
    minutes_proj, opp_allowed_to_pos, usage_rate, pace,
    surface, rank, opp_rank, h2h_winrate
"""

from __future__ import annotations

import datetime as dt
import os
from pathlib import Path
from typing import Any, Iterable

import pandas as pd
import streamlit as st


SUPPORTED_SPORTS = ["NBA", "Tennis"]

REPO_ROOT = Path(__file__).resolve().parent
NBA_PROJ_DIR = REPO_ROOT / "data" / "nba" / "projections"
TENNIS_PROJ_DIR = REPO_ROOT / "data" / "tennis" / "projections"

# Columns we always want in the output, in display order.
CANONICAL_COLUMNS = [
    "player",
    "stat",
    "side",
    "line",
    "projection",
    "ev_pct",
    "fd_implied_pct",
    "book",
]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


@st.cache_data(ttl=300, show_spinner=False)
def run_scan(
    sport: str,
    game_date: dt.date,
    ev_threshold: float = 0.04,
    max_results: int = 15,
    player_filter: str | None = None,
) -> pd.DataFrame:
    """Run a +EV scan and return a ranked DataFrame.

    Parameters
    ----------
    sport : "NBA" or "Tennis"
    game_date : the game date to scan
    ev_threshold : minimum EV as a fraction (0.04 == 4%)
    max_results : max number of picks to return
    player_filter : optional case-insensitive substring filter on player name
    """
    sport = sport.strip()
    if sport == "NBA":
        raw = _scan_nba(game_date)
    elif sport == "Tennis":
        raw = _scan_tennis(game_date)
    else:
        raise ValueError(f"Unsupported sport: {sport!r}")

    df = _normalize(raw)
    if df.empty:
        return df

    df = df[df["ev_pct"] >= ev_threshold]
    if player_filter:
        df = df[df["player"].str.contains(player_filter, case=False, na=False)]

    df = df.sort_values("ev_pct", ascending=False).head(max_results).reset_index(drop=True)
    return df


@st.cache_data(ttl=600, show_spinner=False)
def list_players(sport: str) -> list[str]:
    """Optional helper: list known players for a given sport.

    Currently unused by the UI but handy for autocomplete extensions.
    """
    sport = sport.strip()
    try:
        if sport == "NBA":
            from src.sports.nba import builder  # type: ignore

            if hasattr(builder, "list_active_players"):
                return list(builder.list_active_players())
        elif sport == "Tennis":
            from src.sports.tennis import builder  # type: ignore

            if hasattr(builder, "list_active_players"):
                return list(builder.list_active_players())
    except Exception:  # noqa: BLE001 — best-effort
        return []
    return []


# ---------------------------------------------------------------------------
# Per-sport scanners
# ---------------------------------------------------------------------------


def _pick_csv(proj_dir: Path, game_date: dt.date, today_fallback: str | None = None) -> Path:
    """Choose which CSV to read for a given date.

    Tries date-specific files first, then a 'today' fallback, then the most
    recently modified CSV in the directory.
    """
    if not proj_dir.exists():
        raise RuntimeError(
            f"Projection directory not found: {proj_dir}. "
            "Run the CLI scanner at least once to produce a CSV."
        )

    dated = proj_dir / f"scan_{game_date.isoformat()}.csv"
    if dated.exists():
        return dated

    if today_fallback:
        f = proj_dir / today_fallback
        if f.exists():
            return f

    csvs = [p for p in proj_dir.glob("*.csv") if p.name != "accuracy_log.csv"]
    if not csvs:
        raise RuntimeError(
            f"No scan CSVs found in {proj_dir}. Run the CLI scanner first."
        )
    return max(csvs, key=lambda p: p.stat().st_mtime)


def _scan_nba(game_date: dt.date) -> pd.DataFrame:
    """Read the most relevant NBA scan CSV produced by the CLI scanner."""
    csv_path = _pick_csv(NBA_PROJ_DIR, game_date, today_fallback="todays_automated_analysis.csv")
    df = pd.read_csv(csv_path)
    df["_source_csv"] = csv_path.name
    return df


def _scan_tennis(game_date: dt.date) -> pd.DataFrame:
    """Read the most relevant Tennis scan CSV produced by the CLI scanner."""
    csv_path = _pick_csv(TENNIS_PROJ_DIR, game_date)
    df = pd.read_csv(csv_path)
    df["_source_csv"] = csv_path.name
    return df


# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------


def _normalize(raw: Iterable[dict[str, Any]] | pd.DataFrame) -> pd.DataFrame:
    """Coerce whatever the scanner returned into a canonical DataFrame.

    Handles common column-name variations and converts EV/probabilities to
    fractions if the scanner returned percentages.
    """
    if isinstance(raw, pd.DataFrame):
        df = raw.copy()
    else:
        df = pd.DataFrame(list(raw))

    if df.empty:
        return df

    # Normalize column names — map common variants (and the bot's CLI CSV
    # schema: REC, NAME, TARGET, AI, PP, EDGE [, SURFACE, RANK]) to canonical
    # names.
    rename_map = {
        "NAME": "player",
        "player_name": "player",
        "name": "player",
        "TARGET": "stat",
        "market": "stat",
        "stat_type": "stat",
        "prop": "stat",
        "direction": "side",
        "ou": "side",
        "over_under": "side",
        "PP": "line",
        "prizepicks_line": "line",
        "pp_line": "line",
        "AI": "projection",
        "model_projection": "projection",
        "proj": "projection",
        "prediction": "projection",
        "ev": "ev_pct",
        "expected_value": "ev_pct",
        "fanduel_implied": "fd_implied_pct",
        "fd_implied": "fd_implied_pct",
        "implied_prob": "fd_implied_pct",
        "sportsbook": "book",
        "source": "book",
    }
    df = df.rename(
        columns={k: v for k, v in rename_map.items() if k in df.columns},
    )

    # Drop rows with no PrizePicks line (PP == 0 in the CSV means "no line").
    if "line" in df.columns:
        df = df[pd.to_numeric(df["line"], errors="coerce").fillna(0) > 0].copy()

    # Derive `side` and `ev_pct` from EDGE if the CSV came from the CLI scanner.
    if "EDGE" in df.columns:
        edge = pd.to_numeric(df["EDGE"], errors="coerce")
        if "side" not in df.columns or df["side"].isna().all():
            df["side"] = edge.apply(lambda e: "Over" if e > 0 else ("Under" if e < 0 else pd.NA))
        if "ev_pct" not in df.columns or df["ev_pct"].isna().all():
            line = pd.to_numeric(df["line"], errors="coerce")
            denom = line.where(line > 2.0, 2.0)
            df["ev_pct"] = (edge / denom).abs()

    # Default book if none provided — CLI CSVs only carry PrizePicks lines.
    if "book" not in df.columns or df["book"].isna().all():
        df["book"] = "PrizePicks"

    # Make sure all canonical columns exist. Use float NaN for numeric
    # columns (so downstream .round()/arithmetic works) and pd.NA elsewhere.
    _NUMERIC_CANON = {"line", "projection", "ev_pct", "fd_implied_pct"}
    for col in CANONICAL_COLUMNS:
        if col not in df.columns:
            df[col] = float("nan") if col in _NUMERIC_CANON else pd.NA

    # If EV / implied prob came back as percentages (e.g. 7.0 for 7%), convert
    # to fractions so the UI can format consistently.
    for col in ("ev_pct", "fd_implied_pct"):
        s = pd.to_numeric(df[col], errors="coerce")
        if s.notna().any() and s.abs().max() > 1.5:
            df[col] = s / 100.0

    # Force side to title-case Over/Under for display.
    if "side" in df.columns:
        df["side"] = df["side"].astype(str).str.title().replace({"Over": "Over", "Under": "Under"})

    return df
