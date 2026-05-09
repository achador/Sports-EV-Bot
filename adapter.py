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

import numpy as np
import pandas as pd
import streamlit as st


SUPPORTED_SPORTS = ["NBA"]

REPO_ROOT = Path(__file__).resolve().parent
NBA_PROJ_DIR = REPO_ROOT / "data" / "nba" / "projections"
TENNIS_PROJ_DIR = REPO_ROOT / "data" / "tennis" / "projections"


def _projection_dir_for(sport: str) -> Path:
    if sport == "NBA":
        return NBA_PROJ_DIR
    if sport == "Tennis":
        return TENNIS_PROJ_DIR
    raise ValueError(f"Unsupported sport: {sport!r}")


def _date_from_fixture_name(p: Path) -> dt.date | None:
    """Extract YYYY-MM-DD from a `scan_YYYY-MM-DD.csv` filename, or None."""
    name = p.stem
    if name.startswith("scan_"):
        try:
            return dt.date.fromisoformat(name[len("scan_"):])
        except ValueError:
            return None
    return None


def best_fixture_date(sport: str) -> dt.date | None:
    """Return the fixture date whose CSV has the most data (largest file).

    Used by app.py to set a default for the date picker so that a casual
    user clicking 'Find +EV bets' immediately sees a populated table — not
    the most-recent date (which can be a thin day-ahead scan with no
    picks above the EV threshold).
    """
    proj_dir = _projection_dir_for(sport)
    if not proj_dir.exists():
        return None
    candidates: list[tuple[int, dt.date]] = []
    for p in proj_dir.glob("scan_*.csv"):
        d = _date_from_fixture_name(p)
        if d is not None:
            candidates.append((p.stat().st_size, d))
    if not candidates:
        return None
    return max(candidates, key=lambda t: t[0])[1]


# Back-compat alias (older imports may still reference the old name).
latest_fixture_date = best_fixture_date

# Columns we always want in the output, in display order.
CANONICAL_COLUMNS = [
    "player",
    "stat",
    "side",
    "line",
    "projection",
    "ev_pct",
    "fd_implied_pct",
    "confidence",     # P(bet hits) from our_meta confidence classifier
    "kelly_fraction", # fraction of bankroll suggested (full Kelly, signed)
    "book",
]


# Training-mean fallbacks for features the saved scan CSV doesn't carry.
# Pulled from `models/our_meta/feature_meta.json` after training.
_FEATURE_FALLBACKS = {
    "days_rest": 2.0,
    "is_home": 0.5,
    "min_l10": 25.0,
}

_CONFIDENCE_ARTIFACT = None


def _load_confidence_artifact():
    """Lazy-load the pickled confidence classifier (kind, model, scaler, features)."""
    global _CONFIDENCE_ARTIFACT
    if _CONFIDENCE_ARTIFACT is not None:
        return _CONFIDENCE_ARTIFACT
    import pickle

    pkl = REPO_ROOT / "models" / "our_meta" / "confidence.pkl"
    if not pkl.exists():
        _CONFIDENCE_ARTIFACT = {"missing": True}
        return _CONFIDENCE_ARTIFACT
    with open(pkl, "rb") as f:
        _CONFIDENCE_ARTIFACT = pickle.load(f)
    _CONFIDENCE_ARTIFACT["missing"] = False
    return _CONFIDENCE_ARTIFACT


def _build_confidence_features(df: pd.DataFrame) -> pd.DataFrame | None:
    """Build the 5-feature input matrix the classifier expects from a normalized scan df.

    Returns None if the dataframe lacks `projection` and `line`; otherwise returns
    a DataFrame with the canonical feature columns (training-mean fallback for
    schedule features the CSV doesn't carry).
    """
    art = _load_confidence_artifact()
    if art.get("missing"):
        return None
    if "projection" not in df.columns or "line" not in df.columns:
        return None

    proj = pd.to_numeric(df["projection"], errors="coerce")
    line = pd.to_numeric(df["line"], errors="coerce")
    edge_pct = (proj - line) / line.clip(lower=2.0)

    feats = pd.DataFrame({
        "edge_pct": edge_pct,
        "line_magnitude": np.log1p(line.clip(lower=0)),
        "days_rest": _FEATURE_FALLBACKS["days_rest"],
        "is_home": _FEATURE_FALLBACKS["is_home"],
        "min_l10": _FEATURE_FALLBACKS["min_l10"],
    })
    return feats[art["features"]]  # ensure correct column order


def _attach_confidence(df: pd.DataFrame) -> pd.DataFrame:
    """Add `confidence` column = P(bet lands on predicted side) from our model.

    The saved model's positive class is "projection direction matched outcome",
    which directly answers "does the side this row recommends actually hit?".
    """
    if df.empty:
        df["confidence"] = pd.NA
        return df

    art = _load_confidence_artifact()
    feats = _build_confidence_features(df)
    if feats is None:
        df["confidence"] = pd.NA
        return df

    if art["kind"] == "logistic":
        X = art["scaler"].transform(feats.values)
    else:
        X = feats.values
    proba = art["model"].predict_proba(X)[:, 1]
    df = df.copy()
    df["confidence"] = proba
    return df


def _attach_shap_topk(df: pd.DataFrame, k: int = 3) -> pd.DataFrame:
    """Add `shap_top3` column listing the top-k feature contributions to confidence.

    Uses XGBoost's built-in `pred_contribs=True` (the same SHAP values shap.Explainer
    would compute, but without the extra dependency). For logistic regression, falls
    back to coefficient × standardized feature value as the contribution.
    """
    if df.empty:
        df["shap_top3"] = pd.NA
        return df

    art = _load_confidence_artifact()
    feats = _build_confidence_features(df)
    if feats is None:
        df["shap_top3"] = pd.NA
        return df

    feature_names = art["features"]
    if art["kind"] == "xgboost":
        import xgboost as xgb

        booster = art["model"].get_booster()
        dmat = xgb.DMatrix(feats.values, feature_names=feature_names)
        contribs = booster.predict(dmat, pred_contribs=True)
        # Last column is the bias term — drop it.
        feature_contribs = contribs[:, :-1]
    else:
        # Logistic regression: contribution = coef * standardized_feature
        scaler = art["scaler"]
        z = scaler.transform(feats.values)
        coef = art["model"].coef_[0]
        feature_contribs = z * coef

    top_lists: list[list[tuple[str, float]]] = []
    for row in feature_contribs:
        idx = np.argsort(-np.abs(row))[:k]
        top_lists.append([(feature_names[i], float(row[i])) for i in idx])

    df = df.copy()
    df["shap_top3"] = top_lists
    return df


def _attach_kelly(df: pd.DataFrame, payout_decimal: float = 2.0) -> pd.DataFrame:
    """Add `kelly_fraction` = full-Kelly fraction of bankroll given our `confidence`.

    Assumes `payout_decimal=2.0` (1:1 even-money payout, the default for
    PrizePicks 2-pick power plays). Negative kelly_fraction means the model
    advises against the bet (P(win) below break-even).
    """
    if df.empty or "confidence" not in df.columns:
        df["kelly_fraction"] = pd.NA
        return df
    p = pd.to_numeric(df["confidence"], errors="coerce")
    b = payout_decimal - 1.0  # net odds; for 2.0 decimal odds, b=1.0
    kelly = (p * b - (1.0 - p)) / b
    df = df.copy()
    df["kelly_fraction"] = kelly.clip(lower=-1.0, upper=1.0)
    return df


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
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Run a +EV scan and return (ranked DataFrame, scan_info).

    `scan_info` carries fallback metadata so the UI can warn the user when
    we substituted demo data for an unavailable date:
        - is_fallback: bool — True if game_date had no fixture
        - data_date: date | None — the actual date of the CSV used
        - requested_date: date — what the user asked for
        - source_csv: str — basename of the CSV used
    """
    sport = sport.strip()
    if sport == "NBA":
        raw, info = _scan_nba(game_date)
    elif sport == "Tennis":
        raw, info = _scan_tennis(game_date)
    else:
        raise ValueError(f"Unsupported sport: {sport!r}")

    df = _normalize(raw)
    if df.empty:
        return df, info

    df = df[df["ev_pct"] >= ev_threshold]

    # Sanity cap — drop picks with implausibly large edges. Real prop
    # edges are 2-15%. An "edge" of 90%+ almost always means the
    # upstream projection model assigned a near-zero projection because
    # the player is inactive (injured, garbage time, DNP-CD), while
    # PrizePicks' line was scraped before news broke. Books like PP
    # would never publish those lines once roster status updates, so
    # surfacing them produces obviously-bogus recommendations. We cap
    # at 25% to keep only realistic prop opportunities.
    MAX_REALISTIC_EV = 0.25
    df = df[df["ev_pct"] <= MAX_REALISTIC_EV]

    if player_filter:
        df = df[df["player"].str.contains(player_filter, case=False, na=False)]

    df = df.sort_values("ev_pct", ascending=False).head(max_results).reset_index(drop=True)
    df = _attach_confidence(df)
    df = _attach_shap_topk(df, k=3)
    df = _attach_kelly(df)
    return df, info


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


def _pick_csv(proj_dir: Path, game_date: dt.date) -> tuple[Path, bool, dt.date | None]:
    """Choose which CSV to read for a given date.

    Returns (path, is_fallback, data_date). Tries `scan_<date>.csv` first;
    if absent, falls back to the *largest* `scan_*.csv` in the directory
    (same heuristic as the default-date picker — gives the demo the
    densest data when the requested date isn't available).
    """
    if not proj_dir.exists():
        raise RuntimeError(
            f"Projection directory not found: {proj_dir}. "
            "Run the CLI scanner at least once to produce a CSV."
        )

    exact = proj_dir / f"scan_{game_date.isoformat()}.csv"
    if exact.exists():
        return exact, False, game_date

    sized: list[tuple[int, dt.date, Path]] = []
    for p in proj_dir.glob("scan_*.csv"):
        d = _date_from_fixture_name(p)
        if d is not None:
            sized.append((p.stat().st_size, d, p))

    if not sized:
        raise RuntimeError(
            f"No fixture CSVs found in {proj_dir} (expected scan_YYYY-MM-DD.csv). "
            "Run the CLI scanner first or bundle a demo fixture."
        )

    _, fallback_date, fallback_path = max(sized, key=lambda t: t[0])
    return fallback_path, True, fallback_date


def _scan_nba(game_date: dt.date) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Load NBA scan CSV (date-specific if available, else most-recent fixture)."""
    csv_path, is_fallback, data_date = _pick_csv(NBA_PROJ_DIR, game_date)
    df = pd.read_csv(csv_path)
    df["_source_csv"] = csv_path.name
    info = {
        "is_fallback": is_fallback,
        "data_date": data_date,
        "requested_date": game_date,
        "source_csv": csv_path.name,
    }
    return df, info


def _scan_tennis(game_date: dt.date) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Load Tennis scan CSV (date-specific if available, else most-recent fixture)."""
    csv_path, is_fallback, data_date = _pick_csv(TENNIS_PROJ_DIR, game_date)
    df = pd.read_csv(csv_path)
    df["_source_csv"] = csv_path.name
    info = {
        "is_fallback": is_fallback,
        "data_date": data_date,
        "requested_date": game_date,
        "source_csv": csv_path.name,
    }
    return df, info


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
    _NUMERIC_CANON = {"line", "projection", "ev_pct", "fd_implied_pct",
                      "confidence", "kelly_fraction"}
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
