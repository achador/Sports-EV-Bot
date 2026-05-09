"""
Sports +EV Bot — Streamlit web UI

Wraps the existing XGBoost projection models in src/sports/{nba,tennis} with a
web interface and an LLM-powered explanation layer (Anthropic Claude).

Drop this file at the repo root, alongside `main.py`. Run locally with:
    streamlit run app.py
"""

from __future__ import annotations

import datetime as dt
import os
import traceback
from typing import Any

import pandas as pd
import streamlit as st

from adapter import (
    SUPPORTED_SPORTS,
    list_players,
    run_scan,
)
from explainer import explain_pick, explainer_available

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Sports +EV Bot",
    page_icon="🏀",
    layout="wide",
)

st.title("🏀🎾 Sports +EV Bot")
st.caption(
    "XGBoost projections vs. PrizePicks lines and FanDuel odds, "
    "with Claude-generated reasoning for every recommendation."
)

# ---------------------------------------------------------------------------
# Sidebar — controls
# ---------------------------------------------------------------------------

with st.sidebar:
    st.header("Scan settings")

    sport = st.selectbox(
        "Sport",
        options=SUPPORTED_SPORTS,
        index=0,
        help="Which sport's models to use.",
    )

    target_date = st.date_input(
        "Game date",
        value=dt.date.today(),
        help="Date of the games to scan.",
    )

    ev_threshold = st.slider(
        "Minimum +EV (%)",
        min_value=0.0,
        max_value=20.0,
        value=4.0,
        step=0.5,
        help="Only show picks with EV at or above this percentage.",
    )

    max_results = st.slider(
        "Max picks to return",
        min_value=1,
        max_value=50,
        value=15,
    )

    player_filter = st.text_input(
        "Filter by player (optional)",
        value="",
        placeholder="e.g. Jokic",
    )

    explain_picks = st.checkbox(
        "Generate Claude explanations",
        value=True,
        help="Calls the Anthropic API once per pick. Adds ~1s/pick.",
    )

    if explain_picks and not explainer_available():
        st.warning(
            "ANTHROPIC_API_KEY not found. "
            "Add it to `.streamlit/secrets.toml` or your environment to enable explanations."
        )

    run_button = st.button("🔍 Find +EV bets", type="primary", use_container_width=True)

    with st.expander("ℹ️ About this tool"):
        st.markdown(
            """
            **A. ML model:** XGBoost regressors trained on historical
            `nba_api` game logs (NBA, 17 stats) and Jeff Sackmann's open
            tennis data (Tennis, 7 markets), with 100+ engineered
            features per player.

            **B. AI component:** Anthropic Claude takes each model
            projection plus its supporting features and writes a short
            human-readable explanation of *why* the bet has positive EV.

            **C. Output:** A ranked table of Over/Under recommendations
            with EV%, the bookmaker line, the model's projection, and
            Claude's reasoning. Bets are sorted by EV.
            """
        )

# ---------------------------------------------------------------------------
# Main panel
# ---------------------------------------------------------------------------

if not run_button:
    st.info(
        "👈 Configure your scan in the sidebar and hit **Find +EV bets** to start."
    )
    st.markdown("### How it works")
    st.markdown(
        """
        1. Pick a sport and date.
        2. We pull live PrizePicks lines and FanDuel odds.
        3. For each player/stat, our XGBoost model produces a projection.
        4. We compute expected value vs. the bookmaker's implied probability.
        5. Claude writes a 2-sentence explanation grounded in the model's features.
        """
    )
    st.stop()

# --- Run the scan ----------------------------------------------------------

with st.status("Running scan…", expanded=True) as status:
    try:
        st.write(f"Loading {sport} models and pulling lines for {target_date.isoformat()}…")
        picks_df: pd.DataFrame = run_scan(
            sport=sport,
            game_date=target_date,
            ev_threshold=ev_threshold / 100.0,
            max_results=max_results,
            player_filter=player_filter or None,
        )
        st.write(f"Found **{len(picks_df)}** picks above the {ev_threshold:.1f}% EV threshold.")
        status.update(label="Scan complete", state="complete")
    except Exception as exc:  # noqa: BLE001
        status.update(label="Scan failed", state="error")
        st.error(f"Scan failed: {exc}")
        with st.expander("Traceback"):
            st.code(traceback.format_exc())
        st.stop()

if picks_df.empty:
    st.warning(
        "No picks cleared the EV threshold. Try lowering the threshold "
        "or picking a different date."
    )
    st.stop()

# --- Results table ---------------------------------------------------------

st.subheader("Top picks")
display_cols = [
    "player",
    "stat",
    "side",
    "line",
    "projection",
    "ev_pct",
    "fd_implied_pct",
    "book",
]
display_df = picks_df[[c for c in display_cols if c in picks_df.columns]].copy()
if "ev_pct" in display_df.columns:
    display_df["ev_pct"] = (display_df["ev_pct"] * 100).round(2)
if "fd_implied_pct" in display_df.columns:
    display_df["fd_implied_pct"] = (display_df["fd_implied_pct"] * 100).round(1)
if "projection" in display_df.columns:
    display_df["projection"] = display_df["projection"].round(2)

st.dataframe(display_df, use_container_width=True, hide_index=True)

# --- Per-pick explanations -------------------------------------------------

st.subheader("Reasoning for each pick")

for _, pick in picks_df.iterrows():
    pick_dict: dict[str, Any] = pick.to_dict()
    header = (
        f"**{pick_dict.get('player', '?')}** — "
        f"{pick_dict.get('side', '?')} {pick_dict.get('line', '?')} "
        f"{pick_dict.get('stat', '?')}  "
        f"(EV {pick_dict.get('ev_pct', 0) * 100:.2f}%)"
    )
    with st.expander(header, expanded=False):
        col1, col2 = st.columns([1, 2])
        with col1:
            st.metric("Projection", f"{pick_dict.get('projection', float('nan')):.2f}")
            st.metric("Line", f"{pick_dict.get('line', float('nan'))}")
            st.metric(
                "FD implied %",
                f"{pick_dict.get('fd_implied_pct', 0) * 100:.1f}%",
            )
        with col2:
            if explain_picks and explainer_available():
                with st.spinner("Asking Claude…"):
                    explanation = explain_pick(pick_dict, sport=sport)
                st.markdown(explanation)
            else:
                st.markdown(
                    "_Explanations disabled. Enable in the sidebar to see Claude's reasoning._"
                )

# --- Footer ----------------------------------------------------------------

st.divider()
st.caption(
    "⚠️ For educational use only. Bets carry risk; the model's projections "
    "are estimates, not guarantees. Do not bet more than you can afford to lose."
)
