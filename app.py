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
    latest_fixture_date,
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

    _default_date = latest_fixture_date(sport) or dt.date.today()
    target_date = st.date_input(
        "Game date",
        value=_default_date,
        help="Date of the games to scan. The default is the most recent date "
             "with bundled demo data.",
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

    st.divider()
    st.subheader("Bet sizing")
    bankroll = st.number_input(
        "Bankroll ($)",
        min_value=10,
        max_value=100_000,
        value=1000,
        step=50,
        help="Used to compute the Kelly bet size for each pick.",
    )
    kelly_multiplier = st.select_slider(
        "Fractional Kelly",
        options=["1/8", "1/4", "1/2", "Full"],
        value="1/4",
        help="Real bettors use fractional Kelly to absorb model calibration "
             "error. 1/4 Kelly is standard. Full Kelly is mathematically "
             "optimal under perfect calibration but ruinous under any error.",
    )
    _kelly_lookup = {"1/8": 0.125, "1/4": 0.25, "1/2": 0.5, "Full": 1.0}
    kelly_scale = _kelly_lookup[kelly_multiplier]

    st.divider()
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
        picks_df, scan_info = run_scan(
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

if scan_info.get("is_fallback"):
    fallback_date = scan_info.get("data_date")
    fallback_str = fallback_date.isoformat() if fallback_date else "an earlier date"
    st.warning(
        f"No live data for {target_date.isoformat()}. "
        f"Showing demo data from **{fallback_str}** so you can still explore the tool."
    )

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
    "confidence",
    "bet_size",
    "book",
]

picks_df = picks_df.copy()
# Apply fractional-Kelly multiplier and convert to dollar amount.
# Negative kelly = model says don't bet → display as $0.
if "kelly_fraction" in picks_df.columns:
    sized = pd.to_numeric(picks_df["kelly_fraction"], errors="coerce").clip(lower=0)
    picks_df["bet_size"] = (sized * kelly_scale * bankroll).round(0)
else:
    picks_df["bet_size"] = 0

display_df = picks_df[[c for c in display_cols if c in picks_df.columns]].copy()
if "ev_pct" in display_df.columns:
    display_df["ev_pct"] = (display_df["ev_pct"] * 100).round(2)
if "confidence" in display_df.columns:
    display_df["confidence"] = (pd.to_numeric(display_df["confidence"], errors="coerce") * 100).round(1)
if "projection" in display_df.columns:
    display_df["projection"] = pd.to_numeric(display_df["projection"], errors="coerce").round(2)
if "bet_size" in display_df.columns:
    display_df["bet_size"] = display_df["bet_size"].astype("Int64")

# Rename columns for nicer display headers
display_df = display_df.rename(columns={
    "ev_pct": "EV %",
    "confidence": "Confidence %",
    "bet_size": f"Bet ($, {kelly_multiplier} Kelly)",
})

st.dataframe(display_df, use_container_width=True, hide_index=True)
st.caption(
    f"**Confidence %** is from our meta-classifier (XGBoost on `nba_api` "
    f"player-game logs, AUC 0.57 on held-out). **Bet size** = "
    f"{kelly_multiplier} of full Kelly fraction × ${bankroll} bankroll. "
    "Suggested $0 means our model says don't bet."
)

# --- Per-pick explanations -------------------------------------------------

st.subheader("Reasoning for each pick")

for _, pick in picks_df.iterrows():
    pick_dict: dict[str, Any] = pick.to_dict()
    conf_pct = (pick_dict.get("confidence") or 0) * 100
    bet_amt = pick_dict.get("bet_size") or 0
    header = (
        f"**{pick_dict.get('player', '?')}** — "
        f"{pick_dict.get('side', '?')} {pick_dict.get('line', '?')} "
        f"{pick_dict.get('stat', '?')}  "
        f"(Confidence {conf_pct:.0f}% · Bet ${bet_amt:.0f})"
    )
    with st.expander(header, expanded=False):
        col1, col2 = st.columns([1, 2])
        with col1:
            st.metric("Projection", f"{pick_dict.get('projection', float('nan')):.2f}")
            st.metric("Line", f"{pick_dict.get('line', float('nan'))}")
            st.metric("Confidence", f"{conf_pct:.1f}%")
            st.metric(f"Suggested bet ({kelly_multiplier} Kelly)", f"${bet_amt:.0f}")
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
