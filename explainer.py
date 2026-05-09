"""
LLM explanation layer.

For each +EV pick produced by the XGBoost models, ask Claude to write a short
plain-English explanation of *why* the bet has positive expected value, grounded
in the model's features (projection vs. line, recent form, opponent allowed,
rest, home/away, etc.).

This is the "AI component" in the project rubric: it goes beyond the prediction
model alone by translating numeric features into reasoning a human can act on.
"""

from __future__ import annotations

import os
from typing import Any

import streamlit as st

try:
    from anthropic import Anthropic
except ImportError:  # pragma: no cover
    Anthropic = None  # type: ignore[assignment]


MODEL_NAME = "claude-haiku-4-5-20251001"
MAX_TOKENS = 220
SYSTEM_PROMPT = (
    "You are an analyst on a sports prop-betting desk. You are given a single "
    "model-recommended pick: the player, the stat, the bookmaker line, the "
    "model's projection, the EV%, and a handful of supporting features. "
    "Write a 2-3 sentence explanation of why this pick has positive expected "
    "value. Be specific and quantitative — cite the projection vs. the line, "
    "and reference the most relevant supporting feature(s). Do not hedge with "
    "generic disclaimers. Do not exceed 60 words."
)


def _get_api_key() -> str | None:
    """Read the Anthropic key from Streamlit secrets or the environment."""
    try:
        if "ANTHROPIC_API_KEY" in st.secrets:
            return st.secrets["ANTHROPIC_API_KEY"]
    except Exception:  # noqa: BLE001 — secrets file may not exist locally
        pass
    return os.environ.get("ANTHROPIC_API_KEY")


def explainer_available() -> bool:
    """True if both the SDK is importable and the API key is set."""
    return Anthropic is not None and _get_api_key() is not None


@st.cache_resource(show_spinner=False)
def _client() -> Any:
    api_key = _get_api_key()
    if Anthropic is None or api_key is None:
        raise RuntimeError("Anthropic client not configured")
    return Anthropic(api_key=api_key)


def _format_pick(pick: dict[str, Any], sport: str) -> str:
    """Pack the pick dict into a tight, model-friendly prompt body."""
    lines = [
        f"Sport: {sport}",
        f"Player: {pick.get('player', 'unknown')}",
        f"Stat / market: {pick.get('stat', 'unknown')}",
        f"Recommendation: {pick.get('side', '?')} {pick.get('line', '?')}",
        f"Model projection: {pick.get('projection', '?')}",
        f"EV: {round(float(pick.get('ev_pct', 0)) * 100, 2)}%",
        f"FanDuel implied probability: "
        f"{round(float(pick.get('fd_implied_pct', 0)) * 100, 1)}%",
    ]
    # Optional supporting features. Only include those that are present and
    # non-null to keep the prompt focused.
    extras_keys = [
        "opponent",
        "is_home",
        "rest_days",
        "season_avg",
        "last5_avg",
        "last10_avg",
        "minutes_proj",
        "opp_allowed_to_pos",
        "usage_rate",
        "pace",
        "surface",
        "rank",
        "opp_rank",
        "h2h_winrate",
    ]
    extras = {k: pick[k] for k in extras_keys if k in pick and pick[k] is not None}
    if extras:
        lines.append("Supporting features:")
        for k, v in extras.items():
            lines.append(f"  - {k}: {v}")
    return "\n".join(lines)


def explain_pick(pick: dict[str, Any], sport: str = "NBA") -> str:
    """Return a 2-3 sentence explanation for a single pick.

    Falls back to a deterministic template if the API call fails so the UI
    never breaks during the demo.
    """
    if not explainer_available():
        return _fallback(pick)

    body = _format_pick(pick, sport)
    try:
        msg = _client().messages.create(
            model=MODEL_NAME,
            max_tokens=MAX_TOKENS,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": body}],
        )
        # The SDK returns a list of content blocks; we want the text.
        parts = [
            block.text  # type: ignore[attr-defined]
            for block in msg.content
            if getattr(block, "type", None) == "text"
        ]
        text = "\n".join(parts).strip()
        return text or _fallback(pick)
    except Exception as exc:  # noqa: BLE001
        return f"_Claude call failed ({exc.__class__.__name__}). Fallback:_\n\n{_fallback(pick)}"


def _fallback(pick: dict[str, Any]) -> str:
    """Deterministic, no-API-required reasoning so the demo never breaks."""
    proj = pick.get("projection")
    line = pick.get("line")
    side = pick.get("side", "?")
    stat = pick.get("stat", "stat")
    ev = float(pick.get("ev_pct", 0)) * 100
    delta = None
    try:
        delta = float(proj) - float(line)
    except (TypeError, ValueError):
        pass
    delta_str = f"{delta:+.2f}" if delta is not None else "?"
    return (
        f"Model projects **{proj}** {stat} vs. a line of **{line}** "
        f"({delta_str}), implying a clear lean to the **{side}**. "
        f"Net of the FanDuel-implied probability, expected value is **{ev:.2f}%**."
    )
