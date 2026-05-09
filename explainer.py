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
    "upstream projection, our meta-model's confidence, the suggested Kelly "
    "stake, and the top-3 features driving our confidence (with signed SHAP "
    "contributions). Write a 2-3 sentence explanation of why this pick has "
    "positive expected value. Be specific and quantitative — cite the "
    "projection vs. the line, the confidence number, and the most-relevant "
    "SHAP contributor. Do not hedge with generic disclaimers. Do not exceed "
    "70 words."
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
    conf = pick.get("confidence")
    conf_str = f"{round(float(conf) * 100, 1)}%" if conf is not None else "n/a"
    bet_amt = pick.get("bet_size")
    bet_str = f"${int(bet_amt)}" if bet_amt is not None else "n/a"

    lines = [
        f"Sport: {sport}",
        f"Player: {pick.get('player', 'unknown')}",
        f"Stat / market: {pick.get('stat', 'unknown')}",
        f"Recommendation: {pick.get('side', '?')} {pick.get('line', '?')}",
        f"Upstream projection: {pick.get('projection', '?')}",
        f"Heuristic EV (proj vs. line): "
        f"{round(float(pick.get('ev_pct', 0)) * 100, 2)}%",
        f"Our meta-model confidence (P[bet hits]): {conf_str}",
        f"Suggested fractional-Kelly stake: {bet_str}",
    ]

    shap_top = pick.get("shap_top3")
    if shap_top:
        lines.append("Top-3 features driving confidence (signed SHAP contribution):")
        try:
            for name, contrib in shap_top:
                lines.append(f"  - {name}: {contrib:+.3f}")
        except Exception:  # noqa: BLE001 — defensive against shape changes
            pass

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
