"""
Confidence Classifier — Component A (ours, not forked).

Trained on player-game logs pulled from nba_api. Predicts the probability
that a player ends up on the same side of their own L20 median as the
side suggested by their L10 mean — i.e., whether recent hot/cold form
predicts continued out- or under-performance.

This is intentionally a *small, interpretable* model on a *narrow*
feature set. It is meant to live downstream of the upstream XGBoost
projection regressors (Devansh's, in `src/sports/nba/`), not to replace
them. The output is a calibrated probability the system uses for
fractional-Kelly bet sizing.

Why a synthetic L20-median target instead of historical PrizePicks
lines? Because historical PrizePicks lines aren't archived — same
constraint the upstream codebase faces. Player-specific median is a
documented, defensible proxy.

Outputs:
    models/our_meta/confidence.pkl      — picked classifier (better Brier)
    models/our_meta/metrics.csv         — both classifiers' eval metrics
    models/our_meta/feature_meta.json   — feature list, training info

Usage:
    $ python -m our_meta.train
"""

from __future__ import annotations

import json
import os
import pickle
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler

REPO_ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = REPO_ROOT / "models" / "our_meta"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

SEASONS = ["2023-24", "2024-25"]
TARGET_STAT = "PTS"

FEATURES = [
    "edge_pct",       # (projection - line) / max(line, 2)
    "line_magnitude", # log1p(line) — stat-agnostic scale
    "days_rest",      # days since previous game (clipped to [0, 14])
    "is_home",        # 1 if home, 0 if away
    "min_l10",        # average minutes last 10 games — exposure proxy
]


def fetch_player_logs() -> pd.DataFrame:
    """Pull player-level game logs for SEASONS via nba_api."""
    from nba_api.stats.endpoints import leaguegamelog

    frames = []
    for season in SEASONS:
        print(f"  fetching {season}...", flush=True)
        t0 = time.time()
        gl = leaguegamelog.LeagueGameLog(
            league_id="00",
            season=season,
            season_type_all_star="Regular Season",
            player_or_team_abbreviation="P",
        )
        df = gl.get_data_frames()[0]
        print(f"    -> {len(df):,} rows in {time.time()-t0:.1f}s", flush=True)
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Engineer the feature set + target.

    Strict leakage prevention: all rolling stats use `.shift(1)` so the
    current game is excluded from its own feature window.
    """
    df = df.copy()
    df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"])
    df = df.sort_values(["PLAYER_ID", "GAME_DATE"]).reset_index(drop=True)

    df["is_home"] = df["MATCHUP"].str.contains(" vs. ").astype(int)

    # days_rest: gap from previous game per player, in days
    df["prev_date"] = df.groupby("PLAYER_ID")["GAME_DATE"].shift(1)
    df["days_rest"] = (df["GAME_DATE"] - df["prev_date"]).dt.days.clip(0, 14)

    grouped = df.groupby("PLAYER_ID", group_keys=False)

    # L10 mean of target stat = our "projection" proxy
    df["projection"] = grouped[TARGET_STAT].transform(
        lambda x: x.shift(1).rolling(10, min_periods=5).mean()
    )

    # L20 median of target stat = our "line" proxy
    df["line"] = grouped[TARGET_STAT].transform(
        lambda x: x.shift(1).rolling(20, min_periods=10).median()
    )

    df["edge_pct"] = (df["projection"] - df["line"]) / df["line"].clip(lower=2.0)
    df["line_magnitude"] = np.log1p(df["line"].clip(lower=0))

    # exposure proxy: minutes recent
    df["min_l10"] = grouped["MIN"].transform(
        lambda x: x.shift(1).rolling(10, min_periods=5).mean()
    )

    # target: did the actual outcome land on the side the projection predicted?
    pred_over = df["projection"] > df["line"]
    actual_over = df[TARGET_STAT] > df["line"]
    df["target"] = (pred_over == actual_over).astype(int)

    # drop rows missing any feature or the target
    needed = FEATURES + ["target", "GAME_DATE"]
    df = df.dropna(subset=needed).reset_index(drop=True)

    # drop rows where projection == line exactly (no signal — sign is degenerate)
    df = df[df["projection"] != df["line"]].reset_index(drop=True)

    return df


def time_split(df: pd.DataFrame, test_frac: float = 0.20):
    df = df.sort_values("GAME_DATE").reset_index(drop=True)
    split = int(len(df) * (1 - test_frac))
    return df.iloc[:split], df.iloc[split:]


def evaluate(name, model, X, y, scaler=None):
    Xp = X if scaler is None else scaler.transform(X)
    proba = model.predict_proba(Xp)[:, 1]
    pred = (proba >= 0.5).astype(int)
    return {
        "model": name,
        "accuracy": float(accuracy_score(y, pred)),
        "auc": float(roc_auc_score(y, proba)),
        "log_loss": float(log_loss(y, np.clip(proba, 1e-6, 1 - 1e-6))),
        "brier": float(brier_score_loss(y, proba)),
        "n": int(len(y)),
        "base_rate": float(y.mean()),
    }


def main():
    print("=" * 60)
    print("  CONFIDENCE CLASSIFIER TRAINING")
    print("=" * 60)

    print("[1/4] Fetching player game logs from nba_api...")
    raw = fetch_player_logs()
    print(f"      total rows: {len(raw):,}")

    print("[2/4] Engineering features + target...")
    df = build_features(raw)
    print(f"      usable rows: {len(df):,}")
    print(f"      base rate (target=1): {df['target'].mean():.3f}")

    print("[3/4] Time-based 80/20 split + training both classifiers...")
    train_df, test_df = time_split(df, test_frac=0.20)
    X_train = train_df[FEATURES].values
    y_train = train_df["target"].values
    X_test = test_df[FEATURES].values
    y_test = test_df["target"].values
    print(f"      train: {len(train_df):,}  test: {len(test_df):,}")
    print(f"      train dates: {train_df['GAME_DATE'].min().date()} → {train_df['GAME_DATE'].max().date()}")
    print(f"      test  dates: {test_df['GAME_DATE'].min().date()}  → {test_df['GAME_DATE'].max().date()}")

    # Logistic regression
    scaler = StandardScaler().fit(X_train)
    logreg = LogisticRegression(max_iter=1000, C=1.0)
    logreg.fit(scaler.transform(X_train), y_train)

    # XGBoost classifier
    xgbc = xgb.XGBClassifier(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=4,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=10,
        reg_lambda=1.5,
        eval_metric="logloss",
        random_state=42,
        n_jobs=-1,
    )
    xgbc.fit(X_train, y_train)

    metrics = [
        evaluate("logistic_regression", logreg, X_test, y_test, scaler),
        evaluate("xgboost_classifier", xgbc, X_test, y_test),
    ]

    print()
    print(f"  {'model':<22} {'acc':>6} {'auc':>6} {'logloss':>8} {'brier':>7}")
    print("  " + "-" * 52)
    for m in metrics:
        print(f"  {m['model']:<22} {m['accuracy']:>6.3f} {m['auc']:>6.3f} {m['log_loss']:>8.4f} {m['brier']:>7.4f}")

    # Pick the better-calibrated model (lower Brier wins)
    winner = min(metrics, key=lambda m: m["brier"])
    winner_name = winner["model"]
    print(f"\n  → winner (lowest Brier): {winner_name}")

    print("[4/4] Saving artifacts...")
    if winner_name == "logistic_regression":
        artifact = {"kind": "logistic", "model": logreg, "scaler": scaler, "features": FEATURES}
    else:
        artifact = {"kind": "xgboost", "model": xgbc, "scaler": None, "features": FEATURES}
    with open(MODEL_DIR / "confidence.pkl", "wb") as f:
        pickle.dump(artifact, f)

    pd.DataFrame(metrics).to_csv(MODEL_DIR / "metrics.csv", index=False)

    meta = {
        "trained_at": datetime.now().isoformat(timespec="seconds"),
        "seasons": SEASONS,
        "target_stat": TARGET_STAT,
        "features": FEATURES,
        "winner": winner_name,
        "train_rows": int(len(train_df)),
        "test_rows": int(len(test_df)),
        "test_metrics": {m["model"]: {k: v for k, v in m.items() if k != "model"} for m in metrics},
    }
    with open(MODEL_DIR / "feature_meta.json", "w") as f:
        json.dump(meta, f, indent=2, default=str)

    print(f"  ✓ saved {MODEL_DIR / 'confidence.pkl'}")
    print(f"  ✓ saved {MODEL_DIR / 'metrics.csv'}")
    print(f"  ✓ saved {MODEL_DIR / 'feature_meta.json'}")


if __name__ == "__main__":
    main()
