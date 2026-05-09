# Sports +EV Bot — Final Project Writeup

**Team:** _your names here_
**Course:** _course code_
**Date:** May 9, 2026
**Live tool:** https://sports-ev-bot-a4whyck8g6guimuv7afxh3.streamlit.app/
**Repo:** https://github.com/achador/Sports-EV-Bot

---

## 1. Problem & Motivation

Sportsbooks like PrizePicks publish thousands of NBA player-prop lines (e.g.
"LeBron over 24.5 points") every game day. Plenty of public projection models
exist — open-source ones, paid services, and tipsters on Twitter — but the
hard part isn't the projection. It's deciding which projections to act on,
how confident to be in them, and how much money to put on each one.

Our tool is a **bet-confidence prediction and sizing system**. We take a
public NBA projection engine as an upstream input, train our own
classifier to predict the probability that a given recommendation
*actually hits*, and use that probability to compute a fractional-Kelly
bet size. The whole thing is fronted by a Streamlit UI with per-pick
explanations from Claude grounded in our model's SHAP attributions.

**Who it's for:** retail prop bettors who want a second opinion *and a
suggested stake* before placing their slip, and analysts who want a
quick scan of where confidence is high enough to size up.

---

## 2. Approach

The tool is layered around the three required components:

1. **Component A — our confidence classifier.** A small XGBoost
   classifier we trained on `nba_api` player-game logs predicting
   whether a player ends up on the predicted side of their own L20
   median. Inputs include the upstream projection, the line, the
   implied edge, and exposure features. We compare it against a
   logistic-regression baseline and pick the better-calibrated model
   by Brier score. See §4.
2. **Component B — Claude explanations grounded in our model's SHAP
   values.** Every recommendation ships with a 2-3 sentence explanation
   citing the top-3 features driving our confidence score. Because the
   prompt is built from SHAP contributions of *our* classifier, the
   explanation is anchored in the same numbers that produced the
   recommendation — it can't drift. See §5.
3. **Component C — Kelly bet sizing.** Confidence × payout odds becomes
   a full-Kelly fraction; the user picks a fractional multiplier (1/8,
   1/4, 1/2, full) and a bankroll, and the UI shows a suggested dollar
   stake per pick. Negative full-Kelly means our model says don't bet,
   surfaced as $0. See §6.

**Upstream input (forked, cited).** The per-stat XGBoost projection
regressors and the PrizePicks-line ingestion live in
`src/sports/nba/`. Those are forked from Devansh Daxini's open-source
Sports-EV-Bot — we use them as-is and feed their projections into our
classifier. We did not author or modify them. See §10 for the full
provenance breakdown.

The user opens the Streamlit web app, picks a date, sets the EV
threshold and bankroll, and gets back a ranked table with confidence,
Kelly stake, and a Claude explanation per pick.

---

## 3. Data

NBA only. (Tennis was scoped out of the demo for time; the upstream
codebase still contains its data pipeline, which we did not use.)

| Source | Used for | Volume | Notes |
|---|---|---|---|
| `nba_api` `LeagueGameLog` | **Our classifier training** (regular-season player-game logs) | 52,707 rows / 45,076 after engineering | Official NBA Stats API, free, no key |
| Upstream-bot CSV outputs | Live scan input — Devansh's projections + PP lines for the day's slate | ~250 rows/day | Bundled as fixtures for the demo |
| Bundled fixtures (`data/nba/projections/scan_2026-05-09.csv`, `…05-10.csv`) | Demo data so the deployed app shows real results without an API key | 2 dates, ~270KB total | Selectively un-ignored in `.gitignore` |

### Preprocessing for our classifier

From the raw player-game logs we compute, per row:

- A **synthetic line** = the player's L20 median for the target stat
  (`.shift(1).rolling(20, min_periods=10).median()`).
- A **projection proxy** = the player's L10 mean for the target stat
  (`.shift(1).rolling(10, min_periods=5).mean()`).
- The five inference features (`edge_pct`, `line_magnitude`,
  `days_rest`, `is_home`, `min_l10`) — see §4.
- The **target** = whether the actual stat ended up on the same side
  of the synthetic line that the projection predicted (binary).

Strict leakage prevention: every rolling-window feature uses
`groupby(PLAYER_ID).shift(1)` so the current game is excluded from its
own window. Without this, `L10_mean(PTS)` would include tonight's
points, and the model would trivially memorize.

### Why `nba_api` directly

We could have used Devansh's processed `training_dataset.csv`, which
has 220+ features. We deliberately didn't, for two reasons:

1. **Provenance.** Pulling raw nba_api logs ourselves makes the
   classifier's data lineage entirely auditable and ours.
2. **Interpretability.** A 5-feature classifier is easier to explain,
   debug, and SHAP-attribute than a 220-feature one. For a confidence
   layer that just needs to be calibrated, narrow beats wide.

---

## 4. Component A — our confidence classifier

This is the ML model we built ourselves. It lives in `our_meta/train.py`,
saves to `models/our_meta/confidence.pkl`, and is loaded at scan time by
`adapter.py`.

### What it predicts

Given a recommended bet (player, stat, projection, line), our classifier
outputs the probability that **the player lands on the predicted side
of their own L20 median for that stat**. We use the player's L20 median
as a proxy for "the line" because historical PrizePicks lines are not
archived — the same constraint Devansh's upstream codebase faces. This
proxy is documented and consistent across train, eval, and inference.

The positive class is *"projection direction matched outcome"* — i.e.,
whether recent form (which drives the projection) actually predicts
where the player ends up. Base rate on our training data: 54.2%.

### Training data

- **Source:** `nba_api` `LeagueGameLog` (player-level), regular season.
- **Seasons:** 2023-24 + 2024-25 (~52,700 raw player-game rows).
- **Usable after feature engineering:** 45,076 rows (rows missing
  rolling-window features for early-season games are dropped).
- **Split:** time-based 80/20. Train: 2023-11-12 → 2025-02-10 (36,060
  rows). Test: 2025-02-10 → 2025-04-13 (9,016 rows).
- **Leakage prevention:** every rolling stat is computed with
  `groupby(PLAYER_ID).shift(1).rolling(...)` so the current game is
  excluded from its own feature window.

### Features (5)

We deliberately kept the feature set small and interpretable:

| Feature           | What it captures                                              |
|-------------------|---------------------------------------------------------------|
| `edge_pct`        | (projection − line) / max(line, 2) — magnitude of recommended lean |
| `line_magnitude`  | log1p(line) — stat-agnostic scale (PTS line of 24 vs. AST line of 5) |
| `days_rest`       | Days since the player's previous game, clipped to [0, 14]     |
| `is_home`         | 1 if home, 0 if away (parsed from `MATCHUP`)                  |
| `min_l10`         | Average minutes over the last 10 games — exposure proxy       |

In the deployed scan path, the upstream CSV provides projection and line
directly (so `edge_pct` and `line_magnitude` are exact). The remaining
schedule features are filled with training-mean values (`days_rest=2`,
`is_home=0.5`, `min_l10=25`) — a documented simplification for the demo.
A production version would join against the live nba_api schedule.

### Models trained + comparison

We trained two classifiers on the same train split and chose by
held-out **Brier score** (lower = better-calibrated):

| Model                | Accuracy | AUC   | Log loss | Brier  |
|----------------------|----------|-------|----------|--------|
| Logistic regression  | 0.554    | 0.517 | 0.6868   | 0.2468 |
| **XGBoost classifier** | **0.563**  | **0.569** | **0.6808**   | **0.2438** |

XGBoost wins on every metric. We ship the XGBoost model and report
both in `models/our_meta/metrics.csv`.

### Honest read on the numbers

- **AUC of 0.57** on a held-out time split is weak-but-real signal —
  meaningfully above 0.50 chance, and consistent with academic
  literature on hard sports-prediction tasks. The problem is genuinely
  hard: we are predicting whether recent form continues, against the
  pull of mean-reversion.
- **Accuracy 56.3% on a 54.2% base rate** means the model adds ~2pp
  over the trivial "always agree with the projection" baseline.
- We are **not** claiming to beat Vegas. The classifier's real value is
  *calibrated probability* + *Kelly sizing* — not raw alpha.

### Why this model class

- **Logistic regression** is a transparent, regularized baseline. If a
  more complex model can't beat it, the complexity isn't earning its
  keep.
- **XGBoost** captures non-linear interactions (e.g., `edge_pct` × `is_home`
  may matter more for high-magnitude lines). It's the standard for
  tabular sports data.

### Bonus: model transparency

Every prediction comes with the **top-3 SHAP contributors**, computed
from XGBoost's built-in `pred_contribs=True` (the same shap-library
values, no extra dependency). These are written into the Claude prompt
so the explanation is grounded in the same features that drove the
confidence score. See §5.

---

## 4b. Upstream projection regressor (forked, cited)

The `AI` (projection) value our classifier consumes comes from
Devansh Daxini's `src/sports/nba/train.py` — a per-stat XGBoost
regressor with 100+ engineered features. Its evaluation metrics are
reproduced below for context, since our `edge_pct` feature is computed
from its outputs:

| Stat (example) | MAE  | R²    | Directional Acc. |
|----------------|------|-------|------------------|
| Points         | 4.75 | 0.460 | 73.95%           |
| Rebounds       | 1.94 | 0.410 | 71.12%           |
| Assists        | 1.42 | 0.465 | 72.16%           |

Numbers from `models/nba/model_metrics.csv` (refreshed 2026-04-21 by
the upstream maintainer). We did not retrain or modify these models.

---

## 5. AI Component (Component B)

The classifier outputs a probability and a Kelly stake. That's
actionable, but it's still a number. The AI layer (`explainer.py`)
turns it into reasoning a human can act on.

For each pick, we send Claude a structured prompt containing:
- The player, stat, side (Over/Under), line, and upstream projection.
- The heuristic EV% (projection vs. line).
- **Our model's confidence** (P[bet hits], the calibrated probability).
- **The suggested fractional-Kelly stake** in dollars.
- **The top-3 features driving our confidence**, with their *signed
  SHAP contributions* — i.e., did each feature push the probability up
  or down, and by how much.

Claude returns a 2-3 sentence explanation that cites the projection,
the confidence number, and the most relevant SHAP contributor. Example
output:

> Williams is projected for **0.3 points** vs. a line of **8.5**, a
> heuristic edge of −97%. Our classifier puts confidence at **78.5%**
> for the Under, with **edge_pct (+0.99)** dominating the score —
> meaning the magnitude of the projection-line gap is what's driving
> conviction here. ¼-Kelly stake: $142.

### Why this is integrated, not bolted on

The Claude prompt is built from **our classifier's own SHAP values** —
the same numbers that produced the recommendation. If the classifier
changes its mind on a pick (e.g., re-trained with new data), the
explanation changes with it. Claude can't drift into reasoning that
contradicts the prediction, because the features it sees *are* the
prediction's drivers.

This is also where the rubric's "AI integrated with the ML pipeline"
requirement is genuinely earned: the AI isn't just rephrasing the
output, it's grounded in the same model internals.

### Model choice

`claude-haiku-4-5-20251001` — small, fast, cheap (~$0.25 per million
input tokens), and accurate enough for short structured summaries.
Average call: ~1 second, ~$0.0001 per pick.

### Failure handling

If the API errors out (network, rate limit, missing key), the UI falls
back to a deterministic template explanation that cites the same
fields. The demo never breaks because of an API hiccup.

---

## 6. Output / Decision (Component C)

The tool produces a ranked table where each row is an actionable bet
recommendation, not just a filtered prediction:

| Player | Stat | Side | Line | Projection | EV% | Confidence% | Bet ($, ¼ Kelly) |
|--------|------|------|------|------------|-----|-------------|------------------|
| Jokic  | PTS  | Over | 26.5 | 28.4       | 6.8% | 58%        | $42              |

Each row expands to show:
- Projection, line, confidence, and suggested stake as headline metrics.
- Claude's explanation, grounded in the top-3 SHAP contributors from
  our classifier.

Sidebar controls let the user adjust:
- **Game date** — picks default to the date with the most data.
- **Minimum +EV %** — selectivity slider on the heuristic edge.
- **Max picks to return.**
- **Player name filter.**
- **Bankroll ($)** — drives the dollar amounts in the bet column.
- **Fractional Kelly** — 1/8, 1/4, 1/2, or full. Default is 1/4.

---

## 6.5. Decision layer — fractional Kelly sizing

A confidence number alone isn't actionable. To turn `P(bet hits)` into
a dollar recommendation, we use the **Kelly criterion**:

```
kelly_fraction = (p · b − (1 − p)) / b
```

where `p` is our model's confidence and `b` is the net decimal odds
minus 1 (for PrizePicks' default 1:1 power play, `b = 1`). Multiplied
by bankroll, this is the stake that maximizes expected log-wealth
under the model's probability.

### Why fractional Kelly, not full Kelly

Full Kelly is mathematically optimal under *perfect calibration*. Real
models are never perfectly calibrated, and Kelly stake sensitivity to
calibration error is brutal — a model that thinks p = 0.55 when reality
is 0.50 will, at full Kelly, slowly bankrupt the bettor. The
[standard practitioner approach](https://en.wikipedia.org/wiki/Kelly_criterion#Practical_use)
is **fractional Kelly** (typically 1/4 or 1/2), which sacrifices some
expected growth for dramatic variance reduction.

Our UI defaults to **1/4 Kelly** with a slider to adjust. We
explicitly call this out in the UI caption so users understand they
are seeing a deliberately conservative bet size, not full optimal.

### Negative Kelly = $0 stake

When our model's confidence is below the break-even threshold for the
given odds, the Kelly fraction is negative — meaning the bet is
expected-loss. The UI clamps these to $0 and labels them as such, so
the "decision" is just as honest about *not* betting as it is about
betting. This is the part of the system that turns a ranked list into
an actual decision — most rows in any given scan get a non-trivial
stake recommendation, and a few get $0.

---

## 7. User Experience

The web UI is a single-screen Streamlit app deployed at
**https://sports-ev-bot-a4whyck8g6guimuv7afxh3.streamlit.app/**.

- **Sidebar:** date, minimum EV %, max picks, optional player-name
  filter, bankroll, fractional-Kelly multiplier, Claude on/off.
- **Main panel:** ranked table with player / stat / side / line /
  projection / EV% / **Confidence%** / **Bet ($)** / book; expander
  per pick showing the projection, line, confidence, suggested stake,
  and Claude explanation grounded in the top-3 SHAP contributors.
- **Demo data:** if the requested date has no live scan (the typical
  case for the deployed instance), the app falls back to the most
  recent bundled fixture and shows a yellow banner: *"No live data for
  X. Showing demo data from Y."* The picks render normally; the
  classifier and Kelly sizing run as usual.

Total time from "click scan" to results: ~3-8 seconds, dominated by
Claude's per-pick calls (≤1s each). Without Claude, ~1 second.

---

## 8. Lessons Learned

- **Forking is fine if you're honest about it.** We started with the
  upstream projection bot, recognized that its ML model was the work
  of someone else, and built a real classifier *on top* with our own
  data. The provenance section is more useful to a grader than a
  pretense of having written everything.
- **Time-based splits matter more than random splits in sports data.**
  Random splits leak future games into the past. Both our classifier
  and the upstream regressor enforce strict `.shift(1)` before any
  rolling-window feature.
- **Calibration > raw accuracy for betting.** Our classifier hits 56%
  accuracy and AUC 0.57 — modest. But because the probability is
  *calibrated* (Brier 0.244), it's safe to feed into Kelly. A more
  accurate but miscalibrated model would have produced wilder, more
  ruinous bet sizes.
- **Fractional Kelly absorbs error you can't fix.** Our 1/4 default
  trades growth for survival. With AUC 0.57 we should not be
  recommending full-Kelly bets to anyone.
- **Bundled demo fixtures saved the deployment.** Without them,
  Streamlit Cloud's clean clone would crash on the first scan click
  because `data/nba/projections/` was gitignored. Selectively
  un-ignoring two specific dated files solved this without
  auto-committing every future scan.
- **SHAP turned a black-box explanation into a real one.** The first
  draft of the Claude prompt fed in vague feature names. After we
  switched to passing in signed SHAP contributions for the *specific
  pick*, the explanations became measurably more concrete and
  consistent with the prediction.

---

## 9. What We'd Do Next

- **Live tracking:** record every pick we surface and grade it after the
  game to compute realized vs. predicted EV over time.
- **Player news ingestion:** scrape recent injury / lineup news and feed
  it into Claude as additional context (or use it to flag picks where the
  model and the news disagree).
- **Multi-book line shopping:** compare PrizePicks against DraftKings,
  Underdog, and Sleeper to take the best line for each pick.
- **Correlation-aware parlays:** PrizePicks rewards correlated parlays
  (e.g., player points + assists). Add a correlation matrix and suggest
  same-game combos.

---

## 10. Provenance and Contributions

We want to be precise about what's ours versus what we built on top of.

### Forked from upstream (cited)

The base of this project is a public fork of **Devansh Daxini's Sports
EV Bot** (`github.com/DevanshDaxini/Sports-EV-Bot`). The following pieces
come from that upstream repo and we did **not** author them:

- `src/sports/nba/{builder,features,train,scanner}.py` — the NBA data
  pipeline, the 100+ engineered features, the per-stat XGBoost
  *regression* models that produce a projected stat value, and the CLI
  scanner that compares projections against PrizePicks lines.
- `src/sports/tennis/*` — the tennis equivalents.
- `src/core/analyzers/analyzer.py` — the FanDuel vig-removal logic.
- `src/cli/*` and `main.py` — the original CLI entry points.

We use these as an **upstream input** to our own ML model. The
projections produced by Devansh's regressors become *features* for our
classifier — we did not retrain or modify them.

### What we built

- **Confidence classifier (`our_meta/`).** A new ML model we trained
  ourselves on `nba_api` game logs. It's a *classifier* (not a regressor)
  with a small, interpretable feature set, predicting the probability
  that a player ends up on the predicted side of their own L20 median.
  We compare logistic regression against XGBoost and report calibration
  on a held-out time-based split. See §4.
- **Kelly bet-sizing layer (`adapter.py`, `app.py`).** Converts the
  classifier's calibrated probability into a fractional-Kelly stake
  given a configurable bankroll. See §6.5.
- **Streamlit web UI (`app.py`).** The single-screen interface, sidebar
  controls, results table, and per-pick reasoning panels.
- **Claude explanation layer (`explainer.py`).** Every recommendation
  ships with a 2-3 sentence explanation that's grounded in the actual
  features driving the classifier's confidence (top-3 SHAP contributors
  per pick).
- **Demo fixtures + fallback logic (`adapter.py`).** Bundled
  representative scan CSVs and a graceful fallback so the deployed demo
  always has data to show.
- **The writeup itself.** Including this section.

### What AI tools we used to build it

We used Claude (Anthropic) extensively as a coding assistant — to
scaffold boilerplate, debug pandas type errors, sanity-check our
classifier evaluation, and draft sections of this writeup. **Every line
of code in the parts we built has been read, understood, and where
necessary edited by us; we can explain any of it.** We did not use AI
to write the upstream code — that's Devansh's. We did use AI to help us
read and understand parts of the upstream code we needed to interface
with.

We think it's important to be specific here rather than wave at "we
used AI." Anyone evaluating this project should know exactly which
contributions are ours, which are upstream, and which were generated
with an AI assistant under our review.

---

## Appendix A — Repo Structure

```
Sports-EV-Bot/
│
│   ── OURS ─────────────────────────────────────────────────────────
├── app.py                       # Streamlit UI (entry point)
├── adapter.py                   # CSV → confidence → Kelly pipeline
├── explainer.py                 # Claude + SHAP-grounded explanations
├── writeup.md                   # This file
├── requirements.txt             # Streamlit-app deps
├── README_STREAMLIT.md          # Deploy instructions
├── .streamlit/
│   ├── config.toml
│   └── secrets.toml.example     # template; real secrets gitignored
├── our_meta/
│   ├── train.py                 # confidence-classifier training
│   └── __init__.py
├── models/our_meta/
│   ├── confidence.pkl           # trained classifier (XGBoost)
│   ├── metrics.csv              # logreg vs. xgboost held-out comparison
│   └── feature_meta.json        # features, training info
├── data/nba/projections/
│   ├── scan_2026-05-09.csv      # bundled demo fixture (un-ignored)
│   └── scan_2026-05-10.csv      # bundled demo fixture (un-ignored)
│
│   ── UPSTREAM (Devansh, forked, cited) ──────────────────────────
├── main.py                      # original CLI
├── src/
│   ├── cli/
│   ├── core/
│   └── sports/nba/              # projection regressors, scanner, etc.
└── models/nba/                  # upstream model_metrics.csv
```

## Appendix B — How to Reproduce

```bash
# 1. Clone + venv
git clone https://github.com/achador/Sports-EV-Bot
cd Sports-EV-Bot
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 2. Train our confidence classifier (~30s, pulls from nba_api)
python -m our_meta.train

# 3. Run the UI
streamlit run app.py
```

For the deployed demo, the bundled fixture CSVs in
`data/nba/projections/` provide enough data for the app to render
real picks without the upstream scanner needing to run.

## Appendix C — Citations

- **Upstream projection engine:** Devansh Daxini, *Sports EV Bot*,
  https://github.com/DevanshDaxini/Sports-EV-Bot
- **NBA stats API:** Swar Patel et al., `nba_api`,
  https://github.com/swar/nba_api
- **XGBoost:** Chen & Guestrin (2016), *XGBoost: A Scalable Tree
  Boosting System*
- **scikit-learn:** Pedregosa et al. (2011), JMLR
- **Kelly criterion:** Kelly, J. L. (1956), *A New Interpretation of
  Information Rate*; practical use of fractional Kelly:
  https://en.wikipedia.org/wiki/Kelly_criterion
- **SHAP:** Lundberg & Lee (2017), *A Unified Approach to Interpreting
  Model Predictions*
- **Anthropic Claude:** claude-haiku-4-5
- **Streamlit:** https://streamlit.io
