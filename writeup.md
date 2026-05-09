# Sports +EV Bot — Final Project Writeup

**Team:** _your names here_
**Course:** _course code_
**Date:** May 9, 2026
**Live tool:** https://sports-ev-bot-a4whyck8g6guimuv7afxh3.streamlit.app/
**Repo:** https://github.com/achador/Sports-EV-Bot

---

## Rubric Mapping

| Rubric criterion | Points | Where it's earned in this writeup |
|---|---|---|
| A. ML Model | 30 | §3 — our XGBoost + logistic-regression confidence classifier, with held-out metrics, model comparison, time-based split |
| B. AI Component | 25 | §4 — Claude Haiku integrated via SHAP-grounded prompts; deterministic fallback for messiness |
| C. Output / Decision | 25 | §5 — fractional Kelly bet sizing turning P(hit) into a dollar stake |
| User Experience | 10 | §6 — public Streamlit Cloud deploy, sub-10s end-to-end |
| Writeup | 10 | this document, plus §7 (Provenance) and §8 (Citations) |
| Bonus — model transparency | — | SHAP top-3 per pick (§4) |
| Bonus — model comparison | — | logistic vs. XGBoost head-to-head (§3) |
| Bonus — deployed | — | live URL above |

---

## 1. Problem & Motivation

Sportsbooks publish thousands of NBA player-prop lines per night ("Jokic
over 26.5 points"). Public projection models exist, but the hard part
isn't predicting the stat — it's **deciding which projections are
trustworthy enough to bet, and how much to bet on each one**.

Our tool is a *bet-confidence prediction and sizing system*: it takes a
public NBA projection as input, predicts the probability the bet
actually hits, and computes a fractional-Kelly bet size. Every
recommendation comes with a Claude-generated explanation grounded in
the model's own SHAP attributions.

**User input → output:** the user picks a date, sets bankroll and EV
threshold, hits "Find +EV bets." The tool returns a ranked table of
recommendations with confidence %, dollar stake, and per-pick reasoning.

---

## 2. Approach

Three components, each addressing one rubric criterion:

1. **Component A — confidence classifier.** XGBoost binary classifier
   trained on `nba_api` player-game logs. Inputs include the upstream
   projection, the line, and exposure features. Output is a calibrated
   `P(bet lands on predicted side)`.
2. **Component B — Claude explanation grounded in our SHAP values.**
   Per-pick prompt includes the top-3 SHAP feature contributions from
   Component A, so the explanation is anchored in the same numbers the
   classifier used.
3. **Component C — fractional Kelly bet sizing.** Confidence + payout
   odds → Kelly fraction → dollar stake. Default ¼ Kelly to absorb
   model calibration error.

**Upstream input (forked, cited).** The per-stat projection regressors
in `src/sports/nba/` come from Devansh Daxini's open-source
Sports-EV-Bot. We use them as a feature input only; we did not author
or modify them. See §7 for the explicit provenance breakdown.

---

## 3. Component A — Confidence Classifier (30 pts)

This is the ML model we built. Code: `our_meta/train.py`. Artifact:
`models/our_meta/confidence.pkl`.

### Data source

- **`nba_api` `LeagueGameLog`** — the official NBA Stats API,
  player-level regular-season game logs. Free, no key required.
- **Seasons:** 2023-24 + 2024-25.
- **Raw rows:** 52,707. **Usable after feature engineering:** 45,076
  (rows missing rolling-window features for early-season games are
  dropped).

### Feature engineering

For each player-game, we compute:

| Feature           | Definition |
|-------------------|------------|
| `edge_pct`        | (projection − line) / max(line, 2) — magnitude of the recommended lean |
| `line_magnitude`  | log1p(line) — stat-agnostic scale handling |
| `days_rest`       | Days since previous game, clipped to [0, 14] |
| `is_home`         | 1 if home, 0 if away (parsed from `MATCHUP`) |
| `min_l10`         | Average minutes over last 10 games — exposure proxy |

**`projection`** = player's L10 mean of the target stat.
**`line`** = player's L20 median of the target stat (synthetic line —
historical PrizePicks lines aren't archived; this is the same proxy
the upstream model uses for directional accuracy).

**Leakage prevention:** every rolling-window feature uses
`groupby(PLAYER_ID).shift(1).rolling(...)`, so the current game is
excluded from its own feature window. Without this, the model would
trivially memorize tonight's stats.

### Target

Binary: **1 if the actual stat lands on the side the projection
predicted, else 0.** Base rate on training data: 0.542.

### Model selection — comparison

We trained two classifiers on the same time-based 80/20 split, then
chose the better-calibrated by Brier score:

| Model                  | Accuracy | AUC   | Log loss | **Brier**  |
|------------------------|----------|-------|----------|------------|
| Logistic regression    | 0.554    | 0.517 | 0.6868   | 0.2468     |
| **XGBoost classifier** | **0.563**  | **0.569** | **0.6808** | **0.2438** |

Numbers from `models/our_meta/metrics.csv` (held-out test split). XGBoost
wins on every metric; we ship XGBoost.

### Hyperparameters (XGBoost)

`n_estimators=300, learning_rate=0.05, max_depth=4, subsample=0.8,
colsample_bytree=0.8, min_child_weight=10, reg_lambda=1.5,
random_state=42`. Chosen for regularization (small max_depth, high
min_child_weight) — the dataset is noisy and we want a smooth,
calibrated probability.

### Why this is interesting (not just numbers)

AUC 0.569 is "weak but real" signal — meaningfully above the 0.50
chance baseline on a genuinely hard problem (predicting whether recent
form continues against the pull of mean-reversion). Accuracy 0.563 vs.
base rate 0.542 means the model adds ~2 percentage points over the
trivial baseline. We're not claiming to beat Vegas; we're producing a
*calibrated probability* that's safe to feed into Kelly sizing.

### Model transparency (bonus criterion)

For each prediction, we extract the top-3 signed SHAP feature
contributions using XGBoost's built-in `pred_contribs=True`. These are
displayed in the UI and fed to Component B's prompt, so the
explanation cites the same features the classifier used.

---

## 4. Component B — AI Explanation Layer (25 pts)

Code: `explainer.py`. Calls Anthropic's Claude Haiku.

### What the AI does

For each pick the classifier surfaces, we send Claude a structured
prompt containing:

- Player, stat, side (Over/Under), line, upstream projection.
- Heuristic EV% (projection vs. line).
- **Our model's confidence** (the calibrated probability).
- **Suggested fractional-Kelly stake** in dollars.
- **Top-3 SHAP contributors** with their signed values.

Claude returns a 2-3 sentence explanation that cites specific numbers
and the dominant SHAP contributor. Example:

> Williams is projected for **0.3 points** vs. a line of **8.5** —
> heuristic edge of −97%. Our classifier puts confidence at **78.5%**
> for the Under, with `edge_pct (+0.99 SHAP)` dominating — the
> magnitude of the projection-line gap is what's driving the score.
> ¼-Kelly stake: $142.

### Why this is "integrated, not bolted on" (rubric criterion)

The AI prompt is **built from our classifier's own SHAP values** — the
same numbers that produced the recommendation. If the classifier
changes its mind on a pick (re-trained, new features), the explanation
changes with it. Claude can't drift into reasoning that contradicts the
prediction, because the features it sees *are* the prediction's
drivers. The AI isn't rephrasing model output; it's reading the model's
internals.

### Real-world messiness handling (rubric criterion)

The AI layer is wrapped in robust failure handling:

- **Model fallback chain.** If `claude-haiku-4-5` returns
  `BadRequestError` (e.g., model unavailable), we transparently retry
  with `claude-3-5-haiku-latest`, then `claude-3-5-haiku-20241022`. The
  user never sees model selection logic.
- **Deterministic template fallback.** If every model in the chain
  fails (network, missing key, rate limit, billing), the UI falls back
  to a template explanation built from the same fields. The demo
  doesn't break — it just looks slightly less polished. The full error
  is surfaced (truncated to 200 chars) so the operator can fix it.
- **Empty-string guard.** If Claude returns whitespace, we substitute
  the template. Prevents blank cards.

### Model choice

`claude-haiku-4-5` (with date-stamped fallbacks). Small, fast (~1s per
call), inexpensive (~$0.0001 per pick at this prompt size). Adequate
for short structured summaries. We don't need Opus or Sonnet for this
task.

---

## 5. Component C — Output / Decision (25 pts)

Code: `adapter.py` and `app.py`. The decision layer turns probabilities
into dollar bets.

### The output

A ranked table where each row is an actionable recommendation:

| Player | Stat | Side  | Line | Projection | EV %  | Confidence % | Bet ($, ¼ Kelly) |
|--------|------|-------|------|------------|-------|--------------|------------------|
| LeBron James  | PA   | Under | 32.5 | 25.20      | 22.46 | 57.5 | $38 |
| Klay Thompson | FG3M | Under | 2.5  | 1.93       | 22.80 | 63.8 | $69 |
| Donovan Clingan | RA | Under | 15.0 | 11.25      | 25.00 | 68.3 | $92 |

Each row expands to show the same metrics plus Claude's per-pick
explanation.

### Data-quality filter — why the picks are realistic

The upstream scanner CSV occasionally contains rows where a player's
projection is near zero against an active PrizePicks line. This
happens when news of a player being inactive (injured, DNP-CD,
garbage-time minutes) breaks *after* PrizePicks scraped the line but
*before* the upstream projection ran. The result is a fake "96% edge"
that doesn't represent a real betting opportunity — the line would be
pulled the moment the book got the update.

We filter these out with a hard cap: **`ev_pct` must be in [4%, 25%]**.
The lower bound is the user's selectivity threshold (default 4%, slider).
The upper bound is the data-quality screen — real prop edges sit in
the low-to-mid teens, and anything claiming 25%+ is almost always a
roster-status data lag we don't want to surface.

This isn't a hyperparameter we tuned; it's a domain-knowledge decision.
A grader, an analyst, or a real bettor would all reject "edge = 96%"
out of hand.

### How ML + AI combine into a coherent answer

1. Upstream regressor produces a projection (point estimate).
2. **Our classifier** consumes (projection, line, schedule context) →
   confidence probability.
3. **Kelly formula** consumes confidence + payout odds → dollar stake.
4. **Claude** consumes confidence, stake, and SHAP top-3 → English
   explanation.

Steps 2-4 are all in *our* code path. The output is a single number
per pick (the dollar stake) plus a sentence (the explanation), both
derived from the same model's internals.

### Decision math — Kelly criterion

```
kelly_fraction = (p · b − (1 − p)) / b
```

where `p` is our confidence and `b` is net decimal odds minus 1
(PrizePicks default `b = 1`). Stake = `bankroll × kelly_multiplier ×
max(kelly_fraction, 0)`.

**Why fractional, not full Kelly.** Full Kelly is mathematically
optimal under perfect calibration — and ruinous under any miscalibration.
A model that thinks `p = 0.55` when reality is `0.50` will, at full
Kelly, slowly bankrupt the bettor. Standard practitioner approach is
fractional Kelly (¼ or ½). We default to **¼ Kelly** with a slider; the
UI caption explicitly tells users why.

**Negative Kelly = $0.** When confidence is below the break-even
threshold for the given odds, Kelly returns negative — meaning the bet
is expected-loss. The UI clamps to $0 and labels the row as
"don't bet." This is the part of the system that turns a ranked list
into an actual *decision*, not just a filter.

---

## 6. User Experience (10 pts)

**Deployed publicly:** https://sports-ev-bot-a4whyck8g6guimuv7afxh3.streamlit.app/.
No login, no install. Anyone with the link can use it.

**Workflow:**
1. Sidebar — pick a date (defaults to the densest fixture available),
   set EV threshold, max picks, bankroll, fractional-Kelly multiplier.
2. Click "Find +EV bets."
3. See the ranked table + per-pick expanders.

**Performance:** ~3-8 seconds end-to-end depending on how many Claude
explanations are requested. Without explanations: ~1 second. Well
inside the rubric's <30s budget.

**Robustness:**
- **Fallback fixtures.** Two demo CSVs are bundled so the deployed
  app works without an upstream live-scan. Clicking a date with no
  fixture shows a yellow banner: *"No live data for X. Showing demo
  data from Y."* Picks still render.
- **Empty results handling.** If no picks clear the threshold, the UI
  shows a hint to lower the threshold.
- **API failure handling.** See §4 — Claude failures degrade
  gracefully to template explanations.

---

## 7. Provenance and Contributions

We are precise about what we wrote and what we forked.

### Forked from upstream (cited)

Public fork of **Devansh Daxini's Sports EV Bot**
(`github.com/DevanshDaxini/Sports-EV-Bot`). The following files come
from upstream and we did **not** author them:

- `src/sports/nba/{builder,features,train,scanner}.py` — NBA pipeline,
  100+ engineered features, per-stat XGBoost regressors, CLI scanner.
- `src/sports/tennis/*` — tennis equivalents (not used in the demo).
- `src/core/analyzers/analyzer.py` — FanDuel vig-removal logic.
- `main.py`, `src/cli/*` — original CLI entry points.

The upstream regressor's projections become *features* for our
classifier. We did not retrain or modify the upstream models. Their
metrics (`models/nba/model_metrics.csv`, refreshed 2026-04-21) are
preserved and cited here for context: NBA Points MAE 4.75, R² 0.460,
73.95% directional accuracy.

### What we built

| File / module                          | What it does |
|----------------------------------------|--------------|
| `our_meta/train.py`                    | Confidence classifier training (this is Component A) |
| `models/our_meta/{confidence.pkl, metrics.csv, feature_meta.json}` | Trained artifact + held-out comparison |
| `adapter.py`                           | CSV ingest, normalization, confidence scoring, SHAP, Kelly |
| `app.py`                               | Streamlit UI |
| `explainer.py`                         | Claude integration with model fallback chain + deterministic template |
| `data/nba/projections/scan_2026-05-{09,10}.csv` | Bundled demo fixtures |
| `writeup.md`                           | This document |

### AI tools used in development

We used Claude (Anthropic) extensively as a coding assistant — to
scaffold boilerplate, debug pandas type errors, sanity-check our
classifier evaluation, and draft sections of this writeup. **Every
line of code in the parts we built has been read, understood, and
where necessary edited by us; we can explain any of it.** We did not
use AI to write the upstream code (that's Devansh's). We did use AI to
help understand parts of the upstream code we needed to interface with.

---

## 8. Citations

- **Upstream projection engine:** Devansh Daxini, *Sports EV Bot* —
  https://github.com/DevanshDaxini/Sports-EV-Bot
- **NBA stats API:** Swar Patel et al., `nba_api` —
  https://github.com/swar/nba_api
- **XGBoost:** Chen & Guestrin (2016), *XGBoost: A Scalable Tree Boosting System*
- **scikit-learn:** Pedregosa et al. (2011), *JMLR* 12 (logistic
  regression baseline)
- **Kelly criterion:** Kelly, J. L. (1956), *A New Interpretation of
  Information Rate*. Practical fractional Kelly:
  https://en.wikipedia.org/wiki/Kelly_criterion
- **SHAP:** Lundberg & Lee (2017), *A Unified Approach to Interpreting
  Model Predictions* — XGBoost's built-in `pred_contribs=True` produces
  the same values, no extra dependency.
- **Anthropic Claude:** `claude-haiku-4-5` (with fallback chain)
- **Streamlit:** https://streamlit.io

---

## Appendix — Reproduce Locally

```bash
git clone https://github.com/achador/Sports-EV-Bot
cd Sports-EV-Bot
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Train our classifier (~30s, pulls live from nba_api)
python -m our_meta.train

# Run the UI
streamlit run app.py
```

For the deployed demo, the bundled fixture CSVs in
`data/nba/projections/` provide enough data for the app to render real
picks without the upstream scanner needing to run.
