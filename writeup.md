# Sports +EV Bot — Final Project Writeup

**Team:** _your names here_
**Course:** _course code_
**Date:** May 2026
**Live tool:** _paste your Streamlit Cloud URL_
**Repo:** https://github.com/achador/Sports-EV-Bot

---

## 1. Problem & Motivation

Sportsbooks like PrizePicks publish thousands of player-prop lines (e.g.
"LeBron over 24.5 points") every game day. Most casual bettors evaluate these
one at a time, by feel. But every line implies a probability, and any
projection model that consistently disagrees with the book's number — by more
than the vig — is producing positive expected value (+EV).

Our tool automates that comparison across every player and every market on a
given slate, ranks the bets by EV, and uses an LLM to translate the model's
reasoning into a plain-English explanation a human can act on.

**Who it's for:** retail prop bettors who want a second opinion before placing
their slip, and analysts who want a quick scan of where the model thinks the
book has mispriced the market.

---

## 2. Approach

The tool has three layers:

1. **Projection models (XGBoost regressors)** — one per stat per sport.
2. **Bookmaker line ingestion** — live pulls from PrizePicks (lines) and
   FanDuel (odds → implied probability).
3. **EV computation + LLM explanation** — for each player/stat, compare the
   projection to the line, compute EV, then ask Claude to summarize *why*
   the bet has value.

The user opens the Streamlit web app, picks a sport and date, sets an EV
threshold, and gets back a ranked table of recommended bets, each with a
short reasoning paragraph.

---

## 3. Data

| Sport | Source | Volume | Notes |
|---|---|---|---|
| NBA | `nba_api` game logs | ~30k player-games over recent seasons | Official, free |
| Tennis | Jeff Sackmann's open match data | ~50k matches across ATP/WTA | Github repo, public domain |
| Live lines | PrizePicks API | varies by slate | Pulled at scan time |
| Live odds | FanDuel | varies by slate | Pulled at scan time |

### Preprocessing

For NBA we engineer 100+ features per player-game:

- Rolling averages (last 5, 10, season-to-date) for the target stat and
  related stats (e.g., assists when projecting points).
- Opponent-allowed stats by position (using lineup metadata).
- Usage rate, pace, minutes projection.
- Rest days, home/away split, back-to-back flag.
- Injury-adjusted teammate availability.

For Tennis we engineer 150+ features per player-match: surface, current rank,
H2H record, recent form, fatigue (matches/sets played in trailing window),
service stats, return stats.

Every model trains on a held-out time-based split (most recent ~15% of games
held out for evaluation) to avoid leakage from future games into past
predictions.

---

## 4. Models (Component A)

We train one XGBoost regressor per market: 17 for NBA (points, rebounds,
assists, threes, steals, blocks, turnovers, PRA, PR, PA, RA, etc.) and 7 for
Tennis (total games, sets, aces, double faults, etc.).

**Why XGBoost:** non-linear, handles missing values natively, robust to mixed
feature types (counts, rates, categoricals), and trains fast enough to retune
weekly.

**Hyperparameters:** tuned via cross-validation grid over `max_depth` (4–8),
`learning_rate` (0.03–0.1), `n_estimators` (200–800), and
`min_child_weight` (1–5).

### Evaluation

Each model is evaluated on the held-out time-based test split:

| Sport  | Stat (example) | MAE  | R²    | Directional Acc. |
|--------|----------------|------|-------|------------------|
| NBA    | Points         | 4.75 | 0.460 | 73.95%           |
| NBA    | Rebounds       | 1.94 | 0.410 | 71.12%           |
| NBA    | Assists        | 1.42 | 0.465 | 72.16%           |
| Tennis | Total games    | 2.99 | 0.697 | 84.02%           |
| Tennis | Aces           | 2.45 | 0.476 | 76.43%           |

We track MAE, R², and a directional-accuracy score (the percentage of test
samples where the model and the held-out actual end up on the same side of a
proxy for the prop line). Directional accuracy is more decision-relevant for
prop betting than RMSE, since the bet outcome is binary (Over/Under), so it
replaces the squared-error metric in our reporting. Numbers above come
straight from `models/{nba,tennis}/model_metrics.csv` (NBA last refreshed
2026-04-21, Tennis 2026-02-19).

We also did a sanity check by comparing model picks against actual outcomes
on a recent slate — _add hit rate here once the live grader has run for a
full week_.

---

## 5. AI Component (Component B)

The XGBoost models output a number. That's enough to compute EV, but not
enough to *justify* a bet to a human. The AI layer (`explainer.py`) closes
that gap.

For each pick, we send Claude a structured prompt containing:
- The player, stat, side (Over/Under), line, and projection.
- The computed EV% and FanDuel implied probability.
- Up to ~10 supporting features (recent form, opponent allowed, rest,
  surface, etc.).

Claude returns a 2-3 sentence explanation that cites the actual numbers and
the most relevant feature(s). Example:

> Jokic is projected for **28.4 points** vs. a line of **26.5**. He's
> averaging 29.1 over his last five and faces a Wizards defense allowing
> the 3rd-most points to opposing centers this month. EV: **6.8%**.

**Why this is integrated, not bolted on:** the explanation is grounded in
the same features the model uses, not in scraped news or generic chatter.
If the model changes its mind, the explanation changes with it — there's
no chance of the AI saying something that contradicts the prediction.

**Model choice:** `claude-haiku-4-5-20251001` — small, fast, cheap (~$0.25
per million input tokens), and accurate enough for short structured
summaries. Average call: ~1 second, ~$0.0001.

**Failure handling:** if the API errors out (network, rate limit, missing
key), the UI falls back to a deterministic template explanation so the
demo never breaks.

---

## 6. Output / Decision (Component C)

The tool produces a ranked table:

| Player | Stat | Side | Line | Projection | EV% | FD implied | Book |
|---|---|---|---|---|---|---|---|
| Jokic | Points | Over | 26.5 | 28.4 | 6.8% | 49% | PrizePicks |

Each row expands to show:
- Projection, line, and FD implied probability as headline metrics.
- Claude's explanation.

The user can adjust an **EV threshold slider** (default 4%) to control how
selective the picks are, a **max results** cap, and an optional **player
name filter** to focus on specific players.

---

## 7. User Experience

The web UI is a single-screen Streamlit app:
- Sidebar: sport, date, EV threshold, result cap, player filter, AI toggle.
- Main: status indicator while scanning, ranked table, expandable
  per-pick explanations.
- Total time from "click scan" to results: ~10–25 seconds depending on
  slate size and how many explanations the user requests.

Deployed publicly on Streamlit Community Cloud at the URL above. No login,
no install, no code required to use.

---

## 8. Lessons Learned

_Pick 3-5 honest ones from this list, edit to match your actual experience:_

- Time-based splits matter more than random splits in sports data — random
  splits inflate metrics by leaking future games into the past.
- Opponent-adjusted features moved the needle more than additional rolling
  windows. Defense matters.
- The hardest part of this project wasn't the model — it was reliably
  pulling live PrizePicks lines without getting rate-limited.
- Claude was much more useful as an explanation layer than as a
  replacement for the model. It can articulate reasoning the model can't,
  but its raw stat predictions are worse than a tuned XGBoost.
- Streamlit caching (`@st.cache_data`) cut scan latency by ~80% on
  repeated runs by avoiding re-pulling the slate.
- Failure handling matters more than feature richness. We added a
  fallback explanation template specifically because we didn't want the
  demo to break if the API call timed out.

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

## 10. Honest Notes on AI Use

We used Claude (this very same tool) to scaffold the Streamlit wrapper,
the explainer module, and the adapter layer. We wrote the XGBoost
training pipeline ourselves. Every line of code in the final repo has
been read, understood, and (where needed) edited by us — we can explain
any part of it.

---

## Appendix A — Repo Structure

```
Sports-EV-Bot/
├── app.py                    # Streamlit entry point (NEW)
├── adapter.py                # UI ↔ scanner bridge (NEW)
├── explainer.py              # Anthropic Claude integration (NEW)
├── requirements.txt          # Web app deps (NEW)
├── README_STREAMLIT.md       # Deploy instructions (NEW)
├── writeup.md                # This file (NEW)
├── .streamlit/
│   ├── config.toml
│   └── secrets.toml          # gitignored — holds ANTHROPIC_API_KEY
├── main.py                   # Original CLI entry point
└── src/
    ├── cli/
    ├── core/
    └── sports/
        ├── nba/              # NBA models, features, scanner
        └── tennis/           # Tennis models, features, scanner
```

## Appendix B — How to Reproduce

1. Clone repo, `pip install -r requirements.txt`.
2. Train (one-time, slow): run the per-sport training scripts in
   `src/sports/{nba,tennis}/train.py`. This builds the model artifacts.
3. Run locally: `streamlit run app.py`.
4. Or visit the deployed URL above.

## Appendix C — Citations

- NBA stats: `nba_api` (Swar Patel et al.), https://github.com/swar/nba_api
- Tennis data: Jeff Sackmann, https://github.com/JeffSackmann
- XGBoost: Chen & Guestrin (2016), "XGBoost: A Scalable Tree Boosting System"
- Anthropic Claude: claude-haiku-4-5
- Streamlit: https://streamlit.io
