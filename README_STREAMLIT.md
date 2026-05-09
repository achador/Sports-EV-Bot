# Sports +EV Bot — Web App

Streamlit front-end for the existing XGBoost prop-betting models, with an
Anthropic Claude layer that explains each pick in plain English.

This satisfies all three required components of the AI/ML decision tool:

| Rubric component | Where it lives |
|---|---|
| **A. ML model** | XGBoost projection models in `src/sports/{nba,tennis}/` |
| **B. AI component** | `explainer.py` — Claude generates a 2-3 sentence reasoning per pick |
| **C. Decision/output** | Ranked Over/Under recommendations with EV%, line, projection, and explanation |
| **User interaction** | `app.py` — Streamlit web UI |

## Files added by this scaffold

```
app.py                          # Streamlit entry point
adapter.py                      # Bridges UI ↔ existing scanner code
explainer.py                    # Anthropic Claude integration
requirements.txt                # Python deps (Streamlit + Anthropic + ML libs)
.streamlit/config.toml          # Theme + server settings
.streamlit/secrets.toml.example # Template for the API key (copy → secrets.toml)
.gitignore.additions            # Lines to append to .gitignore
```

## Local setup

```bash
# 1. Clone / cd into the repo
cd Sports-EV-Bot

# 2. Drop the scaffold files at the repo root (alongside main.py)

# 3. Create + activate a virtualenv
python -m venv .venv
source .venv/bin/activate    # Windows: .venv\Scripts\activate

# 4. Install deps
pip install -r requirements.txt

# 5. Set your Anthropic API key
mkdir -p .streamlit
cp .streamlit/secrets.toml.example .streamlit/secrets.toml
# Then edit .streamlit/secrets.toml and paste in your real key

# 6. Run
streamlit run app.py
```

The app opens at <http://localhost:8501>.

### Wire the adapter to your real scanner

Open `adapter.py` and check the `_scan_nba` and `_scan_tennis` functions. They
try a handful of common entry-point names (`scan`, `run_scan`, `scan_date`,
`find_ev`). If none of those match the function in your `src/sports/nba/scanner.py`
or `src/sports/tennis/scanner.py`, edit those two functions to call the right
one.

The expected return shape is a list of dicts (or a DataFrame) with these keys:

```
player, stat, side ("Over"/"Under"), line, projection,
ev_pct (fraction, 0.07 = 7%), fd_implied_pct (fraction), book
```

Optional supporting keys (used to enrich Claude's explanation):
`opponent`, `is_home`, `rest_days`, `season_avg`, `last5_avg`, `last10_avg`,
`minutes_proj`, `opp_allowed_to_pos`, `usage_rate`, `pace`,
`surface`, `rank`, `opp_rank`, `h2h_winrate`.

The adapter normalizes a bunch of common variant column names automatically —
check `adapter._normalize` if your scanner uses a different naming convention.

## Deploy to Streamlit Community Cloud (free)

1. Push the repo to GitHub (`.streamlit/secrets.toml` MUST stay gitignored).
2. Go to <https://share.streamlit.io> and sign in with GitHub.
3. Click **New app** → pick this repo, branch `main`, file path `app.py`.
4. In **Advanced settings → Secrets**, paste:
   ```
   ANTHROPIC_API_KEY = "sk-ant-..."
   ```
5. Hit **Deploy**. First boot installs the wheels (~2-3 minutes).
6. Streamlit gives you a public URL like
   `https://sports-ev-bot-<hash>.streamlit.app` — that's what you submit.

### Free-tier gotchas to know about

- Apps sleep after 7 days of inactivity; first request wakes them up (~30s).
- Memory cap is 1 GB. If your XGBoost model files are huge, consider
  loading them lazily and clearing `st.cache_resource` between sports.
- The `nba_api` package occasionally rate-limits from cloud IPs. If you see
  empty results in the deployed app but not locally, this is the cause —
  add retry/backoff in your scanner or cache yesterday's slate.

## Smoke test before the demo

```bash
# 1. App starts without errors
streamlit run app.py

# 2. Click "Find +EV bets" with default settings (NBA, today, 4% EV)
# 3. Confirm at least one pick rendered + Claude wrote an explanation
# 4. Toggle "Generate Claude explanations" off → confirm fallback text appears
# 5. Switch sport to Tennis → confirm scan still works
# 6. Test on the deployed URL from a different network (phone hotspot)
```

## Demo script (~3 min)

1. "This is a multi-sport prop betting tool — XGBoost models project player
   stats, compare to PrizePicks lines, find +EV picks."
2. Open the deployed URL. Pick **NBA**, today's date, 4% EV threshold.
3. Hit **Find +EV bets**. While it loads: "Behind the scenes we're loading
   17 trained models, pulling live lines, and projecting every market."
4. Walk through one row: "Here's the projection (XGBoost), here's the line
   (PrizePicks), here's the FD implied probability — and here's the
   recommendation with EV."
5. Expand a pick: "And this is the AI layer — Claude takes the model's
   features and writes a plain-English explanation. The reasoning cites the
   actual numbers, not generic boilerplate."
6. Switch to **Tennis**, run again, show same flow on a different sport.
