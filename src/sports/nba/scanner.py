"""
NBA Props Scanner - AI-Powered Prediction System

Scans upcoming NBA games, generates player performance predictions using
trained XGBoost models, and identifies profitable betting opportunities
by comparing predictions against PrizePicks lines.

Usage:
    $ python3 -m src.sports.nba.scanner
"""

import pandas as pd
import xgboost as xgb
import os
import sys
import time
import warnings
import unicodedata
import re
from datetime import datetime, timedelta
import requests

from nba_api.stats.endpoints import ScoreboardV2, LeagueGameLog

from src.core.odds_providers.prizepicks import PrizePicksClient
from src.sports.nba.config   import STAT_MAP, MODEL_QUALITY, ACTIVE_TARGETS, ABSORPTION_RATES
from src.sports.nba.injuries import get_injury_report

# --- CONFIGURATION ---
# scanner.py lives at src/sports/nba/scanner.py → root is 4 levels up
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
MODEL_DIR = os.path.join(BASE_DIR, 'models', 'nba')
DATA_FILE = os.path.join(BASE_DIR, 'data',   'nba', 'processed', 'training_dataset.csv')
PROJ_DIR  = os.path.join(BASE_DIR, 'data',   'nba', 'projections')

TODAY_SCAN_FILE    = os.path.join(PROJ_DIR, 'todays_automated_analysis.csv')
TOMORROW_SCAN_FILE = os.path.join(PROJ_DIR, 'tomorrows_automated_analysis.csv')
ACCURACY_LOG_FILE  = os.path.join(PROJ_DIR, 'accuracy_log.csv')

warnings.filterwarnings('ignore')

# Injury cache — refreshed before each scan (see refresh_injuries)
INJURY_DATA = {}


def refresh_injuries():
    """Fetch fresh injury report and update global cache. Call before each scan."""
    global INJURY_DATA
    INJURY_DATA = get_injury_report()

TARGETS = ACTIVE_TARGETS

FEATURES = [
    'PTS_L5', 'PTS_L20', 'PTS_Season',
    'REB_L5', 'REB_L20', 'REB_Season',
    'AST_L5', 'AST_L20', 'AST_Season',
    'FG3M_L5', 'FG3M_L20', 'FG3M_Season',
    'FG3A_L5', 'FG3A_L20', 'FG3A_Season',
    'STL_L5', 'STL_L20', 'STL_Season',
    'BLK_L5', 'BLK_L20', 'BLK_Season',
    'TOV_L5', 'TOV_L20', 'TOV_Season',
    'FGM_L5', 'FGM_L20', 'FGM_Season',
    'FTM_L5', 'FTM_L20', 'FTM_Season',
    'MIN_L5', 'MIN_L20', 'MIN_Season',
    'GAME_SCORE_L5', 'GAME_SCORE_L20', 'GAME_SCORE_Season',
    'USAGE_RATE_L5', 'USAGE_RATE_L20', 'USAGE_RATE_Season',
    'MISSING_USAGE',
    'TS_PCT', 'DAYS_REST', 'IS_HOME',
    'GAMES_7D', 'IS_4_IN_6', 'IS_B2B', 'IS_FRESH',
    'PACE_ROLLING', 'FGA_PER_MIN', 'TOV_PER_USAGE',
    'USAGE_VACUUM', 'STAR_COUNT',
    # Location splits + opponent context (synced with train.py)
    'PTS_LOC_MEAN', 'REB_LOC_MEAN', 'AST_LOC_MEAN', 'FG3M_LOC_MEAN', 'PRA_LOC_MEAN',
    'OPP_WIN_PCT', 'IS_VS_ELITE_TEAM'
]

for combo in ['PRA', 'PR', 'PA', 'RA', 'SB']:
    FEATURES.extend([f'{combo}_L5', f'{combo}_L20', f'{combo}_Season'])

for stat in ['PTS', 'REB', 'AST', 'FG3M', 'FGA', 'BLK', 'STL', 'TOV', 'FGM', 'FTM', 'FTA']:
    FEATURES.append(f'OPP_{stat}_ALLOWED')
    FEATURES.append(f'OPP_{stat}_ALLOWED_DIFF')  # DvP Diff (synced with train.py)

for combo in ['PRA', 'PR', 'PA', 'RA', 'SB']:
    FEATURES.append(f'OPP_{combo}_ALLOWED')
    FEATURES.append(f'OPP_{combo}_ALLOWED_DIFF')  # DvP Diff (synced with train.py)


def normalize_name(name):
    if not name: return ""
    n = unicodedata.normalize('NFKD', name)
    clean = "".join([c for c in n if not unicodedata.combining(c)])
    clean = re.sub(r'[^a-zA-Z\s]', '', clean)
    for s in ['Jr', 'Sr', 'III', 'II', 'IV']:
        clean = clean.replace(f" {s}", "")
    return " ".join(clean.lower().split())


def get_player_status(name, injury_data=None):
    """
    Check if player is on injury report. Uses exact match + last-name fallback
    (ESPN may use 'Patrick Williams' while our data has 'Patrick Williams II').
    """
    if injury_data is None:
        injury_data = INJURY_DATA
        
    norm_name = normalize_name(name)
    for injured_name, status in injury_data.items():
        if normalize_name(injured_name) == norm_name:
            return status
            
    # Fallback: last-name match when exactly one injured player shares that last name
    parts = norm_name.split()
    if len(parts) >= 2:
        last_name = parts[-1]
        matches = [
            s for inj_name, s in injury_data.items()
            if normalize_name(inj_name).split()[-1] == last_name
        ]
        if len(matches) == 1:
            return matches[0]
    return "Active"


def prepare_features(player_row, is_home=0, days_rest=2, missing_usage=0):
    if isinstance(player_row, dict):
        features = player_row.copy()
    else:
        features = player_row.to_dict()
    
    features['IS_HOME']       = 1 if is_home else 0
    features['DAYS_REST']     = days_rest
    features['IS_B2B']        = 1 if days_rest == 1 else 0
    features['MISSING_USAGE'] = missing_usage
    return pd.DataFrame([features])


def get_betting_indicator(proj, line):
    if line is None or line <= 0: return "⚪ NO LINE"
    diff = proj - line
    if diff > 0: return f"🟢 OVER (+{diff:.2f})"
    else:        return f"🔴 UNDER ({diff:.2f})"


def calculate_hit_rates(df_history, player_id, stat, line):
    """
    Calculate L5, L10, L20 hit rates against a specific line for a player.
    Returns: (l5_rate, l10_rate, l20_rate) as floats between 0.0 and 1.0.
    """
    if line is None or line <= 0:
        return 0.0, 0.0, 0.0
        
    # Get player's history sorted by date
    player_logs = df_history[df_history['PLAYER_ID'] == player_id].sort_values('GAME_DATE')
    
    # We might not have the raw stat name if it's a combo stat (like PRA)
    # Ensure combo stats are calculated if missing
    if stat not in player_logs.columns:
        if stat == 'PRA' and all(c in player_logs.columns for c in ['PTS', 'REB', 'AST']):
            player_logs['PRA'] = player_logs['PTS'] + player_logs['REB'] + player_logs['AST']
        elif stat == 'PR' and all(c in player_logs.columns for c in ['PTS', 'REB']):
            player_logs['PR'] = player_logs['PTS'] + player_logs['REB']
        elif stat == 'PA' and all(c in player_logs.columns for c in ['PTS', 'AST']):
            player_logs['PA'] = player_logs['PTS'] + player_logs['AST']
        elif stat == 'RA' and all(c in player_logs.columns for c in ['REB', 'AST']):
            player_logs['RA'] = player_logs['REB'] + player_logs['AST']
        elif stat == 'SB' and all(c in player_logs.columns for c in ['STL', 'BLK']):
            player_logs['SB'] = player_logs['STL'] + player_logs['BLK']
        else:
            return 0.0, 0.0, 0.0
            
    recent_20 = player_logs.tail(20)
    if recent_20.empty:
        return 0.0, 0.0, 0.0
        
    # How many times did they hit OVER the line?
    hits_20 = (recent_20[stat] > line).sum()
    hits_10 = (recent_20.tail(10)[stat] > line).sum()
    hits_5  = (recent_20.tail(5)[stat] > line).sum()
    
    count_20 = len(recent_20)
    count_10 = len(recent_20.tail(10))
    count_5  = len(recent_20.tail(5))
    
    l20_rate = hits_20 / count_20 if count_20 > 0 else 0.0
    l10_rate = hits_10 / count_10 if count_10 > 0 else 0.0
    l5_rate  = hits_5 / count_5 if count_5 > 0 else 0.0
    
    return l5_rate, l10_rate, l20_rate


def calculate_confidence_score(edge_pct, l10_hit, opponent_win_pct=None, is_role_expansion=False):
    """
    Generate an Elite Confidence Score (0-100).
    Components:
    - 60%: Model Edge %
    - 20%: L10 Hit Rate
    - 20%: Matchup Quality
    """
    # Max out edge score at 15% Edge (scales 0-60)
    edge_cap = min(abs(edge_pct), 15.0)
    edge_score = (edge_cap / 15.0) * 60.0
    
    # Hit rate score (Hit rate for overs, miss rate for unders)
    hit_score = l10_hit * 20.0
    
    # Matchup Score (Simple version based on Opponent strength)
    matchup_score = 10.0 # Base average
    if opponent_win_pct is not None:
        if opponent_win_pct < 0.40: matchup_score = 20.0
        elif opponent_win_pct > 0.60: matchup_score = 5.0
        
    total_score = edge_score + hit_score + matchup_score
    
    # HUMAN HANDICAPPER OVERRIDE: 
    # Do not bet UNDERS on bench players suddenly getting massive minutes or usage.
    # The variance is too high, even if historical hit-rate is 0%.
    if is_role_expansion and edge_pct < 0:
        total_score *= 0.5  # Slash confidence by half for dangerous Unders
        
    return min(total_score, 100.0)


def load_data():
    if not os.path.exists(DATA_FILE): return None
    df = pd.read_csv(DATA_FILE)
    df.columns = [c.strip() for c in df.columns]
    df['GAME_DATE'] = pd.to_datetime(df['GAME_DATE'])
    return df


# ---------------------------------------------------------------------------
# DATA CACHING (Optimization)
# ---------------------------------------------------------------------------

def build_data_cache(df_history):
    """
    Pre-indexes the dataframe for O(1) lookups.
    Returns:
        latest_rows_map: {pid: row_dict}
        team_rosters_map: {team_id: [pid, pid, ...]}
    """
    print("...Building data cache for fast lookups")
    
    # Filter to current season only — prevents traded players from inflating
    # team rosters (e.g. Deandre Ayton still mapping to the Suns)
    season_col = 'SEASON_YEAR' if 'SEASON_YEAR' in df_history.columns else 'SEASON_ID'
    current_season = df_history[season_col].max()
    df_current = df_history[df_history[season_col] == current_season]
    print(f"   Using season {current_season} ({df_current['PLAYER_ID'].nunique()} players)")
    
    # Sort by date, take latest row per player
    df_sorted = df_current.sort_values(['PLAYER_ID', 'GAME_DATE'])
    df_latest = df_sorted.drop_duplicates(subset=['PLAYER_ID'], keep='last')
    
    # 1. Latest Rows Map
    latest_rows_map = df_latest.set_index('PLAYER_ID').to_dict('index')
    
    # 2. Team Rosters Map (current season only)
    team_rosters_map = df_latest.groupby('TEAM_ID')['PLAYER_ID'].apply(list).to_dict()
    
    return latest_rows_map, team_rosters_map


def load_models():
    models = {}
    for target in TARGETS:
        path = os.path.join(MODEL_DIR, f"{target}_model.json")
        if os.path.exists(path):
            m = xgb.XGBRegressor()
            m.load_model(path)
            models[target] = m
    return models


# stats.nba.com is often slow/unreliable - use timeout + retries
NBA_API_TIMEOUT = 45   # Fail faster, rely on retries
NBA_API_RETRIES = 3
NBA_API_RETRY_DELAY = 5

# cdn.nba.com endpoints — fast and reliable
CDN_SCOREBOARD_URL = "https://cdn.nba.com/static/json/liveData/scoreboard/todaysScoreboard_00.json"
CDN_SCHEDULE_URL   = "https://cdn.nba.com/static/json/staticData/scheduleLeagueV2.json"
CDN_TIMEOUT = 15

CDN_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                  "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36",
    "Accept": "application/json",
}

# Catch all request-related errors (timeout, connection, etc.)
_REQUEST_ERRORS = (
    requests.exceptions.Timeout,
    requests.exceptions.ReadTimeout,
    requests.exceptions.ConnectionError,
    requests.exceptions.RequestException,
    ConnectionError,
    OSError,
)


def _cdn_scoreboard_to_df(games_list):
    """Convert cdn.nba.com games list to a DataFrame matching ScoreboardV2 format."""
    rows = []
    for g in games_list:
        rows.append({
            'GAME_ID':         g.get('gameId', ''),
            'GAME_STATUS_ID':  g.get('gameStatus', 1),
            'HOME_TEAM_ID':    g.get('homeTeam', {}).get('teamId', 0),
            'VISITOR_TEAM_ID': g.get('awayTeam', {}).get('teamId', 0),
        })
    return pd.DataFrame(rows) if rows else pd.DataFrame()


def _fetch_scoreboard_cdn(game_date):
    """
    Fetch scoreboard via cdn.nba.com (primary, fast).

    - For today's games: uses the live scoreboard endpoint.
    - For any date: uses the full-season schedule endpoint.
    """
    target = datetime.strptime(game_date, '%Y-%m-%d').date()
    today  = datetime.now().date()

    # Fast path: today's games via the live scoreboard
    if target == today:
        try:
            print(f"   Fetching today's scoreboard...", flush=True)
            resp = requests.get(CDN_SCOREBOARD_URL, headers=CDN_HEADERS, timeout=CDN_TIMEOUT)
            resp.raise_for_status()
            data = resp.json()
            games = data.get('scoreboard', {}).get('games', [])
            df = _cdn_scoreboard_to_df(games)
            if not df.empty:
                return df
        except Exception as e:
            print(f"   CDN scoreboard failed, trying schedule...", flush=True)

    # Fallback / future dates: use the full schedule
    return _fetch_schedule_cdn(game_date)


def _fetch_schedule_cdn(game_date):
    """Fetch games for a specific date from the cdn.nba.com season schedule."""
    from datetime import datetime as _dt
    target = _dt.strptime(game_date, '%Y-%m-%d').date()
    try:
        print(f"   Fetching schedule...", flush=True)
        resp = requests.get(CDN_SCHEDULE_URL, headers=CDN_HEADERS, timeout=CDN_TIMEOUT)
        resp.raise_for_status()
        data = resp.json()
        for gd in data.get('leagueSchedule', {}).get('gameDates', []):
            date_str = gd.get('gameDate', '')
            try:
                gd_date = _dt.strptime(date_str, '%m/%d/%Y %H:%M:%S').date()
            except ValueError:
                continue
            if gd_date == target:
                return _cdn_scoreboard_to_df(gd.get('games', []))
        # Date exists but no games scheduled
        return pd.DataFrame()
    except Exception as e:
        print(f"   CDN schedule failed", flush=True)
        return None


def _fetch_scoreboard_statsapi(game_date, retries=NBA_API_RETRIES):
    """Fetch scoreboard from stats.nba.com (fallback — often slow/unreliable)."""
    for attempt in range(retries):
        try:
            if attempt > 0:
                print(f"   Retry {attempt + 1}/{retries}...", flush=True)
            else:
                print(f"   Trying stats.nba.com fallback...", flush=True)
            sys.stdout.flush()
            board = ScoreboardV2(
                game_date=game_date, league_id='00', day_offset=0, timeout=NBA_API_TIMEOUT
            )
            return board.game_header.get_data_frame()
        except _REQUEST_ERRORS as e:
            if attempt < retries - 1:
                print(f"   Request failed, retrying in {NBA_API_RETRY_DELAY}s...", flush=True)
                time.sleep(NBA_API_RETRY_DELAY)
            else:
                raise
    return None


def _fetch_scoreboard(game_date, retries=NBA_API_RETRIES):
    """
    Fetch scoreboard with automatic failover:
      1. cdn.nba.com  (fast, reliable)
      2. stats.nba.com (legacy fallback)
    """
    # --- Primary: cdn.nba.com ---
    try:
        df = _fetch_scoreboard_cdn(game_date)
        if df is not None and not df.empty:
            print(f"   Got {len(df)} games from cdn.nba.com", flush=True)
            return df
        elif df is not None:
            # CDN responded but no games on this date
            return df
    except Exception as e:
        print(f"   cdn.nba.com failed, falling back...", flush=True)

    # --- Fallback: stats.nba.com ---
    try:
        return _fetch_scoreboard_statsapi(game_date, retries=retries)
    except Exception as e:
        print(f"   Both cdn.nba.com and stats.nba.com failed", flush=True)
        raise


def get_games(date_offset=0, require_scheduled=True, max_days_forward=7):
    """
    Fetch games for a specific date, with fallback to search forward.
    
    Args:
        date_offset (int): Days from today (0=today, 1=tomorrow, etc.)
        require_scheduled (bool): Only return games not yet started
        max_days_forward (int): Maximum days to search forward if no games found
        
    Returns:
        tuple: (team_map, actual_date_used)
            team_map: dict of {team_id: {'is_home': bool, 'opp': opponent_id}}
            actual_date_used: str of date where games were found
            
    Workflow:
        1. Try the requested date (today, tomorrow, etc.)
        2. If no games found, search forward day-by-day
        3. Stop at first date with games (up to max_days_forward)
        4. Return games + the date they were found on
        
    Example:
        # Today is Monday, no games today/tomorrow
        # Thursday has games
        team_map, date = get_games(date_offset=0)
        # Returns: (thursday_games, '2026-02-20')
        # Prints: "No games today. Found games on 2026-02-20 (Thursday)"
    """
    # Try the initially requested date
    initial_date = datetime.now() + timedelta(days=date_offset)
    target_date = initial_date.strftime('%Y-%m-%d')
    
    print(f"...Checking for games on {target_date}")
    
    try:
        games = _fetch_scoreboard(target_date)
        
        if not games.empty:
            if require_scheduled:
                scheduled_games = games[games['GAME_STATUS_ID'] == 1]
                if not scheduled_games.empty:
                    print(f"Found {len(scheduled_games)} scheduled games on {target_date}")
                    return _build_team_map(scheduled_games), target_date
            else:
                print(f"Found {len(games)} games on {target_date}")
                return _build_team_map(games), target_date
        
        # No games on requested date - search forward
        print(f"   No games on {target_date}. Searching forward...")
        
        for days_ahead in range(1, max_days_forward + 1):
            search_date = (initial_date + timedelta(days=days_ahead)).strftime('%Y-%m-%d')
            
            # Show progress every 2 days
            if days_ahead % 2 == 0 or days_ahead == 1:
                print(f"   Checking {search_date}...", end='\r')
            
            try:
                games = _fetch_scoreboard(search_date)
                
                if not games.empty:
                    if require_scheduled:
                        scheduled_games = games[games['GAME_STATUS_ID'] == 1]
                        if not scheduled_games.empty:
                            # Found games!
                            day_name = (initial_date + timedelta(days=days_ahead)).strftime('%A')
                            print(f"\nFound {len(scheduled_games)} games on {search_date} ({day_name})")
                            print(f"   📅 That's {days_ahead} day{'s' if days_ahead > 1 else ''} from now")
                            return _build_team_map(scheduled_games), search_date
                    else:
                        if not games.empty:
                            day_name = (initial_date + timedelta(days=days_ahead)).strftime('%A')
                            print(f"\nFound {len(games)} games on {search_date} ({day_name})")
                            return _build_team_map(games), search_date
            
            except Exception as e:
                # Skip this date if error
                continue
        
        # No games found in entire search window
        print(f"\nNo scheduled games found in the next {max_days_forward} days")
        return {}, None
        
    except Exception as e:
        print(f"Error fetching games: {e}")
        return {}, None


def _build_team_map(games_df):
    """
    Helper function to build team mapping from games DataFrame.
    
    Args:
        games_df: DataFrame with HOME_TEAM_ID and VISITOR_TEAM_ID columns
        
    Returns:
        dict: {team_id: {'is_home': bool, 'opp': opponent_id}}
    """
    team_map = {}
    for _, g in games_df.iterrows():
        team_map[g['HOME_TEAM_ID']] = {
            'is_home': True,
            'opp': g['VISITOR_TEAM_ID']
        }
        team_map[g['VISITOR_TEAM_ID']] = {
            'is_home': False,
            'opp': g['HOME_TEAM_ID']
        }
    return team_map


# ============================================================================
# UPDATED scan_all FUNCTION (to use the new return format)
# ============================================================================

def scan_all(df_history, models, is_tomorrow=False, max_days_forward=7):
    """
    Batch analysis of all games, with automatic forward search (optional).
    
    OPTIMIZED: Uses pre-built dictionaries for O(1) lookups instead of O(N) DataFrame filtering.
    """
    refresh_injuries()
    offset = 1 if is_tomorrow else 0
    
    # NEW: get_games now returns (team_map, actual_date)
    todays_teams, actual_date = get_games(
        date_offset=offset,
        require_scheduled=True,
        max_days_forward=max_days_forward
    )
    
    if not todays_teams:
        print("No scheduled games found in the next 7 days.")
        input("\nPress Enter to continue...")
        return
    
    # Show what date we're actually scanning
    if actual_date:
        scan_date_obj = datetime.strptime(actual_date, '%Y-%m-%d')
        day_name = scan_date_obj.strftime('%A, %B %d, %Y')
        print(f"\n📅 Scanning games for: {day_name}")
    
    print("\nFetching PrizePicks lines...")
    pp_client = PrizePicksClient(stat_map=STAT_MAP)
    live_lines = pp_client.fetch_lines_dict(league_filter='NBA')

    if live_lines:
        print(f"   Loaded {len(live_lines)} players from PrizePicks")
    else:
        print("   Warning: empty response from PrizePicks")

    norm_lines = {normalize_name(k): v for k, v in live_lines.items()}

    print("Building data cache & scanning markets...")
    # --- PRE-BUILD CACHE (O(N) once) ---
    latest_rows_map, team_rosters_map = build_data_cache(df_history)

    best_bets = []
    all_projections = []

    for team_id, info in todays_teams.items():
        # O(1) Lookup for roster
        team_players = team_rosters_map.get(team_id, [])

        # Calculate missing usage (injured players)
        missing_usage_today = 0.0
        for pid in team_players:
            # O(1) Lookup for player data
            last_row = latest_rows_map.get(pid)
            if not last_row: continue
            
            pname = last_row['PLAYER_NAME']
            if get_player_status(pname) == 'OUT':
                usage = last_row.get('USAGE_RATE_Season', 0)
                if usage > 15:
                    missing_usage_today += usage

        # Generate predictions for each player
        for pid in team_players:
            last_row = latest_rows_map.get(pid)
            if not last_row: continue
            
            player_name = last_row['PLAYER_NAME']

            if get_player_status(player_name) == 'OUT':
                continue
                
            input_row = prepare_features(
                last_row,
                is_home=info['is_home'],
                missing_usage=missing_usage_today
            )

            player_predictions = {}
            features_ok = True

            for target, model in models.items():
                if not features_ok: break
                try:
                    # Filter for features actually used by THIS model
                    model_features = [f for f in model.feature_names_in_]
                    valid_input = input_row.reindex(columns=model_features, fill_value=0)
                    proj = float(model.predict(valid_input)[0])
                    player_predictions[target] = proj
                except Exception:
                    # Model mismatch or missing features
                    features_ok = False
            
            if not features_ok: continue

            # NOTE: Injury redistribution (Layer 2) intentionally removed.
            # MISSING_USAGE is fed into XGBoost as a feature (Layer 1), and the
            # model already learned the injury boost from historical training data.
            # Adding a second manual bump on top caused double-counting and
            # inflated projections when teammates were out.

            # Apply correlation constraints
            if 'PRA' in player_predictions:
                pts = player_predictions.get('PTS', 0)
                reb = player_predictions.get('REB', 0)
                ast = player_predictions.get('AST', 0)
                player_predictions['PRA'] = max(player_predictions['PRA'], pts + reb + ast)

            if 'PR' in player_predictions:
                player_predictions['PR'] = max(
                    player_predictions['PR'],
                    player_predictions.get('PTS', 0) + player_predictions.get('REB', 0)
                )

            if 'PA' in player_predictions:
                player_predictions['PA'] = max(
                    player_predictions['PA'],
                    player_predictions.get('PTS', 0) + player_predictions.get('AST', 0)
                )

            if 'RA' in player_predictions:
                player_predictions['RA'] = max(
                    player_predictions['RA'],
                    player_predictions.get('REB', 0) + player_predictions.get('AST', 0)
                )

            if 'SB' in player_predictions:
                player_predictions['SB'] = max(
                    player_predictions['SB'],
                    player_predictions.get('STL', 0) + player_predictions.get('BLK', 0)
                )

            # Create recommendations
            for target, proj in player_predictions.items():
                line = norm_lines.get(normalize_name(player_name), {}).get(target)
                rec = get_betting_indicator(proj, line)
                
                # Default empty pp/edge if no line
                pp_val   = round(line, 2) if line else 0
                edge_val = round(proj - line, 2) if line else 0

                # Calculate Hit Rates
                l5_hit, l10_hit, l20_hit = calculate_hit_rates(df_history, pid, target, line)

                all_projections.append({
                    'REC': rec,
                    'NAME': player_name,
                    'TARGET': target,
                    'AI': round(proj, 2),
                    'PP': pp_val,
                    'EDGE': edge_val,
                    'L5_HIT': l5_hit,
                    'L10_HIT': l10_hit,
                    'L20_HIT': l20_hit
                })

                # Calculate Confidence Score
                line_diff_for_hit = l10_hit if edge_val > 0 else (1 - l10_hit)
                
                # Detect Role Expansion (High variance traps for Unders)
                is_role_expansion = False
                min_l5 = last_row.get('MIN_L5', 0)
                min_season = last_row.get('MIN_Season', 0)
                
                # If they are stepping into huge minutes relative to season avg, OR massive team injuries
                if (min_season > 5 and min_l5 / min_season > 1.4) or missing_usage_today > 25.0:
                    is_role_expansion = True
                
                # Fetch opponent context for Matchup Score if available in valid_input
                opp_win_pct = None
                if 'OPP_WIN_PCT' in valid_input.columns:
                    opp_win_pct = valid_input['OPP_WIN_PCT'].iloc[0]

                # Low-Line Mathematical Variance Fix:
                # Dividing by 0.5 can inflate a 0.38 block edge into a 76% edge.
                # We enforce a baseline statistical denominator of 2.0.
                safe_denominator = max(line, 2.0) if line else 2.0
                
                pct_edge_safe = (edge_val / safe_denominator) * 100
                conf_score = calculate_confidence_score(pct_edge_safe, line_diff_for_hit, opp_win_pct, is_role_expansion)

                if line is not None and line > 0:
                    edge = proj - line
                    pct_edge = (edge / safe_denominator) * 100

                    tier_info = MODEL_QUALITY.get(target, {})

                    best_bets.append({
                        'REC': rec,
                        'NAME': player_name,
                        'TARGET': target,
                        'AI': round(proj, 2),
                        'PP': round(line, 2),
                        'EDGE': edge,
                        'PCT_EDGE': pct_edge,
                        'TIER': tier_info.get('tier', 'UNKNOWN'),
                        'THRESHOLD': tier_info.get('threshold', 2.5),
                        'L5_HIT': l5_hit,
                        'L10_HIT': l10_hit,
                        'L20_HIT': l20_hit,
                        'CONFIDENCE': conf_score
                    })

    # Display results
    if best_bets:
        # ✅ DEDUPLICATE: Remove duplicate player+stat+line combinations
        seen = set()
        deduped_bets = []

        for bet in best_bets:
            key = (bet['NAME'], bet['TARGET'], bet['PP'])
            if key not in seen:
                seen.add(key)
                deduped_bets.append(bet)

        print(f"   Removed {len(best_bets) - len(deduped_bets)} duplicate entries")

        # ── Per-market display ────────────────────────────────────────────────
        PLAYS_PER_SIDE = 5   # ← change to show more/fewer per direction

        # Group all bets by market (TARGET)
        from collections import defaultdict
        by_market = defaultdict(lambda: {'OVER': [], 'UNDER': []})
        for bet in deduped_bets:
            direction = 'OVER' if bet['EDGE'] > 0 else 'UNDER'
            by_market[bet['TARGET']][direction].append(bet)

        # Sort within each direction by absolute edge (best first)
        for mkt in by_market:
            by_market[mkt]['OVER'].sort(key=lambda b: -b['PCT_EDGE'])
            by_market[mkt]['UNDER'].sort(key=lambda b: b['PCT_EDGE'])  # most negative first

        # Determine print order: markets sorted by combined top-edge strength
        def _market_strength(mkt):
            bets = by_market[mkt]
            top_over  = bets['OVER'][0]['PCT_EDGE']  if bets['OVER']  else 0
            top_under = abs(bets['UNDER'][0]['PCT_EDGE']) if bets['UNDER'] else 0
            return -(top_over + top_under)

        ordered_markets = sorted(by_market.keys(), key=_market_strength)

        col_w = 108
        sep   = "─" * col_w
        hdr   = f" {'TIER':<10} | {'PLAYER':<22} | {'STAT':<10} | {'AI vs PP':^17} | {'EDGE %':>8} | {'O/U':^5} | {'L10 HIT':>7} | {'CONF':>5}"

        total_shown = 0
        print(f"\n  📊 BEST PLAYS BY MARKET  (top {PLAYS_PER_SIDE} OVERs + {PLAYS_PER_SIDE} UNDERs each)\n")

        for mkt in ordered_markets:
            mkt_label = mkt.replace('_1H', ' 1H').replace('FPTS', 'FSCR')
            overs  = by_market[mkt]['OVER'][:PLAYS_PER_SIDE]
            unders = by_market[mkt]['UNDER'][:PLAYS_PER_SIDE]

            if not overs and not unders:
                continue

            print(f"  ══ {mkt_label} ══")
            print(hdr)
            print(sep)

            for bet in overs:
                l10_pct   = f"{bet['L10_HIT']*100:.0f}%"
                edge_str  = f"{bet['PCT_EDGE']:.1f}%"
                target_str = bet['TARGET'].replace('_1H', ' 1H').replace('FPTS', 'FSCR')
                print(f" {bet['TIER']:<10} | {bet['NAME'][:22]:<22} | {target_str:<10} | "
                      f"{bet['AI']:>7.2f} vs {bet['PP']:>6.2f} | {edge_str:>8} | {'OVER':^5} | {l10_pct:>7} | {bet['CONFIDENCE']:>5.1f}")
                total_shown += 1

            if overs and unders:
                print(f" {'':10}   {'─── UNDERs ───':^22}")

            for bet in unders:
                l10_pct   = f"{(1 - bet['L10_HIT'])*100:.0f}%"   # miss rate = under hit rate
                edge_str  = f"{bet['PCT_EDGE']:.1f}%"
                target_str = bet['TARGET'].replace('_1H', ' 1H').replace('FPTS', 'FSCR')
                print(f" {bet['TIER']:<10} | {bet['NAME'][:22]:<22} | {target_str:<10} | "
                      f"{bet['AI']:>7.2f} vs {bet['PP']:>6.2f} | {edge_str:>8} | {'UNDER':^5} | {l10_pct:>7} | {bet['CONFIDENCE']:>5.1f}")
                total_shown += 1

            print()

        # Determine save filename based on actual date used
        if actual_date:
            save_path = os.path.join(PROJ_DIR, f"scan_{actual_date}.csv")
        else:
            save_path = TOMORROW_SCAN_FILE if is_tomorrow else TODAY_SCAN_FILE

        pd.DataFrame(all_projections).to_csv(save_path, index=False)
        print(f"\nFull analysis ({len(all_projections)} rows) saved to {save_path}")
    else:
        print("\nNo active lines found.")

    input("\nPress Enter to continue...")


def prepare_features(player_row, is_home=0, days_rest=2, missing_usage=0):
    if isinstance(player_row, dict):
        features = player_row.copy()
    else:
        features = player_row.to_dict()
        
    features['IS_HOME']       = 1 if is_home else 0
    features['DAYS_REST']     = days_rest
    features['IS_B2B']        = 1 if days_rest == 1 else 0
    features['MISSING_USAGE'] = missing_usage
    return pd.DataFrame([features])


# --- STAT COLUMNS used for injury redistribution ---
# NOTE: Column names match the training CSV (mixed case from features.py)
_STAT_SEASON_COLS = {
    'PTS': 'PTS_Season', 'REB': 'REB_Season', 'AST': 'AST_Season',
    'FG3M': 'FG3M_Season', 'FGM': 'FGM_Season', 'FGA': 'FGA_Season',
    'FG3A': 'FG3A_Season', 'FTM': 'FTM_Season', 'FTA': 'FTA_Season',
    'STL': 'STL_Season', 'BLK': 'BLK_Season', 'TOV': 'TOV_Season',
}


def _get_position_category(pos):
    """Map raw NBA positions to Guard, Wing, Big for injury tracking."""
    pos = str(pos).upper()
    if 'C' in pos: return 'Big'
    if 'F' in pos: return 'Wing'
    return 'Guard'


def _calculate_injury_adjustments_fast(latest_rows_map, team_players, active_pid):
    """
    Optimized version of injury adjustments using pre-computed cache.
    Distributes 50% globally across usage, 50% positionally across usage.
    """
    out_production_global = {}
    out_production_pos = {'Guard': {}, 'Wing': {}, 'Big': {}}
    
    active_usage_global = {}
    active_usage_pos = {'Guard': {}, 'Wing': {}, 'Big': {}}

    active_last = latest_rows_map.get(active_pid, {})
    active_cat = _get_position_category(active_last.get('POSITION', 'G'))

    for pid in team_players:
        last = latest_rows_map.get(pid)
        if not last: continue
        
        pname = last['PLAYER_NAME']
        usage = last.get('USAGE_RATE_Season', 0)
        pos = last.get('POSITION', 'G')
        cat = _get_position_category(pos)

        if get_player_status(pname) == 'OUT':
            # Accumulate missing production
            for stat, col in _STAT_SEASON_COLS.items():
                val = last.get(col, 0)
                if pd.notna(val) and val > 0:
                     out_production_global[stat] = out_production_global.get(stat, 0) + val
                     out_production_pos[cat][stat] = out_production_pos[cat].get(stat, 0) + val
        else:
            if usage > 0:
                active_usage_global[pid] = usage
                active_usage_pos[cat][pid] = usage

    if not out_production_global or active_pid not in active_usage_global:
        return {}  # nothing to adjust

    # BLEND: 50% Positional, 50% Global
    total_active_usage = sum(active_usage_global.values())
    global_share = active_usage_global[active_pid] / total_active_usage if total_active_usage > 0 else 0

    cat_usage = sum(active_usage_pos[active_cat].values())
    pos_share = active_usage_pos[active_cat][active_pid] / cat_usage if cat_usage > 0 else 0

    adjustments = {}
    for stat in _STAT_SEASON_COLS.keys():
        rate = ABSORPTION_RATES.get(stat, 0.40)
        
        missing_global = out_production_global.get(stat, 0) * 0.5
        adj_global = missing_global * global_share * rate
        
        missing_pos = out_production_pos[active_cat].get(stat, 0) * 0.5
        adj_pos = missing_pos * pos_share * rate

        adj = adj_global + adj_pos
        if abs(adj) > 0.01:
            adjustments[stat] = round(adj, 2)

    # Derive combo-stat adjustments from components
    pts_adj = adjustments.get('PTS', 0)
    reb_adj = adjustments.get('REB', 0)
    ast_adj = adjustments.get('AST', 0)
    stl_adj = adjustments.get('STL', 0)
    blk_adj = adjustments.get('BLK', 0)

    if pts_adj or reb_adj or ast_adj:
        adjustments['PRA'] = round(pts_adj + reb_adj + ast_adj, 2)
        adjustments['PR']  = round(pts_adj + reb_adj, 2)
        adjustments['PA']  = round(pts_adj + ast_adj, 2)
        adjustments['RA']  = round(reb_adj + ast_adj, 2)
    if stl_adj or blk_adj:
        adjustments['SB'] = round(stl_adj + blk_adj, 2)

    return adjustments


def _calculate_injury_adjustments(df_history, team_id, active_pid):
    """
    Calculate per-stat injury adjustments for a single active player.

    For each OUT teammate, sum their season-average production per stat.
    Distribute that missing production to active players proportional to
    each active player's usage share.  Scale by ABSORPTION_RATE per stat.

    Args:
        df_history: full historical DataFrame
        team_id:    team to analyse
        active_pid: PLAYER_ID of the player getting the adjustment

    Returns:
        dict  {stat: adjustment_value, ...}
              e.g. {'PTS': +2.31, 'REB': +1.80, 'AST': +0.42, ...}
    """
    # Filter to current season to avoid counting traded players
    season_col = 'SEASON_YEAR' if 'SEASON_YEAR' in df_history.columns else 'SEASON_ID'
    current_season = df_history[season_col].max()
    team_df = df_history[(df_history[season_col] == current_season) & (df_history['TEAM_ID'] == team_id)]
    all_pids = team_df['PLAYER_ID'].unique()

    # Get active player's category
    active_cat = 'Guard'
    active_rows = team_df[team_df['PLAYER_ID'] == active_pid].sort_values('GAME_DATE')
    if not active_rows.empty:
        active_cat = _get_position_category(active_rows.iloc[-1].get('POSITION', 'G'))

    # --- Gather last-row data for every teammate ---
    out_production_global = {}
    out_production_pos = {'Guard': {}, 'Wing': {}, 'Big': {}}
    
    active_usage_global = {}
    active_usage_pos = {'Guard': {}, 'Wing': {}, 'Big': {}}

    for pid in all_pids:
        p_rows = team_df[team_df['PLAYER_ID'] == pid].sort_values('GAME_DATE')
        if p_rows.empty:
            continue
        last = p_rows.iloc[-1]
        pname = last['PLAYER_NAME']
        usage = last.get('USAGE_RATE_Season', 0)
        pos = last.get('POSITION', 'G')
        cat = _get_position_category(pos)

        if get_player_status(pname) == 'OUT':
            # Accumulate missing production
            for stat, col in _STAT_SEASON_COLS.items():
                val = last.get(col, 0)
                if pd.notna(val) and val > 0:
                    out_production_global[stat] = out_production_global.get(stat, 0) + val
                    out_production_pos[cat][stat] = out_production_pos[cat].get(stat, 0) + val
        else:
            if usage > 0:
                active_usage_global[pid] = usage
                active_usage_pos[cat][pid] = usage

    if not out_production_global or active_pid not in active_usage_global:
        return {}  # nothing to adjust

    # BLEND: 50% Positional, 50% Global
    total_active_usage = sum(active_usage_global.values())
    global_share = active_usage_global[active_pid] / total_active_usage if total_active_usage > 0 else 0

    cat_usage = sum(active_usage_pos[active_cat].values())
    pos_share = active_usage_pos[active_cat][active_pid] / cat_usage if cat_usage > 0 else 0

    adjustments = {}
    for stat in _STAT_SEASON_COLS.keys():
        rate = ABSORPTION_RATES.get(stat, 0.40)
        
        missing_global = out_production_global.get(stat, 0) * 0.5
        adj_global = missing_global * global_share * rate
        
        missing_pos = out_production_pos[active_cat].get(stat, 0) * 0.5
        adj_pos = missing_pos * pos_share * rate

        adj = adj_global + adj_pos
        if abs(adj) > 0.01:
            adjustments[stat] = round(adj, 2)

    # Derive combo-stat adjustments from components
    pts_adj = adjustments.get('PTS', 0)
    reb_adj = adjustments.get('REB', 0)
    ast_adj = adjustments.get('AST', 0)
    stl_adj = adjustments.get('STL', 0)
    blk_adj = adjustments.get('BLK', 0)

    if pts_adj or reb_adj or ast_adj:
        adjustments['PRA'] = round(pts_adj + reb_adj + ast_adj, 2)
        adjustments['PR']  = round(pts_adj + reb_adj, 2)
        adjustments['PA']  = round(pts_adj + ast_adj, 2)
        adjustments['RA']  = round(reb_adj + ast_adj, 2)
    if stl_adj or blk_adj:
        adjustments['SB'] = round(stl_adj + blk_adj, 2)

    return adjustments


def scout_player(df_history, models):
    print("\n--- PLAYER SCOUT ---")
    refresh_injuries()
    d_choice = input("Select Start Date (1=Today, 2=Tomorrow): ").strip()
    offset = 1 if d_choice == '2' else 0
    
    # Use the improved get_games logic to find the next available games
    todays_teams, actual_date = get_games(
        date_offset=offset, 
        require_scheduled=True, 
        max_days_forward=7
    )
    
    if not todays_teams:
        print("No scheduled games found in the next 7 days.")
        return

    # Display the date being scouted
    scan_date_obj = datetime.strptime(actual_date, '%Y-%m-%d')
    print(f"\n📅 Scouting for games on: {scan_date_obj.strftime('%A, %B %d, %Y')}")

    pp_client  = PrizePicksClient(stat_map=STAT_MAP)

    # --- Load FanDuel cache from disk (NO API call) ---
    import json, math
    fd_cache_file = os.path.join('fanduel_cache', 'fanduel_cache_nba.json')
    fd_odds_by_player = {}   # {normalized_name: {stat_code: {'over': odds, 'under': odds, 'line': float}}}
    fd_cache_age_str = "N/A"

    # Reverse map: FanDuel display name -> scanner target code
    _FD_TO_CODE = {
        'Points': 'PTS', 'Rebounds': 'REB', 'Assists': 'AST',
        '3-Pt Made': 'FG3M', 'Pts+Rebs+Asts': 'PRA', 'Pts+Rebs': 'PR',
        'Pts+Asts': 'PA', 'Rebs+Asts': 'RA', 'Blks+Stls': 'SB',
        'Blocks': 'BLK', 'Steals': 'STL', 'Turnovers': 'TOV',
        'Field Goals Made': 'FGM', 'Free Throws Made': 'FTM',
        'Free Throws Attempted': 'FTA', 'Field Goals Attempted': 'FGA',
    }

    if os.path.exists(fd_cache_file):
        try:
            cache_age_mins = (time.time() - os.path.getmtime(fd_cache_file)) / 60
            if cache_age_mins < 60:
                fd_cache_age_str = f"{int(cache_age_mins)} min(s) ago"
            else:
                fd_cache_age_str = f"{cache_age_mins / 60:.1f} hr(s) ago"
            with open(fd_cache_file, 'r') as f:
                fd_raw = json.load(f)
            for entry in fd_raw:
                p_name = normalize_name(entry.get('Player', ''))
                raw_stat = entry.get('Stat', '')
                stat_code = _FD_TO_CODE.get(raw_stat, raw_stat)
                side = entry.get('Side', '')
                odds = entry.get('Odds', 0)
                line = entry.get('Line', 0)
                if not p_name or not stat_code:
                    continue
                if p_name not in fd_odds_by_player:
                    fd_odds_by_player[p_name] = {}
                if stat_code not in fd_odds_by_player[p_name]:
                    fd_odds_by_player[p_name][stat_code] = {'over': 0, 'under': 0, 'line': line}
                if side == 'Over':
                    fd_odds_by_player[p_name][stat_code]['over'] = odds
                elif side == 'Under':
                    fd_odds_by_player[p_name][stat_code]['under'] = odds
            print(f"   FanDuel cache loaded ({len(fd_odds_by_player)} players, {fd_cache_age_str})")
        except Exception as e:
            print(f"   Warning: could not load FanDuel cache: {e}")
    else:
        print("   No FanDuel cache found -- run Odds Scanner first to populate it")

    # Helper: calculate vig-removed true probability from American odds
    def _odds_to_prob(odds):
        if odds < 0:
            return (-odds) / ((-odds) + 100)
        else:
            return 100 / (odds + 100)

    # Line adjustment factors (same as analyzer.py)
    _LINE_ADJ = {
        'PTS': 0.035, 'PRA': 0.025, 'PR': 0.030, 'PA': 0.030,
        'REB': 0.040, 'AST': 0.045, 'RA': 0.035,
        'FG3M': 0.055, 'STL': 0.060, 'BLK': 0.060, 'SB': 0.055,
        'TOV': 0.045, 'FGM': 0.040, 'FGA': 0.040, 'FTM': 0.050, 'FTA': 0.050,
    }

    scouting   = True

    while scouting:
        print("\n(Type '0' to return to Main Menu)")
        query = input("Enter player name: ").strip().lower()
        if query == '0':
            break

        try:
            matches = df_history[df_history['PLAYER_NAME'].str.lower().str.contains(query)]
        except Exception as e:
            print(f"Search error: {e}")
            continue

        if matches.empty:
            print(f"No players found matching '{query}'.")
            continue

        unique_players = matches[['PLAYER_ID', 'PLAYER_NAME']].drop_duplicates()
        if len(unique_players) > 1:
            print(unique_players.to_string(index=False))
            try:
                pid = int(input("Enter PLAYER_ID: "))
                matches = matches[matches['PLAYER_ID'] == pid]
                if matches.empty:
                    print(f"No data found for PLAYER_ID {pid}.")
                    continue
            except ValueError:
                print("Invalid PLAYER_ID.")
                continue

        # Fetch lines for the identified date (includes goblin/demon alt lines)
        print("Fetching PrizePicks lines...")
        live_lines_full = pp_client.fetch_lines_with_type(league_filter='NBA')
        norm_lines_full = {normalize_name(k): v for k, v in live_lines_full.items()}

        try:
            player_data = matches.sort_values('GAME_DATE').iloc[-1]
        except IndexError:
            print("No recent history found for this player.")
            continue

        name    = player_data['PLAYER_NAME']
        team_id = player_data['TEAM_ID']
        
        # Check if the player's team is in the team_map for the 'actual_date'
        if team_id not in todays_teams:
            print(f"{name} is not scheduled to play on {actual_date}.")
            continue

        is_home = todays_teams[team_id]['is_home']

        # Calculate injury impact for the target date (current season only)
        season_col = 'SEASON_YEAR' if 'SEASON_YEAR' in df_history.columns else 'SEASON_ID'
        current_season = df_history[season_col].max()
        df_current = df_history[(df_history[season_col] == current_season) & (df_history['TEAM_ID'] == team_id)]
        team_players = df_current['PLAYER_ID'].unique()
        missing_usage_today = 0.0
        for pid in team_players:
            p_rows = df_current[df_current['PLAYER_ID'] == pid].sort_values('GAME_DATE')
            if p_rows.empty: continue
            last_row = p_rows.iloc[-1]
            if get_player_status(last_row['PLAYER_NAME']) == 'OUT':
                usage = last_row.get('USAGE_RATE_Season', 0)
                if usage > 15: missing_usage_today += usage

        print(f"\nSCOUTING REPORT: {name} ({actual_date})")
        print(f"Injury Impact: {missing_usage_today:.1f}% Missing Usage")

        # NOTE: Manual injury redistribution removed — matches scan_all behaviour.
        # MISSING_USAGE is fed into XGBoost as a feature and the model handles the
        # boost internally.  Adding a second manual bump caused double-counting.
        inj_adj = {}

        # Show FanDuel cache age
        print(f"FanDuel odds: cached {fd_cache_age_str}")

        import unicodedata
        def _term_width(s):
            """Terminal display width of a string (handles emojis correctly)."""
            w = 0
            for ch in s:
                cat = unicodedata.category(ch)
                if cat in ('Mn', 'Me', 'Cf'):   # zero-width: marks, variation selectors
                    continue
                eaw = unicodedata.east_asian_width(ch)
                w += 2 if eaw in ('W', 'F') else 1
            return w

        TIER_WIDTH = 4  # desired terminal cols for tier column

        # (G) = goblin alt line, (D) = demon alt line
        print()
        print(f" {'TIER':<10} | {'STAT':<8} | {'PROJ':<14} | {'LINE':^11} | {'WIN%':>6} | SIDE")
        print("-" * 82)

        input_row = prepare_features(player_data, is_home=is_home, missing_usage=missing_usage_today)

        # Lookup FanDuel odds and PrizePicks lines for this player
        player_fd = fd_odds_by_player.get(normalize_name(name), {})
        player_pp = norm_lines_full.get(normalize_name(name), {})

        for target in TARGETS:
            if target in models:
                tier_text     = MODEL_QUALITY.get(target, {}).get('tier', 'UNKNOWN')
                model_features = [f for f in models[target].feature_names_in_]
                valid_input   = input_row.reindex(columns=model_features, fill_value=0)
                pred          = float(models[target].predict(valid_input)[0])

                # Get PP line with type info
                pp_info = player_pp.get(target)
                line      = pp_info['line'] if pp_info else None
                line_type = pp_info['type'] if pp_info else None

                # Format line string — fixed 11 chars wide
                if line is not None:
                    num_str = f"{line:.2f}"
                    if line_type == 'goblin':
                        line_str = f"{num_str:>6} (G) "
                    elif line_type == 'demon':
                        line_str = f"{num_str:>6} (D) "
                    else:
                        line_str = f"{num_str:>6}     "
                else:
                    line_str = "     --    "

                # --- Calculate WIN% from FanDuel odds — fixed 6 chars ---
                win_pct_str = "    --"
                if line is not None and target in player_fd:
                    fd_info = player_fd[target]
                    fd_over_odds = fd_info['over']
                    fd_under_odds = fd_info['under']
                    fd_line = fd_info['line']
                    if fd_over_odds != 0 and fd_under_odds != 0:
                        prob_o = _odds_to_prob(fd_over_odds)
                        prob_u = _odds_to_prob(fd_under_odds)
                        total = prob_o + prob_u
                        true_over = prob_o / total
                        true_under = prob_u / total
                        # Adjust for PP vs FD line difference
                        line_diff = line - fd_line
                        if line_diff != 0:
                            factor = _LINE_ADJ.get(target, 0.035)
                            adjustment = factor * math.log(1 + abs(line_diff)) / math.log(2)
                            if line_diff < 0:
                                true_over = min(true_over + adjustment, 0.90)
                                true_under = max(true_under - adjustment, 0.10)
                            else:
                                true_over = max(true_over - adjustment, 0.10)
                                true_under = min(true_under + adjustment, 0.90)
                            norm = true_over + true_under
                            true_over /= norm
                            true_under /= norm
                        # Show the win% for the projected side
                        if pred > line:
                            win_pct_str = f"{true_over * 100:5.1f}%"
                        else:
                            win_pct_str = f"{true_under * 100:5.1f}%"

                # --- ▲/▼ indicator ---
                if line is None or line <= 0:
                    side_str = "--"
                else:
                    diff = pred - line
                    if diff > 0:
                        side_str = f"▲ Over (+{diff:.2f})"
                    else:
                        side_str = f"▼ Under ({diff:.2f})"

                # Format Tier and Target natively without emojis
                tier_col = f"{tier_text:<10}"
                target_str = target.replace('_1H', ' 1H').replace('FPTS', 'FSCR')
                proj_str = f"{pred:>6.2f}       "
                print(f" {tier_col} | {target_str:<8} | {proj_str:<14} | {line_str} | {win_pct_str} | {side_str}")

        if input("\nScout another player? (y/n): ").lower() != 'y':
            scouting = False


def main():
    print("...Initializing System")
    refresh_injuries()
    df     = load_data()
    models = load_models()
    if df is None or not models:
        print("Setup failed.")
        return

    while True:
        print("\n" + "="*30 + "\n   NBA AI SCANNER\n" + "="*30)
        print("1. Scan TODAY's Games")
        print("2. Scan NEXT Match")
        print("3. Scout Specific Player")
        print("0. Exit")
        choice = input("\nSelect: ").strip()
        if choice == '1':   scan_all(df, models, is_tomorrow=False, max_days_forward=0)
        elif choice == '2': scan_all(df, models, is_tomorrow=True, max_days_forward=7)
        elif choice == '3': scout_player(df, models)
        elif choice == '0': break


if __name__ == "__main__":
    main()