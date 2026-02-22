"""
NBA Configuration Constants

All NBA-specific settings: API keys, sport mappings, stat names,
model quality tiers, and scanning modes.

For shared cross-sport settings (SLIP_CONFIG) see:
    src/core/config.py

Environment Variables:
    .env file must contain: ODDS_API_KEY=your_key_here

Usage:
    from src.sports.nba.config import STAT_MAP, MODEL_QUALITY, ACTIVE_TARGETS
"""

import os
from dotenv import load_dotenv

load_dotenv()

# 1. API Configuration
ODDS_API_KEY = os.getenv('ODDS_API_KEY')
if not ODDS_API_KEY:
    raise ValueError("API Key not found! Make sure you have a .env file with ODDS_API_KEY inside.")

# 2. Sport Constants
SPORT_MAP = {
    'NBA': 'basketball_nba',
}

REGIONS = 'us'

MARKETS = (
    'player_points,player_rebounds,'
    'player_assists,player_threes,player_blocks,'
    'player_steals,player_blocks_steals,player_turnovers,'
    'player_points_rebounds_assists,player_points_rebounds,'
    'player_points_assists,player_rebounds_assists,'
    'player_field_goals,player_frees_made,player_frees_attempts,'
)

ODDS_FORMAT = 'american'
DATE_FORMAT = 'iso'

# 3. Stat Name Map  (PrizePicks display name -> internal abbreviation)
STAT_MAP = {
    'Points': 'PTS',
    'Rebounds': 'REB',
    'Assists': 'AST',
    '3-PT Made': 'FG3M',
    '3-PT Attempted': 'FG3A',
    'Blocked Shots': 'BLK',
    'Steals': 'STL',
    'Turnovers': 'TOV',
    'FG Made': 'FGM',
    'FG Attempted': 'FGA',
    'Free Throws Made': 'FTM',
    'Free Throws Attempted': 'FTA',
    'Pts+Rebs+Asts': 'PRA',
    'Pts+Rebs': 'PR',
    'Pts+Asts': 'PA',
    'Rebs+Asts': 'RA',
    'Blks+Stls': 'SB',
    'Fantasy Score': 'FPTS',
    # 1st Half (1H) Markets
    '1H Pts+Rebs+Asts': 'PRA_1H',
    '1H Points': 'PTS_1H',
    '1H Fantasy Score': 'FPTS_1H'
}

# 4. Model Quality Tiers (Based on Actual Directional Accuracy)
# Updated: 2026-02-13
MODEL_TIERS = {
    'ELITE': {
        'models': ['PTS', 'FGM', 'PA', 'PR', 'PRA', 'FPTS', 'PTS_1H', 'PRA_1H'],
        'accuracy_range': '85-90%',
        'edge_threshold': 1.5,
        'description': '⭐ Highest confidence - bet heavily',
        'emoji': '⭐'
    },
    'STRONG': {
        'models': ['FG3A', 'FGA', 'FPTS_1H'],
        'accuracy_range': '80-85%',
        'edge_threshold': 2.0,
        'description': '✔ Good confidence - bet selectively',
        'emoji': '✔'
    },
    'DECENT': {
        'models': ['FG3M', 'FTA', 'RA', 'FTM', 'REB', 'AST'],
        'accuracy_range': '72-80%',
        'edge_threshold': 2.5,
        'description': '~ Moderate confidence - bet carefully',
        'emoji': '~'
    },
    'RISKY': {
        'models': ['STL', 'TOV'],
        'accuracy_range': '61-71%',
        'edge_threshold': 3.0,
        'description': '⚠️ High variance - bet only large edges',
        'emoji': '⚠️'
    },
    'AVOID': {
        'models': ['BLK', 'SB'],
        'accuracy_range': '35-53%',
        'edge_threshold': 10.0,
        'description': '❌ Too random - avoid unless huge edge',
        'emoji': '❌'
    }
}

# Quick lookup dict: model name -> tier info
MODEL_QUALITY = {}
for tier, data in MODEL_TIERS.items():
    for model in data['models']:
        MODEL_QUALITY[model] = {
            'tier': tier,
            'threshold': data['edge_threshold'],
            'emoji': data['emoji']
        }

# 5. Scanning Mode
# Controls which model tiers to include in a scan run.
# Options: 'ELITE_ONLY', 'SAFE', 'BALANCED', 'AGGRESSIVE', 'ALL'
SCANNING_MODE = 'ALL'

SCANNING_MODES = {
    'ELITE_ONLY': MODEL_TIERS['ELITE']['models'],
    'SAFE':       MODEL_TIERS['ELITE']['models'] + MODEL_TIERS['STRONG']['models'],
    'BALANCED':   MODEL_TIERS['ELITE']['models'] + MODEL_TIERS['STRONG']['models'] + MODEL_TIERS['DECENT']['models'],
    'AGGRESSIVE': MODEL_TIERS['ELITE']['models'] + MODEL_TIERS['STRONG']['models'] + MODEL_TIERS['DECENT']['models'] + MODEL_TIERS['RISKY']['models'],
    'ALL':        list(MODEL_QUALITY.keys())
}

ACTIVE_TARGETS = SCANNING_MODES.get(SCANNING_MODE, SCANNING_MODES['BALANCED'])

mode_descriptions = {
    'ELITE_ONLY': "⭐ ELITE ONLY - 5 models (85%+), max accuracy",
    'SAFE':       "✔ SAFE MODE - 7 models (80%+), high confidence",
    'BALANCED':   "📊 BALANCED - 13 models (72%+), includes REB/AST",
    'AGGRESSIVE': "⚡ AGGRESSIVE - 15 models (61%+), includes STL/TOV",
    'ALL':        "🎲 ALL MODELS - 17 models (includes BLK/SB)"
}

print(f"⚙️  {mode_descriptions.get(SCANNING_MODE, 'UNKNOWN MODE')}")
print(f"   Scanning: {', '.join(ACTIVE_TARGETS)}")
if SCANNING_MODE == 'BALANCED':
    print(f"   Excluded: {', '.join(MODEL_TIERS['RISKY']['models'] + MODEL_TIERS['AVOID']['models'])}")

# 6. Injury Adjustment — Absorption Rates
# What fraction of a missing player's production redistributes to active teammates.
# Lower = more player-specific skill (hard to replace),  Higher = more opportunity-based.
ABSORPTION_RATES = {
    'PTS':  0.50,   # Scoring redistributes moderately
    'FGM':  0.45,   # Shot-making is partially skill-dependent
    'FGA':  0.55,   # Shot attempts redistribute well
    'FG3M': 0.30,   # 3PT shooting is very skill-dependent
    'FG3A': 0.45,   # 3PT attempts redistribute somewhat
    'REB':  0.65,   # Rebounds strongly redistribute (someone grabs them)
    'AST':  0.35,   # Assists are playmaker-specific
    'STL':  0.20,   # Steals are position/skill-dependent
    'BLK':  0.20,   # Blocks are heavily position-dependent
    'TOV':  0.25,   # Turnovers don't "redistribute" meaningfully
    'FTM':  0.35,   # Free throws depend on who drives to rim
    'FTA':  0.40,   # FT attempts redistribute a bit
    'PRA':  0.50,   # Combo stat — average of components
    'PR':   0.55,   # PTS + REB — rebounds help
    'PA':   0.42,   # PTS + AST — assists drag it down
    'RA':   0.48,   # REB + AST — mixed
    'SB':   0.20,   # Steals + Blocks — very position-dependent
}
