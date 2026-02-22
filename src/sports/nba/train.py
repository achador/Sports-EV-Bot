"""
XGBoost Model Training Pipeline

Trains separate regression models for 17 NBA statistics using time-series split
validation. Implements feature leakage prevention.

Output:
    models/nba/{TARGET}_model.json
    models/nba/model_metrics.csv

Usage:
    $ python3 -m src.sports.nba.train
"""

import pandas as pd
import xgboost as xgb
import os
import csv
from datetime import datetime
from sklearn.metrics import mean_absolute_error, r2_score

# --- CONFIGURATION ---
BASE_DIR    = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
DATA_FILE   = os.path.join(BASE_DIR, 'data',   'nba', 'processed', 'training_dataset.csv')
MODEL_DIR   = os.path.join(BASE_DIR, 'models', 'nba')
# TEST_START_DATE removed in favor of dynamic split

TARGETS = [
    'PTS', 'REB', 'AST', 'FG3M', 'FG3A', 'BLK', 'STL', 'TOV',
    'PRA', 'PR', 'PA', 'RA', 'SB',
    'FGM', 'FGA', 'FTM', 'FTA', 'FPTS',
    'PTS_1H', 'PRA_1H', 'FPTS_1H'
]

def get_features_for_target(target):
    """
    Dynamically select features based on the target to reduce noise.
    For example, the AST model doesn't need to look at BLK_L5.
    """
    # Core context features that help every model
    core = [
        'MIN_Season', 'MIN_L5', 'MIN_L10', 'USAGE_RATE_Season', 'USAGE_RATE_L5', 'USAGE_RATE_L10',
        'DAYS_REST', 'IS_HOME', 'GAMES_7D', 'IS_4_IN_6', 'IS_B2B', 'IS_FRESH',
        'PACE_ROLLING', 'USAGE_VACUUM', 'STAR_COUNT', 'GAME_SCORE_Season', 'GAME_SCORE_L5'
    ]
    
    # Determine which statistical families to include
    target_stats = []
    
    if target in ['PTS', 'FGM', 'FGA', 'FTM', 'FTA', 'FG3M', 'FG3A']:
        target_stats = ['PTS', 'FGM', 'FGA', 'FTM', 'FTA', 'FG3M', 'FG3A']
    elif target in ['REB']:
        target_stats = ['REB', 'PTS', 'FGA']
    elif target in ['AST']:
        target_stats = ['AST', 'PTS', 'FGA', 'TOV']
    elif target in ['STL', 'BLK', 'TOV']:
        target_stats = ['STL', 'BLK', 'TOV']
    elif target in ['PRA', 'PR', 'PA', 'RA']:
        target_stats = ['PTS', 'REB', 'AST', 'PRA', 'PR', 'PA', 'RA', 'FGA']
    elif target in ['SB']:
        target_stats = ['STL', 'BLK', 'SB']
    elif target in ['FPTS']:
        target_stats = ['FPTS', 'PTS', 'REB', 'AST', 'BLK', 'STL', 'TOV']
    elif target in ['PTS_1H', 'PRA_1H', 'FPTS_1H']:
        target_stats = ['PTS_1H', 'PTS', 'FGA_1H', 'MIN_1H']
        if target == 'PRA_1H': target_stats.extend(['PRA_1H', 'PRA'])
        if target == 'FPTS_1H': target_stats.extend(['FPTS_1H', 'FPTS'])
    
    features = list(core)
    
    # Generate the rolling variants for these specific stats
    for stat in target_stats:
        for variant in [f'{stat}_Season', f'{stat}_L5', f'{stat}_L10', f'{stat}_L20', f'{stat}_L5_Median', f'{stat}_L10_Median']:
            features.append(variant)
            
        # Add Defensive Matchup (DvP) if applicable
        if stat in ['PTS', 'REB', 'AST', 'FG3M', 'FGA', 'BLK', 'STL', 'TOV', 'FGM', 'FTM', 'FTA', 'PRA', 'PR', 'PA', 'RA', 'SB']:
            features.append(f'OPP_{stat}_ALLOWED')
            features.append(f'OPP_{stat}_ALLOWED_DIFF')
            
    # Add Location means if predicting main stats
    if target in ['PTS', 'REB', 'AST', 'FG3M', 'PRA']:
        features.append(f'{target}_LOC_MEAN')
        
    return list(dict.fromkeys(features))


def ensure_combo_stats(df):
    df = df.copy()
    if 'PRA' not in df.columns: df['PRA'] = df['PTS'] + df['REB'] + df['AST']
    if 'PR'  not in df.columns: df['PR']  = df['PTS'] + df['REB']
    if 'PA'  not in df.columns: df['PA']  = df['PTS'] + df['AST']
    if 'RA'  not in df.columns: df['RA']  = df['REB'] + df['AST']
    if 'SB'  not in df.columns: df['SB']  = df['STL'] + df['BLK']
    
    if 'PRA_1H' not in df.columns and 'PTS_1H' in df.columns: df['PRA_1H'] = df['PTS_1H'] + df['REB_1H'] + df['AST_1H']
    if 'PR_1H'  not in df.columns and 'PTS_1H' in df.columns: df['PR_1H']  = df['PTS_1H'] + df['REB_1H']
    if 'PA_1H'  not in df.columns and 'PTS_1H' in df.columns: df['PA_1H']  = df['PTS_1H'] + df['AST_1H']
    if 'RA_1H'  not in df.columns and 'REB_1H' in df.columns: df['RA_1H']  = df['REB_1H'] + df['AST_1H']
    if 'SB_1H'  not in df.columns and 'STL_1H' in df.columns: df['SB_1H']  = df['STL_1H'] + df['BLK_1H']
    return df


def train_and_evaluate():
    print("--- STARTING TRAINING PIPELINE ---")

    if not os.path.exists(DATA_FILE):
        print(f"ERROR: Training data not found at {DATA_FILE}. Run features.py first.")
        return

    df = pd.read_csv(DATA_FILE)
    df['GAME_DATE'] = pd.to_datetime(df['GAME_DATE'])
    df = ensure_combo_stats(df)
    
    # Sort by date for time-series split
    df = df.sort_values('GAME_DATE').reset_index(drop=True)
    
    # Dynamic 70/30 Split
    split_idx = int(len(df) * 0.70)
    train_df = df.iloc[:split_idx]
    test_df  = df.iloc[split_idx:]
    
    # Print date ranges for verification
    print(f"Train Date Range: {train_df['GAME_DATE'].min().date()} -> {train_df['GAME_DATE'].max().date()}")
    print(f"Test Date Range:  {test_df['GAME_DATE'].min().date()} -> {test_df['GAME_DATE'].max().date()}")

    print(f"Training Set: {len(train_df)} games")
    print(f"Testing Set:  {len(test_df)} games")

    os.makedirs(MODEL_DIR, exist_ok=True)

    all_metrics = []

    for target in TARGETS:
        print(f"\nTraining Model for: {target}...")

        if target not in df.columns:
            print(f" -> SKIPPING {target} (Column not found in data)")
            continue

        # Filter requested features to ensure they actually exist in the dataframe
        raw_features = get_features_for_target(target)
        features_to_use = [f for f in raw_features if f in df.columns]
        
        # Log how many features were pruned
        print(f"   -> Using {len(features_to_use)} specialized features for {target} (Pruned {len(df.columns) - len(features_to_use)})")

        X_train = train_df[features_to_use]
        y_train = train_df[target]
        X_test  = test_df[features_to_use]
        y_test  = test_df[target]

        model = xgb.XGBRegressor(
            n_estimators=1000,
            learning_rate=0.05,
            max_depth=5,
            early_stopping_rounds=50,
            n_jobs=-1
        )
        model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)

        predictions = model.predict(X_test)
        mae = mean_absolute_error(y_test, predictions)
        r2  = r2_score(y_test, predictions)

        test_median       = y_test.median()
        actual_over       = (y_test > test_median).astype(int)
        predicted_over    = (predictions > test_median).astype(int)
        directional_accuracy = (actual_over == predicted_over).mean()

        all_metrics.append({
            'Target': target,
            'MAE': round(mae, 4),
            'R2': round(r2, 4),
            'Directional_Accuracy': round(directional_accuracy * 100, 2),
            'Last_Updated': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        })

        print(f" -> MAE: {mae:.2f}")
        print(f" -> R2 Score: {r2:.3f}")
        print(f" -> Directional Accuracy: {directional_accuracy:.1%}")

        model_path = os.path.join(MODEL_DIR, f"{target}_model.json")
        model.save_model(model_path)
        print(f" -> Saved to {model_path}")

    metrics_file = os.path.join(MODEL_DIR, 'model_metrics.csv')
    keys = all_metrics[0].keys()
    with open(metrics_file, 'w', newline='') as f:
        dict_writer = csv.DictWriter(f, fieldnames=keys)
        dict_writer.writeheader()
        dict_writer.writerows(all_metrics)
    print(f"\n✅ Performance metrics saved to {metrics_file}")
    print("\n--- ALL MODELS TRAINED ---")


if __name__ == "__main__":
    train_and_evaluate()
