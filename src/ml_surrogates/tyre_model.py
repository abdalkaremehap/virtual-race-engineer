import os
import warnings

import fastf1
import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from loguru import logger
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings("ignore")

def load_and_prep_data():
    logger.info("🏎️ Loading FastF1 data with Advanced Driver & Stint Features...")
    
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
    cache_dir = os.path.join(project_root, "data", "fastf1_cache")
    os.makedirs(cache_dir, exist_ok=True)
    fastf1.Cache.enable_cache(cache_dir)
    
    session = fastf1.get_session(2024, 'Bahrain', 'R')
    session.load(telemetry=False, weather=True) 
    
    laps = session.laps
    weather_data = laps.get_weather_data()
    
    laps = laps.reset_index(drop=True)
    weather_data = weather_data.reset_index(drop=True)
    laps_with_weather = pd.concat([laps, weather_data], axis=1)
    
    clean_laps = laps_with_weather.pick_accurate().pick_quicklaps().pick_track_status('1')
    
    required_cols = ['LapTime', 'TyreLife', 'Compound', 'TrackTemp', 'AirTemp', 'Driver', 'Stint', 'FreshTyre']
    clean_laps = clean_laps.dropna(subset=required_cols)
    
    # Fuel correction
    raw_lap_time = clean_laps['LapTime'].dt.total_seconds()
    fuel_effect = 0.06 * clean_laps['LapNumber']
    fuel_corrected_lap_time = raw_lap_time + fuel_effect
    
    df = pd.DataFrame({
        'fuel_corrected_lap_time': fuel_corrected_lap_time,
        'tyre_life': clean_laps['TyreLife'],
        'compound': clean_laps['Compound'],
        'lap_number': clean_laps['LapNumber'],
        'track_temp': clean_laps['TrackTemp'],
        'air_temp': clean_laps['AirTemp'],
        'driver': clean_laps['Driver'],
        'stint': clean_laps['Stint'],
        'fresh_tyre': clean_laps['FreshTyre'].astype(int),
        'circuit': 'Bahrain',
        'raw_lap_time': raw_lap_time
    })
    
    df['stint_group'] = df['driver'].astype(str) + "_" + df['stint'].astype(str)
    df = df.sort_values(by=['driver', 'lap_number'])
    
    # Calculate base pace and relative degradation early for filtering
    base_pace = df.groupby(['driver', 'compound'])['fuel_corrected_lap_time'].transform(lambda x: x.nsmallest(3).mean())
    df['relative_degradation'] = df['fuel_corrected_lap_time'] - base_pace
    
    # Clean traffic/lockup anomalies using a causal backward-looking rolling median with asymmetric bounds
    # so genuine sudden tyre cliffs (+0.20s to +1.25s/lap drop-off) are preserved rather than clipped.
    rolling_median = df.groupby('stint_group')['relative_degradation'].transform(lambda x: x.rolling(3, center=False, min_periods=1).median())
    diff_from_med = df['relative_degradation'] - rolling_median
    df = df[(diff_from_med >= -0.50) & (diff_from_med <= 1.25)].copy()
    
    logger.info(f"🧹 Advanced Data ready! {len(df)} laps available.")
    return df


def train_tyre_model(df):
    logger.info("🧠 Training Physics-Informed Tyre Degradation Model...")
    
    df = df.sort_values(by=['driver', 'lap_number'])
    
    # 1. Rolling Weather
    df['track_temp_rolling'] = df.groupby('driver')['track_temp'].transform(lambda x: x.rolling(window=3, min_periods=1).mean())
    df['air_temp_rolling'] = df.groupby('driver')['air_temp'].transform(lambda x: x.rolling(window=3, min_periods=1).mean())
    
    # 2. Track Rubber Evolution
    total_laps = df['lap_number'].max()
    df['track_evolution_sqrt'] = np.sqrt(df['lap_number'])
    df['track_evolution_log'] = np.log1p(df['lap_number'])
    df['track_evolution_lin'] = df['lap_number'] / total_laps
    
    # 3. Tyre Degradation Non-Linearities
    df['tyre_life_sq'] = df['tyre_life'] ** 2
    df['tyre_life_log'] = np.log1p(df['tyre_life'])
    df['tyre_life_cube'] = df['tyre_life'] ** 3
    df['tyre_life_sqrt'] = np.sqrt(df['tyre_life'])
    df['tyre_life_exp'] = np.exp(df['tyre_life'] / 15.0)
    
    df['fuel_kg_remaining'] = np.maximum(0, 110.0 - (110.0 / total_laps) * df['lap_number'])

    base_pace = df.groupby(['driver', 'compound'])['fuel_corrected_lap_time'].transform(lambda x: x.nsmallest(3).mean())
    df['relative_degradation'] = df['fuel_corrected_lap_time'] - base_pace
    
    # Lag features represent current known tyre state
    # df['prev_lap_degradation'] = df.groupby('stint_group')['relative_degradation'].shift(1).fillna(0)
    
    # More interactions
    df['temp_interaction'] = df['track_temp'] * df['air_temp']
    df['tyre_life_x_track_temp'] = df['tyre_life'] * df['track_temp']
    df['tyre_life_x_fuel'] = df['tyre_life'] * df['fuel_kg_remaining']
    df['fuel_x_track_evolution'] = df['fuel_kg_remaining'] * df['track_evolution_lin']
    df['tyre_life_x_track_evolution'] = df['tyre_life'] * df['track_evolution_lin']

    y = df['relative_degradation']

    encoders = {}
    for col in ['compound', 'circuit']:
        le = LabelEncoder()
        df[f'{col}_encoded'] = le.fit_transform(df[col])
        encoders[col] = le

    features = [
        'tyre_life', 'tyre_life_sq', 'tyre_life_log', 'tyre_life_cube', 'tyre_life_sqrt', 'tyre_life_exp',
        'compound_encoded',
        'track_temp', 'air_temp', 'track_temp_rolling', 'air_temp_rolling',
        'temp_interaction', 'tyre_life_x_track_temp', 'tyre_life_x_fuel', 'fuel_x_track_evolution', 'tyre_life_x_track_evolution',
        'circuit_encoded',
        'stint', 'fresh_tyre',
        'track_evolution_sqrt', 'track_evolution_log', 'track_evolution_lin',
        'fuel_kg_remaining'
    ]

    X = df[features]

    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, test_idx = next(gss.split(X, y, groups=df['stint_group']))

    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    train_data = lgb.Dataset(X_train, label=y_train, categorical_feature=['compound_encoded', 'circuit_encoded'])

    # Monotonic constraints: first 6 are tyre life features (must be +1). 
    # tyre_life_x_track_temp, tyre_life_x_fuel, tyre_life_x_track_evolution can also be +1 or 0. We leave as 0 to be safe.
    # Total features: 23
    monotone_constraints = [1, 1, 1, 1, 1, 1] + [0] * 17

    params = {
        'objective': 'regression',  
        'metric': 'mae',
        'learning_rate': 0.005,
        'num_leaves': 63,
        'max_depth': 8,
        'verbose': -1,
        'monotone_constraints': monotone_constraints,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 5
    }

    model = lgb.train(params, train_data, num_boost_round=3000)

    y_pred = model.predict(X_test)
    rmse = mean_squared_error(y_test, y_pred) ** 0.5
    mae = mean_absolute_error(y_test, y_pred)

    logger.success("✅ Physics Model trained successfully!")
    logger.info("📊 --- TRUE OUT-OF-SAMPLE METRICS ---")
    logger.info(f"   RMSE: {rmse:.3f} seconds per lap")
    logger.info(f"   MAE:  {mae:.3f} seconds per lap (This is the real, un-cheated error!)")

    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
    models_dir = os.path.join(project_root, "models")
    os.makedirs(models_dir, exist_ok=True)

    model.save_model(os.path.join(models_dir, "tyre_deg_model.txt"))
    joblib.dump(encoders, os.path.join(models_dir, "encoders.joblib"))
    logger.success("💾 Model and Encoders saved securely for production inference!")

if __name__ == "__main__":
    df = load_and_prep_data()
    train_tyre_model(df)