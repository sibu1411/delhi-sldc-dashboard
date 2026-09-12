"""
Delhi SLDC - 3D AI Grid Load Prediction Engine
Handles synthetic thermodynamic load curve generation, feature engineering,
model training (Random Forest, Gradient Boosting, XGBoost, Ridge), evaluation,
and 3D response surface mesh grid extraction.
"""

from typing import Dict, Tuple
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error, r2_score

# Optional XGBoost integration with graceful fallback
try:
    from xgboost import XGBRegressor
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False


def calculate_heat_index(temp_c: np.ndarray, rh: np.ndarray) -> np.ndarray:
    """Calculates an approximated Heat Index (°C) given Temperature and Relative Humidity."""
    # Simplified Rothfusz regression adaptation for metric units
    temp_f = (temp_c * 9 / 5) + 32
    hi_f = -42.379 + 2.04901523 * temp_f + 10.14333127 * rh - 0.22475541 * temp_f * rh
    hi_c = (hi_f - 32) * 5 / 9
    return np.where(temp_c >= 26, hi_c, temp_c)


def generate_synthetic_delhi_data(days: int = 45) -> pd.DataFrame:
    """Generates realistic high-frequency historical grid load and meteorological data

    modeling Delhi's summer thermodynamic response profile.
    """
    np.random.seed(42)
    dates = pd.date_range(end=pd.Timestamp.now().floor("h"), periods=24 * days, freq="h")
    hours = dates.hour
    months = dates.month

    # Diurnal temperature cycle for Delhi (24°C night to ~44°C daytime peak)
    base_temp = 32 + 9 * np.sin((hours - 9) * np.pi / 12) + 2 * np.sin((months - 5) * np.pi / 6)
    temp_noise = np.random.normal(0, 1.1, len(dates))
    temperature = np.clip(base_temp + temp_noise, 18.0, 48.5)

    # Relative humidity (inverse relation with temperature)
    humidity = np.clip(85 - (temperature - 18) * 1.5 + np.random.normal(0, 3.5, len(dates)), 15.0, 95.0)
    heat_index = calculate_heat_index(temperature, humidity)

    # Base load + Non-linear HVAC exponential cooling response + Hourly diurnal curve
    base_grid_load = 4100 + np.random.normal(0, 90, len(dates))
    diurnal_profile = 1450 * np.sin((hours - 5) * np.pi / 12) ** 2
    hvac_load = np.where(heat_index > 26, ((heat_index - 26) ** 1.65) * 125, 0)
    weekend_reduction = np.where(dates.dayofweek.isin([5, 6]), -350, 0)

    demand_mw = base_grid_load + diurnal_profile + hvac_load + weekend_reduction

    df = pd.DataFrame({
        "timestamp": dates,
        "demand_mw": demand_mw,
        "temperature": temperature,
        "humidity": humidity,
        "heat_index": heat_index,
        "hour": hours,
        "dayofweek": dates.dayofweek,
        "is_weekend": dates.dayofweek.isin([5, 6]).astype(int),
        "sin_hour": np.sin(2 * np.pi * hours / 24),
        "cos_hour": np.cos(2 * np.pi * hours / 24),
    })

    # Lag features for temporal continuity
    df["load_lag_1h"] = df["demand_mw"].shift(1).bfill()
    df["load_lag_24h"] = df["demand_mw"].shift(24).bfill()

    return df


def run_enhanced_pipeline(
    model_choice: str = "Random Forest",
    temp_offset: float = 0.0,
    humidity_offset: float = 0.0,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, float], Dict[str, float], Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Trains the selected load forecast model and produces a 72-hour ahead prediction

    along with 3D thermodynamic surface grid matrices.
    """
    df = generate_synthetic_delhi_data(days=60)

    feature_cols = [
        "temperature", "humidity", "heat_index",
        "hour", "dayofweek", "is_weekend",
        "sin_hour", "cos_hour", "load_lag_1h", "load_lag_24h"
    ]

    X = df[feature_cols]
    y = df["demand_mw"]

    # 85/15 Time-Series Split
    split_idx = int(len(df) * 0.85)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    # Model Engine Selection
    if model_choice == "XGBoost" and HAS_XGBOOST:
        model = XGBRegressor(n_estimators=180, learning_rate=0.05, max_depth=6, random_state=42)
    elif model_choice == "Gradient Boosting":
        model = GradientBoostingRegressor(n_estimators=160, learning_rate=0.06, max_depth=5, random_state=42)
    elif model_choice == "Linear Regression":
        model = Ridge(alpha=1.0)
    else:  # Default to Random Forest
        model = RandomForestRegressor(n_estimators=140, max_depth=12, random_state=42, n_jobs=-1)

    model.fit(X_train, y_train)

    # Evaluation Metrics
    y_pred = model.predict(X_test)
    metrics = {
        "R2": r2_score(y_test, y_pred),
        "MAE": mean_absolute_error(y_test, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y_test, y_pred)),
        "MAPE": mean_absolute_percentage_error(y_test, y_pred) * 100,
    }

    # Feature Importance Analysis
    if hasattr(model, "feature_importances_"):
        importances = dict(zip(feature_cols, model.feature_importances_))
    else:
        coef_abs = np.abs(model.coef_)
        importances = dict(zip(feature_cols, coef_abs / np.sum(coef_abs)))

    # 72-Hour Future Load Forecast Simulation
    last_time = df["timestamp"].iloc[-1]
    future_dates = pd.date_range(start=last_time + pd.Timedelta(hours=1), periods=72, freq="h")
    future_hours = future_dates.hour

    future_temp = 32 + 9 * np.sin((future_hours - 9) * np.pi / 12) + temp_offset
    future_hum = np.clip(80 - (future_temp - 18) * 1.5 + humidity_offset, 15, 95)
    future_hi = calculate_heat_index(future_temp, future_hum)

    forecast_df = pd.DataFrame({
        "timestamp": future_dates,
        "temperature": future_temp,
        "humidity": future_hum,
        "heat_index": future_hi,
        "hour": future_hours,
        "dayofweek": future_dates.dayofweek,
        "is_weekend": future_dates.dayofweek.isin([5, 6]).astype(int),
        "sin_hour": np.sin(2 * np.pi * future_hours / 24),
        "cos_hour": np.cos(2 * np.pi * future_hours / 24),
    })

    # Iterative multi-step prediction for lag continuation
    predictions = []
    last_lag_1h = df["demand_mw"].iloc[-1]
    last_lag_24h_series = list(df["demand_mw"].iloc[-24:].values)

    for i in range(len(forecast_df)):
        row = forecast_df.iloc[i].copy()
        row["load_lag_1h"] = last_lag_1h
        row["load_lag_24h"] = last_lag_24h_series[i % 24]

        pred_val = model.predict(pd.DataFrame([row[feature_cols]]))[0]
        predictions.append(pred_val)
        last_lag_1h = pred_val

    forecast_df["predicted_demand_mw"] = predictions

    # 3D Load Topology Surface Meshgrid Generation
    t_axis = np.linspace(20, 50, 31)  # Temperature Range (°C)
    h_axis = np.arange(0, 24, 1)      # Hour of Day (0-23)
    T, H = np.meshgrid(t_axis, h_axis)

    mesh_humidity = np.clip(80 - (T - 18) * 1.5 + humidity_offset, 15, 95)
    mesh_hi = calculate_heat_index(T, mesh_humidity)

    mesh_df = pd.DataFrame({
        "temperature": T.ravel(),
        "humidity": mesh_humidity.ravel(),
        "heat_index": mesh_hi.ravel(),
        "hour": H.ravel(),
        "dayofweek": 2,
        "is_weekend": 0,
        "sin_hour": np.sin(2 * np.pi * H.ravel() / 24),
        "cos_hour": np.cos(2 * np.pi * H.ravel() / 24),
        "load_lag_1h": df["demand_mw"].mean(),
        "load_lag_24h": df["demand_mw"].mean(),
    })[feature_cols]

    z_matrix = model.predict(mesh_df).reshape(T.shape)

    return df, forecast_df, metrics, importances, (t_axis, h_axis, z_matrix)
     
