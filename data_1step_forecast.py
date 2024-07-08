from sklearn.preprocessing import MinMaxScaler

from xgboost import XGBRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import TimeSeriesSplit
from scipy.fft import fft

from sklearn.metrics import root_mean_squared_error, r2_score
from xgboost import plot_importance, plot_tree

import numpy as np
import pandas as pd
from itertools import combinations




perimeter = ['ALL', 'BAL', 'YSL', 'GIV']

features = {
    'BAL': ['qty', 'qty_V_3', 'kering_stock_price_rol_mean', 'qty_V_2', 'day', 'qty_rol_std', 'qty-1', 'qty_V_1', 'kering_stock_price_rol_std', 'qty_rol_mean', 'qty-2', 'kering_stock_price_rol_max', 'qty_rol_max', 'kering_stock_price_rol_min', 'qty_rol_min', 'month'],
    'YSL': ['qty', 'qty_rol_mean', 'qty-1', 'qty_rol_std', 'qty_V_3', 'kering_stock_price_rol_std', 'qty_V_1', 'qty_rol_min', 'day', 'qty_V_2', 'qty-2', 'kering_stock_price_rol_mean', 'qty_rol_max', 'Google_imm_rol_std', 'kering_stock_price_rol_min', 'month'],
    'GIV': ['qty', 'kering_stock_price_rol_std', 'kering_stock_price_rol_mean', 'qty-1', 'day', 'qty_V_2', 'qty-2', 'qty_V_1', 'qty_rol_mean', 'qty_V_3', 'qty_rol_max', 'qty_rol_std', 'Google_imm_rol_mean', 'qty-3', 'Google_imm_rol_std', 'month'],
    'ALL': ['qty', 'qty-1', 'qty_V_1', 'qty_rol_std', 'kering_stock_price_rol_std', 'kering_stock_price_rol_mean', 'qty_V_2', 'qty_V_3', 'day', 'qty_rol_max', 'qty_rol_mean', 'qty-3', 'month', 'qty-2']
}

params = {
    'BAL': {'colsample_bytree': 0.7, 'learning_rate': 0.07, 'max_depth': 6, 'min_child_weight': 4, 'n_estimators': 1000, 'nthread': 4, 'objective': 'reg:squarederror', 'silent': 1, 'subsample': 0.7},
    'YSL': {'colsample_bytree': 0.7, 'learning_rate': 0.07, 'max_depth': 7, 'min_child_weight': 4, 'n_estimators': 1000, 'nthread': 4, 'objective': 'reg:squarederror', 'silent': 1, 'subsample': 0.7},
    'GIV': {'colsample_bytree': 0.7, 'learning_rate': 0.08, 'max_depth': 7, 'min_child_weight': 4, 'n_estimators': 1000, 'nthread': 4, 'objective': 'reg:squarederror', 'silent': 1, 'subsample': 0.7},
    'ALL': {'colsample_bytree': 0.7, 'learning_rate': 0.07, 'max_depth': 6, 'min_child_weight': 4, 'n_estimators': 1000, 'nthread': 4, 'objective': 'reg:squarederror', 'silent': 1, 'subsample': 0.7}
}

target = 'qty'


for brand in perimeter:

    df = pd.read_csv(f'forecast/{brand}.csv', index_col='date')

    df['month'].astype("category")
    df['day'].astype("category")

    df = df[features.get(brand)]

    cat_cols = ['month', 'day']
    X_cat_cols = [col for col in df.columns if col in cat_cols]
    X_scalar_cols = [col for col in df.columns if col != target and col not in X_cat_cols]

    xgb = XGBRegressor(**params, verbosity = 0)

    X_cols = [col for col in df.columns if col != target]
    X = df[X_cols]
    y = df[[target]]

    X_scaler = MinMaxScaler()
    scaled_data = X_scaler.fit_transform(X[X_scalar_cols])
    X_scaled = pd.DataFrame(scaled_data, columns=X_scalar_cols, index=X.index)
    if len(X_cat_cols) > 0:
        X_scaled = pd.concat([X_scaled, X[X_cat_cols]], axis=1)            

    target_scaler = MinMaxScaler()
    scaled_data = target_scaler.fit_transform(y[[target]])
    y_scaled = pd.DataFrame(scaled_data, columns=[target], index=y.index)

    X_train_scaled = X_scaled.iloc[:-1]
    y_train_scaled = y_scaled.iloc[:-1]

    X_forecast_scaled = X_scaled.tail(1)

    xgb.fit(X_train_scaled, y_train_scaled)
    
    y_pred_scaled = pd.DataFrame(xgb.predict(X_forecast_scaled), columns = [target])
    
    y_pred = target_scaler.inverse_transform(y_pred_scaled)

    last_week = df.index[-1]
    last_week_qty = df.iloc[-1]['qty-1']
    next_week_forecast = y_pred[0][0]

    print(f"Brand {brand} quantities in the week {last_week} were: {last_week_qty}.\nIn the subsequent week the quantities expected are around: {next_week_forecast}") 
        