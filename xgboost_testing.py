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

#'qty', 'price', 'Google_imm', 'lvmh_stock_price', 'kering_stock_price', 'leather_price'
features = ['qty', 'kering_stock_price', 'Google_imm']
"""
BAL -> kering_stock_price
YSL -> kering_stock_price
GIV -> 'Google_imm', 'kering_stock_price'
"""

perimeter = ['GIV'] #, 'YSL', 'GIV', 'ALL'

target = 'qty'

rolling_features = features
rolling_look_back = 4

look_back = 4

tscv = TimeSeriesSplit(n_splits=3)  

parameters = {'nthread':[4],
              'objective':['reg:squarederror'],
              'learning_rate': [.04, 0.05, .06],
              'max_depth': [4, 5, 6],
              'min_child_weight': [4],
              'silent': [1],
              'subsample': [0.5], #0.7
              'colsample_bytree': [0.7],
              'n_estimators': [500]}


def add_rolling_features(df, cols, rolling_look_back):
    for col in cols:
        df[col+'_rol_mean'] = df[col].shift(1).rolling(rolling_look_back).mean()
        df[col+'_rol_std'] = df[col].shift(1).rolling(rolling_look_back).std()
        df[col+'_rol_max'] = df[col].shift(1).rolling(rolling_look_back).max()
        df[col+'_rol_min'] = df[col].shift(1).rolling(rolling_look_back).min()
        if col != target:
            df.drop(col, axis = 1, inplace = True)
    return df

for brand in perimeter:

    #preprocessing

    df = pd.read_csv(f'{brand}.csv', index_col='date')
    df = df[features]

    #look back target
    for i in range(1,look_back):
        df[target+'-'+str(i)] = df[target].shift(i)
        df[target+'_V_'+str(i)] = (df[target].shift(i) - df[target].shift(i + 1)) / df[target].shift(i + 1)

    #rolling windows
    df = add_rolling_features(df, rolling_features, rolling_look_back)
    
    df.dropna(inplace = True)

    X_scalar_cols = [col for col in df.columns if col != target]
    
    df.index = pd.to_datetime(df.index)
    df['date'] = df.index
    df['month'] = df['date'].dt.month
    df['month'].astype("category")
    df['day'] = df['date'].dt.day
    df['day'].astype("category")
    df.set_index('date', drop = True, inplace = True)

    #training

    xgb_model = XGBRegressor(verbosity = 0)
    xgb = GridSearchCV(xgb_model, parameters, cv = 2, n_jobs = 5, verbose=True)

    X_cols = [col for col in df.columns if col != target]
    X = df[X_cols]
    y = df[[target]]

    r2_values = []
    rmse_values = []

    for train_index, test_index in tscv.split(X):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        y_train_mean = y_train[target].mean()

        #scaling

        X_scaler = MinMaxScaler()
        scaled_data = X_scaler.fit_transform(X_train[X_scalar_cols])
        X_train_scaled = pd.DataFrame(scaled_data, columns=X_scalar_cols, index=X_train.index)

        scaled_data = X_scaler.transform(X_test[X_scalar_cols])
        X_test_scaled = pd.DataFrame(scaled_data, columns=X_scalar_cols, index=X_test.index)
        
        target_scaler = MinMaxScaler()
        scaled_data = target_scaler.fit_transform(y_train[[target]])
        y_train_scaled = pd.DataFrame(scaled_data, columns=[target], index=y_train.index)

        scaled_data = target_scaler.transform(y_test[[target]])
        y_test_scaled = pd.DataFrame(scaled_data, columns=[target], index=y_test.index)
        
        #fitting and pred

        xgb.fit(X_train_scaled, y_train_scaled)
        
        y_pred_scaled = pd.DataFrame(xgb.predict(X_test_scaled), columns = [target])
        
        y_pred = target_scaler.inverse_transform(y_pred_scaled)
        y_test = target_scaler.inverse_transform(y_test_scaled)

        #Evaluation

        r2 = r2_score(y_test, y_pred)
        rmse = root_mean_squared_error(y_test, y_pred)
        r2_values.append(r2)
        rmse_values.append(rmse)
    
    avg_r2 = np.mean(r2_values)
    avg_rmse = np.mean(rmse_values)

    print(f"Brand {brand}:")
    print(f"Gave RMSE ratio: {avg_rmse/y_train_mean}")
    print(f"And R2: {avg_r2}")
    print(f"Parameters: {xgb.best_params_}")
        