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




perimeter = ['BAL'] 

target = 'qty+1' #'qty+1'


tscv = TimeSeriesSplit(n_splits=3)  

parameters = {'nthread':[4],
              'objective':['reg:squarederror'],
              'learning_rate': [.06, .07, .08], #.04, 0.05,
              'max_depth': [6, 7, 8],
              'min_child_weight': [4],
              'silent': [1],
              'subsample': [0.8, 0.7], #0.7
              'colsample_bytree': [0.7],
              'n_estimators': [1000]}


for brand in perimeter:

    df = pd.read_csv(f'train_month_change/{brand}.csv', index_col='date')

    df['month'].astype("category")
    df['day'].astype("category")

    df = df[[target] + [
        'qty_rol_mean',
        'qty_rol_std',
        'qty_rol_max',
        #'qty_rol_min',
        'Google_imm_rol_mean',
        'Google_imm_rol_std',
        #'Google_imm_rol_max',
        #'Google_imm_rol_min',
        'lvmh_stock_price_rol_mean',
        'lvmh_stock_price_rol_std',
        #'lvmh_stock_price_rol_max',
        #'lvmh_stock_price_rol_min',
        'kering_stock_price_rol_mean',
        'kering_stock_price_rol_std',
        #'kering_stock_price_rol_max',
        #'kering_stock_price_rol_min',
        'qty-0',
        'qty_V_0',
        'qty-1',
        'qty_V_1',
        'qty-2',
        'qty_V_2',
        'qty-3',
        'qty_V_3',
        #'qty-4',
        #'qty_V_4',
        #'qty-5',
        #'qty_V_5',
        #'qty-6',
        #'qty_V_6',
        #'qty-7',
        #'qty_V_7',
        #'Google_imm-0',
        #'Google_imm_V_0',
        #'Google_imm-1',
        #'Google_imm_V_1',
        #'Google_imm-2',
        #'Google_imm_V_2',
        #'Google_imm-3',
        #'Google_imm_V_3',
        #'Google_imm-4',
        #'Google_imm_V_4',
        #'Google_imm-5',
        #'Google_imm_V_5',
        #'Google_imm-6',
        #'Google_imm_V_6',
        #'Google_imm-7',
        #'Google_imm_V_7',
        #'lvmh_stock_price-0',
        #'lvmh_stock_price_V_0',
        #'lvmh_stock_price-1',
        #'lvmh_stock_price_V_1',
        #'lvmh_stock_price-2',
        #'lvmh_stock_price_V_2',
        #'lvmh_stock_price-3',
        #'lvmh_stock_price_V_3',
        #'lvmh_stock_price-4',
        #'lvmh_stock_price_V_4',
        #'lvmh_stock_price-5',
        #'lvmh_stock_price_V_5',
        #'lvmh_stock_price-6',
        #'lvmh_stock_price_V_6',
        #'lvmh_stock_price-7',
        #'lvmh_stock_price_V_7',
        #'kering_stock_price-0',
        #'kering_stock_price_V_0',
        #'kering_stock_price',
        #'kering_stock_price-1',
        #'kering_stock_price_V_1',
        #'kering_stock_price-2',
        #'kering_stock_price_V_2',
        #'kering_stock_price-3',
        #'kering_stock_price_V_3',
        #'kering_stock_price-4',
        #'kering_stock_price_V_4',
        #'kering_stock_price-5',
        #'kering_stock_price_V_5',
        #'kering_stock_price-6',
        #'kering_stock_price_V_6',
        #'kering_stock_price-7',
        #'kering_stock_price_V_7',
        'month',
        'day'
    ]]


    cat_cols = ['month', 'day']
    X_cat_cols = [col for col in df.columns if col in cat_cols]
    X_scalar_cols = [col for col in df.columns if col != target and col not in X_cat_cols]

    xgb_model = XGBRegressor(verbosity = 0)
    xgb = GridSearchCV(xgb_model, parameters, cv = 2, n_jobs = 5, verbose=False)

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
        if len(X_cat_cols) > 0:
            X_train_scaled = pd.concat([X_train_scaled, X_train[X_cat_cols]], axis=1)            

        scaled_data = X_scaler.transform(X_test[X_scalar_cols])
        X_test_scaled = pd.DataFrame(scaled_data, columns=X_scalar_cols, index=X_test.index)
        if len(X_cat_cols) > 0:
            X_test_scaled = pd.concat([X_test_scaled, X_test[X_cat_cols]], axis=1)
        
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
        feature_scores   = xgb.best_estimator_.get_booster().get_score(importance_type='weight')
        sorted_features = sorted(feature_scores.items(), key=lambda x: x[1], reverse=True)
        top_features = sorted_features#[:10]
        
    
    avg_r2 = np.mean(r2_values)
    avg_rmse = np.mean(rmse_values)

    print(f"###########################Brand {brand}:")
    print(f"Gave RMSE ratio: {avg_rmse/y_train_mean}")
    print(f"And R2: {avg_r2}")
    print(f"Most important features: {list(pd.DataFrame(top_features)[0])}")
    print(f"Parameters: {xgb.best_params_}")
        