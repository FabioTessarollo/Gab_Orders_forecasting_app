from sklearn.preprocessing import MinMaxScaler

from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import TimeSeriesSplit
from scipy.fft import fft

from sklearn.metrics import f1_score
from xgboost import plot_importance, plot_tree

import numpy as np
import pandas as pd
from itertools import combinations


perimeter = ['BAL', 'YSL', 'GIV', 'ALL']
train_result = pd.read_csv('train/scores_df.csv')
target = 'month_variation'

for brand in perimeter:
    brand_p = train_result[train_result['brand'] == brand]
    features = eval(brand_p['features'].values[0])
    params = eval(brand_p['params'].values[0])

    df = pd.read_csv(f'train_month_change/{brand}.csv', index_col='date')
    X_pred = pd.read_csv(f'train_month_change/{brand}_X_forecast.csv', index_col='date')
    df = pd.concat([df, X_pred], ignore_index=True)

    df['month'] = df['month'].astype("category")
    df['day'] = df['day'].astype("category")
    df['month_variation'] = df['month_variation'].astype("category")

    df = df[features + [target]]

    cat_cols = ['month', 'day']
    X_cat_cols = [col for col in df.columns if col in cat_cols]
    X_scalar_cols = [col for col in df.columns if col != target and col not in X_cat_cols]

    xgb = XGBClassifier(**params, verbosity = 0, scoring = 'f1')

    X_cols = [col for col in df.columns if col != target]
    X = df[X_cols]
    y = df[[target]]

    X_scaler = MinMaxScaler()
    scaled_data = X_scaler.fit_transform(X[X_scalar_cols])
    X_scaled = pd.DataFrame(scaled_data, columns=X_scalar_cols, index=X.index)
    if len(X_cat_cols) > 0:
        X_scaled = pd.concat([X_scaled, X[X_cat_cols]], axis=1)

    X_train = X.iloc[:-1] 
    X_pred  = X.tail(1)
    y_train = y.iloc[:-1]            

    xgb.fit(X_train, y_train)
    
    y_pred = pd.DataFrame(xgb.predict(X_pred), columns = [target])

    #Evaluation

    print(f"{brand}: {y_pred.values[0]}")


