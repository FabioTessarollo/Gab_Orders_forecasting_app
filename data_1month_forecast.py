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


perimeter = ['ALL', 'BAL', 'YSL', 'GIV']
train_result = pd.read_csv('train/scores_df.csv')
target = 'month_variation'

for brand in perimeter:
    brand_p = train_result[train_result['brand'] == brand]
    features = eval(brand_p['features'].values[0])
    params = eval(brand_p['params'].values[0])

    df = pd.read_csv(f'train_month_change/{brand}.csv', index_col='date')

    df['month'].astype("category")
    df['day'].astype("category")
    df['month_variation'].astype("category")

    df = df[features + [target]]

    cat_cols = ['month', 'day']
    X_cat_cols = [col for col in df.columns if col in cat_cols]
    X_scalar_cols = [col for col in df.columns if col != target and col not in X_cat_cols]

    xgb = XGBClassifier(**params, verbosity = 0, n_jobs = 5, verbose=False, scoring = 'f1')

    X_cols = [col for col in df.columns if col != target]
    X = df[X_cols]
    y = df[[target]]

    #togliere split, train su tutto dataset
    test_index_start = int(len(X)*0.80)
    train_index = list(range(0, test_index_start))
    test_index = list(range(test_index_start, len(X)))
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    X_scaler = MinMaxScaler()
    scaled_data = X_scaler.fit_transform(X_train[X_scalar_cols])
    X_train_scaled = pd.DataFrame(scaled_data, columns=X_scalar_cols, index=X_train.index)
    if len(X_cat_cols) > 0:
        X_train_scaled = pd.concat([X_train_scaled, X_train[X_cat_cols]], axis=1)            

    scaled_data = X_scaler.transform(X_test[X_scalar_cols])
    X_test_scaled = pd.DataFrame(scaled_data, columns=X_scalar_cols, index=X_test.index)
    if len(X_cat_cols) > 0:
        X_test_scaled = pd.concat([X_test_scaled, X_test[X_cat_cols]], axis=1)

    xgb.fit(X_train_scaled, y_train)
    
    y_pred = pd.DataFrame(xgb.predict(X_test_scaled), columns = [target])

    #Evaluation

    f1 = f1_score(y_test, y_pred, average = 'binary', pos_label=1)
    print(f'{brand} f1 score: {f1}')


