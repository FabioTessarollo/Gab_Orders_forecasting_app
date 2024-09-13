from sklearn.preprocessing import MinMaxScaler

from xgboost import XGBClassifier
import optuna
from sklearn.model_selection import TimeSeriesSplit

from sklearn.metrics import f1_score

import numpy as np
import pandas as pd


perimeter = ['BAL', 'YSL', 'GIV', 'ALL']
train_result = pd.read_csv('train/scores_df_optuna.csv')
target = 'month_variation'
cat_cols = ['month_1','month_2','month_3','month_4','month_5','month_6','month_7','month_8','month_9','month_10','month_11','month_12','week_of_month_1','week_of_month_2','week_of_month_3','week_of_month_4','week_of_month_5']


for brand in perimeter:
    brand_p = train_result[train_result['brand'] == brand]
    features = eval(brand_p['features'].values[0])
    params = eval(brand_p['params'].values[0])

    df = pd.read_csv(f'features_extraction/{brand}.csv', index_col='date')

    for col in cat_cols:
        df[col] = df[col].astype("category")

    df = df[features + [target]]

    X_cat_cols = [col for col in df.columns if col in cat_cols]
    X_scalar_cols = [col for col in df.columns if col != target and col not in X_cat_cols]

    xgb = XGBClassifier(**params, verbosity = 0, scoring = 'f1', enable_categorical = True)

    X_cols = [col for col in df.columns if col != target]
    X = df[X_cols]
    y = df[[target]]

    train_index = X.index <= '2023-07-31'
    test_index = X.index > '2023-07-31'
    X_train, X_test = X.loc[train_index], X.loc[test_index]
    y_train, y_test = y.loc[train_index], y.loc[test_index]         

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

    f1 = f1_score(y_test, y_pred, average = 'binary', pos_label=1)
    print(list(y_test[target]))
    print(list(y_pred[target]))

    print(f"{brand}: {f1}")



        



        