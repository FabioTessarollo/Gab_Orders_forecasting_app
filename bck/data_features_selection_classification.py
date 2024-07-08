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
target = 'month_variation'
initial_features_comb = ['qty_V_0', 'qty-2', 'qty-1', 'Google_imm_rol_mean', 'qty_rol_std', 'qty_V_4', 'qty_V_3', 'qty_V_1', 'Google_imm_rol_std', 'kering_stock_price_rol_mean', 'lvmh_stock_price_rol_mean', 'lvmh_stock_price_rol_std', 'kering_stock_price_rol_std', 'qty-0', 'month']#['qty_rol_mean', 'qty_rol_std', 'qty_rol_max', 'qty_rol_min', 'Google_imm_rol_mean', 'Google_imm_rol_std', 'lvmh_stock_price_rol_mean', 'lvmh_stock_price_rol_std', 'kering_stock_price_rol_mean', 'kering_stock_price_rol_std', 'qty-0', 'qty_V_0', 'qty-1', 'qty_V_1', 'qty-2', 'qty_V_2', 'qty-3', 'qty_V_3', 'qty-4', 'qty_V_4', 'qty-5', 'qty_V_5', 'month', 'day']

scores_df = pd.DataFrame(columns = ['brand', 'score', 'features', 'params']) 

cv_splits = 3
tscv = TimeSeriesSplit(n_splits = cv_splits)  

parameters = {
    'nthread': [4],
    'objective': ['binary:logistic'], 
    'learning_rate': [0.06, 0.07],  # Fine-tune learning rate #, 0.08
    'max_depth': [6, 7, 8],  # Control tree depth
    'min_child_weight': [4, 5],  # Additional values to test #, 6
    'silent': [1],
    'subsample': [0.7, 0.8],  # More values to test #, 0.9
    'colsample_bytree': [0.7],  # More values to test , 0.8
    'n_estimators': [500, 1000],  # Test fewer trees as well
    'scale_pos_weight': [2, 3]  # Important for handling imbalance  1 ,  4, 5
}

def add_row_to_df(df, dict_row):
    return pd.concat([df, pd.DataFrame(dict_row)], ignore_index=True)


def get_feature_comb_score(brand, features_comb):

    print(f"Features combination: {' '.join(features_comb)}")

    df = pd.read_csv(f'train_month_change/{brand}.csv', index_col='date')

    #df = df[df.index <= '2022-03-31'] ###########################--------------------------------overfitting extra test

    df['month'].astype("category")
    df['day'].astype("category")
    df['month_variation'].astype("category")

    df = df[[target] + [c for c in features_comb if c in df.columns]]

    cat_cols = ['month', 'day']
    X_cat_cols = [col for col in df.columns if col in cat_cols]
    X_scalar_cols = [col for col in df.columns if col != target and col not in X_cat_cols]

    xgb_model = XGBClassifier(verbosity = 0)
    xgb = GridSearchCV(xgb_model, parameters, cv = 3, n_jobs = 5, verbose=False, scoring = 'f1' )

    X_cols = [col for col in df.columns if col != target]
    X = df[X_cols]
    y = df[[target]]

    f1_values = []
    weight = 0
    divider = 0

    for train_index, test_index in tscv.split(X):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
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

        #fitting and pred

        xgb.fit(X_train_scaled, y_train)
        
        y_pred = pd.DataFrame(xgb.predict(X_test_scaled), columns = [target])

        #Evaluation

        f1 = f1_score(y_test, y_pred, average = 'binary', pos_label=1)
        weight = weight + 1.05**weight
        divider = divider + weight
        f1_values.append(f1*weight)
        feature_scores   = xgb.best_estimator_.get_booster().get_score(importance_type='weight')
        sorted_features = sorted(feature_scores.items(), key=lambda x: x[1], reverse=True)
        features_by_importance = list(pd.DataFrame(sorted_features)[0])

    #print(f"Performed with a F1 of: {f1_values}")
    #print(f"Most important features: {features_by_importance}")
    best_params = xgb.best_params_
    print(f"Parameters: {best_params}")
    final_score = sum(f1_values)/divider
    print(f"F1 score: {final_score}")
    return features_by_importance, best_params, final_score


for brand in perimeter:
    print(f"\n###########################Brand {brand}:")
    initial_features, params, score = get_feature_comb_score(brand, initial_features_comb)
    best = {'brand': brand, 'score': score, 'features': [initial_features], 'params': [params]}
    max_score = score
    for n in range(2, 11):
        comb = initial_features[:n]
        features, params, score = get_feature_comb_score(brand, comb)
        if score > max_score:
            best = {'brand': brand, 'score': score, 'features': [features], 'params': [params]}
            max_score = score
    scores_df = add_row_to_df(scores_df, best)

scores_df.to_csv('train/scores_df.csv')

#'brand', 'score', 'features', 'params'


        



        