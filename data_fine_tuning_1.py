from sklearn.preprocessing import MinMaxScaler

from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import TimeSeriesSplit
from scipy.fft import fft

from sklearn.metrics import f1_score
from xgboost import plot_importance, plot_tree

import numpy as np
import pandas as pd
import itertools

#TOGLIERE CASI ISOLATI DI MONTH VARIATION!


perimeter = ['BAL', 'YSL', 'GIV', 'ALL']
target = 'month_variation'
#initial_features_comb = ['qty_V_0', 'qty-2', 'qty-1', 'Google_imm_rol_mean', 'qty_rol_std', 'qty_V_4', 'qty_V_3', 'qty_V_1', 'Google_imm_rol_std', 'kering_stock_price_rol_mean', 'lvmh_stock_price_rol_mean', 'lvmh_stock_price_rol_std', 'kering_stock_price_rol_std', 'qty-0', 'month']
#initial_features_comb = ['qty_rol_mean', 'qty_rol_std', 'qty_rol_max', 'qty_rol_min', 'Google_imm_rol_mean', 'Google_imm_rol_std', 'lvmh_stock_price_rol_mean', 'lvmh_stock_price_rol_std', 'kering_stock_price_rol_mean', 'kering_stock_price_rol_std', 'qty-0', 'qty_V_0', 'qty-1', 'qty_V_1', 'qty-2', 'qty_V_2', 'qty-3', 'qty_V_3', 'qty-4', 'qty_V_4', 'qty-5', 'qty_V_5', 'month', 'day']
initial_features_comb = ['qty_rol_mean', 'qty_rol_std', 'Google_imm_rol_mean', 'Google_imm_rol_std', 'lvmh_stock_price_rol_mean', 'lvmh_stock_price_rol_std', 'kering_stock_price_rol_mean', 'kering_stock_price_rol_std', 'qty-0', 'qty_V_0', 'qty-1', 'qty_V_1', 'qty-2', 'qty_V_2', 'qty_V_3', 'qty_V_4', 'qty_V_5']
cat_cols = ['month_1','month_2','month_3','month_4','month_5','month_6','month_7','month_8','month_9','month_10','month_11','month_12','week_of_month_1','week_of_month_2','week_of_month_3','week_of_month_4','week_of_month_5']
initial_features_comb = initial_features_comb + cat_cols

scores_df = pd.DataFrame(columns = ['brand', 'score', 'features', 'params']) 

cv_splits = 3
tscv = TimeSeriesSplit(n_splits = cv_splits)  

parameters = {
    'nthread': [4],
    'objective': ['binary:logistic'], 
    'learning_rate': [0.06, 0.07, 0.08],
    'max_depth': [6, 7, 8],
    'min_child_weight': [4, 5, 6],
    'subsample': [0.7, 0.8, 0.9],
    'colsample_bytree': [0.7, 0.8],
    'n_estimators': [500, 1000],
    'scale_pos_weight': [1, 2, 3, 4, 5]
}

param_combinations = list(itertools.product(
    parameters['nthread'],
    parameters['objective'],
    parameters['learning_rate'],
    parameters['max_depth'],
    parameters['min_child_weight'],
    parameters['subsample'],
    parameters['colsample_bytree'],
    parameters['n_estimators'],
    parameters['scale_pos_weight']
))

def add_row_to_df(df, dict_row):
    return pd.concat([df, pd.DataFrame(dict_row)], ignore_index=True)


def get_feature_comb_score(brand, features_comb):

    print(f"Features combination: {' '.join(features_comb)}")

    df = pd.read_csv(f'train_month_change/{brand}.csv', index_col='date')

    #df = df[df.index <= '2022-03-31'] ###########################--------------------------------overfitting extra test

    for col in cat_cols:
        df[col] = df[col].astype("category")

    df = df[[target] + [c for c in features_comb if c in df.columns]]

    X_cat_cols = [col for col in df.columns if col in cat_cols]
    X_scalar_cols = [col for col in df.columns if col != target and col not in X_cat_cols]

    X_cols = [col for col in df.columns if col != target]
    X = df[X_cols]
    y = df[[target]]

    best_score = 0
    #divider = 0
    #weight = 0
    best_params = None
    best_features_by_importance = []

    for combo in param_combinations:

        params = {
            'nthread': combo[0],
            'objective': combo[1],
            'learning_rate': combo[2],
            'max_depth': combo[3],
            'min_child_weight': combo[4],
            'subsample': combo[5],
            'colsample_bytree': combo[6],
            'n_estimators': combo[7],
            'scale_pos_weight': combo[8]
        }

        f1_values = []

        xgb = XGBClassifier(**params, verbosity = 0, scoring = 'f1', enable_categorical = True)

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
            #weight = weight + 1.05**weight
            #divider = divider + weight
            f1_values.append(f1) #*weight

        final_score = final_score = np.mean(f1_values)#sum(f1_values)/divider

        if final_score > best_score:
            best_score = final_score
            best_params = params
            feature_scores = xgb.get_booster().get_score(importance_type='weight') #basata soltanto sull'ultimo split!
            sorted_features = sorted(feature_scores.items(), key=lambda x: x[1], reverse=True)
            best_features_by_importance = list(pd.DataFrame(sorted_features)[0])
            y_pred_best = y_pred
            y_test_best = y_test

    #print(f"Performed with a F1 of: {f1_values}")
    #print(f"Most important features: {features_by_importance}")
    print(list(y_pred_best[target]))
    print(list(y_test_best[target]))
    print(f"Parameters: {best_params}")
    print(f"F1 score: {best_score}")
    return best_features_by_importance, best_params, best_score


for brand in perimeter:
    print(f"\n###########################Brand {brand}:")
    initial_features, params, score = get_feature_comb_score(brand, initial_features_comb)
    best = {'brand': brand, 'score': score, 'features': [initial_features], 'params': [params]}
    max_score = score
    for n in range(4, 11):
        comb = initial_features[:n]
        features, params, score = get_feature_comb_score(brand, comb)
        if score > max_score:
            best = {'brand': brand, 'score': score, 'features': [features], 'params': [params]}
            max_score = score
    scores_df = add_row_to_df(scores_df, best)

scores_df.to_csv('train/scores_df.csv')

#'brand', 'score', 'features', 'params'


        



        