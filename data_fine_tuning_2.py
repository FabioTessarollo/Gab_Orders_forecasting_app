from sklearn.preprocessing import MinMaxScaler

from xgboost import XGBClassifier
import optuna
from sklearn.model_selection import TimeSeriesSplit

from sklearn.metrics import f1_score

import numpy as np
import pandas as pd


perimeter = ['BAL', 'YSL', 'GIV', 'ALL']
scores_df = pd.read_csv('train/scores_df.csv')
target = 'month_variation'
cat_cols = ['month_1','month_2','month_3','month_4','month_5','month_6','month_7','month_8','month_9','month_10','month_11','month_12','week_of_month_1','week_of_month_2','week_of_month_3','week_of_month_4','week_of_month_5']


scores_df_optuna = pd.DataFrame(columns = ['brand', 'score', 'features', 'params']) 

def add_row_to_df(df, dict_row):
    return pd.concat([df, pd.DataFrame(dict_row)], ignore_index=True)


def get_bounds(data_dict, key, is_int):
    if key not in data_dict:
        return None
    
    value = data_dict[key]
    
    if not isinstance(value, (int, float)):
        raise ValueError("Value must be an int or float")

    if is_int:
        lower_bound = int(round(value * 0.8))
        upper_bound = int(round(value * 1.2))
    else:
        lower_bound = round(value * 0.8, 8)
        upper_bound = round(value * 1.2, 8)
    
    return lower_bound, upper_bound


for brand in perimeter:
    brand_p = scores_df[scores_df['brand'] == brand]
    features = eval(brand_p['features'].values[0])
    params_grid = eval(brand_p['params'].values[0])

    df = pd.read_csv(f'features_extraction/{brand}.csv', index_col='date')

    for col in cat_cols:
        df[col] = df[col].astype("category")

    df = df[features + [target]]

    X_cat_cols = [col for col in df.columns if col in cat_cols]
    X_scalar_cols = [col for col in df.columns if col != target and col not in X_cat_cols]

    X_cols = [col for col in df.columns if col != target]
    X = df[X_cols]
    y = df[[target]]

    test_index_start = int(len(X)*0.74)
    train_index = list(range(0, test_index_start))
    test_index = list(range(test_index_start, len(X)))
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]


    #test_index_start = int(len(X)*0.66)
    #validation_index_start = int(len(X)*0.77)
    #test_index = list(range(test_index_start, validation_index_start))
    #validation_index = list(range(validation_index_start, len(X)))
    #X_train, X_test, X_validation = X.iloc[train_index], X.iloc[test_index], X.iloc[validation_index]
    #y_train, y_test, X_validation = y.iloc[train_index], y.iloc[test_index], y.iloc[validation_index]

    X_scaler = MinMaxScaler()
    scaled_data = X_scaler.fit_transform(X_train[X_scalar_cols])
    X_train_scaled = pd.DataFrame(scaled_data, columns=X_scalar_cols, index=X_train.index)
    if len(X_cat_cols) > 0:
        X_train_scaled = pd.concat([X_train_scaled, X_train[X_cat_cols]], axis=1)            

    scaled_data = X_scaler.transform(X_test[X_scalar_cols])
    X_test_scaled = pd.DataFrame(scaled_data, columns=X_scalar_cols, index=X_test.index)
    if len(X_cat_cols) > 0:
        X_test_scaled = pd.concat([X_test_scaled, X_test[X_cat_cols]], axis=1)

    def objective(trial):
        params = {
            'nthread': 4,
            'objective': 'binary:logistic', 
            'n_estimators': trial.suggest_int('n_estimators', *get_bounds(params_grid, 'n_estimators', True)), 
            'max_depth': trial.suggest_int('max_depth', *get_bounds(params_grid, 'max_depth', True)),
            'learning_rate': trial.suggest_float('learning_rate', *get_bounds(params_grid, 'learning_rate', False)),
            'subsample': trial.suggest_float('subsample', get_bounds(params_grid, 'subsample', False)[0], 1),
            'colsample_bytree': trial.suggest_float('colsample_bytree', *get_bounds(params_grid, 'colsample_bytree', False)),
            'min_child_weight':trial.suggest_int('min_child_weight', *get_bounds(params_grid, 'min_child_weight', True)),
            'scale_pos_weight':trial.suggest_int('scale_pos_weight', *get_bounds(params_grid, 'scale_pos_weight', True))
        }

        xgb = XGBClassifier(**params, verbosity = 0, scoring = 'f1', enable_categorical = True)

        xgb.fit(X_train_scaled, y_train)
        
        y_pred = pd.DataFrame(xgb.predict(X_test_scaled), columns = [target])

        f1 = f1_score(y_test, y_pred, average = 'binary', pos_label=1)

        return f1
    
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=100)

    row = {'brand': brand, 'score': study.best_value, 'features': [features], 'params': [study.best_params]}
    scores_df_optuna = add_row_to_df(scores_df_optuna, row)

scores_df_optuna.to_csv('train/scores_df_optuna.csv')


        



        