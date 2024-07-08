from sklearn.preprocessing import MinMaxScaler

from xgboost import XGBRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import TimeSeriesSplit
from scipy.fft import fft

from sklearn.metrics import root_mean_squared_error, r2_score
from xgboost import plot_importance, plot_tree

import numpy as np
import pandas as pd
from itertools import combination

perimeter = ['GIV', 'YSL', 'GIV', 'ALL']
features = ['qty']
target = 'qty'

rolling_features = features
rolling_look_back = 4

look_back = 4

parameters = {'nthread':[4],
              'objective':['reg:squarederror'],
              'learning_rate': [.04, 0.05, .06],
              'max_depth': [4, 5, 6],
              'min_child_weight': [4],
              'silent': [1],
              'subsample': [0.5], #0.7
              'colsample_bytree': [0.7],
              'n_estimators': [500]}


def add_rolling_features(df, cols, rolling_look_back, shift):
    for col in cols:
        df[col+'_rol_mean'] = df[col].shift(shift).rolling(rolling_look_back).mean()
        df[col+'_rol_std'] = df[col].shift(shift).rolling(rolling_look_back).std()
        df[col+'_rol_max'] = df[col].shift(shift).rolling(rolling_look_back).max()
        df[col+'_rol_min'] = df[col].shift(shift).rolling(rolling_look_back).min()
        if col != target:
            df.drop(col, axis = 1, inplace = True)
    return df

def look_back_target(df, shift):
    for i in range(1,look_back):
        df[target+'-'+str(i)] = df[target].shift(shift)
        df[target+'_V_'+str(i)] = (df[target].shift(shift)) - df[target].shift(shift + 1) / df[target].shift(shift + 1)
    return df

def scale(df):
    X_scaler = MinMaxScaler()
    scaled_data = X_scaler.fit_transform(df)
    return pd.DataFrame(scaled_data, columns=df.columns, index=df.index), X_scaler

def add_time_features(df):
    df.index = pd.to_datetime(df.index)
    df['date'] = df.index
    df['month'] = df['date'].dt.month
    df['month'].astype("category")
    df['day'] = df['date'].dt.day
    df['day'].astype("category")
    df.set_index('date', drop = True, inplace = True)
    return df


def train(df):

    xgb_model = XGBRegressor(verbosity = 0)
    xgb = GridSearchCV(xgb_model, parameters, cv = 2, n_jobs = 5, verbose=True)

    df, scaler = scale(df)

    df = add_time_features(df)

    X_cols = [col for col in df.columns if col != target]

    X = df[X_cols]
    y = df[[target]]

    xgb.fit(X, y)

    return xgb.best_estimator_, scaler


for brand in perimeter:

    #preprocessing
    df_original = pd.read_csv(f'{brand}.csv', index_col='date')
    df = df_original.copy()
    df = df[features]

    #look back target

    #rolling windows
    df = add_rolling_features(df, rolling_features, 6, 1)
    
    df.dropna(inplace = True)

    model, scaler = train(df)

    #remap features 
    df['qty-1'] = df[target]
    for i in range(2,look_back-1):
        df[target+'-'+str(i)] = df[target+'-'+str(i - 1)]
        df[target+'_V_'+str(i)] = df[target+'_V_'+str(i - 1)]
    
    #predict next step
    last_step = df.tail(1)

    last_step = scaler.transform(last_step)

    qty_one_step_ahead_scaled = model.predict(last_step)

    qty_one_step_ahead = scaler.inverse_transform(qty_one_step_ahead_scaled)

    df = pd.concat([df, ])



    #training





        