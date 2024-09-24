from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import TimeSeriesSplit
from scipy.fft import fft
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd

target = 'qty'
perimeter = ['BAL', 'YSL', 'GIV', 'ALL']
rolling_look_back = 4
look_back = 8

def add_rolling_features(df, cols, rolling_look_back):
    for col in cols:
        df[col+'_rol_mean'] = df[col].rolling(rolling_look_back).mean()
        df[col+'_rol_std'] = df[col].rolling(rolling_look_back).std()
        df[col+'_rol_max'] = df[col].rolling(rolling_look_back).max()
        df[col+'_rol_min'] = df[col].rolling(rolling_look_back).min()
    return df

def add_look_back(df, cols, look_back):
    for col in cols:
        for i in range(0,look_back):
            df[col+'-'+str(i)] = df[col].shift(i)
            df[col+'_V_'+str(i)] = (df[col].shift(i) - df[col].shift(i + 1)) / df[col].shift(i + 1)
    return df

def add_month_variation_target(df, target):
    four_weeks = 4
    rolling_back = df[target].rolling(window=four_weeks).sum()#.mean()
    rolling_ahead =  df[target].shift(-1) + df[target].shift(-2) + df[target].shift(-3) + df[target].shift(-4) #df[target].shift(-1) + df[target].shift(-2) + df[target].shift(-3) + df[target].shift(-4)
    month_variation_Series = (rolling_ahead - rolling_back) / rolling_back
    quantile = month_variation_Series.quantile(0.75)
    print(quantile)
    df['month_variation'] = month_variation_Series.apply(lambda x : 1 if x > quantile else 0)
    # df['prev_to_positive'] =  df['month_variation'].shift(1)
    # df['next_to_positive'] =  df['month_variation'].shift(-1)
    # df['isolate_check'] = df['month_variation'] + df['prev_to_positive'] +  df['next_to_positive']
    # df['month_variation'] = df['isolate_check'].apply(lambda x : 1 if x > 1 else 0)
    # df.drop(['prev_to_positive', 'next_to_positive', 'isolate_check'], axis = 1, inplace = True)
    df.dropna(inplace = True)
    df = df.drop(df.tail(4).index)
    return df

def add_one_step_target(df, target):
    df[target +'+1'] = df[target].shift(-1)
    return df

def add_time_features(df):
    df.index = pd.to_datetime(df.index)
    df['date'] = df.index
    df['month'] = df['date'].dt.month
    df['year'] = df['date'].dt.year
    df['week_of_month'] = df.groupby(['year', 'month']).cumcount() + 1
    df = pd.get_dummies(df, columns= ['month'], dtype=float)
    df = pd.get_dummies(df, columns= ['week_of_month'], dtype=float)
    df.drop('year', axis = 1, inplace = True)
    df.set_index('date', drop = True, inplace = True)
    return df

def extract_features(df, raw_features):
    df = add_rolling_features(df, raw_features, rolling_look_back)
    df = add_look_back(df, raw_features, look_back)
    df = add_time_features(df)
    return df

def set_target(df, type_of_model):
    if type_of_model == 'one step':
        df = add_one_step_target(df, target)
        df.drop(target, axis = 1, inplace = True)
        return df
    elif type_of_model == 'month change':
        df = add_month_variation_target(df, target)
        df.drop(target, axis = 1, inplace = True)
        return df


for brand in perimeter:

    print(brand)

    df = pd.read_csv(f'sourcing/{brand}.csv', index_col='date')
    df_features_and_target = df.columns
    df_features = df_features_and_target.drop(target)  

    df = extract_features(df, df_features_and_target)
    df.drop(df_features, axis = 1, inplace = True)

    #df_1 = set_target(df.copy(deep = True), 'one step')
    #df_1.dropna(inplace = True)
    #df_1.to_csv(f'train/{brand}.csv', index_label='date')

    X_forecast = df.tail(1).drop(target, axis = 1)
    X_forecast.to_csv(f'features_extraction/{brand}_X_forecast.csv', index_label='date')
    df_m = set_target(df.copy(deep = True), 'month change')
    df_m.dropna(inplace = True)
    df_m.to_csv(f'features_extraction/{brand}.csv', index_label='date')




