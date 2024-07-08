from sklearn.preprocessing import MinMaxScaler

from xgboost import XGBRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import TimeSeriesSplit
from scipy.fft import fft

from sklearn.metrics import root_mean_squared_error
from sklearn.metrics import r2_score

import numpy as np
import pandas as pd
from itertools import combinations


perimeter = ['BAL'] #, 'YSL', 'GIV', 'ALL'
features = ['qty', 'price', 'Google_imm', 'lvmh_stock_price', 'kering_stock_price'] #, 'leather_price'
max_lag = 4
future = 1
look_back = 4
rolling_look_back = 6
max_exo_features = 4
tscv = TimeSeriesSplit(n_splits=3)  

def generate_combinations(original_list, max_elements):
    result = []
    for r in range(1, min(len(original_list), max_elements) + 1):
        result.extend(combinations(original_list, r))
    return [list(combo) for combo in result]


parameters = {'nthread':[4],
              'objective':['reg:squarederror'],
              'learning_rate': [.03, 0.05, .07], 
              'max_depth': [5, 6, 7],
              'min_child_weight': [4],
              'silent': [1],
              'subsample': [0.7],
              'colsample_bytree': [0.7],
              'n_estimators': [500]}


for brand in perimeter:

    #preprocessing

    df = pd.read_csv(f'{brand}.csv', index_col='date')
    df = df[features]

    for col in df.columns:

        #lagged features and variations
        for i in range(future,look_back):
            col_name = col+'-'+str(i)
            df[col+'-'+str(i)] = df[col].shift(i)
            df[col+'_V_'+str(i)] = (df[col] - df[col].shift(i)) / df[col].shift(i)

        #rolling windows
        df[col+'_rol_mean'] = df[col].shift(1).rolling(rolling_look_back).mean()
        df[col+'_rol_std'] = df[col].shift(1).rolling(rolling_look_back).std()
        df[col+'_rol_max'] = df[col].shift(1).rolling(rolling_look_back).max()
        df[col+'_rol_min'] = df[col].shift(1).rolling(rolling_look_back).min()

        #fft
        #if col == 'qty':
        #    fft_values = fft(df[col])
        #    fft_magnitude = np.abs(fft_values)
        #    fft_freq = np.fft.fftfreq(len(fft_values))
        #    dominant_freq = fft_freq[np.argmax(fft_magnitude)]
        #    df[col+'_dominant_freq'] = dominant_freq

        #removing today data except target
        if col != 'qty':
            df.drop(col, axis = 1, inplace = True)
    
    df.dropna(inplace = True)

    #isolate qty features
    qty_deriv = []
    exo = []
    for col in df.columns:
        if col != 'qty':
            if 'qty' in col and col != 'qty':
                qty_deriv.append(col)
            else:
                exo.append(col)

    #generate combs for exo features
    exo_comb = generate_combinations(exo, max_exo_features)

    best_RMSE = 0

    for exo_cols in exo_comb:

        X_cols = qty_deriv + exo_cols
        y_col = ['qty']
        
        df.index = pd.to_datetime(df.index)
        df['date'] = df.index
        df['month'] = df['date'].dt.month
        df['month'].astype("category")
        df['day'] = df['date'].dt.day
        df['day'].astype("category")
        df.set_index('date', drop = True, inplace = True)

        X_cols_cat = ['month', 'day']

        #training

        xgb_model = XGBRegressor(verbosity = 0)
        xgb = GridSearchCV(xgb_model, parameters, cv = 2, n_jobs = 5, verbose=True)
        print(X_cols)
        X = df[X_cols + X_cols_cat]
        y = df[y_col]

        r2_values = []
        rmse_values = []

        for train_index, test_index in tscv.split(X):
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]

            y_train_mean = y_train.mean()

            #scaling

            X_scaler = MinMaxScaler()
            scaled_data = X_scaler.fit_transform(X_train[X_cols])
            X_train.loc[:,X_cols] = pd.DataFrame(scaled_data, columns=X_cols, index=X_train.index)
            

            target_scaler = MinMaxScaler()
            scaled_data = target_scaler.fit_transform(y_train[y_col])
            y_train.loc[:,X_cols] = pd.DataFrame(scaled_data, columns=y_col, index=y_train.index)
            
            #fitting and pred

            xgb.fit(X_train, y_train)
            
            y_pred = pd.DataFrame(xgb.predict(X_test), columns = ['qty'])

            y_pred = target_scaler.inverse_transform(y_pred)
            y_test = target_scaler.inverse_transform(y_test)

            #€valuation

            r2 = r2_score(y_test, y_pred)
            rmse = root_mean_squared_error(y_test, y_pred)
            r2_values.append(r2)
            rmse_values.append(rmse)
        
        avg_r2 = np.mean(r2_values)
        avg_rmse = np.mean(rmse_values)

        print(f"Training obtained with exog features:\n{exo_cols}")
        print(f"Gave RMSE ratio: {avg_rmse}")
        print(f"And R2: {avg_r2}")

        current_rmse = avg_rmse/y_train_mean
        
        if best_RMSE < current_rmse:
            best_RMSE = current_rmse
        
        print(f"Best performace was obtained with exog features:\n{exo_cols}")
        print(f"WIth RMSE ratio as: {best_RMSE}")



         

#fit only on the train
#inverse 
    #target_min = scaler.data_min_[-1]
    #target_max = scaler.data_max_[-1]
    #y_pred_original = y_pred_scaled * (target_max - target_min) + target_min