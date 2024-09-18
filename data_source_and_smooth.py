import pandas as pd
import matplotlib.pyplot as plt
import datetime
import pyodbc 
import yfinance as yf

perimeter = ['BAL', 'YSL', 'GIV', 'ALL']
scope = 'BOR'

SERVER = 'GABORCL01'
DATABASE = 'RVLKPI'
USERNAME = 'RVLKPI'
PASSWORD = 'RVLKPI!'
connectionString = f'DRIVER={{SQL Server}};SERVER={SERVER};DATABASE={DATABASE};UID={USERNAME};PWD={PASSWORD}'
conn = pyodbc.connect(connectionString)

df_google_im = pd.read_csv('google_trends.csv')
df_google_im.set_index('date', inplace = True)
df_google_im.index = pd.to_datetime(df_google_im.index)

#LVMH
ticker = 'MC.PA'
start_date = '2018-11-25'
end_date = '2024-12-31'
stock_data = yf.download(ticker, start=start_date, end=end_date, interval='1d')
lvmh_stock_price = pd.DataFrame(stock_data['Close'].resample('W-SUN').last())
lvmh_stock_price = lvmh_stock_price.rename(columns= {'Close': 'lvmh_stock_price'})

#KERING
ticker = 'KER.PA'
start_date = '2018-11-25'
end_date = '2024-12-31'
stock_data = yf.download(ticker, start=start_date, end=end_date, interval='1d')
kering_stock_price = pd.DataFrame(stock_data['Close'].resample('W-SUN').last())
kering_stock_price = kering_stock_price.rename(columns= {'Close': 'kering_stock_price'})

#FTSE MIB 
ticker = 'FTSEMIB.MI'
start_date = '2018-11-25'
end_date = '2024-12-31'
stock_data = yf.download(ticker, start=start_date, end=end_date, interval='1d')
ftse_mib_price = pd.DataFrame(stock_data['Close'].resample('W-SUN').last())
ftse_mib_price = ftse_mib_price.rename(columns= {'Close': 'ftse_mib_price'})

def get_sunday_of_week(date):
    if date.weekday() == 6:
        return date
    else:
        for d in range(6):
            date = date + datetime.timedelta(days=1)
            if date.weekday() == 6: 
                return date

for brand in perimeter:
    ######qty and price extraction from GAB db
    if brand == 'ALL':
        cur = conn.cursor()
        query = f"""select  
                    coalesce(DATARIFERINENTOCLIENTE, dataordine) as date, 
                    'ALL' as brand,
                    sum(QUANTITATOTALE) as qty,
                    coalesce(avg(case when QUANTITASPEDITA = 0 then null else VALORESPEDITO/QUANTITASPEDITA end), 0) as price
                    from dbo.Kpi_TestataOrdiniClienti ktoc 
                    left join dbo.Kpi_Clienti kc  on kc.id_cliente = ktoc.id_cliente 
                    left join dbo.Kpi_Brands kb ON KB.ID_BRAND = KTOC.ID_BRAND 
                    left join dbo.Kpi_RigheOrdiniClienti kro on kro.id_testata = ktoc.id_testata 
                    left join dbo.Kpi_Articoli ka   on ka.ID_ARTICOLO = kro.ID_ARTICOLO 
                    left join dbo.Kpi_ClasseArticoli kcl on kcl.ID_classe = ka.ID_classe 
                    where 
                    CODICECLASSE = '{scope}' and TIPORIGA = 'Normale'
                    group by  coalesce(DATARIFERINENTOCLIENTE, dataordine)
                    """
        cur.execute(query)
        res = cur.fetchall()
    else:
        cur = conn.cursor()
        query = f"""select  
                    coalesce(DATARIFERINENTOCLIENTE, dataordine) as date, 
                    CODICEBRAND as brand,
                    sum(QUANTITATOTALE) as qty,
                    coalesce(avg(case when QUANTITASPEDITA = 0 then null else VALORESPEDITO/QUANTITASPEDITA end), 0) as price
                    from dbo.Kpi_TestataOrdiniClienti ktoc 
                    left join dbo.Kpi_Clienti kc  on kc.id_cliente = ktoc.id_cliente 
                    left join dbo.Kpi_Brands kb ON KB.ID_BRAND = KTOC.ID_BRAND 
                    left join dbo.Kpi_RigheOrdiniClienti kro on kro.id_testata = ktoc.id_testata 
                    left join dbo.Kpi_Articoli ka   on ka.ID_ARTICOLO = kro.ID_ARTICOLO 
                    left join dbo.Kpi_ClasseArticoli kcl on kcl.ID_classe = ka.ID_classe 
                    where 
                    CODICEBRAND like '%{brand}%' 
                    and CODICECLASSE = '{scope}' and TIPORIGA = 'Normale'  
                    group by  coalesce(DATARIFERINENTOCLIENTE, dataordine), CODICEBRAND 
                    """
        cur.execute(query)
        res = cur.fetchall()

    res_list = []
    for row in res:
        res_list.append(tuple(row))
    df = pd.DataFrame(res_list, columns=['date', 'brand', 'qty', 'price'])
    df.drop('price', axis = 1, inplace = True)

    ########resampling and smoothing
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values(by = 'date')
    df['date'] = df['date'].apply(get_sunday_of_week)
    df = df.groupby(['date', 'brand']).agg({
        'qty':'sum',
    }).reset_index()
    df.set_index('date', inplace = True)

    start_date = df.index.min()
    end_date = df.index.max()

    sundays = pd.date_range(start=start_date, end=end_date, freq='W-SUN')
    df_sundays = pd.DataFrame(index=sundays)

    df = pd.merge(df_sundays, df, how = 'left', left_index=True, right_index=True)
    
    df['qty'] = df['qty'].ewm(halflife = 4, adjust=True).mean()

    df = df.replace(0, None)
    df = df.ffill()

    df['qty'] = df['qty'].ewm(alpha = 0.5, adjust=True).mean()

    ########google data
    if brand != 'ALL':
        df = pd.merge(df, df_google_im[[brand]], how = 'left', left_index=True, right_index=True)
        df = df.rename(columns = {brand: 'Google_imm'})

    #######stocks
    df = pd.merge(df, lvmh_stock_price, how = 'left', left_index=True, right_index=True)
    df = pd.merge(df, kering_stock_price, how = 'left', left_index=True, right_index=True)
    df = pd.merge(df, ftse_mib_price, how = 'left', left_index=True, right_index=True)

    df.drop('brand', axis = 1, inplace = True)

    df = df.ffill()

    df.to_csv(f'sourcing/{brand}.csv', index_label='date')

    #long format
    #df_long = pd.DataFrame(columns = ['ds', 'value'])
    #df['ds'] = df.index
    #for col in df.columns:
    #    if col != 'ds':
    #        df_long_portion = df[['ds', col]].rename(columns = {col:'value'})
    #        df_long_portion['unique_id'] = col 
    #        df_long_portion = df_long_portion.set_index('unique_id', drop = True)
    #        df_long = pd.concat([df_long, df_long_portion])
    #df_long.to_csv(f'{brand}_long_format.csv', index=True, index_label='unique_id')