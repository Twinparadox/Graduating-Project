import pandas as pd
import datetime

def SS_converter(df):        
    df = df.drop(['id'],axis=1)
    df_prev = df[df['date'] < '2018-05-04']
    df_prev[['close', 'open', 'high', 'low']] = df_prev[['close', 'open', 'high', 'low']] / 50
    df_prev['volume'] = df_prev['volume'] * 50
    df[df['date'] < '2018-05-04'] = df_prev
    df['MA5'] = df['close'].rolling(window=5, min_periods=1).mean()
    df['MA10'] = df['close'].rolling(window=10, min_periods=1).mean()
    
    df['diff'] = df['close'].diff()
    
    df_train = df[df['date'] < '2017-07-01']
    df_train = df_train[df_train['date'] > '2009-12-31']
    df_valid = df[df['date'] > '2017-06-30']
    df_valid = df_valid[df_valid['date'] < '2018-07-01']
    df_test = df[df['date'] > '2018-06-30']
    
    df_train.to_csv('stockdata/SS.csv', index=False)
    df_valid.to_csv('stockdata/SS_2018.csv', index=False)
    df_test.to_csv('stockdata/SS_2019.csv', index=False)
    
def NAVER_Converter(df):
    df = df.drop(['id'],axis=1)
    df[['close', 'open', 'high', 'low', 'volume']] = df[['close', 'open', 'high', 'low', 'volume']].astype(float)
    df_prev = df[df['date'] < '2018-10-12']
    df_prev[['close', 'open', 'high', 'low']] = df_prev[['close', 'open', 'high', 'low']] / 5
    df_prev['volume'] = df_prev['volume'] * 5
    df[df['date'] < '2018-10-12'] = df_prev
    df['MA5'] = df['close'].rolling(window=5, min_periods=1).mean()
    df['MA10'] = df['close'].rolling(window=10, min_periods=1).mean()
    
    df['diff'] = df['close'].diff()
    
    df_train = df[df['date'] < '2017-07-01']
    df_train = df_train[df_train['date'] > '2009-12-31']
    df_valid = df[df['date'] > '2017-06-30']
    df_valid = df_valid[df_valid['date'] < '2018-07-01']
    df_test = df[df['date'] > '2018-06-30']
    
    df_train.to_csv('stockdata/NV.csv', index=False)
    df_valid.to_csv('stockdata/NV_2018.csv', index=False)
    df_test.to_csv('stockdata/NV_2019.csv', index=False)
    
    
    
df_SS = pd.read_csv('stockdata/삼성전자.csv')
SS_converter(df_SS)

df_NAVER = pd.read_csv('stockdata/NAVER.csv')
NAVER_Converter(df_NAVER)