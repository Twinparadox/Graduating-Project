import pandas as pd
import datetime

def SS_converter(df):        
    df = df.drop(['id'],axis=1)
    df_prev = df[df['date'] < '2018-05-04']
    df_prev[['close', 'open', 'high', 'low']] = df_prev[['close', 'open', 'high', 'low']] / 50
    df_prev['volume'] = df_prev['volume'] * 50
    df[df['date'] < '2018-05-04'] = df_prev
    
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
    df_prev = df[df['date'] < '2018-05-04']
    df_prev[['close', 'open', 'high', 'low']] = df_prev[['close', 'open', 'high', 'low']] / 50
    df_prev['volume'] = df_prev['volume'] * 50
    df[df['date'] < '2018-05-04'] = df_prev
    
    df['diff'] = df['close'].diff()
    
    df_train = df[df['date'] < '2017-07-01']
    df_train = df_train[df_train['date'] > '2009-12-31']
    df_valid = df[df['date'] > '2017-06-30']
    df_valid = df_valid[df_valid['date'] < '2018-07-01']
    df_test = df[df['date'] > '2018-06-30']
    
    df_train.to_csv('stockdata/SS.csv')
    df_valid.to_csv('stockdata/SS_2018.csv')
    df_test.to_csv('stockdata/SS_2019.csv')
    
    
    
df_SS = pd.read_csv('stockdata/삼성전자.csv')
SS_converter(df_SS)

#df_NAVER = pd.read_csv('stockdata/NAVER.csv')