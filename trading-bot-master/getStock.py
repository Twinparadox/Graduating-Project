from pandas_datareader import data
import yfinance as yf
yf.pdr_override()

start_date = '2019-01-01'
end_date = '2020-06-13'

path = './stockdata/'

# 주가 목록은 여기에 있음
stock_list = ['002790.KS', '090430.KS', '035420.KS',
              '005380.KS', '068270.KS', '051900.KS',
              '017670.KS', '005490.KS']
file_list = ['SS_2020_06_05.csv','Amore_2020_06_05.csv', 'Celltrion_2020_06_05.csv',
             'HyundaiMotor_2020_06_05.csv', 'LG_H&H_2020_06_05.csv', 'NAVER_2020_06_05.csv',
             'Posco_2020_06_05.csv', 'SK_2020_06_05.csv']

length = len(file_list)
for idx in range(length):
    stock = data.get_data_yahoo(stock_list[idx],start_date,end_date, interval='1h')
    stock.to_csv(path+file_list[idx],mode='w')