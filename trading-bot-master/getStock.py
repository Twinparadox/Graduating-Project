from pandas_datareader import data
import yfinance as yf
yf.pdr_override()

start_date = '2019-01-01'
end_date = '2020-07-04'
date = '_2020_07_03.csv'

path = './stockdata/'

# 주가 목록은 여기에 있음
stock_list = ['002790.KS', '090430.KS', '035420.KS', '005380.KS', '068270.KS', '051900.KS', '017670.KS', '005490.KS']
file_list = ['SS', 'Amore', 'Naver', 'HyundaiMotor', 'Celltrion', 'LG_HH', 'SK', 'Posco']

length = len(file_list)
for idx in range(length):
    stock = data.get_data_yahoo(stock_list[idx],start_date,end_date, interval='1h')
    stock.to_csv(path+file_list[idx]+date,mode='w')