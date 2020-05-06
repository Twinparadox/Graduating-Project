from pandas_datareader import data
import yfinance as yf
yf.pdr_override()

start_date = '2020-03-18'
end_date = '2020-03-24'

path = './stockdata/'

# 주가 목록은 여기에 있음
stock_list = ['005930.KS']
file_list = ['SS_1_2.csv']

length = len(file_list)
for idx in range(length):
    stock = data.get_data_yahoo(stock_list[idx],start_date,end_date,interval='5m')
    stock.to_csv(path+file_list[idx],mode='w')