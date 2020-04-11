from pandas_datareader import data
import yfinance as yf
yf.pdr_override()

start_date = '2000-1-1'
end_date = '2020-03-31'

path = '../stockdata/'

# 주가 목록은 여기에 있음
stock_list = ['005930.KS', '035420.KS', '005380.KS', '005490.KS', '035720.KS',
              '017670.KS', '090430.KS', '055550.KS', '251270.KS', '068270.KS',
              '000660.KS', '105560.KS', '051900.KS', '000270.KS', '030200.KS',
              '051910.KS', '011170.KS', '006400.KS', '036570.KS', '003490.KS',
              '000100.KS', '053800.KQ', '035900.KQ', '007310.KS', '139480.KS']
file_list = ['SAMSUNG_Electronics.csv', 'NAVER.csv', 'HYUNDAI_Motors.csv', 'POSCO.csv', 'Kakao.csv',
             'SK_Telecom.csv', 'AMOREPACIFIC.csv', 'SHINHAN_Financial.csv', 'Netmarble.csv', 'Celltrion.csv',
             'SK_Hynix.csv', 'KB_Financial.csv', 'LG_Household.csv', 'KIA_Motors.csv', 'KT.csv',
             'LG_Chem.csv', 'Lotte_Chem.csv', 'SAMSUNG_SDI.csv', 'NCSoft.csv', 'KoreanAir.csv',
             'Yuhan.csv', 'AhnLab.csv', 'JYP_Ent.csv', 'Ottogi.csv', 'Emart.csv']

length = len(file_list)
for idx in range(length):
    stock = data.get_data_yahoo(stock_list[idx],start_date,end_date)
    stock.to_csv(path+file_list[idx],mode='w')