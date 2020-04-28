import matplotlib.pyplot as plt
import datetime
import pandas as pd

start_date = '2000-1-1'
end_date = '2020-03-31'

path = '../stockdata/'

if __name__== "__main__":
    df = pd.read_csv('../stockdata/SAMSUNG_Electronics.csv', engine='c')
