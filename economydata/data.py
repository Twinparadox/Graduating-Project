import datetime
import pandas as pd
import math
from dateutil.relativedelta import relativedelta

def sigmoid(x):
    """Performs sigmoid operation
    """
    try:
        if x < 0:
            return 1 - 1 / (1 + math.exp(x))
        return 1 / (1 + math.exp(-x))
    except Exception as err:
        print("Error in sigmoid: " + err)

df = pd.read_csv('economy_leading_2005.csv')

data = '2003-05'

d1 = data[:4]
d2 = data[5:7]

df['Date'] = df['Date'].astype(str)
df['Date'] = df['Date'].str.replace(". ", "-", regex=False)
df['Date'] = pd.to_datetime(df['Date'], format="%Y-%m")
print(df['Date'])

current = datetime.datetime.now()
prev = current - relativedelta(months=3)
prev2 = prev - relativedelta(months=3)

prev = str(prev.date().replace(day=1))
prev2 = str(prev2.date().replace(day=1))
print(prev)
print(df['Date'])
print(df.info())
df = df.set_index('Date')

datas = df.loc[prev]
print(df.loc[prev])

print(list(df.columns))
print(prev2)

df2 = df[prev2:prev]


economy_block = []
for column in df.columns:
    print(df2[column][0] - df2[column][1])

for column in df2.columns:
    for i in range(len(df2)-1):
        economy_block.append(sigmoid(df2[column][i+1] - df2[column][i]))