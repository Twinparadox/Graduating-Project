import csv
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('result.csv')

train = data['train']

plt.plot(train, label = 'asset')
plt.xlabel('episode')
plt.ylabel('asset')

plt.title('A stock DQN')
plt.legend()

plt.savefig('result.png')
plt.show()