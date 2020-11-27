import csv
import pandas as pd
import matplotlib.pyplot as plt

dqn = pd.read_csv('삼성_dqn 결과/result.csv')
ddqn = pd.read_csv('삼성_ddqn 결과/result.csv')

dqn_result = dqn['train']
ddqn_result = ddqn['train']

plt.plot(dqn_result, label = 'DQN')
plt.plot(ddqn_result, label = 'DDQN')
plt.xlabel('episode')
plt.ylabel('asset')

plt.title('A stock train')
plt.legend()

plt.savefig('삼성 result.png')
plt.show()