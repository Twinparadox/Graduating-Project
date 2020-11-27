import csv
import pandas as pd
import matplotlib.pyplot as plt



dqn = pd.read_csv('삼성_dqn 결과/result.csv')
ddqn = pd.read_csv('삼성_ddqn 결과/result.csv')

dqn_result = dqn['train']
ddqn_result = ddqn['train']

plt.plot(dqn_result, label = 'DQN')
plt.plot(ddqn_result, label = 'DDQN')
plt.xlabel('Episode')
plt.ylabel('Asset')

plt.title('Asset for Stock A')
plt.legend()

plt.savefig('삼성 result.png')
plt.show()
''''''
dqn = pd.read_csv('카카오_dqn 결과/result.csv')
ddqn = pd.read_csv('카카오_ddqn 결과/result.csv')

dqn_result = dqn['train']
ddqn_result = ddqn['train']

plt.plot(dqn_result, label = 'DQN')
plt.plot(ddqn_result, label = 'DDQN')
plt.xlabel('Episode')
plt.ylabel('Asset')

plt.title('Asset for Stock B')
plt.legend()

plt.savefig('카카오 result.png')
plt.show()
''''''
dqn = pd.read_csv('현대자동차_dqn 결과/result.csv')
ddqn = pd.read_csv('현대자동차_ddqn 결과/result.csv')

dqn_result = dqn['train']
ddqn_result = ddqn['train']

plt.plot(dqn_result, label = 'DQN')
plt.plot(ddqn_result, label = 'DDQN')
plt.xlabel('Episode')
plt.ylabel('Asset')

plt.title('Asset for Stock C')
plt.legend()

plt.savefig('현대자동차 result.png')
plt.show()
''''''
dqn = pd.read_csv('엔씨소프트_dqn 결과/result.csv')
ddqn = pd.read_csv('엔씨소프트_ddqn 결과/result.csv')

dqn_result = dqn['train']
ddqn_result = ddqn['train']

plt.plot(dqn_result, label = 'DQN')
plt.plot(ddqn_result, label = 'DDQN')
plt.xlabel('Episode')
plt.ylabel('Asset')

plt.title('Asset for Stock D')
plt.legend()

plt.savefig('엔씨소프츠 result.png')
plt.show()
''''''
dqn = pd.read_csv('SK텔레콤_dqn 결과/result.csv')
ddqn = pd.read_csv('SK텔레콤_ddqn 결과/result.csv')

dqn_result = dqn['train']
ddqn_result = ddqn['train']

plt.plot(dqn_result, label = 'DQN')
plt.plot(ddqn_result, label = 'DDQN')
plt.xlabel('Episode')
plt.ylabel('Asset')

plt.title('Asset for Stock E')
plt.legend()

plt.savefig('SK텔레콤 result.png')
plt.show()