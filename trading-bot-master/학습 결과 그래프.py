import csv
import pandas as pd
import matplotlib.pyplot as plt



fig = plt.figure()

ax1 = fig.add_subplot(2, 1, 1)
ax2 = fig.add_subplot(2, 1, 2)
dqn = pd.read_csv('삼성_dqn 결과/train_result_20.csv')
ddqn = pd.read_csv('삼성_ddqn 결과/train_result_20.csv')

close = dqn['close']
dqn_result = dqn['asset']
ddqn_result = ddqn['asset']

ax1.plot(dqn_result, label = 'DQN')
ax1.plot(ddqn_result, label = 'DDQN')
ax1.set_xlabel('Date')
ax1.set_ylabel('Asset')

ax1.set_title('Asset for Stock A')
ax1.legend()

ax2.plot(close)
ax2.set_xlabel('Date')
ax2.set_ylabel('Price')
ax2.set_title('Price of Stock A')

plt.tight_layout()
plt.savefig('삼성 train_result.png')
plt.show()
''''''
fig = plt.figure()

ax1 = fig.add_subplot(2, 1, 1)
ax2 = fig.add_subplot(2, 1, 2)
dqn = pd.read_csv('카카오_dqn 결과/train_result_20.csv')
ddqn = pd.read_csv('카카오_ddqn 결과/train_result_20.csv')

close = dqn['close']
dqn_result = dqn['asset']
ddqn_result = ddqn['asset']

ax1.plot(dqn_result, label = 'DQN')
ax1.plot(ddqn_result, label = 'DDQN')
ax1.set_xlabel('Date')
ax1.set_ylabel('Asset')

ax1.set_title('Asset for Stock B')
ax1.legend()

ax2.plot(close)
ax2.set_xlabel('Date')
ax2.set_ylabel('Price')
ax2.set_title('Price of Stock B')

plt.tight_layout()
plt.savefig('카카오 train_result.png')
plt.show()
''''''
fig = plt.figure()

ax1 = fig.add_subplot(2, 1, 1)
ax2 = fig.add_subplot(2, 1, 2)
dqn = pd.read_csv('현대자동차_dqn 결과/train_result_20.csv')
ddqn = pd.read_csv('현대자동차_ddqn 결과/train_result_20.csv')

close = dqn['close']
dqn_result = dqn['asset']
ddqn_result = ddqn['asset']

ax1.plot(dqn_result, label = 'DQN')
ax1.plot(ddqn_result, label = 'DDQN')
ax1.set_xlabel('Date')
ax1.set_ylabel('Asset')

ax1.set_title('Asset for Stock C')
ax1.legend()

ax2.plot(close)
ax2.set_xlabel('Date')
ax2.set_ylabel('Price')
ax2.set_title('Price of Stock C')

plt.tight_layout()
plt.savefig('현대자동차 train_result.png')
plt.show()
''''''
fig = plt.figure()

ax1 = fig.add_subplot(2, 1, 1)
ax2 = fig.add_subplot(2, 1, 2)
dqn = pd.read_csv('엔씨소프트_dqn 결과/train_result_20.csv')
ddqn = pd.read_csv('엔씨소프트_ddqn 결과/train_result_20.csv')

close = dqn['close']
dqn_result = dqn['asset']
ddqn_result = ddqn['asset']

ax1.plot(dqn_result, label = 'DQN')
ax1.plot(ddqn_result, label = 'DDQN')
ax1.set_xlabel('Date')
ax1.set_ylabel('Asset')

ax1.set_title('Asset for Stock D')
ax1.legend()

ax2.plot(close)
ax2.set_xlabel('Date')
ax2.set_ylabel('Price')
ax2.set_title('Price of Stock D')

plt.tight_layout()
plt.savefig('엔씨소프트 train_result.png')
plt.show()
''''''
fig = plt.figure()

ax1 = fig.add_subplot(2, 1, 1)
ax2 = fig.add_subplot(2, 1, 2)
dqn = pd.read_csv('SK텔레콤_dqn 결과/train_result_20.csv')
ddqn = pd.read_csv('SK텔레콤_ddqn 결과/train_result_20.csv')

close = dqn['close']
dqn_result = dqn['asset']
ddqn_result = ddqn['asset']

ax1.plot(dqn_result, label = 'DQN')
ax1.plot(ddqn_result, label = 'DDQN')
ax1.set_xlabel('Date')
ax1.set_ylabel('Asset')

ax1.set_title('Asset for Stock E')
ax1.legend()

ax2.plot(close)
ax2.set_xlabel('Date')
ax2.set_ylabel('Price')
ax2.set_title('Price of Stock E')

plt.tight_layout()
plt.savefig('SK텔레콤 train_result.png')
plt.show()