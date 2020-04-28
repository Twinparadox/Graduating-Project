import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_squared_error

from keras.models import Sequential
from keras.models import load_model
from keras.layers import Dense, LSTM, Dropout, Activation
from keras.optimizers import Adam, RMSprop
from keras import backend as K

from scipy.signal import savgol_filter

scaler = StandardScaler()
target_scaler = StandardScaler()
look_back = 1

def root_mean_squared_error(y_true, y_pred):
        return K.sqrt(K.mean(K.square(y_pred - y_true), axis=-1)) 

def create_dataset(dataset, look_back):
    pass

def train(train, name, look_back=1):
    X_train = train[['Open', 'High', 'Low', 'Close', 'Volume']]
    Y_train = X_train['Close']
    
    X_train = np.array(X_train)
    
    X_train = scaler.transform(X_train)
    X_train = pd.DataFrame(data=X_train, columns=['Open', 'High', 'Low', 'Close', 'Volume'])

    print(X_train.describe())
    print(Y_train.describe())

    X_train = X_train.astype(float)
    column_list = list(X_train)
    
    print(column_list)

    for s in range(1, look_back+1):
        tmp_train = X_train[column_list].shift(s)
        tmp_train.columns = "shift_"+tmp_train.columns+"_"+str(s)
        X_train[tmp_train.columns] = X_train[column_list].shift(s)

    X_train = X_train[look_back:]
    X_train = X_train.drop(['Open', 'High', 'Low', 'Close', 'Volume'], axis=1)
    Y_train = Y_train[look_back:]
    
    print(X_train)
    print(Y_train)
    print(X_train.shape)
    
    X_train = np.array(X_train)
    X_train = X_train.reshape(-1,look_back,int(X_train.shape[1]/look_back))
    print(X_train.shape)

    batch_size = 320
    epochs = 100
    optimizer = RMSprop()

    model = Sequential()
    model.add(LSTM(input_shape=(X_train.shape[1], X_train.shape[2]),
                   output_dim=50, return_sequences=True))
    model.add(LSTM(output_dim=50, return_sequences=True))
    model.add(LSTM(output_dim=50, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(output_dim=1))
    model.compile(optimizer=optimizer, loss='mape', metrics=['mape'])
    model.summary()

    history = model.fit(X_train, Y_train, batch_size=batch_size, epochs=epochs)
    loss = history.history['loss']

    x_epochs = range(1, len(loss) + 1)

    plt.plot(x_epochs, loss, 'b', label='Training loss')
    plt.title('Loss')
    plt.legend()
    plt.show()

    model.save('../models/stock_model_'+name+'.h5')

def eval(data, look_back):
    pass

def test(data, name, look_back=1):    
    X_data = data[['Open', 'High', 'Low', 'Close', 'Volume']]
    Y_data = X_data['Close']
    
    X_data = np.array(X_data)
    X_data = scaler.transform(X_data)
    X_data = pd.DataFrame(X_data, columns=['Open', 'High', 'Low', 'Close', 'Volume'])

    print(X_data.describe())
    print(Y_data.describe())

    X_data = X_data.astype(float)
    column_list = list(X_data)

    for s in range(1, look_back+1):
        tmp_data = X_data[column_list].shift(s)
        tmp_data.columns = "shift_" + tmp_data.columns + "_" + str(s)
        X_data[tmp_data.columns] = X_data[column_list].shift(s)

    X_data = X_data[look_back:]
    X_data = X_data.drop(['Open', 'High', 'Low', 'Close', 'Volume'], axis=1)
    Y_data = Y_data[look_back:]
    
    X_data = np.array(X_data)
    X_data = X_data.reshape(-1,look_back,int(X_data.shape[1]/look_back))

    model = load_model('../models/stock_model_'+name+'.h5')

    pred = model.predict(X_data)
    
    x_coord = range(1,1+Y_data.shape[0])
    plt.plot(x_coord, Y_data, 'b', label='Ground Truth')
    plt.plot(pred, 'r', label='LSTM Predict')
    plt.title('Compare')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    data_types ={'Date':'str'}
    '''
    df_stock = pd.read_csv('../stockdata/date/SAMSUNG_Electronics.csv',
                           engine='c')
    df_test = pd.read_csv('../stockdata/date_test/SAMSUNG_Electronics.csv',
                          engine='c')
    '''
    #df_stock['Date'] = df_stock['Date'].dt.strftime('%Y-%m-%d')
    #df_economy = pd.read_csv('../data/', engine='c')
    
    df_stock = pd.read_csv('../stockdata/date_whole/SAMSUNG_Electronics.csv',
                           dtype=data_types, engine='c', parse_dates=['Date'])
    df_stock['Date'] = df_stock['Date'].dt.strftime('%Y-%m-%d')
    df_stock = df_stock.drop(['Adj Close'], axis=1)
    
    print(df_stock.describe())
    
    df_train = df_stock[df_stock['Date'] < '2019-01-01']
    df_train = df_train.drop(['Date'], axis=1)
    df_test = df_stock[df_stock['Date'] > '2018-12-31']
    df_test = df_test.drop(['Date'], axis=1)
    
    scaler.fit(df_train)
    print(df_train.describe())
    
    print(scaler.mean_)
    
    #train(df_train, "SSE", 7)
    #test(df_test, "SSE", 7)