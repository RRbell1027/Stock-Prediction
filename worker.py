from PyQt5.QtCore import QThread, pyqtSignal

import yfinance as yf # 股市爬蟲函式庫
from sklearn.preprocessing import MinMaxScaler

import os
import numpy as np
import matplotlib.pyplot as plt

# 深度學習模型函式庫
import keras.layers as layers
from keras.models import Model
from tensorflow.keras.optimizers import Adam

# 實作執行緒
class Worker(QThread):
    trigger = pyqtSignal(str)
    finished = pyqtSignal(np.ndarray)
    error = pyqtSignal()
    ticker = ''
    
    def __init__(self):
        super().__init__()
        
    def run(self):
        self.stop = False
        time_length = 40
        epochs = 100
        batch_size = 128
        
        # 網路爬蟲與資料預處理
        self.trigger.emit('正在從網上抓資料')
        x_data, y_data, mm_list = self.preprocessing(self.ticker, time_length)
        
        # 資料洗牌
        if self.stop: return
        x_shuffle, y_shuffle = self.shuffle(x_data, y_data)
        
        # 建立模型
        if self.stop: return
        self.trigger.emit('正在建立模型')
        self.model = self.build_model(time_length, 0.001)
        
        # 訓練模型
        if self.stop: return
        self.trigger.emit('正在訓練模型(0/100)')
        self.fit(x_shuffle, y_shuffle, batch_size, epochs)
        
        # 預測未來7天的股票指數
        if self.stop: return
        self.trigger.emit('正在預測股票')
        pred = self.predict(x_data[-1])
        
        # 還原數據大小
        if self.stop: return
        pred = self.inverse(pred, mm_list)
        
        # 畫出折線圖
        if self.stop: return
        self.trigger.emit('正在繪製折線圖')
        self.plot(pred)
        
        # 正在回傳資料
        self.finished.emit(pred)
        
    def preprocessing(self, ticker, time_length):
        # 從網上載入股票數據
        df = yf.Ticker(f'{ticker}.TW').history(period='max')
        df = df[df.notna()]
        
        # 提取資料庫數據
        datavalue = df.to_numpy()[:,:4]
        
        # 如果資料不存在，結束執行緒
        if datavalue.size == 0:
            self.error.emit()
            self.stop = True
            return None, None, None
        
        # 歸一化
        mm_list = []
        for i in range(datavalue.shape[1]):
            mm = MinMaxScaler()
            datavalue[:,i] = mm.fit_transform(datavalue[:,i].reshape(-1,1)).reshape(-1)
            mm_list.append(mm)
        
        # 將資料每20天一組，步長為 1 
        x_data, y_data = [], []
        for i in range(len(datavalue)-time_length):
            x_data.append(datavalue[i:i+time_length])
            y_data.append(datavalue[i+time_length])

        x_data = np.array(x_data, dtype=np.float64)
        y_data = np.array(y_data, dtype=np.float64)
        
        return x_data, y_data, mm_list
    
    def shuffle(self, x_data, y_data):
        
        # 資料洗牌
        s = np.arange(len(x_data))
        np.random.shuffle(s)
        x_shuffle = x_data[s]
        y_shuffle = y_data[s]
        
        return x_shuffle, y_shuffle
        
    def build_model(self, time_length, lr):
    
        input_layer = layers.Input(shape=(time_length, 4))

        # 第一層(LSTM)
        layer1 = layers.LSTM(256, return_sequences=True)(input_layer)
        layer1 = layers.Dropout(0.3)(layer1)

        # 第二層(LSTM)
        layer2 = layers.LSTM(256, return_sequences=False)(layer1)
        layer2 = layers.Dropout(0.3)(layer2)

        # 第三層(Dense)
        layer3 = layers.Dense(16,kernel_initializer="uniform",activation='relu')(layer2)

        # 第四層(Dense)
        layer4 = layers.Dense(4,kernel_initializer="uniform",activation='linear')(layer3)

        model = Model(inputs=input_layer, outputs=layer4)

        opt = Adam(learning_rate=lr)
        model.compile(loss='mse', optimizer=opt, metrics=['mae'])

        return model
    
    def fit(self, x_train_shuffle, y_train_shuffle, batch_size, epochs):
        for e in range(epochs):
            if self.stop: return
            self.trigger.emit(f'正在訓練模型({e+1}/100)')
            self.model.fit(x_train_shuffle, y_train_shuffle, batch_size=batch_size, epochs=1)
            
    
    def predict(self, x_start):
        # 用預測出來的結果預測之後的未來
        x = x_start
        res = []
        for i in range(7):
            if self.stop: return
            p = self.model.predict(x.reshape(1,-1,4), verbose=0)
            res.append(p)
            x = np.concatenate((x[1:], p))
        return np.array(res)
    
    def inverse(self, pred, mm_list):
        for i, mm in enumerate(mm_list):
            pred[:,0,i] = mm.inverse_transform(pred[:,0,i].reshape(-1, 1)).reshape(-1)
        return pred[:,0,:]
    
    def plot(self, pred):        
        plt.figure(figsize=(10,8))
        plt.plot(pred[:,3])
        plt.xlabel('time')
        plt.ylabel('stock price')
        plt.savefig('./temp.png')
        plt.close()