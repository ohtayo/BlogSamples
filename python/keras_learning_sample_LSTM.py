# GPUを使わない場合
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import keras

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from keras.layers.recurrent import LSTM
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.optimizers import Adam

# 再現性を持たせるために乱数のシードを固定する
np.random.seed(0)

#sinとcosの波形を作る
cycle = 2 # 周期の数
period = 100 # 1周期の時間
x = np.arange(0, cycle*period)
sin = np.sin(2.0 * np.pi * x / period)
cos = np.cos(2.0 * np.pi * x / period)

# numpy のndarrayを横に連結
y = np.vstack((sin, cos))

# グラフ描画
plt.plot(x, y[0])
plt.plot(x, y[1])

# ノイズを乗せる
noise_range = [-0.05, 0.05]
noise = np.random.uniform(noise_range[0], noise_range[1], size=y.shape)
y = y + noise
plt.plot(x, y[0])
plt.plot(x, y[1])

# ---------------------------------------
input_length = 10 # 1入力データの長さを20個とする
X = np.zeros([len(x)-input_length, input_length-1, 2])
Y = np.zeros([len(x)-input_length, 2])
for i in range(0, len(X)):
    X[i, :, :] = y[:,i:i+input_length-1].T
    Y[i, :] = y[:,i+input_length-1].T
print(X.shape)  #(190, 9, 2)
print(Y.shape)  #(190, 2)

# モデルの作成
model = Sequential()
# LSTM層の追加，隠れユニット100，入力は10×2サイズ
num_in_units = 2
num_hidden_units = 100
model.add(
    LSTM(units=num_hidden_units, 
         input_shape=(input_length-1, num_in_units), 
         kernel_initializer="random_normal", 
         stateful=False, 
         return_sequences=False
    )
)
# 全結合出力層の追加，出力は2ユニット
num_out_units = 2
model.add(Dense(units=num_out_units, kernel_initializer="random_normal"))
# 活性化関数は線形関数
model.add(Activation('linear'))
# MREを誤差とし，Adamで最適化する
model.compile(loss="mean_squared_error", optimizer=Adam(lr=0.01, beta_1=0.9, beta_2=0.999))

# モデルの学習
model.fit(X, Y, epochs=20, batch_size=32, validation_split=0.1)

# 推測
Y_predict = model.predict(X)

# 学習データを使った推測値のグラフ描画
# sin
plt.figure()
plt.plot(Y[:, 0])
plt.plot(Y_predict[:, 0], ".")
# cos
plt.figure()
plt.plot(Y[:, 1])
plt.plot(Y_predict[:, 1], ".")

# 出力を次の入力に使用した再帰予測
Y_future = np.zeros([300, 2])
Y_future[:input_length-1, :] = X[-1, :, :]

for i in range(input_length-1,len(Y_future)):
    x_temp = Y_future[i-input_length+1:i, :].reshape(1, input_length-1, -1)
    y_temp = model.predict(x_temp)
    Y_future[i, :] = y_temp

plt.figure()
plt.plot(Y_future[input_length:]) 

