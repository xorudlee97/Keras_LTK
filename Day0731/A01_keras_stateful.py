import numpy as np

import keras
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout, BatchNormalization
from keras.callbacks import EarlyStopping, TensorBoard
import matplotlib.pyplot as plt

a = np.array(range(1, 101))
bat_size = 1
split_num = 4
window_size = 5
def split_5(seq, size):
    aaa = []
    for i in range(len(a) - size +1):
        subset = a[i:(i+size)]
        aaa.append([item for item in subset])
    print(type(aaa))
    return np.array(aaa)

dataset = split_5(a, window_size)
# print("==========================")
# print(dataset)
# print(dataset.shape)
# print("==========================")

x_train = dataset[:,0:4]
y_train = dataset[:,4]

x_train = np.reshape(x_train, (len(x_train), window_size -1 , 1))

x_test = x_train + 100
y_test = y_train + 100

# print(x_train.shape)
# print(y_train.shape)
# print(x_test.shape)
# print(y_test.shape)

# 2. 모델 구성
model = Sequential()
model.add(LSTM(128, batch_input_shape=(bat_size,4,1), stateful=True))
model.add(Dense(64, activation= 'relu'))
model.add(Dense(32))
model.add(Dense(16))
model.add(Dense(8))
# model.add(BatchNormalization())
# model.add(Dense(256))
# model.add(Dense(128))
# model.add(Dropout(0.2))
# model.add(Dense(512))
# model.add(Dense(256))
# model.add(Dropout(0.4))
model.add(Dense(1))


model.summary()

model.compile(loss='mse', optimizer='adam', metrics=['mse'])

num_epochs = 10

tb_hist = TensorBoard(log_dir='./graph',histogram_freq=0, write_graph = True, write_images=True)
early_stopping = EarlyStopping(monitor='mean_squared_error', patience=50, mode='auto')
for epoch_idx in range(num_epochs):
    print('epochs:' + str(epoch_idx))
    model.fit(x_train, y_train, epochs=300, batch_size=bat_size, 
              verbose=2, shuffle=False,
              validation_data=(x_test, y_test),
              callbacks=[early_stopping])#, tb_hist])
    model.reset_states()

mse, _ = model.evaluate(x_train, y_train, batch_size = bat_size)
print("mse :", mse)
model.reset_states()

y_predict = model.predict(x_test, batch_size=1)
print(y_predict[0:10])

# RMSE 구하기
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))

print("RMSE : ", RMSE(y_test, y_predict))
r2_y_predict = r2_score(y_test, y_predict)
print("R2   : ", r2_y_predict)

# model.summary()
'''
1. mse 값을 1이하로 만들것
        -> 3 개 이상의 히든 레이어 추가할것/ 
        -> 드랍 아웃 또는 batnoramliztion 적용
2. RMSE 함수 적용
3. R2 함수 적용
4. EarlyStopping 기능 적용
5. tensorboard 적용
6. matplotlib 이미지 적용 mse/epochs
'''