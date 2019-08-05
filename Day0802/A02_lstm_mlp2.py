import numpy as np
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, LSTM
from keras.callbacks import EarlyStopping

data_array = range(1,101)

split_size = 8
slice_size = 4
def split_5(np_array, size):
    re_array = []
    for i in range(len(np_array) - size + 1):
        subset = np_array[i:(i+size)]
        re_array.append([item for item in subset])
    return np.array(re_array)

dataset = split_5(data_array, split_size)
print("==========================")
# print(dataset)

X = dataset[:,0:slice_size]
Y = dataset[:,slice_size:]
# print(X.shape)
# print(Y.shape)

x_train, x_test, y_train, y_test = train_test_split(
    X, Y, random_state=66, test_size = 0.4, shuffle = True
)

# print(x_train.shape)
# print(y_train.shape)
# print(x_test.shape)
# print(y_test.shape)

x_train = np.reshape(x_train, (x_train.shape[0], 2, 2))
x_test = np.reshape(x_test, (x_test.shape[0], 2, 2))

print(x_train.shape)
print(x_test.shape)

model = Sequential()
model.add(LSTM(64, activation='relu', input_shape=(2, 2)))
model.add(Dense(32, activation='relu'))
model.add(Dense(16))
model.add(Dense(8))
model.add(Dense(4))

model.compile(loss = 'mse', optimizer='adam', metrics=['mse'])

early_stopping = EarlyStopping(monitor='loss', patience=100, mode='auto')

model.fit(x_train, y_train, epochs=1000, batch_size=6, verbose=1, callbacks=[early_stopping])

loss, acc = model.evaluate(x_test, y_test)
y_predict = model.predict(x_test)

# print('loss : ', loss)
# print('acc  : ', acc)

y_predict = model.predict(x_test)
print(y_predict)

# RMSE 구하기
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))
print("RMSE : ", RMSE(y_test, y_predict))
r2_y_predict = r2_score(y_test, y_predict)
print("R2   : ", r2_y_predict)
# model.summary()
