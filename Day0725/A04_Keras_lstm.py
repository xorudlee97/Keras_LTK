import numpy as np
from keras.models import Sequential
from keras.layers import Dense, LSTM
from keras.callbacks import EarlyStopping

a = np.array(range(1, 11))

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
print("==========================")
print(dataset)

x_train = dataset[:,0:split_num]
y_train = dataset[:,split_num]
print(x_train.shape)
print(y_train.shape)

# x_train = np.reshape(x_train, (6,4,1))
x_train = np.reshape(x_train, (len(a)-window_size+1,split_num,1))

print(x_train.shape)

x_test = np.array([
    [[11],[12],[13],[14]],
    [[12],[13],[14],[15]],
    [[13],[14],[15],[16]],
    [[14],[15],[16],[17]],
])

y_test = np.array(
    [15,16,17,18]
)

print(x_test.shape)
print(y_test.shape)

# 2. 모델 구성
model = Sequential()
model.add(LSTM(32, input_shape=(4,1), return_sequences=True))
model.add(LSTM(10, return_sequences=True))
model.add(LSTM(10, return_sequences=True))
model.add(LSTM(10, return_sequences=True))
model.add(LSTM(10, return_sequences=True))
model.add(LSTM(10, return_sequences=True))
model.add(LSTM(10, return_sequences=True))
model.add(LSTM(10, return_sequences=True))
model.add(LSTM(10, return_sequences=True))
model.add(LSTM(10))

model.add(Dense(5, activation='relu'))
model.add(Dense(5, activation='relu'))
model.add(Dense(5, activation='relu'))
model.add(Dense(5, activation='relu'))
model.add(Dense(5, activation='relu'))
model.add(Dense(5, activation='relu'))
model.add(Dense(5, activation='relu'))
model.add(Dense(5, activation='relu'))
model.add(Dense(5, activation='relu'))
model.add(Dense(1))
model.summary()

# 3. 훈련
model.compile(loss = 'mse', optimizer='adam', metrics=['mse'])

early_stopping = EarlyStopping(monitor='loss', patience=1000, mode='auto')

model.fit(x_train, y_train, epochs=100000, batch_size=6, verbose=1, callbacks=[early_stopping])

loss, acc = model.evaluate(x_test, y_test)
y_predict = model.predict(x_test)

print('loss : ', loss)
print('acc  : ', acc)
print(y_predict)