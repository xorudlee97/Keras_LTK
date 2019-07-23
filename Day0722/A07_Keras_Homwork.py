import numpy as np

def append_np_result(a,b):
    temp = []
    for i in range(a,b+1):
        temp.append(i)
    list = np.array(temp)
    return list

def append_np_list(list, a,b):
    temp = []
    for i in range(a,b+1):
        temp.append(i)
        if i % 4 == 0:
            list.append(temp)
            temp = []
        
    return list
def list_np(a , b):
    temp = []
    temp = append_np_list(temp, a, b)
    list = np.array(temp);
    return list

'''
데이터 전처리의 문제
정제된 데이터를 믿지 마라
'''
x_train = append_np_result(1, 100)
y_train = append_np_result(501, 600)
x_test = append_np_result(1001, 1100)
y_test = append_np_result(1101, 1200)
print(y_test)

from keras.models import Sequential
from keras.layers import Dense
model = Sequential()

model.add(Dense(100, input_dim = 1, activation = 'relu'))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])
model.fit(x_train, y_train, epochs=1000, batch_size=20)
loss, acc = model.evaluate(x_test, y_test, batch_size=3)
print("acc: ", acc)
y_predict_temp = model.predict(x_test)
print(y_predict_temp)

# y_temp = y_predict_temp
# x_temp = x_test

# for i in range(5):
#     x_temp_range = x_temp
#     y_temp_range = y_temp
#     model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])
#     model.fit(x_temp_range, y_temp_range, epochs=1000, batch_size=20)
#     loss, acc = model.evaluate(x_test, y_test, batch_size=3)
#     print("acc: ", acc)
#     y_predict_temp = model.predict(x_test)
#     y_temp = y_predict_temp
#     x_temp = x_test

# loss, acc = model.evaluate(x_test, y_test, batch_size=3)
# y_predict = model.predict(x_test)
# print(y_predict)
# model.summary()