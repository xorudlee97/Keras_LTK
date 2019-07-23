import numpy as np

# 행무시, 열 우선
# x_train = np.array([1,2,3,4,5,6,7,8,9,10])
# y_train = np.array([1,2,3,4,5,6,7,8,9,10])
# x_test = np.array([11,12,13,14,15,16,17,18,19,20])
# y_test = np.array([11,12,13,14,15,16,17,18,19,20])
# x3 = np.array([101,102,103,104,105,106])
# x4 = np.array([5,10,15,20,25,30,35,40,45,50,55,60,65,70,85,90])
x_train = np.array(range(1, 11))
y_train = np.array(range(1, 11))
x_test = np.array(range(11, 21))
y_test = np.array(range(11, 21))
x3 = np.array(range(101, 107))
x4 = np.array(range(5, 96, 5))
x5 = np.array(range(30, 50))

from keras.models import Sequential
from keras.layers import Dense
model = Sequential()

# 배열의 갯수와 상관 없이 중첩된 배열의 갯수를 입력한다.
# model.add(Dense(3, input_dim = 1, activation = 'relu'))
model.add(Dense(3, input_shape = (1,), activation = 'relu'))
model.add(Dense(14))
model.add(Dense(15))
model.add(Dense(9))
model.add(Dense(26))
model.add(Dense(5))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])
model.fit(x_train, y_train, epochs=2000, batch_size=10)
loss, acc = model.evaluate(x_test, y_test, batch_size=3)
print("acc: ", acc)

y_predict = model.predict(x_test)
print(y_predict)

# RMSE 구하기
from sklearn.metrics import mean_squared_error
def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))
print("RMSE : ", RMSE(y_test, y_predict))
# model.summary()
'''
INF = 알수 없음
input_shape =(1,INF)
input_shape =(열,행)

sklearn = 옛날 모델
RMSE = Root(MSE)
MSE = 평균 제곱   오차 에러
MAE = 평균 절대값 오차 에러
'''