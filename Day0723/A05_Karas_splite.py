# 1. 데이터
import numpy as np

def splite_np(data, a, b):
    temp = data[a:b]
    list = np.array(temp)
    return list

x = np.array(range(1, 101))
y = np.array(range(1, 101))

x_train = splite_np(x, 0, 60)
x_val = splite_np(x, 60, 80)
x_test = splite_np(x, 80, 100)

y_train = splite_np(y, 0, 60)
y_val = splite_np(y, 60, 80)
y_test = splite_np(y, 80, 100)

# 2. 모델 만들기
from keras.models import Sequential
from keras.layers import Dense
model = Sequential()

# 배열의 갯수와 상관 없이 중첩된 배열의 갯수를 입력한다.
model.add(Dense(3, input_dim = 1, activation = 'relu'))
# model.add(Dense(3, input_shape = (1,), activation = 'relu'))
model.add(Dense(14))
model.add(Dense(15))
model.add(Dense(9))
model.add(Dense(26))
model.add(Dense(5))
model.add(Dense(1))

# 3. 훈련
model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])
model.fit(x_train, y_train, epochs=100, batch_size=1,
          validation_data=(x_val, y_val))
loss, acc = model.evaluate(x_test, y_test, batch_size=3)
print("acc: ", acc)

# 4. 평가 및 예측
y_predict = model.predict(x_test)
print(y_predict)

# 평가
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
INF = 알수 없음
input_shape =(1,INF)
input_shape =(열,행)

sklearn = 옛날 모델
RMSE = Root(MSE)
MSE = 평균 제곱   오차 에러
MAE = 평균 절대값 오차 에러

x_train , y_train       1. 훈련데이터
x_val   , y_val         2. 기계 검증
x_test  , y_test        3. 사람 검증
x_new   , y_predict     4. 새로운 데이터 예측
'''