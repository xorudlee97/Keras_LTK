# 1. 데이터
import numpy as np

# x = np.array(range(1, 101))
# y = np.array(range(1, 101))

x = np.array([range(100), range(311,411), range(100)])
y = np.array([range(501,601)])

print(x.shape)
print(y.shape)

# 행렬 교환 list(x)
# np.transpose(x)
# x.reshape(100, 2)

x = np.transpose(x)
y = np.transpose(y)

print(x.shape)
print(y.shape)

from sklearn.model_selection import train_test_split

# 비율로 자르기 때문에 변환이 필요 없다.
# train : test = 6 : 4
# test : val = 5 : 5
# train : test : val = 6 : 2 : 2
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=66, test_size = 0.4)
x_val, x_test , y_val, y_test = train_test_split(x_test, y_test, random_state=66, test_size = 0.5)

# 2. 모델 만들기
from keras.models import Sequential
from keras.layers import Dense
model = Sequential()

# 배열의 갯수와 상관 없이 중첩된 배열의 갯수를 입력한다.
model.add(Dense(3, input_dim = 3, activation = 'relu'))
# model.add(Dense(3, input_shape = (2,), activation = 'relu'))
model.add(Dense(14))
model.add(Dense(15))
model.add(Dense(9))
model.add(Dense(26))
model.add(Dense(5))
model.add(Dense(1))

# 3. 훈련
model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])
model.fit(x_train, y_train, epochs=1000, batch_size=10,
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
'''
주어진 데이터를 정제 하는 방법
1. 문자 일경우 규칙적인 숫자로 바꾼다.
2. 데이터의 누락인 경우
    - 행삭제
    - 대체할 수 있는 데이터의 평균 값을 삽입한다.
'''