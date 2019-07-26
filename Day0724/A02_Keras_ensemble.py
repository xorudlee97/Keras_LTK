# 1. 데이터
import numpy as np

# x = np.array(range(1, 101))
# y = np.array(range(1, 101))

x1 = np.array([range(100), range(311,411), range(100)])
y1 = np.array([range(501,601),range(711,811),range(100)])

x2 = np.array([range(100, 200), range(311,411), range(100)])
y2 = np.array([range(501,601),range(711,811),range(100)])

# 행렬 교환 list(x)
# np.transpose(x)
# x.reshape(100, 2)

x1 = np.transpose(x1)
y1 = np.transpose(y1)
x2 = np.transpose(x2)
y2 = np.transpose(y2)

from sklearn.model_selection import train_test_split

# 비율로 자르기 때문에 변환이 필요 없다.
# train : test = 6 : 4
# test : val = 5 : 5
# train : test : val = 6 : 2 : 2
x1_train, x1_test, y1_train, y1_test = train_test_split(x1, y1, random_state=66, test_size = 0.4)
x1_val, x1_test , y1_val, y1_test = train_test_split(x1_test, y1_test, random_state=66, test_size = 0.5)
x2_train, x2_test, y2_train, y2_test = train_test_split(x2, y2, random_state=66, test_size = 0.4)
x2_val, x2_test , y2_val, y2_test = train_test_split(x2_test, y2_test, random_state=66, test_size = 0.5)

# 2. 모델 만들기
from keras.models import Sequential, Model
from keras.layers import Dense, Input

# 함수형 모델
# 모델 합치기에 유능
# 다중 모델 코딩에 유능하다.
input1 = Input(shape=(3,))
dense1_1 = Dense(100, activation='relu')(input1)
dense1_2 = Dense(30)(dense1_1)
dense1_3 = Dense(17)(dense1_2)
dense1_4 = Dense(35)(dense1_3)
dense1_5 = Dense(28)(dense1_4)

input2 = Input(shape=(3,))
dense2_1 = Dense(50, activation='relu')(input2)
dense2_2 = Dense(30)(dense2_1)
dense2_3 = Dense(37)(dense2_2)
dense2_4 = Dense(30)(dense2_3)
dense2_5 = Dense(30)(dense2_4)

from keras.layers.merge import concatenate
merge1 = concatenate([dense1_3, dense2_3])

middle1 = Dense(10)(merge1)
middle2 = Dense(5)(middle1)
middle3 = Dense(7)(middle2)
middle4 = Dense(17)(middle3)
middle5 = Dense(27)(middle4)

############### output 모델

output1 = Dense(30)(middle3)
output1_1 = Dense(7)(output1)
output1_2 = Dense(33)(output1_1)
output1_3 = Dense(33)(output1_2)
output1_4 = Dense(3)(output1_3)

output2 = Dense(20)(middle3)
output2_1 = Dense(70)(output2)
output2_2 = Dense(33)(output2_1)
output2_3 = Dense(33)(output2_2)
output2_4 = Dense(3)(output2_3)

model = Model(input = [input1,input2], 
              output = [output1_4, output2_4])
# model.summary()

# Sequential 모델
# 순차적인 처리에 유능
# model = Sequential()
# 배열의 갯수와 상관 없이 중첩된 배열의 갯수를 입력한다.
# model.add(Dense(3, input_shape = (3,), activation = 'relu'))
# model.add(Dense(14))
# model.add(Dense(15))
# model.add(Dense(9))
# model.add(Dense(26))
# model.add(Dense(5))
# model.add(Dense(1))

# 3. 훈련
model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])
model.fit([x1_train,x2_train], [y1_train, y2_train], epochs=1000, batch_size=10,
          validation_data=([x1_val, x2_val], [y1_val, y2_val]))
_, loss1, loss2, acc1, acc2 = model.evaluate([x1_test, x2_test], [y1_test, y2_test], batch_size=1)
print("acc: ", acc1)
print("acc: ", acc2)

# 4. 평가 및 예측
y_predict1, y_predict2 = model.predict([x1_test, x2_test])
print(y_predict1)
print(y_predict2)

# 수열

# 5 7 9 11 13 16 17 19 23 25 
# 717 721 473 217 428 753 576 427 749

# 평가
# RMSE 구하기
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))

print("Y1_Test RMSE : ", RMSE(y1_test, y_predict1))
r2_y_predict1 = r2_score(y1_test, y_predict1)
print("Y1_Test R2   : ", r2_y_predict1)

print("Y2_Test RMSE : ", RMSE(y2_test, y_predict2))
r2_y_predict2 = r2_score(y2_test, y_predict2)
print("Y2_Test R2   : ", r2_y_predict2)

print("Y_AVG RMSE   : ", (RMSE(y1_test, y_predict1)+RMSE(y2_test, y_predict2)) / 2)
print("Y_AVG R2     : ", (r2_y_predict1 + r2_y_predict2) / 2)
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
