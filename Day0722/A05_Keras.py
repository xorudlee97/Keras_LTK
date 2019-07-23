import numpy as np

# 1. 훈련 데이터 y = wx + b
# 그래프 그리기
x_train = np.array([1,2,3,4,5,6,7,8,9,10])
y_train = np.array([1,2,3,4,5,6,7,8,9,10])
x_test = np.array([11,12,13,14,15,16,17,18,19,20])
y_test = np.array([11,12,13,14,15,16,17,18,19,20])

#2. 모델 구성
from keras.models import Sequential
from keras.layers import Dense
model = Sequential()

# model.add(Dense(5, input_dim = 1, activation = 'relu'))
# model.add(Dense(3))
# model.add(Dense(4))
# model.add(Dense(1))

model.add(Dense(1, input_dim = 1, activation = 'relu'))
#output node
model.add(Dense(1))
# hidden_node
'''
테스트1
피보나치 수열 가능
batch_size = 32, 10
테스트2
hidden_1st = 3
hidden = [14, 15, 92, 26, 53, 58, 97, 93]
batsize = 3
테스트2
hidden_1st = 10
hidden = [20, 30, 40, 50, 40, 30, 20, 10]
훈련
batch_size=32, 10
예측
batsize = 3

테스트3
hidden_1st = 64
hidden = [32,16,8,8,16,32,64,32,16,8,8,16,32,64]
훈련
batch_size=32
예측
batsize = 3

output = 1
'''



#3. 훈련
model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])
# model.fit 훈련 시키다.
# epochs 위의 모델을 100번 돌린다.
# batch_size 크기 deault 32
# (데이터 양 /batch_size) * epochs = 훈련 횟수
model.fit(x_train, y_train, epochs=5000, batch_size=1)

# 4. 평가 예측
loss, acc = model.evaluate(x_test, y_test, batch_size=3)
print("acc: ", acc)

y_predict = model.predict(x_test)
print(y_predict)

model.summary()

'''
DNN (Deep Nural Network)
선형 회귀 모델 [회기모델] = 1차함수
1,2,3 이후의 다음 수를 구성

머신러닝이란
그래프를 이용한 처리

딥러닝이란
정제된 데이터를 주고 노드의 개수와 Layer의 깊이를 통해 최적의 weight값을 찾는 기술
x, y 값을 주어주고
y = wx +b 중 
w,b의 값을 구하는 문제를 준다.
그 중 나온 w 값에 새로운 데이터(x)와의 결합
y' = w'x
y' 값을 예측한다. [데이터의 정리]

장점
정확도가 높아진다.
단점
모든 노드를 거쳐 들어가게되면 처리가 느리게 된다.

딥러닝의 종류
선형 회귀 모델 vs 분류 모델
'''
'''
수정 가능 한 값
1. 모델의 깊이
2. 노드의 갯수
3. epochs 값
4. batch_size 값

summary = 
(데이터 양 /batch_size) * epochs = 훈련 횟수

evaluate = 평가하다.
train과 test의 분할 
[중간고사 답지와 문제지를 다르게 한다.]

x_tarin = 1 ~ 100
y_tarin = 501 ~ 600
x_test = 1001 ~ 1100
y_test = 1101 ~ 1200
'''