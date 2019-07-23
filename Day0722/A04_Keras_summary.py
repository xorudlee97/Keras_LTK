import numpy as np

# 1. 훈련 데이터 y = wx + b
# 그래프 그리기
x = np.array([1,2,3,4,5,6,7,8,9,10])
y = np.array([1,2,3,4,5,6,7,8,9,10])
# x2 = np.array([47,13,31])


#2. 모델 구성
from keras.models import Sequential
from keras.layers import Dense
model = Sequential()

# # 여기서 Layer의 깊이와 노드의 갯수를 조절한다. 
# # [취미생활:데이터 모델링]
# # 1차원 직선함수 구하기
# # input and hidden_node
model.add(Dense(5, input_dim = 1, activation = 'relu'))
model.add(Dense(3))
model.add(Dense(4))
model.add(Dense(1))
# model.add(Dense(64, input_dim = 1, activation = 'relu'))
# # hidden_node
# '''
# 테스트1
# hidden_1st  = 5
# hidden = [10, 15, 30, 10, 50,40,30,20,10]
# 테스트2
# hidden_1st = 10
# hidden = [20, 30, 40, 50, 40, 30, 20, 10]
# 테스트3
# hidden_1st = 64
# hidden = [32,16,8,8,16,32,64,32,16,8,8,16,32,64]

# output = 1
# '''
# model.add(Dense(32))
# model.add(Dense(16))
# model.add(Dense(8))
# model.add(Dense(8))
# model.add(Dense(16))
# model.add(Dense(32))
# model.add(Dense(64))
# model.add(Dense(32))
# model.add(Dense(16))
# model.add(Dense(8))
# model.add(Dense(8))
# model.add(Dense(16))
# model.add(Dense(32))
# model.add(Dense(64))
# #output node
# model.add(Dense(1))

# Summary()
# 계산식
# x * line + bias = y
# line * (x + 1) = y 
# 강사님 소스 계산식
# (x + b) * w = y
# (1 + 1) * 5= 10
# (5 + 1) * 3 = 18
# (3 + 1) * 4 = 16
# (4 + 1) * 1 = 5
# 내 소스 계산식
# (1 + 1) * 64 = 128
# (64 + 1) * 32 = 2080
# (32 + 1)* 16 = 528
# (16 + 1) * 8 = 136
# (8 + 1) * 8 = 72
# (8 + 1) * 16 = 144
# (16 + 1) * 32 = 544
# (32 + 1) * 64 = 2112
# (64 + 1) * 1 = 65
model.summary()

'''
#3. 훈련
model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])
# model.fit 훈련 시키다.
# epochs 위의 모델을 100번 돌린다.
# batch_size 크기 deault 32
# (데이터 양 /batch_size) * epochs = 훈련 횟수
model.fit(x, y, epochs=100)

# 4. 평가 예측
loss, acc = model.evaluate(x, y, batch_size=5)
print("acc: ", acc)

y_predict = model.predict(x2)
print(y_predict)
'''
'''
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
