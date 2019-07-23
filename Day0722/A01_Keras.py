import numpy as np

# 1. 훈련 데이터 y = wx + b
# 그래프 그리기
x = np.array([1,2,3])
y = np.array([2,4,6])

#2. 모델 구성
from keras.models import Sequential
from keras.layers import Dense
model = Sequential()

# 여기서 Layer의 깊이와 노드의 갯수를 조절한다. 
# [취미생활:데이터 모델링]
# 1차원 직선함수 구하기
model.add(Dense(5, input_dim = 1, activation = 'relu'))
model.add(Dense(3))
model.add(Dense(4))
model.add(Dense(1))

#3. 훈련
model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])
# epochs 위의 모델을 100번 돌린다.
model.fit(x, y, epochs=100, batch_size=1)

# 4. 평가 예측
loss, acc = model.evaluate(x, y, batch_size=1)
print("acc: ", acc)

'''
모델
1,2,3 이후의 다음 수를 구성

머신러닝이란
그래프를 이용한 처리

딥러닝이란
정제된 데이터를 주고 노드의 개수와 Layer의 깊이를 통해 최적의 weight값을 찾는 기술
x, y 값을 주어주고
y = wx +b 중 
w,b의 값을 구하는 문제를 준다.

장점
정확도가 높아진다.
단점
모든 노드를 거쳐 들어가게되면 처리가 느리게 된다.
'''