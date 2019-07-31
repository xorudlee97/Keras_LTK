
# 2. 모델 만들기
from keras.models import Sequential
from keras.layers import Dense, BatchNormalization, Dropout
from keras import regularizers

model = Sequential()

# 배열의 갯수와 상관 없이 중첩된 배열의 갯수를 입력한다.
model.add(Dense(3, input_dim = 3, activation = 'relu',
                kernel_regularizer= regularizers.l1(0.01)))
# model.add(Dense(3, input_shape = (2,), activation = 'relu'))
model.add(Dense(14,kernel_regularizer= regularizers.l2(0.01)))
model.add(Dense(15))
model.add(Dense(9))
model.add(Dense(25))
model.add(Dense(35))
model.add(Dense(1))


# 3. 훈련

model.save('Day0726_savetest01.h5')