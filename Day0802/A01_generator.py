#-*- coding: utf-8 -*-

# train Data 60000
# test Data 10000

from keras.datasets import mnist
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, BatchNormalization
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras import regularizers

import numpy
import os
import tensorflow as tf

# 데이터 불러오기
(X_train, Y_train), (X_test, Y_test) = mnist.load_data()

X_train = X_train[:300]
Y_train = Y_train[:300]
X_test = X_test[:300]
Y_test = Y_test[:300]

X_train = X_train.reshape(X_train.shape[0], 28,28,1).astype('float32') / 255
X_test = X_test.reshape(X_test.shape[0], 28,28,1).astype('float32') / 255

# print(Y_train.shape)
# print(Y_test.shape)

# One Hot InCoding [categorical]
# 하나의 값만 True이고 나머지는 모두 False인 인코딩
# 장점
# 
# X_trian = 7, 3, 5, 6...
#    0 1 2 3 4 5 6 7 8 9
# 7: 0 0 0 0 0 0 0 1 0 0
# 3: 0 0 0 1 0 0 0 0 0 0
# 5: 0 0 0 0 0 1 0 0 0 0
# 6: 0 0 0 0 0 0 1 0 0 0
# ...

Y_train = np_utils.to_categorical(Y_train)
Y_test = np_utils.to_categorical(Y_test)

# print(Y_train.shape)
# print(Y_test.shape)

# print(X_train.shape)
# print(X_test.shape)
# print(Y_train.shape)
# print(Y_test.shape)

#컨볼루션 신경망의 설정
model = Sequential()
model.add(Conv2D(64, kernel_size=(3,3), input_shape=(28,28,1), activation='relu'))
model.add(Conv2D(32, kernel_size=(3,3), activation='relu'))
model.add(MaxPooling2D(pool_size=2))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.25))

# 분류 모델 마지막은 반드시 SortMax를 쓴다.
model.add(Dense(10, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

from keras.preprocessing.image import ImageDataGenerator
data_generator = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.02,
    height_shift_range=0.02,
    horizontal_flip=True
)

history = model.fit_generator(data_generator.flow(X_train, Y_train, batch_size=1),
                    # steps_per_epoch=len(X_train) *32,
                    steps_per_epoch=30000,
                    epochs = 300,
                    validation_data = (X_test, Y_test),
                    verbose=1
                    )

print("\n Test Accuracy %.4f"% (model.evaluate(X_test, Y_test)[1]))