#-*- coding: utf-8 -*-

# train Data 60000
# test Data 10000

from keras.datasets import mnist
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from keras.callbacks import ModelCheckpoint, EarlyStopping

import numpy
import os
import tensorflow as tf

# 데이터 불러오기
(X_train, Y_train), (X_test, Y_test) = mnist.load_data()

# import matplotlib.pyplot as plt

# digit = X_train[7956]
# plt.imshow(digit, cmap=plt.cm.binary)
# plt.show()

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
model.add(Conv2D(32, kernel_size=(3,3), input_shape=(28,28,1), activation='relu'))
model.add(Conv2D(64, (3,3), activation='relu'))
model.add(MaxPooling2D(pool_size=2))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
# 분류 모델 마지막은 반드시 SortMax를 쓴다.
model.add(Dense(10, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

early_stopping_callback = EarlyStopping(monitor='val_loss', patience=10)

# model.summary()

# 모델의 실행
history = model.fit(X_train, Y_train, validation_data=(X_test, Y_test),
                    epochs=1, batch_size=2000, verbose=1,
                    callbacks=[early_stopping_callback])

print("\n Test Accuracy %.4f"% (model.evaluate(X_test, Y_test)[1]))

print(history.history.keys())

import matplotlib.pyplot plt

plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()


# plt작업
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model loss, accuracy')
plt.ylabel('loss, accuracy')
plt.xlabel('epoch')
plt.legend(['train loss', 'test loss', 'train acc', 'test acc'], loc='upper left')
plt.show()