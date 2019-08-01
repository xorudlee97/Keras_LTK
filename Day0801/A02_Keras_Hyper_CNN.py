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

# CNN 모델
X_train = X_train.reshape(X_train.shape[0], 28,28,1).astype('float32') / 255
X_test = X_test.reshape(X_test.shape[0], 28,28,1).astype('float32') / 255
# DNN 모델
# X_train = X_train.reshape(X_train.shape[0], 28 * 28).astype('float32') / 255
# X_test = X_test.reshape(X_test.shape[0], 28 * 28).astype('float32') / 255

Y_train = np_utils.to_categorical(Y_train)
Y_test = np_utils.to_categorical(Y_test)


from keras.models import Sequential, Model
from keras.layers import Dense, LSTM, Dropout, Input
import numpy as np

def build_network(keep_prob=0.5, optimizer='adam'):
    inputs = Input(shape=(28,28,1),name='input')
    x = Conv2D(32, kernel_size=(3,3), activation='relu', name='hidden1')(inputs)
    x = Conv2D(32, kernel_size=(3,3), activation='relu', name='hidden2')(inputs)
    x = Dropout(keep_prob)(x)
    x = Flatten()(x)
    x = Dense(128, activation='relu', name='hidden3')(x)
    x = Dropout(keep_prob)(x)
    prediction = Dense(10, activation='softmax', name = 'output')(x)
    model = Model(inputs=inputs, outputs = prediction)
    model.compile(optimizer=optimizer, loss= 'categorical_crossentropy',metrics=['accuracy'])
    return model

def create_hyperparameters():
    batches = [10, 20, 30, 40, 50]
    optimizers = ['rmsprop', 'adam', 'adadelta']
    dropout = np.linspace(0.1, 0.5, 5)
    return{"batch_size":batches, "optimizer":optimizers, "keep_prob":dropout}

from keras.wrappers.scikit_learn import KerasClassifier

model = KerasClassifier(build_fn= build_network, verbose=1)
hyperparameters = create_hyperparameters()

from sklearn.model_selection import RandomizedSearchCV

search = RandomizedSearchCV(estimator=model,
                            param_distributions=hyperparameters,
                            n_iter=10, n_jobs=1, cv=3, verbose=1)
search.fit(X_train, Y_train)
print(search.best_params_)

model = build_network(search.best_params_["keep_prob"], search.best_params_['optimizer'])
history = model.fit(X_train, Y_train, validation_data=(X_test, Y_test),
                    epochs=30, batch_size=search.best_params_["batch_size"], verbose=1)
print("\n Test Accuracy %.4f"% (model.evaluate(X_test, Y_test)[1]))
'''
머신러닝을 실행 중 조정 가능한 
파라미터에서 가장 좋은 값을 찾아내기 위해 
파라미터 조정을 하는 프로그래밍
리턴 가장 최적의 파라미터를 리턴

# CNN 모델로 바꿀것
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
'''