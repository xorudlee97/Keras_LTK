from keras.datasets import cifar10
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers import BatchNormalization
from keras.callbacks import EarlyStopping, TensorBoard
from keras.optimizers import SGD, Adam, RMSprop
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import numpy as np
import matplotlib.pyplot as plt
from keras import regularizers

# CIFAR_10은 3채널로 구성된 32*32 이미지 60000장을 갖는다.
IMG_CHANNELS = 3
IMG_ROWS = 32
IMG_COLS = 32

# 상수의 정의
BATCH_SIZE = 128
NB_EPOCH = 500
Eearly_Stop = 50
NB_CLASSES = 10
VERBOSE = 1
VALIDATION_SPLIT = 0.2
OPTIM = RMSprop()

# 데이터 셋 불러 오기
(X_train, Y_train), (X_test, Y_test) = cifar10.load_data()
print('X_train shape:', X_train.shape)
print('X_train shape:', X_test.shape)
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')

X_train_Row = X_train.shape[0]
X_test_Row = X_test.shape[0]

X_train = np.reshape(X_train, (X_train_Row * IMG_COLS * IMG_ROWS * IMG_CHANNELS, 1))
X_test = np.reshape(X_test, (X_test_Row * IMG_COLS * IMG_ROWS * IMG_CHANNELS, 1))
# print(X_train.shape)
# print(X_test.shape)


# 이미지 출력
# image_cifar = X_train[7956]
# plt.imshow(image_cifar, cmap=plt.cm.binary)
# plt.show()

# 범주형으로 변환
scaler = MinMaxScaler()
# scaler = StandardScaler()

Y_train = np_utils.to_categorical(Y_train, NB_CLASSES)
Y_test = np_utils.to_categorical(Y_test, NB_CLASSES)

scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_train = np.reshape(X_train, (X_train_Row,IMG_ROWS * IMG_COLS * IMG_CHANNELS))

scaler.fit(X_test)
X_test = scaler.transform(X_test)
X_test = np.reshape(X_test, (X_test_Row,IMG_ROWS * IMG_COLS * IMG_CHANNELS))

# 모델 구성
model = Sequential()
tb_hist = TensorBoard(log_dir='./graph',histogram_freq=0, write_graph = True, write_images=True)
model.add(Dense(512, activation='relu', input_shape=(IMG_COLS*IMG_ROWS* IMG_CHANNELS, )))
model.add(Dense(256))
model.add(Dense(128))
model.add(Dense(NB_CLASSES, activation='softmax'))
model.compile(optimizer=OPTIM,
             loss='categorical_crossentropy', 
             metrics=['accuracy'])
early_stopping_callback = EarlyStopping(monitor='val_acc', patience=Eearly_Stop)
model.fit(X_train, Y_train, batch_size=BATCH_SIZE,
          epochs=NB_EPOCH, verbose=VERBOSE, 
          callbacks=[early_stopping_callback, tb_hist])

model.summary()

# 학습

print("Testing...")
score = model.evaluate(X_test, Y_test)
print("Test accuracy:", score[1])

# 모델 저장
# model_json = model.to_json()
# open('cifar10_architecture.json', 'w').write(model_json)
# model.sample_weights('cifar10_weights.h5', overwrite = True)

# print(history.history.keys())

# # plt작업
# plt.plot(history.history['loss'])
# plt.plot(history.history['val_loss'])
# plt.plot(history.history['acc'])
# plt.plot(history.history['val_acc'])
# plt.title('model loss, accuracy')
# plt.ylabel('loss, accuracy')
# plt.xlabel('epoch')
# plt.legend(['train loss', 'test loss', 'train acc', 'test acc'], loc='upper left')
# plt.show()
