from keras.datasets import cifar10
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D
# from keras.callbacks import EarlyStopping
from keras.layers import BatchNormalization
from keras.optimizers import SGD, Adam, RMSprop

import matplotlib.pyplot as plt
from keras import regularizers

# CIFAR_10은 3채널로 구성된 32*32 이미지 60000장을 갖는다.
IMG_CHANNELS = 3
IMG_ROWS = 32
IMG_COLS = 32

# 상수의 정의
BATCH_SIZE = 128
NB_EPOCH = 100
# Eearly_Stop = 20
NB_CLASSES = 10
VERBOSE = 1
VALIDATION_SPLIT = 0.2
OPTIM = RMSprop()

# 데이터 셋 불러 오기
(X_train, Y_train), (X_test, Y_test) = cifar10.load_data()
print('X_train shape:', X_train.shape)
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')

# 범주형으로 변환
Y_train = np_utils.to_categorical(Y_train, NB_CLASSES)
Y_test = np_utils.to_categorical(Y_test, NB_CLASSES)

X_train = X_train.astype('float32') / 255
X_test = X_test.astype('float32') / 255

# 모델 구성
model = Sequential()
model.add(Conv2D(32, (3,3), padding='same', input_shape=(IMG_ROWS, IMG_COLS, IMG_CHANNELS), kernel_regularizer= regularizers.l2(0.0001)))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(Conv2D(32, (3,3), padding='same', input_shape=(IMG_ROWS, IMG_COLS, IMG_CHANNELS), kernel_regularizer= regularizers.l2(0.0001)))
# model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.2))

model.add(Conv2D(64, (3,3), padding='same', input_shape=(IMG_ROWS, IMG_COLS, IMG_CHANNELS), kernel_regularizer= regularizers.l2(0.0001)))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(Conv2D(64, (3,3), padding='same', input_shape=(IMG_ROWS, IMG_COLS, IMG_CHANNELS), kernel_regularizer= regularizers.l2(0.0001)))
# model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.3))

model.add(Conv2D(128, (3,3), padding='same', input_shape=(IMG_ROWS, IMG_COLS, IMG_CHANNELS), kernel_regularizer= regularizers.l2(0.0001)))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(Conv2D(128, (3,3), padding='same', input_shape=(IMG_ROWS, IMG_COLS, IMG_CHANNELS), kernel_regularizer= regularizers.l2(0.0001)))
# model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.4))

model.add(Flatten())
# model.add(Dense(512))
# model.add(Activation('relu'))
# model.add(Dropout(0.5))
model.add(Dense(NB_CLASSES))
model.add(Activation('softmax'))

model.summary()

# 학습
model.compile(loss='categorical_crossentropy', optimizer=OPTIM,
             metrics=['accuracy'])

# early_stopping_callback = EarlyStopping(monitor='val_acc', patience=Eearly_Stop)
history = model.fit(X_train, Y_train, batch_size=BATCH_SIZE,
                    epochs=NB_EPOCH, validation_split=VALIDATION_SPLIT,
                    verbose=VERBOSE)      # callbacks=[early_stopping_callback])
print("Testing...")
score = model.evaluate(X_test, Y_test, 
                       batch_size=BATCH_SIZE, verbose=VERBOSE)
print("\nTest score:", score[0])
print("Test accuracy:", score[1])

# 모델 저장
# model_json = model.to_json()
# open('cifar10_architecture.json', 'w').write(model_json)
# model.sample_weights('cifar10_weights.h5', overwrite = True)

print(history.history.keys())

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