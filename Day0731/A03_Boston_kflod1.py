from keras.datasets import boston_housing
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import StratifiedKFold
import numpy as np 

(train_data, train_targets), (test_data, test_target) = boston_housing.load_data()

# mean = train_data.mean(axis=0)
# train_data -= mean
# std = train_data.std(axis=0)
# train_data /= std

# test_data -= mean
# test_data /= std

scaler = StandardScaler()
# scaler = MinMaxScaler()

scaler.fit(train_data)
train_data = scaler.transform(train_data)

from keras import models
from keras import layers
nflod_num = 10
print(train_data.shape)
print(test_data.shape)
def build_model():
    # 동일한 모델을 여러번 생성
    model = Sequential()
    model.add(Dense(64, activation = 'relu', input_shape=(train_data.shape[1],)))
    model.add(Dense(64, activation = 'relu'))
    model.add(Dense(32))
    model.add(Dense(16))
    model.add(Dense(1))
    model.compile(optimizer='rmsprop', loss='mse', metrics=['mae'])
    return model



import numpy as np

k = 5
num_val_samples = len(train_data) // k

num_epochs = 100
all_scores = []
for i in range(k):
    print("처리중인 폴드 #", i)
    # 검증 데이터 준비: k번째 분할
    val_data = train_data[i* num_val_samples: (i+1) * num_val_samples]
    val_target = train_targets[i * num_val_samples: (i +1) * num_val_samples]

    # # 훈련 데이터 준비: 다른 분할 전체
    # print("str_arr : ", i * num_val_samples)
    # print("end_arr : ", (i+1)* num_val_samples)
    partial_train_data = np.concatenate([train_data[:i* num_val_samples], train_data[(i+1)* num_val_samples:]],axis = 0)
    prtial_train_target = np.concatenate([train_targets[:i*num_val_samples],train_targets[(i+1)* num_val_samples:]],axis = 0)

    # 케라스 모델 구성
    model = build_model()
    # 모델 훈련
    model.fit(partial_train_data, prtial_train_target, epochs=num_epochs, batch_size=1, verbose=1)
    # 모델 평가
    val_mse, val_mae = model.evaluate(val_data, val_target, verbose=0)
    all_scores.append(val_mae)

print(all_scores)
print(np.mean(all_scores))
