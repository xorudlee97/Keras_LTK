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

# scaler = StandardScaler()
scaler = MinMaxScaler()

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

# from sklearn.model_selection import StratifiedKFold
seed = 77
from keras.wrappers.scikit_learn import KerasClassifier, KerasRegressor
from sklearn.model_selection import KFold, cross_val_score
model = KerasRegressor(build_fn=build_model, epochs=100, batch_size = 1, verbose=1)
kfold = KFold(n_splits=4, shuffle=True, random_state=seed)
results = cross_val_score(model, train_data, train_targets, cv=kfold)

print(results)
print(np.mean(results))