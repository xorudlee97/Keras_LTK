from numpy import array
from keras.models import Sequential
from keras.layers import Dense, LSTM

x = array([[1,2,3],[2,3,4],[3,4,5],[4,5,6],[5,6,7],[6,7,8],[7,8,9],[8,9,10],[9,10,11],[10,11,12],[20,30,40],[30,40,50],[40,50,60]])
# 4 x 3 행렬
y = array([4,5,6,7,8,9,10,11,12,13,50,60,70])

print("x.shape", x.shape)
print("y.shape", y.shape)

# reshape(행 : 열 : 자르는 개수)
x = x.reshape((x.shape[0], x.shape[1], 1))
# 4, 3, 1
print("x.shape:", x.shape)

model = Sequential()
# input_shape = (열 : 자르는 개수)
model.add(LSTM(100, activation='relu', input_shape=(3, 1)))
# params = 4 * ((size_of_input + 1) * size_of_output + size_of_output^2)
# params = (4 * ((10 + 1) * 1 + 1^1)) * 10
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(1))

while True:
    # 3. 훈련
    model.compile(optimizer='adam', loss='mse')
    model.fit(x, y, epochs=2000, batch_size=3)
    x_input = array([25,35,45]) # 1, 3 , ?????
    x_input = x_input.reshape((1, 3, 1))

    yhat = model.predict(x_input)
    print(yhat)
    if yhat <= 55.01 and yhat >= 54.99:
        break;
# model.summary()