from keras.models import Sequential
from keras.layers import MaxPooling2D, Flatten

# 
filter_size = 7
# 이미지를  픽셀 데이터로 3,3으로 자른다.
kernel_size = (2,2)

from keras.layers import Conv2D
model = Sequential()
# 28,28,1 (흑백 사진 처리)
# 28,28,2 (컬러 사진 처리m )
model.add(Conv2D(filter_size, kernel_size, padding='same', input_shape= (10,10,1)))

# 3x3의 이미지를 16장으로 분할 하라
# 파라미터 개수 (3 * 3) * 16 = 464
# model.add(Conv2D(16,(2,2), padding='same'))
# 중복 없이 자르고 가장 큰 값을 특성값으로 뽑아낸다.
# 4, 4 , Maxpooling2D(2,2) 일경우
# 1. 4 * 4 중 2 * 2의 중복값 없는 이미지 4개를 만든다.
# 2. 나누어진 2 * 2 중 가장 큰 값을 뽑아 2 *2로 만들어 4 * 4를 2* 2로 묶는다.
model.add(MaxPooling2D(3, 3))
# model.add(Conv2D(8,(2,2), padding='same'))


# 이미지[2D]를 => 배열[1D]로 표현하기 위해 변환
model.add(Flatten())

model.summary()