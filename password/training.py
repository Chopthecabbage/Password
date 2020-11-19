import os
# Filter out INFO message
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

import tensorflow as tf

# [텐서플로2]
# print(tf.__version__)
# print(tf.keras.__version__)

# 텐서플로와 케라스가 매우 밀접하게 통합되었고, 다양한 데이터셋이 케라스 라이브러리를 통해 활용할 수 있습니다. 아래의 코드를 통해 MNIST 데이터셋을 인터넷을 통해 가져옵니다.
mnist = tf.keras.datasets.mnist

# x_train에는 총 60000개의 28×28 크기의 이미지가 담겨 있으며,
# y_train에는 이 x_train의 60000개에 대한 값(0~9)이 담겨 있는 레이블 데이터셋입니다.
# 테스트 -> x_test과 y_test은 각각 10000개의 이미지와 레이블 데이터셋입니다.
# 먼저 x_train와 y_train을 통해 모델을 학습하고 난 뒤에, x_test, y_test를 이용해 학습된 모델의 정확도를 평가하게 됩니다.

# x_train : 손글씨 숫자 이미지 대입
# y_train : 이미지가 의미하는 숫자 대입
# -> 데이터의 갯수 60000개, 모델을 학습할때 사용

# x_test : 손글씨 숫자 이미지 대입
# y_test : 이미지가 의미하는 숫자 대입
# -> 데이터의 갯수 10000개, 모델의 예층 정확도 평가시 사용

# Train, Test 데이터 Load
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# 손글씨 숫자 이미지 데이터는 0 - 255 사이의 값을 가진다.
# 변경 -> 모델 훈련에 사용하기 전에 0-1 사이 범위를 갖도록...
x_train, x_test = x_train / 255.0, x_test / 255.0

# 케라스의 순차형 모델(tf.keras.models.Sequential)은 정의한 다음 아래의 예제 코드처럼 차례로 계층(layer)을 쌓아 나가면 되는 매우 간단한 구조이다.
# 총 4개의 레이어로 구성된 신경망

# Flatten()은 이미지를 일차원으로 바꿔줍니다.
# 이미 크기 28, 28로 되어 있기에 -> tf.keras.layers.Flatten(input_shape = (28, 28)), # 크기 28 x 28 의 배열을 입력으로 받아 1차원 배열로 변환

# 히든레이어의 노드개수는 512개, 활성화 함수로 relu 사용
# 히든레이어 -> sigmoids -> 비선형 -> relu -> 선형 -> easy
# The derivative of ReLU:
#  1 if x > 0
#  0 otherwise

# 쉬운예로 기출문제만 열심히 풀었을 때 기출문제는 잘 맞췄으나 실제 시험에서는 성적이 나오지 않는 상태를 말합니다. -> solution -> 정규화, 드롭아웃
# 신경망 연결에서 일부를 덜 학습시켜서 과적합(overfiiting)을 방지하는 것입니다. <- 드롭아웃
# 오버피팅 방지, 이전 레이어의 출력을 20% 끈다

# 1번째 레이어는 입력 이미지의 크기가 28×28이므로 이를 1차원 텐서로 펼치는 것이고,
# 2번째 레이어는 1번째 레이어에서 제공되는 784 개의 값(28×28)을 입력받아 128개의 값으로 인코딩해 주는데, 활성함수로 ReLU를 사용하도록 하였습니다.
# 2번째 레이어의 실제 연산은 1번째 레이어에서 제공받은 784개의 값을 784×matrix 행렬과 곱하고 편향값을 더하여 얻은 matrix개의 출력값을 다시 ReLU 함수에 입력해 얻은 matrix개의 출력입니다.
# 3번째는 matrix개의 뉴런 중 무작위로 0.2가 의미하는 20%를 다음 레이어의 입력에서 무시합니다.
# 이렇게 20% 정도가 무시된 값이 4번째 레이어에 입력되어 충 10개의 값을 출력하는데, 여기서 사용되는 활성화 함수는 Softmax가 사용되었습니다.
# Softmax는 마지막 레이어의 결과값을 다중분류를 위한 확률값으로 해석할 수 있도록 하기 위함입니다.
# 10개의 값을 출력하는 이유는 입력 이미지가 0~9까지의 어떤 숫자를 의미하는지에 대한 각각의 확률을 얻고자 함입니다.

matrix = 1024 # 정확도 ↑, 손실률 ↓ -> 시간 ↑ <- 데이터 양이 많으면 많을수록

model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(),
  tf.keras.layers.Dense(matrix, activation=tf.nn.relu),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(10, activation=tf.nn.softmax)
])

# 모델의 학습 중에 역전파를 통한 가중치 최적화를 위한 기울기 방향에 대한 경사하강을 위한 방법으로 Adam을 사용했으며
# 손실함수로 다중 분류의 Cross Entropy Error인 ‘sparse_categorical_crossentropy’를 지정하였습니다.
# 그리고 모델 평가를 위한 평가 지표로 ‘accuracy’를 지정하였습니다. 이제 다음처럼 모델을 학습할 수 있습니다.

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy', # 손실함수로 크로스 엔트로피
              metrics=['accuracy']) # 메트릭은 모델을 평가할때 사용, 정확도

# 1875 * 32 -> 60000
model.fit(x_train, y_train, epochs=10) # 모델 10번 반복 훈련
model.evaluate(x_test, y_test) # 테스트 데이터셋으로 모델 평가

model.save_weights('model_11_03_15_23') # 가중치 저장