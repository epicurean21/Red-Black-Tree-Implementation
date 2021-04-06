import tensorflow.compat.v1 as tf # Tensorflow 딥러닝 Library, version1 사용, placeholder 이용하기위해
import numpy as np # Matrix로 정리하는 Library

# tensorflow 버젼 1로 사용하겠다
tf.disable_v2_behavior()

x_data = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y_data = np.array([[0], [1], [1], [0]])

# Placeholder 3가지 (dataType, shape(입력 데이터 형태), name)
x = tf.placeholder(tf.float32)
y = tf.placeholder(tf.float32)

# 다층 퍼셉트론 (Multi-Layer Perceptron): 입력과 출력 이외의 뉴런이 연결된 모델
# 평면을 휘어서 XOR 문제를 해결한다

# 입력층 - 은닉층 (입력을 받는 층, 입력층과 출력층 사이에 추가된 층이 은닉충)
#hidden layer 은닉층, X랑 Y두개가 들어가는데 중간 은닉층 
w1 = tf.Variable(tf.random_normal([2,4])) # 2 -> 4로 간다
b1 = tf.Variable(tf.random_normal([4]))
layer1 = tf.sigmoid(tf.matmul(x,w1) + b1) # SIGMOID, matmul로 layer를 정리한다

w2 = tf.Variable(tf.random_normal([4,2]))
b2 = tf.Variable(tf.random_normal([2]))
layer2 = tf.sigmoid(tf.matmul(layer1,w2) + b2)

w3 = tf.Variable(tf.random_normal([2,1]))
b3 = tf.Variable(tf.random_normal([1]))
layer3 = tf.nn.sigmoid(tf.matmul(layer2, w3) + b3) # hypothesis

# 신경망 모델 구현
cost = -tf.reduce_sum(y * tf.log(layer3) + (1-y) * tf.log(1-layer3)) #손실함수 공식
train = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(cost) # Optimizer 정의
# cost는 train을 할때 손실이 잃어나면 안되서 손실을 줄이기 위한 공식을 사용
# train은 경사 (gradient) 를 정한다. 라이브러리 사용 Optimizing

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

# epoch가 1이면 모든 데이터를 한번 돌린거 처음부터 끝까지 1만번 트레이닝 한다
for epoch in range(10000):
    sess.run(train, feed_dict={x: x_data, y: y_data}) 
    if epoch % 1000 == 0:
        print("step = ", epoch)


prob = layer3.eval(session=sess, feed_dict={x: x_data})
# prob는 training data를 가지고, x 데이터와 비교한다. Point를 비교

print()
print("output = \n", np.rint(prob)) #가까운 정수로 반올림
