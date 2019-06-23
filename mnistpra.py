# mnist 모델 불러오기
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

import tensorflow as tf
# 그래프 짜기, 실행 아님.
sess = tf.InteractiveSession()
# 입력될 이미지 form 설정
x = tf.placeholder(tf.float32, shape=[None, 784])
y_ = tf.placeholder(tf.float32, shape=[None, 10])
# 변수 가중치, 편향 설정
W = tf.Variable(tf.zeros([784,10]))
b = tf.Variable(tf.zeros([10]))
# 변수 가중치, 편향 초기화
sess.run(tf.global_variables_initializer())
# 모델 설정 y=f(Wx+b)
y = tf.nn.softmax(tf.matmul(x,W) + b)
# cost function 설정 cost function= mean(\Sigma_i=0 to 9 y_*log y)
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))

# cost function을 최소화 시키는 과정. 방법을 gdo 로 설정함. cost function의 기울기를 계산하여 변수를 얼마나 변경해야할지 결정함. 
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)
#위에선 변수를 1번 변경한 것이므로, 차이를 줄이려면 train_step 을 반복하게 시킴 
for i in range(1000):
  # mnist train image에서 50개만 추림.
  batch = mnist.train.next_batch(50)
  train_step.run(feed_dict={x: batch[0], y_: batch[1]})
#평가
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print(accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels}))
#---------------------------------------------------------------------------------------
# W 자체가 0 되는것을 막기위해 pertubation 적용 
def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)
# b 자체가 0 되는것을 막기위해 pertubation 적용
def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)
# convolution 적용
def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')
# max pooling 적용
def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')
# patch 설정1. (가로, 세로, 채널=1, 필터 개수) conv 1 layer
W_conv1 = weight_variable([5, 5, 1, 32])
b_conv1 = bias_variable([32])
# x 이미지 구조 설정. 1자로 나열한걸 [28,28,1] 로 전환 
x_image = tf.reshape(x, [-1,28,28,1])
# 예측 식 설정1.
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)
# patch 설정2. conv 2 layer
W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])
# 예측 식 설정2. 
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)
# patch 설정3. fc layer
W_fc1 = weight_variable([7 * 7 * 64, 1024])
b_fc1 = bias_variable([1024])
# 예측 식 설정3.
h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
# Drop out 설정.
keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
# patch 설정4. softmax layer
W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])
# 예측 식 설정4
y_conv=tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

#평가
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y_conv), reduction_indices=[1]))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
sess.run(tf.global_variables_initializer())
for i in range(20000):
  batch = mnist.train.next_batch(50)
  if i%100 == 0:
    train_accuracy = accuracy.eval(feed_dict={
        x:batch[0], y_: batch[1], keep_prob: 1.0})
    print("step %d, training accuracy %g"%(i, train_accuracy))
  train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})

print("test accuracy %g"%accuracy.eval(feed_dict={
    x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))