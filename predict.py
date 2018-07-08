import tensorflow as tf
from PIL import Image
import numpy as np
import sys
import os
import random

now_path = str(os.getcwd()).replace('\\','/') + "/" #得到当前目录
data_path = now_path + "data/"
model_path = now_path + "model/"

#构建计算图可以在外部运行,计算通常会通过其它语言并用更为高效的代码来实现
x = tf.placeholder("float", shape=[None, 784]) #placeholder占位符
y_ = tf.placeholder("float", shape=[None, 15])

W = tf.Variable(tf.zeros([784,15])) #一个变量代表着TensorFlow计算图中的一个值，能够在计算过程中使用，甚至进行修改
b = tf.Variable(tf.zeros([15]))   #变量需要通过seesion初始化后，才能在session中使用
#定义两个函数用于初始化
def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)
#卷积和池化
def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')
#第一层卷积
W_conv1 = weight_variable([5, 5, 1, 32])
b_conv1 = bias_variable([32])
#把x变成一个4d向量，其第2、第3维对应图片的宽、高，最后一维代表图片的颜色通道数(因为是灰度图所以这里的通道数为1，如果是rgb彩色图，则为3)
x_image = tf.reshape(x, [-1,28,28,1])
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)  #应用ReLU激活函数
h_pool1 = max_pool_2x2(h_conv1)

#第二层卷积
W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])

h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

#密集连接层
W_fc1 = weight_variable([7 * 7 * 64, 1024])
b_fc1 = bias_variable([1024])

h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
#为了减少过拟合，我们在输出层之前加入dropout
keep_prob = tf.placeholder("float")
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

#输出层
W_fc2 = weight_variable([1024, 15])
b_fc2 = bias_variable([15])

y_conv=tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

# 用于保存训练的最佳模型
saver = tf.train.Saver()
#训练和评估
loss = -tf.reduce_sum(y_*tf.log(y_conv))
optimizer = tf.train.AdamOptimizer(1e-4).minimize(loss)
correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

session = tf.InteractiveSession()
session.run(tf.global_variables_initializer())

saver = tf.train.Saver()
test_image_file = data_path + "train/1.png"
images = np.empty((1, 28 * 28))
labels = np.zeros((1, 15))
if not os.path.exists(test_image_file):
    print("error: test_image_file[%s] not exist" % test_image_file)
    sys.exit()
img = Image.open(test_image_file)
img_ndarray = np.asarray(img, dtype='float64') / 255
images[0] = np.ndarray.flatten(img_ndarray)
labels[0][2] = 1

if not os.path.exists(model_path + "checkpoint"):
    print("error: model_file not exist")
model_file=tf.train.latest_checkpoint(model_path)
saver.restore(session, model_file)
test_pred = tf.argmax(y_conv, 1).eval({x: images, keep_prob: 1})
#test_correct = correct_prediction.eval(feed_dict={x: images, y_: labels, keep_prob: 1})
session.close()
print(test_pred[0])
