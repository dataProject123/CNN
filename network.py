import tensorflow as tf
from PIL import Image
import numpy as np
import sys
import os
import random

class Network:
    def __init__(self, data_path, model_path, y_num = 15):
        self.data_path = data_path
        self.model_path = model_path
        self.y_num = y_num
        self.predict_obj = None
   
    def init(self):
        # 构建计算图可以在外部运行,计算通常会通过其它语言并用更为高效的代码来实现
        self.x = tf.placeholder("float", shape=[None, 784]) #placeholder占位符
        self.y_ = tf.placeholder("float", shape=[None, self.y_num])
        self.keep_prob = tf.placeholder("float")
        self.conv_net = self.get_neural_network()
        # 用于保存训练的最佳模型，最多保存三次训练数据
        self.saver = tf.train.Saver(max_to_keep=3)
        self.session = tf.InteractiveSession()   #InteractiveSession能让你在运行图的时候，插入一些计算图
    
    def weight_variable(self, shape):
        initial = tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(initial)
    
    def bias_variable(self, shape):
        initial = tf.constant(0.1, shape=shape)
        return tf.Variable(initial)
    
    # 卷积和池化
    def conv2d(self, x, W):
        return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')
    
    # 池化
    def max_pool_2x2(self, x):
        return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        
    # 加载真值数据
    def get_label(self, file_path):
        if not os.path.exists(file_path):
            print("error: file_path[%s] not exist" % file_path)
            sys.exit()
        with open(file_path) as f:
            line = f.readline()
            line = line.strip('\n')
            label_list = line.split(',')
            i = 0
            while i < len(label_list):
                label_list[i] = int(label_list[i])
                i += 1
            return label_list
    
    # 加载图片数据
    def get_batch_data(self, data_path, total_num, batch_num, label_value):
        batch = [[] for i in range(2)]
        images = np.empty((batch_num, 28 * 28))
        labels = np.zeros((batch_num, 15))
        for i in  range(batch_num):
            file_index = random.randint(0, total_num)
            file_path = data_path + "/" + str(file_index) + ".png"
            if not os.path.exists(file_path):
                print("error: file_path[%s] not exist" % file_path)
                sys.exit()
            img = Image.open(file_path)
            img_ndarray = np.asarray(img, dtype='float64') / 255
            images[i] = np.ndarray.flatten(img_ndarray)
            labels[i][label_value[file_index]] = 1
        batch[0] = images
        batch[1] = labels
        return batch

    def get_neural_network(self): 
        # 一个变量代表着TensorFlow计算图中的一个值，能够在计算过程中使用，甚至进行修改
        # 变量需要通过seesion初始化后，才能在session中使用
        W = tf.Variable(tf.zeros([784, self.y_num]))
        b = tf.Variable(tf.zeros([self.y_num]))
        
        # 第一层卷积
        W_conv1 = self.weight_variable([5, 5, 1, 32])
        b_conv1 = self.bias_variable([32])
        # 把x变成一个4d向量，其第2、第3维对应图片的宽、高
        # 最后一维代表图片的颜色通道数(因为是灰度图所以这里的通道数为1，如果是rgb彩色图，则为3)
        x_image = tf.reshape(self.x, [-1,28,28,1])
        h_conv1 = tf.nn.relu(self.conv2d(x_image, W_conv1) + b_conv1)  #应用ReLU激活函数
        h_pool1 = self.max_pool_2x2(h_conv1)

        # 第二层卷积
        W_conv2 = self.weight_variable([5, 5, 32, 64])
        b_conv2 = self.bias_variable([64])
        h_conv2 = tf.nn.relu(self.conv2d(h_pool1, W_conv2) + b_conv2)
        h_pool2 = self.max_pool_2x2(h_conv2)

        # 全连接层
        W_fc1 = self.weight_variable([7 * 7 * 64, 1024])
        b_fc1 = self.bias_variable([1024])
        h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
        h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

        # 为了减少过拟合，我们在输出层之前加入dropout
        h_fc1_drop = tf.nn.dropout(h_fc1, self.keep_prob)

        # 输出层
        W_fc2 = self.weight_variable([1024, self.y_num])
        b_fc2 = self.bias_variable([self.y_num])
        
        return tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

    def train(self):
        # 加载训练数据真值
        train_label = self.get_label(self.data_path + "train/label.txt")
        # 定义损失函数
        loss = -tf.reduce_sum(self.y_*tf.log(self.conv_net))
        optimizer = tf.train.AdamOptimizer(learning_rate=0.0001).minimize(loss)
        correct_prediction = tf.equal(tf.argmax(self.conv_net, 1), tf.argmax(self.y_, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
        
        # 所有变量都要进行初始化，不能放到init中
        self.session.run(tf.global_variables_initializer())

        # 最高准确率
        max_acc = 0
        model_index = 1
        for i in range(400):
            batch = self.get_batch_data(self.data_path + "train", 9999, 50, train_label)
            self.session.run(optimizer, feed_dict={self.x: batch[0], self.y_: batch[1], self.keep_prob: 0.5})
            val_loss, val_acc = self.session.run([loss, accuracy], feed_dict={self.x: batch[0], self.y_: batch[1], self.keep_prob: 1.0})
            #train_acc = accuracy.eval(feed_dict={x:batch[0], y_: batch[1], keep_prob: 1.0})
            #test_correct = correct_prediction.eval(feed_dict={x: batch[0], y_: batch[1], keep_prob: 1.0})
            print('epoch:%d, val_loss:%f, val_acc:%f'%(i,val_loss,val_acc))
            if val_acc > max_acc:
                max_acc = val_acc
                self.saver.save(self.session, self.model_path + 'mnist.ckpt', global_step=model_index)
                model_index += 1

    def get_predict_obj(self):
        if self.predict_obj == None:
            if not os.path.exists(self.model_path + "checkpoint"):
                print("error: model_file not exist") 
            model_file = tf.train.latest_checkpoint(self.model_path)
            self.saver.restore(self.session, model_file)
            self.predict_obj = tf.argmax(self.conv_net, 1)
            self.session.run(tf.global_variables_initializer())
        
        return self.predict_obj 

    def predict(self, image_file_path):
        if not os.path.exists(image_file_path):
            print("error: image_file_path[%s] not exist" % image_file_path)
            sys.exit()
        images = np.empty((1, 28 * 28))
        img = Image.open(image_file_path)
        img_ndarray = np.asarray(img, dtype='float64') / 255
        images[0] = np.ndarray.flatten(img_ndarray)
        pred = self.get_predict_obj()
        test_pred = pred.eval({self.x: images, self.keep_prob: 1})
        return test_pred[0]
    

if __name__ == "__main__" :
    now_path = str(os.getcwd()).replace('\\','/') + "/" #得到当前目录
    data_path = now_path + "data/"
    model_path = now_path + "model/"
    if not os.path.exists(data_path):
        print("error: data_path[%s] not exist" % data_path)
        sys.exit()
    network_obj = Network(data_path = data_path, model_path = model_path)
    network_obj.init()
    network_obj.train()

