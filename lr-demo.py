# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

num_examples = 50

X = np.array([np.linspace(-2,4,num_examples), np.linspace(-6,6,num_examples)])

X += np.random.randn(2,num_examples)

x,y = X

x_bias = np.array([(1.,a) for a in x]).astype(np.float32)

loss = []
learning_rate = 0.002
training_step = 50

with tf.Session() as sess:
    #Set up all the tensors, variables, and operations.
    input =tf.constant(x_bias)
    # 输入数，这个是一个行向量[1，x[i]]
    target =tf.constant(np.transpose([y]).astype(np.float32))
    #目标target。
    weights = tf.Variable(tf.random_normal([2,1], 0, 0.1))
    #权重w1,w2组成的数组weights。这个是一个随机数，2行1列，平均为0，均方差为0.1
 
    tf.global_variables_initializer().run()
    #tf 全局变量初始化运行。
 
    yhat =tf.matmul(input, weights)
    #根据输入，和权重预测出y帽
    yerror =tf.subtract(yhat, target)
    #y帽与目标的差
    loss = tf.nn.l2_loss(yerror)
    #使用tf ，计算L2_loss 损失
    update_weights= tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)
    #根据梯度下降算法，使用tf 计算更新后的权重。
    #注意，真正允许还没有开始，目前只是定义好“图形”。
 
   for _inrange(training_step):
       update_weights.run()
       loss.append(loss.eval())
    