import numpy as np
import tensorflow as tf

slim = tf.contrib.slim

def block(net, channel=100, scope=None, reuse=None):
	with tf.variable_scope(scope, 'Block_1', [net], reuse=reuse):
		with tf.variable_scope('Branch_0'):
			tower_conv = slim.conv2d(net, 100, [1,1], scope='Conv2d_1x1')
		with tf.variable_scope('Branch_1'):
			tower_conv1_0 = slim.conv2d(net, 100, [1,1], scope='Conv2d_10_1x1')
			tower_conv1_1 = slim.conv2d(tower_conv1_0, 100, [3,1], scope='Conv2d_11_3x1')
		with tf.variable_scope('Branch_2'):
			tower_conv2_0 = slim.conv2d(net, 100, [1,1], scope='Conv2d_20_1x1')
			tower_conv2_1 = slim.conv2d(tower_conv2_0, 100, [3,1], scope='Conv2d_21_3x1')
			tower_conv2_2 = slim.conv2d(tower_conv2_1, 100, [3,1], scope='Conv2d_22_3x1')
			tower_conv2_3 = slim.conv2d(tower_conv2_2, 100, [3,1], scope='Conv2d_23_3x1')
		mixed=tf.concat(axis=3, values=[tower_conv, tower_conv1_1, tower_conv2_3])
	return mixed


def cnn_v(inputs, scope=None, activation_fn=tf.nn.relu):
	with tf.variable_scope(None, 'cnn_net', [inputs]):
		with slim.arg_scope([slim.conv2d, slim.max_pool2d, slim.avg_pool2d],
				stride=1, padding='SAME'):
			net1=slim.repeat(inputs, 4, block,channel=100, scope='net1')
			net2=slim.repeat(inputs, 2, block,channel=100, scope='net2')
			net3=slim.repeat(inputs, 1, block,channel=100, scope='net3')
			net4=tf.concat(axis=3, values=[net1,net2,net3])
			net1_2=slim.repeat(net4, 4, block,channel=100, scope='net1_2')
			net2_2=slim.repeat(net4, 2, block,channel=100, scope='net2_2')
			net3_2=slim.repeat(net4, 1, block,channel=100, scope='net3_2')
			net4_2=tf.concat(axis=3, values=[net1_2,net2_2,net3_2])
			net5=slim.conv2d(net4_2, 100, [11,1], activation_fn=tf.nn.relu,scope='net5')
			net6=slim.conv2d(net5, 100, 1, activation_fn=tf.nn.relu,scope='net6')
			net7=slim.conv2d(net6, 100, 1, activation_fn=tf.nn.relu,scope='net7')
			net8=slim.conv2d(net7, 3, 1, activation_fn=None,scope='net8')
	return net8

