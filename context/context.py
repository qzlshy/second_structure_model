import numpy as np
import tensorflow as tf

slim = tf.contrib.slim

def cnn_context(inputs, scope=None, activation_fn=tf.nn.relu):
	with tf.variable_scope(None, 'cnn_net', [inputs]):
		with slim.arg_scope([slim.conv2d],  stride=1, padding='SAME'):
			net1 = slim.conv2d(inputs, 2048, [3,1],rate=1,scope='Conv2d_net1_3x1')
			net2 = slim.conv2d(net1, 1024, [3,1],rate=1,scope='Conv2d_net2_3x1')
			net3 = slim.conv2d(net2, 512, [3,1],rate=1,scope='Conv2d_net3_3x1')
			net4 = slim.conv2d(net3, 256, [3,1],rate=1,scope='Conv2d_net4_3x1')
			net5 = slim.conv2d(net4, 128, [3,1],rate=2,scope='Conv2d_net5_3x1')
			net6 = slim.conv2d(net5, 64, [3,1],rate=4,scope='Conv2d_net6_3x1')
			net7 = slim.conv2d(net6, 32, [3,1],rate=8,scope='Conv2d_net7_3x1')
			net8 = slim.conv2d(net7, 16, [3,1],rate=16,scope='Conv2d_net8_3x1')
			net9 = tf.concat(axis=3,values=[net1,net2,net3,net4,net5,net6,net7,net8])
			net10 = slim.conv2d(net9, 640, [3,1],rate=1,scope='Conv2d_net10_3x1')
			net11 = slim.conv2d(net10, 8, [3,1],rate=1,scope='Conv2d_net11_3x1')
			net12 = slim.conv2d(net11, 8, [3,1],rate=1,scope='Conv2d_net12_3x1')
			net13 = slim.conv2d(net12, 8, [3,1],rate=1,scope='Conv2d_net13_3x1')
			net14 = slim.conv2d(net13, 8, [3,1],rate=1,scope='Conv2d_net14_3x1')
			net15 = tf.concat(axis=3,values=[net11,net12,net13,net14])
			net16 = slim.conv2d(net15, 32, [3,1],rate=1,scope='Conv2d_net16_3x1')
			net17 = tf.concat(axis=3,values=[inputs,net10,net16])
			net18 = slim.conv2d(net17, 320, [1,1],rate=1,scope='Conv2d_net18_3x1')
			netout = slim.conv2d(net18, 3, [1,1],rate=1,activation_fn=None,scope='Conv2d_netout')

	return netout
