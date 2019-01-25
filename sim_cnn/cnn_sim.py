import numpy as np
import tensorflow as tf

slim = tf.contrib.slim

def cnn_v(inputs, scope=None, activation_fn=tf.nn.relu):
	with tf.variable_scope(None, 'cnn_net', [inputs]):
		with slim.arg_scope([slim.conv2d],  stride=1, padding='SAME'):
			net = slim.repeat(inputs, 12, slim.conv2d, 256, [3, 1], scope='conv2d_3x1')
			net =slim.conv2d(net, 3, [3,1], activation_fn=None, scope='Conv2d_net1_3x1')
	return net
