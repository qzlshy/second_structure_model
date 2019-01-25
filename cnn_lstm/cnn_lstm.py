import numpy as np
import tensorflow as tf

slim = tf.contrib.slim

def cnn_v(inputs, scope=None, activation_fn=tf.nn.relu):
	with tf.variable_scope(None, 'cnn_net', [inputs]):
		with slim.arg_scope([slim.conv2d],  stride=1, padding='SAME'): 
			net1 =slim.conv2d(inputs, 256, [3,1], scope='Conv2d_net1')
			net2 =slim.conv2d(net1, 256, [3,1], scope='Conv2d_net2')
			net3 =slim.conv2d(net2, 256, [3,1], scope='Conv2d_net3')
			net4 =slim.conv2d(net3, 256, [3,1], scope='Conv2d_net4')
			net5=slim.conv2d(net4, 256, [3,1], activation_fn=None, scope='Conv2d_net5')
	return net5
