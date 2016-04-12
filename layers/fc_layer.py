import numpy as np
import tensorflow as tf
from numpy.random import uniform
from toolbox import name2fn

scale = 0.1

class FullyConnected(object):
	"""docstring for FullyConnected"""
	def __init__(self, input_dim, output_dim, activation=None, name=''):
		super(FullyConnected, self).__init__()

		self.input_dim = input_dim
		self.output_dim = output_dim
		self.activation = activation
		self.act_fn = name2fn(activation)
		self.name = name
		self.weights = {}

		# Initialize the weight matrices
		self.weights['W'] = tf.Variable(
			uniform(low=-scale, high=scale, size=(input_dim, output_dim)).astype(np.float32),
			name = '%s_W' % self.name
		)

		self.weights['b'] = tf.Variable(
			(np.zeros(output_dim) + (scale if activation=='relu' else 0)).astype(np.float32),
			name = '%s_b' % self.name
		)


	def __call__(self, x):
		mul = tf.matmul(x, self.weights['W']) + self.weights['b']

		if self.activation not in ['linear', None]:
			return mul

		return self.act_fn(mul)