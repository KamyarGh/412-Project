import numpy as np
import tensorflow as tf
from containers.sequential import Sequential
from layers.conv import ConvLayer
from layers.fc_layer import FullyConnected as FC
from layers.pool import PoolLayer

act_lib = {
	'relu': tf.nn.relu,
	'linear': None,
	'sigmoid': tf.nn.sigmoid,
	'tanh': tf.nn.tanh
}

class CNNClassifier(object):
	"""docstring for CNNClassifier"""
	def __init__(self, input_dim, input_channels, num_classes, conv_params, pool_params, fc_params, name=''):
		super(CNNClassifier, self).__init__()

		conv_params['act_fn'] = map(lambda p: act_lib[p], conv_params['act_fn'])
		fc_params['act_fn'] = map(lambda p: act_lib[p], fc_params['act_fn'])

		self.input_dim = input_dim
		self.input_channels = input_channels
		self.num_classes = num_classes
		self.conv_params = conv_params
		self.pool_params = pool_params
		self.fc_params = fc_params

		with tf.variable_scope(name):
			self._classifier_conv = Sequential('CNN_Classifier_Conv')
			self._classifier_conv += ConvLayer(
				self.input_channels,
				self.conv_params['n_filters'][0],
				self.input_dim,
				self.conv_params['kernels'][0],
				self.conv_params['strides'][0],
				name='classifier_conv_0'
			)
			print('#'*100)
			print(self._classifier_conv.layers[-1].output_dim)
			self._classifier_conv += self.conv_params['act_fn'][0]
			self._classifier_conv += PoolLayer(
				self._classifier_conv.layers[-2].output_dim,
				self.pool_params['kernels'][0],
				self.pool_params['strides'][0],
				name='pool_0'
			)
			print('#'*100)
			print(self._classifier_conv.layers[-1].output_dim)

			for i in xrange(1, len(self.conv_params['kernels'])):
				self._classifier_conv += ConvLayer(
					self.conv_params['n_filters'][i-1],
					self.conv_params['n_filters'][i],
					self._classifier_conv.layers[-1].output_dim,
					self.conv_params['kernels'][i],
					self.conv_params['strides'][i],
					name='classifier_conv_%d' % i
				)
				print('#'*100)
				print(self._classifier_conv.layers[-1].output_dim)
				self._classifier_conv += self.conv_params['act_fn'][i]
				self._classifier_conv += PoolLayer(
					self._classifier_conv.layers[-2].output_dim,
					self.pool_params['kernels'][0],
					self.pool_params['strides'][0],
					name='pool_%d' % i
				)
				print('#'*100)
				print(self._classifier_conv.layers[-1].output_dim)


			self._classifier_fc = Sequential('CNN_Classifier_FC')
			self._classifier_fc += FC(
				np.prod(self._classifier_conv.layers[-1].output_dim) * self.conv_params['n_filters'][-1],
				self.fc_params['dims'][0],
				activation=self.fc_params['act_fn'][0],
				scale=0.01,
				name='classifier_fc_0'
		    )
			print(self._classifier_fc.layers[-1].output_dim)
			for i in xrange(1, len(self.fc_params['dims'])):
				self._classifier_fc += FC(
					self.fc_params['dims'][i-1],
					self.fc_params['dims'][i],
					activation=self.fc_params['act_fn'][i],
					scale=0.01,
					name='classifier_fc_%d' % i
				)
				print(self._classifier_fc.layers[-1].output_dim)

			# self.decay = 0.0
			# for i in xrange(1, len(self.fc_params['dims'])):
			# 	self._classifier_fc += FC(
			# 		self.fc_params['dims'][i-1],
			# 		self.fc_params['dims'][i],
			# 		activation=self.fc_params['act_fn'][i],
			# 		scale=0.01,
			# 		name='classifier_fc_%d' % i
			# 	)
			# 	self.decay += 2.0 * tf.reduce_sum(tf.square(self._classifier_fc.layers[-1].weights['w']))


	def __call__(self, inputs, labels, keep_prob):
		# --------------------------------------
		self._classifier_fc += (lambda p: tf.nn.dropout(p, keep_prob))

		self._classifier_fc += FC(
			self.fc_params['dims'][-1],
			self.num_classes,
			activation=None,
			scale=0.01,
			name='classifier_logit'
		)

		self._classifier_fc += tf.nn.softmax
		# --------------------------------------

		print('Conv output dim')
		for i in range(len(self._classifier_conv.layers)):
			try:
				print(self._classifier_conv.layers[i].output_dim)
			except:
				pass
		conv = self._classifier_conv(inputs)
		print('Conv output dim w/ inputs')
		print(conv.get_shape())
		conv = tf.reshape(
			conv,
			[inputs.get_shape()[0].value, -1]
		)
		print('Conv flattened')
		print(conv.get_shape())
		self.classifier = self._classifier_fc(
			conv	
		)

		self.cost = -tf.mul(labels, tf.log(self.classifier))
		self.cost = tf.reduce_sum(self.cost)
		self.cost *= 1 / float(inputs.get_shape()[0].value)

		return self.cost, self.classifier