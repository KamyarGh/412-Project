import numpy as np
import tensorflow as tf
from containers.sequential import Sequential
from layers.conv import ConvLayer
from layers.fc_layer import FullyConnected as FC

fc_act_fn = tf.nn.relu

class CNNClassifier(object):
	"""docstring for CNNClassifier"""
	def __init__(self, input_dim, input_channels, num_classes, conv_params, fc_params, name=''):
		super(CNNClassifier, self).__init__()

		self.input_dim = input_dim
		self.input_channels = input_channels
		self.num_classes = num_classes
		self.conv_params = conv_params
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

			for i in xrange(1, len(self.conv_params['kernels'])):
				self._classifier_conv += ConvLayer(
					self.conv_params['n_filters'][i-1],
					self.conv_params['n_filters'][i],
					self._classifier_conv.layers[-1].output_dim,
					self.conv_params['kernels'][i],
					self.conv_params['strides'][i],
					name='classifier_conv_%d' % i
				)

			self._classifier_fc = Sequential('CNN_Classifier_FC')
			self._classifier_fc += FC(
				np.prod(self._classifier_conv.layers[-1].output_dim) * self.conv_params['n_filters'][-1],
				self.fc_params['dims'][0],
				activation=fc_act_fn,
				scale=0.01,
				name='classifier_fc_0'
		    )

			for i in xrange(1, len(self.fc_params['dims'])):
				self._classifier_fc += FC(
					self.fc_params['dims'][i-1],
					self.fc_params['dims'][i],
					activation=fc_act_fn,
					scale=0.01,
					name='classifier_fc_%d' % i
				)

			self._classifier_fc += FC(
				self.fc_params['dims'][-1],
				self.num_classes,
				activation=None,
				scale=0.01,
				name='classifier_logit'
			)

			self._classifier_fc += tf.nn.softmax


	def __call__(self, inputs, labels):
		self.classifier = self._classifier_fc(
			tf.reshape(
				self._classifier_conv(inputs),
				[inputs.get_shape()[0].value, -1]
			)
		)

		self.cost = -tf.mul(labels, tf.log(self.classifier))
		self.cost = tf.reduce_sum(self.cost)
		self.cost *= 1 / inputs.get_shape()[0].value

		return self.cost, self.classifier