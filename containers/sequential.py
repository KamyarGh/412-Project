import numpy as np
import tensorflow as tf
from layers.conv import ConvLayer
from layers.fc_layer import ConstFC, FullyConnected

class Sequential(object):
	""" << docstring for Sequential >>
	Sequential container
	"""
	def __init__(self, name):
		super(Sequential, self).__init__()

		self.name = name
		self.layers = []
		self.model = None


	def __add__(self, layer):
		self.layers.append(layer)
		print('Adding {} to {} ...'.format(layer, self.name))
		return self


	def __call__(self, input_var):
		assert len(self.layers) > 0, 'Container has no layers!'

		print(input_var.get_shape())

		print(self.layers[0])
		self.model = self.layers[0](input_var)
		for i in xrange(1, len(self.layers)):
			print(self.layers[i])
			if (isinstance(self.layers[i], FullyConnected) or isinstance(self.layers[i], ConstFC)) and (len(self.model.get_shape()) > 2):
				batch_size = self.model.get_shape()[0].value
				self.model = self.layers[i](
					tf.reshape(
						self.model,
						[batch_size, -1]
					)
				)
			else:
				self.model = self.layers[i](self.model)
			print(self.model.get_shape())

		return self.model

	def build_layer(self, input_var, layer_idx):
		assert len(self.layers) > 0, 'Container has no layers!'

		print('Building Layer %d' % layer_idx)

		batch_size = input_var.get_shape()[0].value
		layer_build = self.layers[0](input_var)
		for i in xrange(1, layer_idx):
			print(self.layers[i])
			if (isinstance(self.layers[i], FullyConnected) or isinstance(self.layers[i], ConstFC)) and (len(layer_build.get_shape()) > 2):
				layer_build = self.layers[i](
					tf.reshape(
						layer_build,
						[batch_size, -1]
					)
				)
			else:
				layer_build = self.layers[i](layer_build)
			print(layer_build.get_shape())

		return layer_build