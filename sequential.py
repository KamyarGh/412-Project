import numpy as np
import tensorflow as tf

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


	def __call__(self, input_var):
		assert len(self.layers) > 0,
			'Enconder has no layers!'

		self.model = self.layers[0](input_var)
		for i in xrange(1, len(self.layers)):
			self.model = self.layers[i](self.model)

		return self.model