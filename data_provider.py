import numpy as np
from numpy.random import shuffle

class DataProvider(object):
	"""docstring for DataProvider"""
	def __init__(self, num_data_points, batch_size, load_fn):
		super(DataProvider, self).__init__()

		self.num_data_points = num_data_points
		self.order = range(num_data)
		self.batch_size = batch_size
		self.batch_idx = 0
		self.load_fn = load_fn

		self.reset()


	def __iter__(self):
		return self


	def reset(self):
		self.batch_idx = -1
		self.order = shuffle(self.order)


	def next(self):
		self.batch_idx += 1
		if self.batch_idx == np.ceil(self.data[0]/self.batch_size):
			self.reset()
			raise StopIteration

		return self.load_fn(
			self.order[
				self.batch_idx * self.batch_size :
				(self.batch_idx+1) * self.batch_size
			]
		)