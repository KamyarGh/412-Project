import numpy as np
import tensorflow as tf
from scipy.misc import imread
from zoo.vae import VAE
import os

layers_dir = './layers/'

fn_dict = {
	'relu': tf.nn.relu,
}


# format:	(folder, file, name)
_cupboard = {
	'vanilla_vae': VAE
}

def cupboard(name):
	assert name in _cupboard, 'Object of interest was not found!'
		
	return _cupboard[name]


def name2fn(activation):
	return fn_dict[activation]


class ImageLoader(object):
	def __init__(self, data_dir, flat=True):
		self.flat = flat
		self.data_dir = data_dir

	def __call__(self, ids):
		if self.flat:
			data = np.array(
				[
					np.load(
						os.path.join(self.data_dir, '%d.npy'%ID)
					).flatten()
					for ID in ids
				]
			)
		else:
			data = np.array(
				[
					np.load(
						os.path.join(self.data_dir + '%d.npy'%ID)
					)
					for ID in ids
				]
			)

		return data