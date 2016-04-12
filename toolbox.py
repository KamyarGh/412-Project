import numpy as np
import tensorflow as tf
from scipy.misc import imread

layers_dir = './layers/'

fn_dict = {
	'relu': tf.nn.relu,
}


# format:	(folder, file, name)
_cupboard = {
	'vanilla_vae': ('zoo', 'vae', 'VAE')
}

def cupboard(name):
	assert name in _cupboard,
		'Object of interest was not found!'
		
	temp = _cupboard[name]
	eval('from %s.%s import %s' % (temp[0], temp[1], temp[2]))
	eval('return %s' % temp[2])


def name2fn(activation):
	return fn_dict[activation]


class ImageLoader(object):
	def __init__(self, path, flat=True):
		self.flat = flat
		self.path = path

	def __call__(self, ids):
		if self.flat:
			data = np.array([imread(data_dir + '%d.png'%ID).flatten() for ID in ids])
		else:
			data = np.array([imread(data_dir + '%d.png'%ID) for ID in ids])

		return data