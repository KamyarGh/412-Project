import numpy as np
import tensorflow as tf
from scipy.misc import imread
from zoo.vae import VAE
from feat_extractors.cnn_classifier import CNNClassifier
import os
import json
from scipy.misc import imread

layers_dir = './layers/'

fn_dict = {
	'relu': tf.nn.relu,
}


# format:	(folder, file, name)
_cupboard = {
	'vanilla_vae': VAE,
	'cnn_classifier': CNNClassifier
}

def cupboard(name):
	assert name in _cupboard, 'Object of interest was not found!'
		
	return _cupboard[name]


def name2fn(activation):
	return fn_dict[activation]



class CIFARLoader(object):
	def __init__(self, data_dir, flat=False):
		self.flat = flat
		self.data_dir = data_dir
		self.num_classes = 10

	def __call__(self, ids):
		if self.flat:
			data = np.array(
				[
					imread(
						os.path.join(self.data_dir, 'patches', '%d.png'%ID)
					).flatten()
					for ID in ids
				]
			)
		else:
			data = np.array(
				[
					imread(
						os.path.join(self.data_dir, 'patches', '%d.png'%ID)
					)
					for ID in ids
				]
			)

		temp_labels = np.array(
			[
				json.load(open(os.path.join(self.data_dir, 'info', '%d.json'%ID), 'r'))['label']
				for ID in ids
			]
		).astype(float)

		labels = np.zeros([temp_labels.shape[0], self.num_classes])
		labels[np.arange(temp_labels.shape[0]).astype(int), temp_labels.astype(int)] = 1

		return data, labels



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