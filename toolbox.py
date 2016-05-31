import numpy as np
import tensorflow as tf
from scipy.misc import imread, imsave
from zoo.vae import VAE
from feat_extractors.cnn_classifier import CNNClassifier
from zoo.ae import ConvAutoEncoder, AutoEncoder
from zoo.feat_vae import FeatVAE
from zoo.vae_gan import VAEGAN
from zoo.disc import FixedConvDisc
import os
import json
from scipy.misc import imread
from evaluate import save_samples, save_ae_samples
from data_provider import DataProvider

layers_dir = './layers/'

fn_dict = {
	'relu': tf.nn.relu,
}


# format:	(folder, file, name)
_cupboard = {
	'vanilla_vae': VAE,
	'cnn_classifier': CNNClassifier,
	'conv_ae': ConvAutoEncoder,
	'ae': AutoEncoder,
	'feat_vae': FeatVAE,
	'vae_gan': VAEGAN,
	'fixed_conv_disc': FixedConvDisc
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
	def __init__(self, data_dir, flat=True, extension='.npy'):
		self.flat = flat
		self.data_dir = data_dir
		self.extension = extension
		self.load_fn = np.load if self.extension=='.npy' else imread

	def __call__(self, ids):
		if self.flat:
			data = np.array(
				[
					self.load_fn(
						os.path.join(self.data_dir, '%d%s'%(ID,self.extension))
					).flatten()
					for ID in ids
				]
			)
		else:
			data = []
			for ID in ids:
				img = self.load_fn(
					os.path.join(self.data_dir + '/%d%s'%(ID,self.extension))
				)
				if len(img.shape) == 2:
					img = np.expand_dims(img, axis=2)
				data.append(img)

		return np.array(data)


class MNISTLoader(object):
	def __init__(self, mode='train', flat=False):
		from tensorflow.examples.tutorials.mnist import input_data
		self.mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

		if mode=='train':
			self.mnist = self.mnist.train
		elif mode=='validation':
			self.mnist = self.mnist.validation
		else:
			self.mnist = self.mnist.test

		self.flat = flat

	def __call__(self, ids):
		if self.flat:
			return (
				self.mnist.images[ids,:],
				self.mnist.labels[ids,:]
			)
		else:
			return (
				np.reshape(self.mnist.images[ids,:], [len(ids), 28, 28, 1]),
				self.mnist.labels[ids,:]
			)


def visualize_square(sampler_mean, sess, dec_mean, dec_log_std_sq, sampler, sampler_input_batch, model_input_batch, enc_mean, enc_log_std_sq, train_provider, val_provider, options, catalog, mean_img, std_img):
	from numpy.random import multivariate_normal as MVN, uniform

	mean_img = mean_img
	std_img = std_img

	imgs = []
	for i in range(5):
		imgs.append(imread('/u/kamyar/report_visu/%d_norm.png'%i).flatten())
	imgs = np.array(imgs)

	encs = sess.run(
		enc_mean + tf.random_normal(enc_mean.get_shape()) * tf.exp(0.5 * enc_log_std_sq),
		feed_dict = {
			model_input_batch: imgs
		}
	)

	results = []
	for i in range(5):
		imgs = []
		begin = encs[i,:]
		end = encs[(i+1) % 5,:]
		step = (end - begin) / 5.0
		for i in range(5):
			imgs.append(begin + i*step)
		imgs = np.array(imgs)
		samples = sess.run(
			sampler_mean,
			feed_dict = {
				sampler_input_batch: imgs
			}
		)
		results.append(samples)
	for i, imgs in enumerate(results):
		print(imgs.shape)
		for j in range(5):
			imsave(
				'/u/kamyar/report_visu/square/%d_sq.png'%(5*i + j),
				mean_img + std_img*np.reshape(imgs[j,:], (48,48))
			)




def visualize(sampler_mean, sess, dec_mean, dec_log_std_sq, sampler, sampler_input_batch, model_input_batch, enc_mean, enc_log_std_sq, train_provider, val_provider, options, catalog, mean_img, std_img):
	from numpy.random import multivariate_normal as MVN, uniform

	mean_img = mean_img.flatten()
	std_img = std_img.flatten()

	# Validation Samples --------------------------------------------------------------------------
	print('Generate Samples from N(0,I)')
	val_samples = sess.run(
	    sampler_mean,
	    feed_dict = {
	        sampler_input_batch: MVN(
	            np.zeros(options['latent_dims']),
	            np.diag(np.ones(options['latent_dims'])),
	            size = options['batch_size']
	        )
	    }
	)
	val_samples = (val_samples * std_img) + mean_img

	for inputs in val_provider:
	    break
	if isinstance(inputs, tuple):
		inputs = inputs[0]
	rec_samples = sess.run(
	    dec_mean,
	    feed_dict = {
	        model_input_batch: inputs
	    }
	)

	# Reconstruction Samples --------------------------------------------------------------------------
	print('Generate Reconstruction Samples')
	print("NOT STUCK HERE!")

	# recons = []
	# for i, temp in enumerate(zip(rec_samples[0], rec_samples[1])):
	#     mean, log_std_sq = temp
	#     std = np.exp(0.5 * log_std_sq)
	#     recons.append(
	#         std * MVN(
	#             np.zeros(mean.shape[0]),
	#             np.diag(np.ones(std.shape[0]))
	#         ) + mean
	#     )
	#     print(i)
	# print("NOT STUCK HERE!")
	# recons = np.array(recons)

	recons = rec_samples
	recons = (recons * std_img) + mean_img

	inputs = (inputs * std_img) + mean_img

	print("NOT STUCK HERE!")

	try:
	    os.mkdir(options['visu_save_dir'])
	except:
	    pass

	save_ae_samples(
	    catalog,
	    np.reshape(recons, [options['batch_size']]+options['img_shape']),
	    np.reshape(inputs, [options['batch_size']]+options['img_shape']),
	    np.reshape(val_samples, [options['batch_size']]+options['img_shape']),
	    100,
	    options['visu_save_dir'],
	    num_to_save=10,
	    save_gray=True
	)
	save_samples(
	    val_samples,
	    int(0),
	    options['visu_save_dir'],
	    True,
	    options['img_shape'],
	    10
	)
	save_samples(
	    recons,
	    int(1),
	    options['visu_save_dir'],
	    True,
	    options['img_shape'],
	    10
	)
	save_samples(
	    inputs,
	    int(2),
	    options['visu_save_dir'],
	    True,
	    options['img_shape'],
	    10
	)

	# Gaussian Sampling --------------------------------------------------------------------------
	print('Fit Gaussian to Samples')
	enc_samples = None
	for i, inputs in enumerate(train_provider):
	    if isinstance(inputs, tuple):
	    	inputs = inputs[0]
	    if i == 11:
	        break
	    
	    encs = sess.run(
	        enc_mean + tf.random_normal(enc_mean.get_shape()) * tf.exp(0.5 * enc_log_std_sq),
	        feed_dict = {
	            model_input_batch: inputs
	        }
	    )

	    # codes = []
	    # for i, temp in enumerate(zip(encs[0], encs[1])):
	    #     mean, log_std_sq = temp
	    #     var = np.exp(log_std_sq)
	    #     codes.append(MVN(
	    #         mean,
	    #         np.diag(var)
	    #     ))
	    # codes = np.array(codes)
	    codes = encs
	    if enc_samples == None:
	        enc_samples = codes
	    else:
	        enc_samples = np.concatenate((enc_samples, codes))

	mean = np.mean(enc_samples, axis=0)
	std = np.std(enc_samples, axis=0)

	print("Generate new samples from Gaussian")
	val_samples = sess.run(
	    sampler_mean,
	    feed_dict = {
	        sampler_input_batch: MVN(
	            mean,
	            np.diag(std),
	            size = options['batch_size']
	        )
	    }
	)
	val_samples = (val_samples * std_img) + mean_img

	save_samples(
	    val_samples,
	    int(3),
	    options['visu_save_dir'],
	    True,
	    options['img_shape'],
	    10
	)



def test_LL_and_DKL(sess, test_provider, model_DKL, model_LL, options, model_input_batch):
	DKL = []
	LL = []
	for inputs,_ in test_provider:
		results = sess.run(
			[model_DKL, model_LL],
			feed_dict = {
				model_input_batch: inputs
			}
		)

		DKL.append(np.mean(results[0]))
		LL.append(np.mean(results[1]))

	DKL = np.mean(DKL)
	LL = np.mean(LL)
	print('DKL: {}'.format(DKL))
	print('LL: {}'.format(LL))

	return DKL, LL



def get_providers(options, log, flat=True):
    # Train provider
    if options['data_dir'] != 'MNIST':
        num_data_points = len(
            os.listdir(
                os.path.join(options['data_dir'], 'train', 'patches')
            )
        )
        num_data_points -= 2
        num_data_points = int(num_data_points * options['data_percentage'])

        train_provider = DataProvider(
            num_data_points,
            options['batch_size'],
            ImageLoader(
                data_dir = os.path.join(options['data_dir'], 'train', 'patches'),
                flat=flat,
                extension=options['file_extension']
            )
        )

        # Valid provider
        num_data_points = len(
            os.listdir(
                os.path.join(options['data_dir'], 'valid', 'patches')
            )
        )
        num_data_points -= 2
        num_data_points = int(num_data_points * options['data_percentage'])

        val_provider = DataProvider(
            num_data_points,
            options['batch_size'],
            ImageLoader(
                data_dir = os.path.join(options['data_dir'], 'valid', 'patches'),
                flat = flat,
                extension=options['file_extension']
            )
        )

        # Test provider
        num_data_points = len(
            os.listdir(
                os.path.join(options['data_dir'], 'test', 'patches')
            )
        )
        num_data_points -= 2
        num_data_points = int(num_data_points * options['data_percentage'])

        test_provider = DataProvider(
            num_data_points,
            options['batch_size'],
            ImageLoader(
                data_dir = os.path.join(options['data_dir'], 'test', 'patches'),
                flat = flat,
                extension=options['file_extension']
            )
        )

    else:
        train_provider = DataProvider(
            int(55000 * options['data_percentage']),
            options['batch_size'],
            MNISTLoader(
                mode='train',
                flat=flat
            )
        )

        val_provider = DataProvider(
            int(5000 * options['data_percentage']),
            options['batch_size'],
            MNISTLoader(
                mode='validation',
                flat = flat
            )
        )

        test_provider = DataProvider(
            int(10000 * options['data_percentage']),
            options['batch_size'],
            MNISTLoader(
                mode='test',
                flat = flat
            )
        )

    log.info('Data providers initialized.')
    return train_provider, val_provider, test_provider