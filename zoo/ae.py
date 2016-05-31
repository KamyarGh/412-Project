import numpy as np
import tensorflow as tf
from sequential import Sequential
from layers.fc_layer import FullyConnected
from layers.conv import ConvLayer
from numpy.random import normal

act_lib = {
	'relu': tf.nn.relu,
	'linear': None,
	'sigmoid': tf.nn.sigmoid,
	'tanh': tf.nn.tanh
}

class AutoEncoder(object):
	""" << docstring for AE >>
	Auto-Encoder with fully connected layers
	"""
	def __init__(self, input_dims, enc_params, dec_params, name=''):
		super(AutoEncoder, self).__init__()

		self.input_dims = input_dims
		self.enc_params = enc_params
		self.dec_params = dec_params

		self.enc_params['act_fn'] = map(lambda p: act_lib[p], self.enc_params['act_fn'])
		self.dec_params['act_fn'] = map(lambda p: act_lib[p], self.dec_params['act_fn'])

		self.name = name

		self._encoder = Sequential(self.name + '_ae_encoder')
		for i in range(len(enc_params['layer_dims'])):
			if i == 0:
				self._encoder += FullyConnected(
					self.input_dims,
					self.enc_params['layer_dims'][i],
					self.enc_params['act_fn'][i],
					name=self.name + '_e_fc_%d'%(i+1)
				)
			else:
				self._encoder += FullyConnected(
					self.enc_params['layer_dims'][i-1],
					self.enc_params['layer_dims'][i],
					self.enc_params['act_fn'][i],
					name=self.name + '_e_fc_%d'%(i+1)
				)

		self._decoder = Sequential(self.name + '_ae_decoder')
		for i in range(len(self.dec_params['layer_dims'])):
			if i == 0:
				self._decoder += FullyConnected(
					self.enc_params['layer_dims'][-1],
					self.dec_params['layer_dims'][i],
					self.dec_params['act_fn'][i],
					name=self.name + '_d_fc_%d'%(i+1)
				)
			else:
				self._decoder += FullyConnected(
					self.dec_params['layer_dims'][i-1],
					self.dec_params['layer_dims'][i],
					self.dec_params['act_fn'][i],
					name=self.name + '_d_fc_%d'%(i+1)
				)

	def __call__(self, clean_input, noisy_input):
		self.encoder = self._encoder(noisy_input)
		self.decoder = self._decoder(self.encoder)

		self.cost = tf.mul(
    		1.0 / (clean_input.get_shape()[0].value),
    		tf.square(
    			tf.sub(
    				self.decoder,
    				clean_input
    			)
    		)
    	)

		self.cost = tf.reduce_sum(self.cost)
		return self.cost

class ConvAutoEncoder(object):
	""" << docstring for AE >>
	Auto-Encoder with Convolutional Encoder layers
	Encoder is purely Convolutional
	Decoder is purely Fully Connected
	"""
	def __init__(self, input_shape, input_channels, enc_params, dec_params, name=''):
		"""
		enc_params:
			- kernels
			- strides
			- num_filters
			- act_fn
		dec_params:
			- layer_dims
			- act_fn
		"""
		super(ConvAutoEncoder, self).__init__()

		self.input_shape = input_shape
		self.input_channels = input_channels
		self.enc_params = enc_params
		self.dec_params = dec_params
		self.name = name

		self.enc_params['act_fn'] = map(lambda p: act_lib[p], self.enc_params['act_fn'])
		self.dec_params['act_fn'] = map(lambda p: act_lib[p], self.dec_params['act_fn'])

		# Build the encoder which is fully convolutional and no pooling
		self._encoder = Sequential(self.name + 'ae_encoder')
		for i in range(len(self.enc_params['kernels'])):
			self._encoder += ConvLayer(
				self.input_channels if i == 0 else self.enc_params['num_filters'][i-1],
				enc_params['num_filters'][i],
				self.input_shape if i == 0 else self._encoder.layers[-2].output_dim,
				self.enc_params['kernels'][i],
				self.enc_params['strides'][i],
				name=self.name+'_enc_conv_%d' % (i+1)
			)
			self._encoder += self.enc_params['act_fn'][i]

		# Build the decoder which is fully connected
		self._decoder = Sequential(self.name + 'ae_decoder')
		for i in range(len(self.dec_params['layer_dims'])):
			self._decoder += FullyConnected(
				self.enc_params['num_filters'][-1] * np.prod(self._encoder.layers[-2].output_dim) if i == 0 \
					else self.dec_params['layer_dims'][i-1],
				self.dec_params['layer_dims'][i],
				self.dec_params['act_fn'][i],
				name=self.name+'_dec_fc_%d' % (i+1)
			)

	def __call__(self, clean_input, noisy_input):
		self.encoder = self._encoder(noisy_input)
		self.decoder = self._decoder(
			tf.reshape(
				self.encoder,
				[noisy_input.get_shape()[0].value, -1]
			)
		)

		self.cost = tf.reduce_sum(
				tf.mul(
	    		1.0 / (clean_input.get_shape()[0].value),
	    		tf.square(
	    			tf.sub(
	    				self.decoder,
	    				tf.reshape(
	    					clean_input,
	    					[clean_input.get_shape()[0].value, -1]
						)
	    			)
	    		)
	    	)
		)

		return self.cost

if __name__ == '__main__':
	from data_provider import DataProvider
	import toolbox
	from evaluate import save_samples
	import os


	options = {
		'data_dir': './TFD',
		'img_shape': [48,48],
		'batch_size': 128,
		'lr': 0.0005,
	}

	num_data_points = len(
	    os.listdir(
	        os.path.join(options['data_dir'], 'train', 'patches')
	    )
	)
	num_data_points -= 2

	train_provider = DataProvider(
		num_data_points,
		options['batch_size'],
		toolbox.ImageLoader(
			data_dir = './TFD/train/patches',
			flat=True
		)
	)

	with tf.device('/gpu:0'):
		model = AutoEncoder(
			input_dims = 2304,
			enc_params = {
				'layer_dims': [1024, 1024, 400],
				'act_fn': ['tanh', 'tanh', 'tanh']
			},
			dec_params = {
				'layer_dims': [1024, 1024, 2304],
				'act_fn': ['tanh', 'tanh', 'linear']
			},
			name='fuck'
		)

		model_input_batch = tf.placeholder(
			tf.float32,
			shape = [options['batch_size'], np.prod(np.array(options['img_shape']))],
			name = 'enc_inputs'
		)
		model_noisy_input_batch = tf.placeholder(
			tf.float32,
			shape = [options['batch_size'], np.prod(np.array(options['img_shape']))],
			name = 'noisy_enc_inputs'
		)

		cost_function = model(model_input_batch, model_noisy_input_batch)

		optimizer = tf.train.AdamOptimizer(
			learning_rate=options['lr']
		)
		train_step = optimizer.minimize(
			cost_function
		)

		init_op = tf.initialize_all_variables()

	with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess:

		sess.run(init_op)

		for i, inputs in enumerate(train_provider):
			result = sess.run(
				(model.decoder, cost_function, train_step),
				feed_dict = {
					model_input_batch: inputs,
					model_noisy_input_batch: inputs + normal(
                                loc=0.0,
                                scale=np.float32(0.01),
                                size=inputs.shape
                            )
				}
			)

			print('\ncost_function: %f' % float(np.mean(result[1])))

			# if i == 400:
			# 	break

	# save_samples(
	# 	result[0],
	# 	0,
	# 	os.path.join('.', 'rand_folder'),
	# 	True,
	# 	options['img_shape'],
	# 	options['batch_size']
	# )