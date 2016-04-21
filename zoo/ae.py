import numpy as np
import tensorflow as tf
from sequential import Sequential
from layers.fc_layer import FullyConnected

class AutoEncoder(object):
	""" << docstring for AE >>
	Auto-Encoder with fully connected layers
	"""
	def __init__(self, enc_layers, dec_layers, act_fn, name=''):
		super(AutoEncoder, self).__init__()

		self.input_dims = enc_layers[0]
		self.code_dims = dec_layers[0]

		self._encoder = Sequential('ae_encoder')
		print(enc_layers)
		for i in range(len(enc_layers)-1):
			self._encoder += FullyConnected(enc_layers[i], enc_layers[i+1], act_fn, name='fc_%d'%(i+1))
			print('{} to {}'.format(enc_layers[i], enc_layers[i+1]))

		self._decoder = Sequential('ae_decoder')
		print(dec_layers)
		self._decoder += FullyConnected(enc_layers[-1], dec_layers[0], act_fn, name='fc_1')
		print('{} to {}'.format(enc_layers[-1], dec_layers[0]))
		for i in range(len(dec_layers)-1):
			self._decoder += FullyConnected(dec_layers[i], dec_layers[i+1], act_fn, name='fc_%d'%(i+2))
			print('{} to {}'.format(dec_layers[i], dec_layers[i+1]))

	def __call__(self, input_var):
		self.encoder = self._encoder(input_var)
		self.decoder = self._decoder(self.encoder)

		self.cost = tf.mul(
    		1.0 / (input_var.get_shape()[0].value),
    		tf.square(
    			tf.sub(
    				self.decoder,
    				input_var
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
	        os.path.join(options['data_dir'], 'train')
	    )
	)
	num_data_points -= 2

	train_provider = DataProvider(
		num_data_points,
		options['batch_size'],
		toolbox.ImageLoader(
			data_dir = os.path.join(options['data_dir'], 'train'),
			flat=True
		)
	)

	with tf.device('/gpu:0'):
		model = AutoEncoder(
			[2304, 1024, 400],
			[1024, 2304],
			tf.nn.sigmoid,
			name = 'AutoEncoder'
		)

		model_input_batch = tf.placeholder(
			tf.float32,
			shape = [options['batch_size'], np.prod(np.array(options['img_shape']))],
			name = 'enc_inputs'
		)

		cost_function = model(model_input_batch)

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
					model_input_batch: inputs
				}
			)

			print('\ncost_function: %f' % float(np.mean(result[1])))

			if i == 400:
				break

	save_samples(
		result[0],
		0,
		os.path.join('.', 'rand_folder'),
		True,
		options['img_shape'],
		options['batch_size']
	)