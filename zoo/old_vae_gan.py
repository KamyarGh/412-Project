import numpy as np
import tensorflow as tf

class VAEGAN(object):
	"""docstring for VAEGAN"""
	def __init__(self, vae, discriminator, img_shape, input_channels, vae_scope, disc_scope, name=''):
		super(VAEGAN, self).__init__()

		self._vae = vae
		self._discriminator = discriminator

		self.img_shape = img_shape
		self.input_channels = input_channels

		self.vae_scope = vae_scope
		self.disc_scope = disc_scope

	def __call__(self, vae_input, sampler_input_batch, optimizer):
		self.n_samples = vae_input.get_shape()[0].value

		with tf.variable_scope(self.vae_scope):
			self.vanilla_cost = self._vae(vae_input)

		recon_mean = self._vae.dec_mean
		recon_std = tf.exp(
			tf.mul(
				0.5,
				self._vae.dec_log_std_sq
			)
		)
		self.vae_recons = tf.add(
			tf.mul(
				tf.random_normal(
					[self.n_samples, vae_input.get_shape()[1].value]
				),
				recon_std
			),
			recon_mean
		)

		with tf.variable_scope(self.vae_scope):
			self.sampler = self._vae.build_sampler(sampler_input_batch)

		disc_inputs = tf.concat(
			0,
			[
				vae_input,
				self.vae_recons,
				self.sampler
			]
		)
		disc_inputs = tf.reshape(
			disc_inputs,
			[disc_inputs.get_shape()[0].value] + self.img_shape + [self.input_channels]
		)

		with tf.variable_scope(self.disc_scope):
			self.discriminator = self._discriminator(
				disc_inputs
			)
			self.discriminator = tf.clip_by_value(self.discriminator, 0.001, 0.999)

		labels = tf.concat(
			0,
			[
				np.ones(vae_input.get_shape()[0].value),
				np.zeros(vae_input.get_shape()[0].value),
				np.zeros(sampler_input_batch.get_shape()[0].value)
			]
		)
		labels = tf.cast(labels, tf.float32)

		# Losses --------------------------------------------------------------------------------

		# Discrimnator Cross-Entropy
		self.disc_CE = (1 / float(labels.get_shape()[0].value)) * tf.reduce_sum(
			-tf.add(
				tf.mul(
					labels,
					tf.log(self.discriminator)
				),
				tf.mul(
					1.0 - labels,
					tf.log(1.0 - self.discriminator)
				)
			)
		)

		# DKL
		self.vae_DKL = -self._vae.DKL # DKL in the vae was accidentally -DKL

		# VAE Reconstruction Loss
		self.vae_rec_cost = -self._vae.rec_loss # Again, accidentally defined as -rec_loss in the vae

		# Generator Loss
		self.gen_loss = -self.disc_CE + self.vae_DKL + self.vae_rec_cost

		# Disc Accuracy
		self.disc_accuracy = (1 / float(labels.get_shape()[0].value)) * tf.reduce_sum(
			tf.cast(
				tf.equal(
					tf.round(self.discriminator),
					labels
				),
				tf.float32
			)
		)

		# Define Optimizers ---------------------------------------------------------------------
		
		self.vae_train_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.vae_scope)
		self.disc_train_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.disc_scope)

		print(self.vae_train_vars)
		print(self.disc_train_vars)

		self.vae_optimizer = optimizer.minimize(self.gen_loss, var_list=self.vae_train_vars)
		self.disc_optimizer = optimizer.minimize(self.disc_CE, var_list=self.disc_train_vars)

		# --------------------------------------------------------------------------------------

		return tf.train.AdamOptimizer(learning_rate=0.0005).minimize(self.vanilla_cost), self.disc_optimizer