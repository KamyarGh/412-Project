import numpy as np
import tensorflow as tf
from layers.pool import PoolLayer

class VAEGAN(object):
	"""docstring for VAEGAN"""
	def __init__(self, vae, discriminator, disc_weight, img_shape, input_channels, vae_scope, disc_scope, name=''):
		super(VAEGAN, self).__init__()

		self._vae = vae
		self._discriminator = discriminator

		self.disc_weight = disc_weight

		self.img_shape = img_shape
		self.input_channels = input_channels

		self.vae_scope = vae_scope
		self.disc_scope = disc_scope

	def __call__(self, vae_input, sampler_input_batch, optimizer):
		self.n_samples = vae_input.get_shape()[0].value

		# Build VAE ---------------------------------------------------------------
		with tf.variable_scope(self.vae_scope):
			self.vae_cost = self._vae(vae_input)

		recon_mean = self._vae.dec_mean
		recon_std = tf.exp(
			tf.mul(
				0.5,
				self._vae.dec_log_std_sq
			)
		)
		self.vae_recons = recon_mean
		self.vae_recons = tf.add(
			tf.mul(
				tf.random_normal(
					[self.n_samples, vae_input.get_shape()[1].value]
				),
				recon_std
			),
			recon_mean
		)

		# Build VAE Sampler -------------------------------------------------------
		with tf.variable_scope(self.vae_scope):
			self.sampler = self._vae.build_sampler(sampler_input_batch)

		# Build Discriminator -----------------------------------------------------
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
			self.discriminator = self._discriminator(disc_inputs)
			self.discriminator = tf.clip_by_value(self.discriminator, 0.00001, 0.99999)

			# Get the feature space activations from the real data
			self.real_pool_feats = []
			for i in range(len(self._discriminator._disc.layers)):
				if isinstance(self._discriminator._disc.layers[i], PoolLayer):
					print('$'*100)
					print(self._discriminator._disc.layers[i])
					print('$'*100)

					self.real_pool_feats.append(
						self._discriminator._disc.build_layer(
							tf.reshape(
								vae_input,
								[vae_input.get_shape()[0].value] + self.img_shape + [self.input_channels]
							),
							i
						)
					)

			# Get the feature space activations from the vae reconstructions
			self.vae_pool_feats = []
			for i in range(len(self._discriminator._disc.layers)):
				if isinstance(self._discriminator._disc.layers[i], PoolLayer):
					self.vae_pool_feats.append(
						self._discriminator._disc.build_layer(
							tf.reshape(
								self.vae_recons,
								[self.vae_recons.get_shape()[0].value] + self.img_shape + [self.input_channels]
							),
							i
						)
					)

		# Labels -----------------------------------------------------------------
		labels = tf.constant(
			np.expand_dims(
				np.concatenate(
					(
						np.ones(vae_input.get_shape()[0].value),
						np.zeros(vae_input.get_shape()[0].value),
						np.zeros(sampler_input_batch.get_shape()[0].value)
					),
					axis=0
				).astype(np.float32),
				axis=1
			)
		)
		labels = tf.cast(labels, tf.float32)
		print(labels.get_shape())
		print('@'*200)

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

		# Define Feature Space Reconstruction Loss
		self.feat_recon_cost = 0.5 * reduce(
			tf.add,
			map(
				lambda p: tf.reduce_sum(
					tf.square(
						tf.sub(p[1], p[0])
					)
				),
				zip(self.real_pool_feats, self.vae_pool_feats)
			)
		)

		# Generator Loss
		self.gen_loss = -self.disc_weight*self.disc_CE + self.vae_DKL + self.feat_recon_cost
		# self.gen_loss = -self.disc_weight*self.disc_CE + self.vae_DKL + self.vae_rec_cost
		# self.gen_loss = -self.disc_weight*self.disc_CE

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
		
		# Get Generator and Disriminator Trainable Variables
		self.vae_train_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.vae_scope)
		self.disc_train_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.disc_scope)

		# Get VAE gradients
		grads = optimizer.compute_gradients(self.gen_loss, self.vae_train_vars)
		grads = [gv for gv in grads if gv[0] != None]
		clip_grads = [(tf.clip_by_norm(gv[0], 5.0, name='vae_grad_clipping'), gv[1]) for gv in grads]
		self.vae_backpass = optimizer.apply_gradients(clip_grads)

		# Get Dsicriminator gradients
		grads = optimizer.compute_gradients(self.disc_CE, self.disc_train_vars)
		grads = [gv for gv in grads if gv[0] != None]
		clip_grads = [(tf.clip_by_norm(gv[0], 5.0, name='disc_grad_clipping'), gv[1]) for gv in grads]
		self.disc_backpass = optimizer.apply_gradients(clip_grads)

		# Get Vanilla VAE gradients (for initial training of the VAE)
		grads = optimizer.compute_gradients(self.vae_cost, self.vae_train_vars)
		grads = [gv for gv in grads if gv[0] != None]
		clip_grads = [(tf.clip_by_norm(gv[0], 5.0, name='vanilla_grad_clipping'), gv[1]) for gv in grads]
		self.vanilla_backpass = optimizer.apply_gradients(clip_grads)

		# --------------------------------------------------------------------------------------

		return self.vae_backpass, self.disc_backpass, self.vanilla_backpass
