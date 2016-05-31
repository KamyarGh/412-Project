import numpy as np
import tensorflow as tf
from sequential import Sequential
from layers.fc_layer import FullyConnected

eps = 0.0

coder_act_fn = tf.nn.tanh
mean_std_act_fn = None
dec_mean_act_fn = None

class VAE(object):
	""" << docstring for VAE >>
	Variational Auto-Encoder with fully connected layers
	"""
	# 	- p_layers:		a list of sizes of each fc layer for the decoder network
	# 	- q_layers:		a list of sizes of each fc layer for the encoder network
	# 	- input_dims:	input data dimensions
	#	- latent_dims:	dimension of latent representation
	def __init__(self, p_layers, q_layers, input_dims, latent_dims, DKL_weight, sigma_clip, name):
		super(VAE, self).__init__()

		# Initialize parameters
		self.p_layers = p_layers
		self.q_layers = q_layers

		self.input_dims = input_dims
		self.latent_dims = latent_dims

		self.DKL_weight = DKL_weight
		self.sigma_clip = sigma_clip

		self.built = False
		self.name = name


	def build_encoder(self, input_var):
		# Build the encoder
		if len(self.q_layers) > 0:
			self._encoder = Sequential('vae_encoder')
			self._encoder += FullyConnected(self.input_dims, self.q_layers[0], coder_act_fn, name='fc_1')
			for i in xrange(1, len(self.q_layers)):
				self._encoder += FullyConnected(self.q_layers[i-1], self.q_layers[i], coder_act_fn, name='fc_%d'%(i+1))

			self.encoder = self._encoder(input_var)

			self.enc_mean = FullyConnected(self.q_layers[-1], self.latent_dims, mean_std_act_fn, name='enc_mean')
			self.enc_mean = self.enc_mean(self.encoder)

		else:
			self.encoder = input_var

			self.enc_mean = FullyConnected(self.input_dims, self.latent_dims, mean_std_act_fn, name='enc_mean')
			self.enc_mean = self.enc_mean(self.encoder)


	def build_decoder(self, input_var):
		# Build the decoder
		if len(self.p_layers) > 0:
			self._decoder = Sequential('vae_decoder')
			self._decoder += FullyConnected(self.latent_dims, self.p_layers[0], coder_act_fn, name='fc_1')
			for i in xrange(1, len(self.p_layers)):
				self._decoder += FullyConnected(self.p_layers[i-1], self.p_layers[i], coder_act_fn, name='fc_%d'%(i+1))

			self.decoder = self._decoder(input_var)

			self._dec_mean = FullyConnected(self.p_layers[-1], self.input_dims, dec_mean_act_fn, name='dec_mean')
			self.dec_mean = self._dec_mean(self.decoder)

		else:
			self.decoder = input_var

			self._dec_mean = FullyConnected(self.latent_dims, self.input_dims, dec_mean_act_fn, name='dec_mean')
			self.dec_mean = self._dec_mean(self.decoder)


	def __call__(self, input_batch):
		self.n_samples = input_batch.get_shape()[0].value
		print(self.latent_dims)

		# Build the encoder and decoder ---------------------------------------
		self.build_encoder(input_batch)

		enc_std = tf.exp(
			tf.mul(
				0.5,
				self.enc_log_std_sq
			)
		)

		self.build_decoder(
			tf.add(
				tf.mul(
					tf.random_normal(
						[self.n_samples, self.latent_dims]
					),
					enc_std
				),
				self.enc_mean
			)
		)

		self.built = True

		# -------------------------- KL part of loss --------------------------
		enc_std_sq = tf.exp(
			self.enc_log_std_sq
		)

		enc_mean_sq = tf.square(
			self.enc_mean
		)

		self.DKL = tf.mul(
			0.5,
			tf.sub(
				tf.add(
					1.0,
					self.enc_log_std_sq
				),
				tf.add(
					enc_std_sq,
					enc_mean_sq
				)
			)
		)

		self.DKL = tf.reduce_sum(
			self.DKL,
			1
		)

		# -------------------------- Reconstruction Loss --------------------------
		inv_dec_std_sq = tf.exp( -self.dec_log_std_sq )

		self.rec_loss = tf.mul(
			-0.5,
			tf.mul(
				inv_dec_std_sq,
				tf.square(
					tf.sub(
						input_batch,
						self.dec_mean
					)
				)
			)
		)

		self.rec_loss = tf.reduce_sum(
			self.rec_loss,
			1
		)

		self.rec_loss = tf.add(
			self.rec_loss,
			tf.mul(
				-0.5,
				tf.reduce_sum(
					self.dec_log_std_sq,
					1
				)
			)
		)

		# -------------------------- Put the Two Parts of the Loss Together --------------------------
		print(self.DKL.get_shape())
		print(self.rec_loss.get_shape())
		self.cost = tf.mul(
			-1.0 / float(self.n_samples),
			tf.reduce_sum(
				tf.add(
					self.DKL_weight * self.DKL,
					self.rec_loss
				)
			)
		)

		self.decay = 0.0
		# for layer in (self._encoder.layers + self._decoder.layers):
		# 	self.decay += tf.reduce_sum(
		# 		tf.square(
		# 			tf.pow(
		# 				layer.weights['w'],
		# 				2
		# 			)
		# 		)
		# 	)

		self.decay_weight = 0.0
		return self.cost + self.decay*self.decay_weight

	def build_sampler(self, input_var):
		assert self.built, 'The encoder and the decoder have not been built yet!'

		temp = self._decoder(input_var)

		self.sampler_mean = self._dec_mean(temp)
		self.sampler_log_std_sq = tf.clip_by_value(
			self._dec_log_std_sq(temp),
			-self.sigma_clip,
			self.sigma_clip
		)

		sampler_std = tf.exp(
			tf.mul(
				0.5,
				self.sampler_log_std_sq
			)
		)

		self.sampler = tf.add(
			tf.mul(
				tf.random_normal(
					[self.n_samples, self.input_dims]
				),
				sampler_std
			),
			self.sampler_mean
		)

		return self.sampler