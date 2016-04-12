import numpy as np
import tensorflow as tf
from sequential import Sequential
from fc_layer import FullyConnected

class VAE(object):
	""" << docstring for VAE >>
	Variational Auto-Encoder with fully connected layers
	"""
	# 	- p_layers:		a list of sizes of each fc layer for the decoder network
	# 	- q_layers:		a list of sizes of each fc layer for the encoder network
	# 	- input_dims:	input data dimensions
	#	- latent_dims:	dimension of latent representation
	def __init__(self, p_layers, q_layers, input_dims, latent_dims, name):
		super(VAE, self).__init__()

		# Initialize parameters
		self.p_layers = p_layers
		self.q_layers = q_layers

		self.input_dims = input_dims
		self.latent_dims = latent_dims
		
		# assert self.q_layers[-1] == self.latent_dims,
		# 		'Encoder output does not match latent dimensions!'
		# assert self.p_layers[-1] == self.input_dims,
		# 		'Decoder output does not match input dimensions!'

		self.built = False
		self.name = name


	def build_encoder(self, input_var):
		# Build the encoder
		self._encoder = Sequential('vae_encoder')
		self._encoder += FullyConnected(self.input_dims, self.q_layers[0], 'relu', name='fc_1')
		for i in xrange(1, len(self.q_layers)):
			self._encoder += FullyConnected(self.q_layers[i-1], self.q_layers[i], 'relu', name='fc_%d'%(i+1))

		self.encoder = self._encoder(input_var)

		self.enc_mean = FullyConnected(self.q_layers[-1], self.latent_dims, 'relu', name='enc_mean')
		self.enc_mean = self.enc_mean(self.encoder)
		self.enc_std = FullyConnected(self.q_layers[-1], self.latent_dims, 'relu', name='enc_std')
		self.enc_std = self.enc_std(self.encoder)


	def build_decoder(self, input_var):
		# Build the decoder
		self._decoder = Sequential('vae_decoder')
		self._decoder += FullyConnected(self.latent_dims, self.p_layers[0], 'relu', name='fc_1')
		for i in xrange(1, len(self.p_layers)):
			self._decoder += FullyConnected(self.p_layers[i-1], self.p_layers[i], 'relu', name='fc_%d'%(i+1))

		self.decoder = self._decoder(input_var)

		self.dec_mean = FullyConnected(self.p_layers[-1], self.input_dims, 'relu', name='dec_mean')
		self.dec_mean = self.dec_mean(self.decoder)
		self.dec_std = FullyConnected(self.p_layers[-1], self.input_dims, 'relu', name='dec_std')
		self.dec_std = self.dec_std(self.decoder)


	def __call__(self, input_batch):
		self.n_samples = input_batch.get_shape()[0].value
		self.build_encoder(input_batch)
		self.build_decoder(
			tf.add(
				tf.mul(
					tf.constant(
						tf.random_normal(
							[self.n_samples, self.latent_dims]
						)
					),
					self.enc_std
				),
				self.enc_mean
			)
		)
		self.built = True

		# -------------------------- KL part of loss --------------------------
		# Square
		enc_std_sq = tf.square(self.enc_std)
		log_enc_std_sq = tf.log(enc_std_sq)
		enc_mean_sq = tf.square(self.enc_mean)

		# Reduce
		enc_std_sq = tf.reduce_sum(enc_std_sq, 1)
		enc_mean_sq = tf.reduce_sum(enc_mean_sq, 1)
		log_plus_one = tf.reduce_sum(
			tf.add(
				tf.float32(1),
				log_enc_std_sq
			),
			1
		)

		# DKL
		DKL = tf.sub(
			log_plus_one,
			tf.add(
				enc_mean_sq,
				log_enc_std_sq
			)
		)

		self.DKL = tf.mul(
			0.5,
			DKL
		)

		# -------------------------- Reconstruction Loss --------------------------
		self.rec_loss = tf.square(
			tf.sub(
				enc_input,
				self.dec_mean
			)
		)

		self.rec_loss = tf.mul(
			self.rec_loss,
			tf.inv(self.dec_std)
		)

		self.rec_loss = tf.reduce_sum(
			self.rec_loss,
			1
		)

		# -------------------------- Put the Two Parts of the Loss Together --------------------------
		self.cost =  tf.mul(
			-1,
			tf.reduce_mean(
				tf.add(
					self.DKL,
					self.rec_loss
				)
			)
		)

		return self.cost


	def build_sampler(self, input_var):
		assert self.built,
			'The encoder and the decoder have not been built yet!'

		temp = self._decoder(input_var)
		self.sampler_mean = FullyConnected(self.p_layers[-1], self.input_dims, 'relu', name='samp_dec_mean')
		self.sampler_mean = self.dec_mean(temp)
		self.sampler_std = FullyConnected(self.p_layers[-1], self.input_dims, 'relu', name='samp_dec_std')
		self.sampler_std = self.dec_std(temp)

		self.sampler = tf.add(
			tf.mul(
				tf.constant(
					tf.random_normal(
						[self.n_samples, self.input_dims]
					)
				),
				self.sampler_std
			),
			self.sampler_mean
		)

		return self.sampler