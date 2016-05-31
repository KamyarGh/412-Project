import numpy as np
import tensorflow as tf

class FeatVAE(object):
	"""docstring for FeatVAE"""
	def __init__(self, vae, feat_model, feat_layer_inds, DKL_weight=1.0, vae_rec_loss_weight=0.05, img_shape=None, input_channels=None, flat=True, name=''):
		super(FeatVAE, self).__init__()

		self.vae = vae
		self.feat_model = feat_model
		self.feat_layer_inds = feat_layer_inds
		self.DKL_weight = DKL_weight
		self.vae_rec_loss_weight = vae_rec_loss_weight

		self.img_shape = img_shape
		self.input_channels = input_channels
		self.flat = flat


	def __call__(self, vae_input):
		self.n_samples = vae_input.get_shape()[0].value
		self.input_dims = vae_input.get_shape()[1].value

		self.vae(vae_input)

		recon_mean = self.vae.dec_mean
		recon_std = tf.exp(
			tf.mul(
				0.5,
				self.vae.dec_log_std_sq
			)
		)
		self.vae_recons = tf.add(
			tf.mul(
				tf.random_normal(
					[self.n_samples, self.input_dims]
				),
				recon_std
			),
			recon_mean
		)

		reshaped_recons = tf.reshape(
			self.vae_recons,
			[self.n_samples] + self.img_shape + [self.input_channels]
		)
		print('#'*100)
		print(self.feat_layer_inds)
		self.vae_feats = map(lambda ind: self.feat_model.build_layer(reshaped_recons, ind), self.feat_layer_inds)

		reshaped_input = tf.reshape(
			vae_input,
			[self.n_samples] + self.img_shape + [self.input_channels]
		)
		self.real_feats = map(lambda ind: self.feat_model.build_layer(reshaped_input, ind), self.feat_layer_inds)

		self.rec_cost = 0.5 * reduce(
			tf.add,
			map(
				lambda p: tf.reduce_sum(
					tf.square(
						tf.sub(p[1], p[0])
					)
				),
				zip(self.real_feats, self.vae_feats)
			)
		)

		self.DKL_cost = -self.vae.DKL # In the vae, DKL is actually neg. DKL

		print(self.DKL_cost.get_shape())

		self.cost = tf.reduce_sum(
			tf.mul(
				1 / float(self.n_samples),
				self.rec_cost + (self.DKL_cost * self.DKL_weight) + (-1 * self.vae.rec_loss)*self.vae_rec_loss_weight
			)
		)

		return self.cost

	def build_sampler(self, sampler_input):
		return self.vae.build_sampler(sampler_input)