import numpy as np
import tensorflow as tf

class AdversarialContainer(object):
	"""docstring for AdversarialContainer"""
	def __init__(self, feature_extractor, generator, discriminator, disc_trainable, gen_scope, disc_scope):
		super(AdversarialContainer, self).__init__()

		self.feature_extractor = feature_extractor
		self._generator = generator
		self._discriminator = discriminator

		self.disc_trainable = disc_trainable

		self.gen_scope = gen_scope
		self.disc_scope = disc_scope

	def __call__(self, gen_input, real_input, optimizer):
		assert gen_input.get_shape().value[0] == real_input.get_shape().value[0], \
			'Generated and Real data must have same batch size!'

		self.generator = self._generator(gen_input)
		self.gen_feats = self.feature_extractor(self.generator)

		self.real_feats = self.feature_extractor(self.real_input)

		self.discriminator = self._discriminator(
			tf.concat(
				0,
				[
					self.gen_feats,
					self.real_feats
				]
			),
			tf.concat(
				0,
				[
					np.zeros(gen_input.get_shape().value[0]),
					np.ones(real_input.get_shape().value[0])
				]
			)
		)

		self.gen_train_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.gen_scope)
		self.disc_train_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.disc_scope)

		self.gen_optimizer = optimizer.minimize(-self.discriminator, var_list=self.gen_train_vars)
		self.disc_optimizer = optimizer.minimize(self.discriminator, var_list=self.disc_train_vars)

		return self.gen_optimizer, self.disc_optimizer