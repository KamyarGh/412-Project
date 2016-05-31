"""
	Author: Lluis Castre
    fully_connected.py
    ~~~~~~~~~~~~~~~~~~

    Fully connected layer.
"""
# Imports ----------------------------------------------------------------------
import numpy as np
import tensorflow as tf
# ------------------------------------------------------------------------------


def uniform_weights(input_dim, output_dim, scale=0.01, name=None):
        """
        Uniform/Orthogonal initialization [-scale, scale].
        """
        if input_dim == output_dim:
            W = np.random.randn(input_dim, output_dim)
            u, s, v = np.linalg.svd(W)
            u = u.astype('float32')
            return tf.Variable(
                u,
                name=name,
            )

        else:
            return tf.Variable(
                np.random.uniform(
                    low=-scale,
                    high=scale,
                    size=(input_dim, output_dim)
                ).astype('float32'),
                name=name,
            )


def zeros(dim, name=None):
    """
    1D tensor of zeros.
    """
    return tf.Variable(
        np.zeros(
            (dim),
            dtype='float32'
        ),
        name=name,
    )


class FullyConnected(object):
    """
    Fully connected layer.
    """
    def __init__(
            self,
            input_dim,
            output_dim,
            activation=None,
            init_weights=None,
            scale=0.01,
            name=''
    ):
        """
        Initialization.
        """
        self.name = name
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.activation = activation
        self.weights = {}
        self.scale = scale
        self.build(init_weights)

    def build(self, init_weights):
        """
        Initialize weights.
        """
        if init_weights:
            self.weights['w'] = tf.Variable(
                init_weights['w'],
                name='{}_w'.format(self.name)
            )
            self.weights['b'] = tf.Variable(
                init_weights['b'],
                name='{}_b'.format(self.name)
            )
        else:
            self.weights['w'] = uniform_weights(
                self.input_dim,
                self.output_dim,
                scale = self.scale,
                name='{}_w'.format(self.name)
            )
            self.weights['b'] = zeros(
                self.output_dim,
                name='{}_b'.format(self.name)
            )

    def re_init_weights(self, sess):
        sess.run(
            self.weights['w'].assign(
                np.random.uniform(
                    low=-self.scale,
                    high=self.scale,
                    size=(self.input_dim, self.output_dim)
                ).astype('float32')
            )
        )
        sess.run(
            self.weights['b'].assign(
                np.zeros(
                    (self.output_dim),
                    dtype='float32'
                )
            )
        )

    def __call__(self, x):
        """
        Forward pass.
        """
        pre_act = tf.matmul(x, self.weights['w']) + self.weights['b']

        if self.activation is None or self.activation == 'linear':
            return pre_act

        return self.activation(pre_act)





class ConstFC(object):
    """
    Fully connected layer.
    """
    def __init__(
            self,
            W,
            b,
            activation=None,
            name=''
    ):
        """
        Initialization.
        """
        self.name = name
        self.activation = activation
        self.weights = {}

        self.weights['w'] = tf.constant(W)
        self.weights['b'] = tf.constant(b)

    def __call__(self, x):
        """
        Forward pass.
        """
        pre_act = tf.matmul(x, self.weights['w']) + self.weights['b']

        if self.activation is None or self.activation == 'linear':
            return pre_act

        return self.activation(pre_act)

















# import numpy as np
# import tensorflow as tf
# from numpy.random import uniform
# import toolbox

# class FullyConnected(object):
# 	"""docstring for FullyConnected"""
# 	def __init__(self, input_dim, output_dim, activation=None, name=''):
# 		super(FullyConnected, self).__init__()

# 		self.input_dim = input_dim
# 		self.output_dim = output_dim
# 		self.activation = activation
# 		self.act_fn = toolbox.name2fn(activation)
# 		self.name = name
# 		self.weights = {}
# 		scale = 1.0 / self.input_dim

# 		# Initialize the weight matrices
# 		self.weights['W'] = tf.Variable(
# 			uniform(low=-scale, high=scale, size=(input_dim, output_dim)).astype(np.float32),
# 			name = '%s_W' % self.name
# 		)

# 		self.weights['b'] = tf.Variable(
# 			(np.zeros(output_dim) + (scale if activation=='relu' else 0)).astype(np.float32),
# 			name = '%s_b' % self.name
# 		)


# 	def __call__(self, x):
# 		mul = tf.matmul(x, self.weights['W']) + self.weights['b']

# 		if self.activation not in ['linear', None]:
# 			return mul

# 		return self.act_fn(mul)