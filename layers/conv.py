"""
    conv.py
    ~~~~~~~
    Original Author: Lluis Castrejon
    Modified By: Kamyar Ghasemipour

    Convolutional layer.
"""
# Imports ----------------------------------------------------------------------
import numpy as np
import tensorflow as tf
# ------------------------------------------------------------------------------


def truncated_normal(shape, name=''):
    """
    Truncated normal initialization.
    """
    return tf.Variable(
        tf.truncated_normal(
            shape,
            stddev=0.1,
        ),
        name=name
    )


class ConvLayer(object):
    """
    Convolutional layer.
    """
    def __init__(self, n_filters_in, n_filters_out,
                 input_dim, filter_dim, strides,
                 padding=[0, 0], name='', fixed=False, fixed_value=0):
        """
        Initialization.
        """
        self.n_filters_in = n_filters_in
        self.n_filters_out = n_filters_out
        self.input_dim = input_dim
        self.filter_dim = filter_dim
        self.strides = strides
        self.padding = padding
        self.name = name
        self.weights = {}
        self.fixed = fixed
        self.fixed_value = fixed_value

        # Check input parameters
        assert len(input_dim) == 2, 'Wrong input size {}'.format(input_dim)
        assert len(filter_dim) == 2, 'Wrong filter size {}'.format(filter_dim)
        assert len(strides) == 2, 'Wrong strides {}'.format(strides)
        assert len(padding) == 2, 'Wrong padding {}'.format(padding)

        # Compute output size
        output_0 = int(np.floor((input_dim[0] + 2*padding[0])/float(strides[0]))) + 1
        output_1 = int(np.floor((input_dim[1] + 2*padding[1])/float(strides[1]))) + 1
        # output_0 = int(np.floor((input_dim[0] + 2*padding[0] - filter_dim[0])/float(strides[0]))) + 1
        # output_1 = int(np.floor((input_dim[1] + 2*padding[1] - filter_dim[1])/float(strides[1]))) + 1
        self.output_dim = [output_0, output_1]

        self.build()

    def build(self):
        """
        Create weight variables.
        """
        if not self.fixed:
            self.weights['W'] = truncated_normal(
                [
                    self.filter_dim[0],
                    self.filter_dim[1],
                    self.n_filters_in,
                    self.n_filters_out
                ],
                name='{}_W'.format(self.name)
            )

            self.weights['b'] = truncated_normal(
                [self.n_filters_out],
                name= '{}_b'.format(self.name)
            )
        else:
            print('\n\n\n\n\n\n\n\n\n\n\n\n\nDOING FIXED CONV %f\n\n\n\n\n\n\n\n\n\n\n' % self.fixed_value)
            assert self.n_filters_in == self.n_filters_out

            cur_weight = np.zeros((
                self.filter_dim[0],
                self.filter_dim[1],
                self.n_filters_in,
                self.n_filters_out
            )).astype(np.float32)

            for i in xrange(self.n_filters_out):
                cur_weight[:,:,i,i] = self.fixed_value

            self.weights['W'] = tf.constant(
                cur_weight,
                dtype=tf.float32,
                name='{}_W'.format(self.name)
            )

            self.weights['b'] = tf.zeros(
                shape=[self.n_filters_out],
                name= '{}_b'.format(self.name)
            )

    def re_init_weights(self, sess):
        sess.run(
            self.weights['W'].assign(
                tf.truncated_normal(
                    [
                        self.filter_dim[0],
                        self.filter_dim[1],
                        self.n_filters_in,
                        self.n_filters_out
                    ],
                    stddev=0.1,
                )
            )
        )
        sess.run(
            self.weights['b'].assign(
                tf.truncated_normal(
                    [self.n_filters_out],
                    stddev=0.1,
                )
            )
        )

    def __call__(self, x):
        """
        Forward pass
        """
        # Pad image
        x_pad = tf.pad(x, [[0, 0], [self.padding[0], self.padding[0]], [self.padding[1], self.padding[1]], [0, 0]])

        # Apply convolution + ReLU
        return tf.nn.conv2d(
            x_pad,
            self.weights['W'],
            strides=[1, self.strides[0], self.strides[1], 1],
            padding='SAME'
        ) + self.weights['b']

# padding='SAME' if self.fixed else 'VALID'