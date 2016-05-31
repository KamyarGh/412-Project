"""
    pool.py
    ~~~~~~~

    Max-pooling operation.
"""
# Imports ----------------------------------------------------------------------
import numpy as np
import tensorflow as tf
# ------------------------------------------------------------------------------


class PoolLayer(object):
    """
    Max pooling layer.
    """
    def __init__(self, input_dim, filter_dim, strides, name=''):
        """
        Initialization.
        """
        self.input_dim = input_dim
        self.filter_dim = filter_dim
        self.strides = strides
        self.name = name

        # Check input parameters
        assert len(input_dim) == 2, 'Wrong input size {}'.format(input_dim)
        assert len(filter_dim) == 2, 'Wrong filter size {}'.format(filter_dim)
        assert len(strides) == 2, 'Wrong strides {}'.format(strides)

        # Compute output size
        output_0 = int(np.floor((input_dim[0] - filter_dim[0])/float(strides[0]))) + 1
        output_1 = int(np.floor((input_dim[1] - filter_dim[1])/float(strides[1]))) + 1
        self.output_dim = [output_0, output_1]

        self.build()

    def build(self):
        pass

    def __call__(self, x):
        return tf.nn.max_pool(
            x,
            ksize=[1, self.filter_dim[0], self.filter_dim[1], 1],
            strides=[1, self.strides[0], self.strides[1], 1],
            padding='VALID',
            name=self.name
        )
