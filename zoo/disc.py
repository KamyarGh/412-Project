import numpy as np
import tensorflow as tf
from containers.sequential import Sequential
from layers.conv import ConvLayer
from layers.fc_layer import FullyConnected, ConstFC
from layers.pool import PoolLayer

class FixedConvDisc(object):
    """docstring for FixedConvDisc"""
    def __init__(self, feat_params, num_feat_layers, name=''):
        super(FixedConvDisc, self).__init__()

        self.feat_params = feat_params
        self.num_feat_layers = num_feat_layers
        self.name = name

    def __call__(self, disc_input):
        feat_params = self.feat_params

        self._disc = Sequential('Fixed_Conv_Disc')
        conv_count, pool_count, fc_count = 0, 0, 0
        for i in xrange(self.num_feat_layers):
            if feat_params[i]['layer_type'] == 'conv':
                self._disc += ConvLayer(
                    feat_params[i]['n_filters_in'],
                    feat_params[i]['n_filters_out'],
                    feat_params[i]['input_dim'],
                    feat_params[i]['filter_dim'],
                    feat_params[i]['strides'],
                    name='classifier_conv_%d' % conv_count
                )
                self._disc.layers[-1].weights['W'] = tf.constant(feat_params[i]['W'])
                self._disc.layers[-1].weights['b'] = tf.constant(feat_params[i]['b'])
                self._disc += feat_params[i]['act_fn']
                conv_count += 1
            elif feat_params[i]['layer_type'] == 'pool':
                self._disc += PoolLayer(
                    feat_params[i]['input_dim'],
                    feat_params[i]['filter_dim'],
                    feat_params[i]['strides'],
                    name='classifier_pool_%d' % i
                )
                pool_count += 1
            elif feat_params[i]['layer_type'] == 'fc':
                # self._disc += FullyConnected(
                #     feat_params[i]['W'].shape[0],
                #     feat_params[i]['W'].shape[1],
                #     activation=tf.nn.tanh,
                #     scale=0.01,
                #     name='classifier_fc_%d' % fc_count
                # )
                self._disc += ConstFC(
                    feat_params[i]['W'],
                    feat_params[i]['b'],
                    activation=feat_params[i]['act_fn'],
                    name='classifier_fc_%d' % fc_count
                )
                fc_count += 1

        if isinstance(self._disc.layers[-1], ConstFC):
            disc_input_dim = self._disc.layers[-1].weights['w'].get_shape()[1].value
        elif isinstance(self._disc.layers[-1], PoolLayer):
            disc_input_dim = np.prod(self._disc.layers[-1].output_dim) * (self._disc.layers[-3].n_filters_out)
        else: # function after conv layer
            disc_input_dim = np.prod(self._disc.layers[-1].output_dim) * (self._disc.layers[-2].n_filters_out)

       # self._disc += FullyConnected(disc_input_dim, 1024, activation=tf.nn.tanh, scale=0.01, name='disc_fc_0')
        self._disc += FullyConnected(disc_input_dim, 1, activation=None, scale=0.01, name='disc_logit')
        self._disc += lambda p: 1.0 / (1.0 + tf.exp(-p))

        self.disc = self._disc(disc_input)

        return self.disc