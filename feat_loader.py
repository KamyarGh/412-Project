"""
Code modified from Lluis Castrjon's Poly-RNN train.py
"""

import numpy as np
import tensorflow as tf
from data_provider import DataProvider
import toolbox
from toolbox import cupboard
import config
import argparse
import utils
import os
from evaluate import save_samples, save_dash_samples, save_ae_samples
from numpy.random import normal

def train(options):
    # Initialize model ------------------------------------------------------------------
    
    # input_shape, input_channels, enc_params, dec_params, name=''
    with tf.device('/gpu:0'):
        if options['model'] == 'cnn_ae':
            model = cupboard(options['model'])(
                options['img_shape'],
                options['input_channels'],
                options['enc_params'],
                options['dec_params'],
                'cnn_ae'
            )

            # Define inputs
            model_clean_input_batch = tf.placeholder(
                tf.float32,
                shape = [options['batch_size']] + options['img_shape'] + [options['input_channels']],
                name = 'clean'
            )
            model_noisy_input_batch = tf.placeholder(
                tf.float32,
                shape = [options['batch_size']] + options['img_shape'] + [options['input_channels']],
                name = 'noisy'
            )
            log.info('Inputs defined')

        else:
            model = cupboard(options['model'])(
                np.prod(options['img_shape']) * options['input_channels'],
                options['enc_params'],
                options['dec_params'],
                'ae'
            )

            # Define inputs
            model_clean_input_batch = tf.placeholder(
                tf.float32,
                shape = [options['batch_size']] + [np.prod(options['img_shape']) * options['input_channels']],
                name = 'clean'
            )
            model_noisy_input_batch = tf.placeholder(
                tf.float32,
                shape = [options['batch_size']] + [np.prod(options['img_shape']) * options['input_channels']],
                name = 'noisy'
            )
            log.info('Inputs defined')

        log.info('Model initialized')

        # Define forward pass
        print(model_clean_input_batch.get_shape())
        print(model_noisy_input_batch.get_shape())
        cost_function = model(model_clean_input_batch, model_noisy_input_batch)
        log.info('Forward pass graph built')

        log.info('Sampler graph built')

        # Define optimizer
        optimizer = tf.train.AdamOptimizer(
            learning_rate=options['lr']
        )
        # optimizer = tf.train.GradientDescentOptimizer(learning_rate=options['lr'])
        train_step = optimizer.minimize(cost_function)
        log.info('Optimizer graph built')

        # # Get gradients
        # grad = optimizer.compute_gradients(cost_function)

        # # Clip gradients
        # clipped_grad = tf.clip_by_norm(grad, 5.0, name='grad_clipping')

        # # Update op
        # backpass = optimizer.apply_gradients(clipped_grad)

        # Define init operation
        init_op = tf.initialize_all_variables()
        log.info('Variable initialization graph built')

    # Define op to save and restore variables
    saver = tf.train.Saver()
    log.info('Save operation built')
    # --------------------------------------------------------------------------

    # Train loop ---------------------------------------------------------------
    with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess:
        log.info('Session started')

        # Initialize shared variables or restore
        if options['reload']:
            saver.restore(sess, os.path.join(options['model_dir'], 'model.ckpt'))
            log.info('Shared variables restored')
        else:
            sess.run(init_op)
            log.info('Shared variables initialized')