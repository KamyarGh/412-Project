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
from evaluate import save_samples
from numpy.random import multivariate_normal as MVN, uniform

def train(options):
    # Get logger
    log = utils.get_logger(os.path.join(options['model_dir'], 'log.txt'))

    # Print options
    utils.print_options(options, log)

    # Load dataset ----------------------------------------------------------------------
    # Train provider
    num_data_points = len(
        os.listdir(
            os.path.join(options['data_dir'], 'train', 'info')
        )
    )
    num_data_points -= 2

    train_provider = DataProvider(
    	num_data_points,
    	options['batch_size'],
    	toolbox.CIFARLoader(
            data_dir = os.path.join(options['data_dir'], 'train'),
            flat=False
        )
    )

    # Valid provider
    num_data_points = len(
        os.listdir(
            os.path.join(options['data_dir'], 'valid')
        )
    )
    num_data_points -= 2

    val_provider = DataProvider(
    	num_data_points,
        options['batch_size'],
        toolbox.CIFARLoader(
        	data_dir = os.path.join(options['data_dir'], 'valid'),
        	flat=False
        )
    )
    log.info('Data providers initialized.')


    # Initialize model ------------------------------------------------------------------
    with tf.device('/gpu:0'):
        model = cupboard(options['model'])(
        	options['img_shape'],
        	options['input_channels'],
            options['num_classes'],
            options['conv_params'],
            options['fc_params'],
        	'CIFAR_classifier'
        )
        log.info('Model initialized')

        # Define inputs
        input_batch = tf.placeholder(
        	tf.float32,
        	shape = [options['batch_size']] + options['img_shape'] + [options['input_channels']],
        	name = 'inputs'
        )
        label_batch = tf.placeholder(
            tf.float32,
            shape = [options['batch_size'], options['num_classes']],
            name = 'labels'
        )
        log.info('Inputs defined')

        # Define forward pass
        cost_function, _ = model(input_batch, label_batch)
        log.info('Forward pass graph built')

        # Define optimizer
        optimizer = tf.train.AdamOptimizer(
            learning_rate=options['lr']
        )
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

        # Define last losses to compute a running average
        last_losses = np.zeros((10))

        batch_abs_idx = 0
        for epoch_idx in xrange(options['n_epochs']):
            batch_rel_idx = 0
            log.info('Epoch {}'.format(epoch_idx + 1))

            for inputs, labels in train_provider:
                batch_abs_idx += 1
                batch_rel_idx += 1

                cost = sess.run(
                    cost_function,
                    feed_dict = {
                        input_batch: inputs,
                        label_batch: labels
                    }
                )

                # Check cost
                if np.isnan(cost) or np.isinf(cost):
                    log.info('NaN detected')
                    return 1., 1., 1.

                # Update last losses
                last_losses = np.roll(last_losses, 1)
                last_losses[0] = cost

                # Display training information
                if np.mod(epoch_idx, options['freq_logging']) == 0:
                    log.info('Epoch {:02}/{:02} Batch {:03} Current Loss: {:0>15.4f} Mean last losses: {:0>15.4f}'.format(
                        epoch_idx + 1,
                        options['n_epochs'],
                        batch_abs_idx,
                        float(cost),
                        np.mean(last_losses)
                    ))

                # Save model
                if np.mod(batch_abs_idx, options['freq_saving']) == 0:
                    saver.save(sess, os.path.join(options['model_dir'], 'model.ckpt'))
                    log.info('Model saved')

                # Validate model
                if np.mod(batch_abs_idx, options['freq_validation']) == 0:

                    valid_costs = []
                    seen_batches = 0
                    for val_batch, val_label in val_provider:

                        # Break if 10 batches seen for now
                        if seen_batches == options['valid_batches']:
                            break

                        val_cost = sess.run(
                            cost_function,
                            feed_dict = {
                                input_batch: val_batch,
                                label_batch: val_label
                            }
                        )
                        valid_costs.append(val_cost)
                        seen_batches += 1

                    # Print results
                    log.info('Validation loss: {:0>15.4f}'.format(
                        float(np.mean(valid_costs))
                    ))

            log.info('End of epoch {}'.format(epoch_idx + 1))
    # --------------------------------------------------------------------------


if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='Vanilla VAE')

	parser.add_argument(
		'experiment',
		type = str,
		help = 'Path to config of the experiment'
	)

	args = parser.parse_args()

	options = config.load_config(args.experiment)

	train(options)