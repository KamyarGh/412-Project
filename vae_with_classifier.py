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
from toolbox import get_providers
from numpy.random import multivariate_normal as MVN, uniform
from layers.fc_layer import FullyConnected as FC
import pickle

def train(options):
    # Get logger
    log = utils.get_logger(os.path.join(options['model_dir'], 'log.txt'))
    options_file = open(os.path.join(options['dashboard_dir'], 'options'), 'w')
    options_file.write(options['description'] + '\n')
    options_file.write(
        'DKL Weight: {}\nLog Sigma^2 clipped to: [{}, {}]\n\n'.format(
            options['DKL_weight'],
            -options['sigma_clip'],
            options['sigma_clip']
        )
    )
    for optn in options:
        options_file.write(optn)
        options_file.write(':\t')
        options_file.write(str(options[optn]))
        options_file.write('\n')
    options_file.close()

    # Dashboard Catalog
    catalog = open(os.path.join(options['dashboard_dir'], 'catalog'), 'w')
    catalog.write(
"""filename,type,name
options,plain,Options
train_loss.csv,csv,Train Loss
val_loss.csv,csv,Validation Loss
"""
    )
    catalog.flush()
    train_log = open(os.path.join(options['dashboard_dir'], 'train_loss.csv'), 'w')
    val_log = open(os.path.join(options['dashboard_dir'], 'val_loss.csv'), 'w')

    train_log.write('step,time,Train Loss\n')
    val_log.write('step,time,Validation Loss\n')

    # Print options
    utils.print_options(options, log)

    # Load dataset ----------------------------------------------------------------------
    # Train provider
    train_provider, val_provider, test_provider = get_providers(options, log, flat=True)

    # Initialize model ------------------------------------------------------------------
    with tf.device('/gpu:0'):
        model = cupboard(options['model'])(
            options['p_layers'],
            options['q_layers'],
            np.prod(options['img_shape']),
            options['latent_dims'],
            options['DKL_weight'],
            options['sigma_clip'],
            'vanilla_vae'
        )
        log.info('Model initialized')

        # Define inputs
        model_input_batch = tf.placeholder(
            tf.float32,
            shape = [options['batch_size'], np.prod(np.array(options['img_shape']))],
            name = 'enc_inputs'
        )
        model_label_batch = tf.placeholder(
            tf.float32,
            shape = [options['batch_size'], options['num_classes']],
            name = 'labels'
        )
        log.info('Inputs defined')

        # Load VAE
        model(model_input_batch)

        feat_params = pickle.load(open(options['feat_params_path'], 'rb'))

        for i in range(len(model._encoder.layers)):
            model._encoder.layers[i].weights['w'] = tf.constant(feat_params[i]['W'])
            model._encoder.layers[i].weights['b'] = tf.constant(feat_params[i]['b'])

        model._enc_mean.weights['w'] = tf.constant(feat_params[-2]['W'])
        model._enc_mean.weights['b'] = tf.constant(feat_params[-2]['b'])

        model._enc_log_std_sq.weights['w'] = tf.constant(feat_params[-1]['W'])
        model._enc_log_std_sq.weights['b'] = tf.constant(feat_params[-1]['b'])

        enc_std = tf.exp(
            tf.mul(
                0.5,
                model.enc_log_std_sq
            )
        )

        classifier = FC(
            model.latent_dims,
            options['num_classes'],
            activation=None,
            scale=0.01,
            name='classifier_fc'
        )(
            tf.add(
                tf.mul(
                    tf.random_normal(
                        [model.n_samples, model.latent_dims]
                    ),
                    enc_std
                ),
                model.enc_mean
            )
        )

        classifier = tf.nn.softmax(classifier)
        cost_function = -tf.mul(model_label_batch, tf.log(classifier))
        cost_function = tf.reduce_sum(cost_function)
        cost_function *= 1 / float(options['batch_size'])

        log.info('Forward pass graph built')

        # Define optimizer
        optimizer = tf.train.AdamOptimizer(
            learning_rate=options['lr']
        )
        # optimizer = tf.train.GradientDescentOptimizer(learning_rate=options['lr'])
        
        # train_step = optimizer.minimize(cost_function)

        # Get gradients
        grads = optimizer.compute_gradients(cost_function)
        grads = [gv for gv in grads if gv[0] != None]
        grad_tensors = [gv[0] for gv in grads]

        # Clip gradients
        clip_grads = [(tf.clip_by_norm(gv[0], 5.0, name='grad_clipping'), gv[1]) for gv in grads]

        # Update op
        backpass = optimizer.apply_gradients(clip_grads)

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
        last_accs = np.zeros((10))

        batch_abs_idx = 0
        for epoch_idx in xrange(options['n_epochs']):
            batch_rel_idx = 0
            log.info('Epoch {}'.format(epoch_idx + 1))

            for inputs, labels in train_provider:

                batch_abs_idx += 1
                batch_rel_idx += 1

                result = sess.run(
                    # (cost_function, train_step, model.enc_std, model.enc_mean, model.encoder, model.dec_std, model.dec_mean, model.decoder, model.rec_loss, model.DKL),
                    #       0           1          2           3               4                     5                       6              7               8            9           10
                    [cost_function, backpass, classifier] + [gv[0] for gv in grads],
                    feed_dict = {
                        model_input_batch: inputs,
                        model_label_batch: labels
                    }
                )

                cost = result[0]

                if batch_abs_idx % 10 == 0:
                    train_log.write('{},{},{}\n'.format(batch_abs_idx, '2016-04-22', np.mean(last_losses)))
                    train_log.flush()

                # Check cost
                if np.isnan(cost) or np.isinf(cost):
                    log.info('NaN detected')
                    for i in range(len(result)):
                        print("\n\nresult[%d]:" % i)
                        try:
                            print(np.any(np.isnan(result[i])))
                        except:
                            pass
                        print(result[i])
                    print(result[3].shape)
                    print(model._encoder.layers[0].weights['w'].eval())
                    print('\n\nAny:')
                    print(np.any(np.isnan(result[8])))
                    print(np.any(np.isnan(result[9])))
                    print(np.any(np.isnan(result[10])))
                    print(inputs)
                    return 1., 1., 1.

                # Update last losses
                last_losses = np.roll(last_losses, 1)
                last_losses[0] = cost

                last_accs = np.roll(last_accs, 1)
                last_accs[0] = np.mean(np.argmax(labels, axis=1) == np.argmax(result[2], axis=1))

                # Display training information
                if np.mod(epoch_idx, options['freq_logging']) == 0:
                    log.info('Epoch {:02}/{:02} Batch {:03} Current Acc.: {:0>15.4f} Mean last accs: {:0>15.4f}'.format(
                        epoch_idx + 1,
                        options['n_epochs'],
                        batch_abs_idx,
                        last_accs[0],
                        np.mean(last_accs)
                    ))
                    log.info('Batch Mean Loss: {:0>15.4f}'.format(np.mean(last_losses)))

                # Save model
                if np.mod(batch_abs_idx, options['freq_saving']) == 0:
                    saver.save(sess, os.path.join(options['model_dir'], 'model_at_%d.ckpt' % batch_abs_idx))
                    log.info('Model saved')

                # Validate model
                if np.mod(batch_abs_idx, options['freq_validation']) == 0:

                    valid_costs = []
                    seen_batches = 0
                    for val_batch, labels in val_provider:

                        val_result = sess.run(
                            [cost_function, classifier],
                            feed_dict = {
                                model_input_batch: val_batch,
                                model_label_batch: labels
                            }
                        )
                        val_cost = np.mean(np.argmax(labels, axis=1) == np.argmax(val_result[1], axis=1))
                        valid_costs.append(val_cost)
                        seen_batches += 1

                        if seen_batches == options['valid_batches']:
                            break

                    # Print results
                    log.info('Validation acc.: {:0>15.4f}'.format(
                        float(np.mean(valid_costs))
                    ))

                    val_log.write('{},{},{}\n'.format(batch_abs_idx, '2016-04-22', np.mean(valid_costs)))
                    val_log.flush()

            log.info('End of epoch {}'.format(epoch_idx + 1))
    # --------------------------------------------------------------------------
        test_results = []

        for inputs, labels in test_provider:
            if isinstance(inputs, tuple):
                inputs = inputs[0]
            batch_results = sess.run(
                [cost_function, classifier],
                feed_dict = {
                    model_input_batch: inputs,
                    model_label_batch: labels
                }
            )

            test_results.append(np.mean(np.argmax(labels, axis=1) == np.argmax(batch_results[1], axis=1)))

        # Print results
        log.info('Test Accuracy: {:0>15.4f}'.format(
            np.mean(test_results)
        ))


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