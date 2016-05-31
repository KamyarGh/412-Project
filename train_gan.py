"""
Code modified from Lluis Castrjon's Poly-RNN train.py
"""

import numpy as np
import tensorflow as tf
from data_provider import DataProvider
import toolbox
from toolbox import cupboard, visualize, test_LL_and_DKL, get_providers
import config
import argparse
import utils
import os
from evaluate import save_samples, save_dash_samples, save_ae_samples
from numpy.random import multivariate_normal as MVN, uniform
import pickle

from layers.fc_layer import FullyConnected
from containers.sequential import Sequential

def train(options):
    # Get logger
    log = utils.get_logger(os.path.join(options['model_dir'], 'log.txt'))
    options_file = open(os.path.join(options['dashboard_dir'], 'options'), 'w')
    options_file.write(options['description'] + '\n')
    options_file.write(
        'Log Sigma^2 clipped to: [{}, {}]\n\n'.format(
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
train_loss.csv,csv,Discriminator Cross-Entropy
train_acc.csv,csv,Discriminator Accuracy
val_loss.csv,csv,Validation Cross-Entropy
val_acc.csv,csv,Validation Accuracy
"""
    )
    catalog.flush()
    train_log = open(os.path.join(options['dashboard_dir'], 'train_loss.csv'), 'w')
    val_log = open(os.path.join(options['dashboard_dir'], 'val_loss.csv'), 'w')
    train_acc = open(os.path.join(options['dashboard_dir'], 'train_acc.csv'), 'w')
    val_acc = open(os.path.join(options['dashboard_dir'], 'val_acc.csv'), 'w')


    train_log.write('step,time,Train CE (Training Vanilla),Train CE (Training Gen.),Train CE (Training Disc.)\n')
    val_log.write('step,time,Validation CE (Training Vanilla),Validation CE (Training Gen.),Validation CE (Training Disc.)\n')
    train_acc.write('step,time,Train CE (Training Vanilla),Train CE (Training Gen.),Train CE (Training Disc.)\n')
    val_acc.write('step,time,Validation CE (Training Vanilla),Validation CE (Training Gen.),Validation Acc. (Training Disc.)\n')

    # Print options
    utils.print_options(options, log)

    # Load dataset ----------------------------------------------------------------------
    # Train provider
    train_provider, val_provider, test_provider = get_providers(options, log, flat=True)


    # Initialize model ------------------------------------------------------------------
    with tf.device('/gpu:0'):
        # Define inputs -------------------------------------------------------------------------
        real_batch = tf.placeholder(
            tf.float32,
            shape = [options['batch_size'], np.prod(np.array(options['img_shape']))],
            name = 'real_inputs'
        )
        sampler_input_batch = tf.placeholder(
            tf.float32,
            shape = [options['batch_size'], options['latent_dims']],
            name = 'noise_channel'
        )
        labels = tf.constant(
            np.expand_dims(
                np.concatenate(
                    (
                        np.ones(options['batch_size']),
                        np.zeros(options['batch_size'])
                    ),
                    axis=0
                ).astype(np.float32),
                axis=1
            )
        )
        labels = tf.cast(labels, tf.float32)
        log.info('Inputs defined')

        # Define model --------------------------------------------------------------------------
        with tf.variable_scope('gen_scope'):
            generator = Sequential('generator')
            generator += FullyConnected(options['latent_dims'], 60, tf.nn.tanh, name='fc_1')
            generator += FullyConnected(60, 60, tf.nn.tanh, name='fc_2')
            generator += FullyConnected(60, np.prod(options['img_shape']), tf.nn.tanh, name='fc_3')

            sampler = generator(sampler_input_batch)

        with tf.variable_scope('disc_scope'):
            disc_model = cupboard('fixed_conv_disc')(
                pickle.load(open(options['disc_params_path'], 'rb')),
                options['num_feat_layers'],
                name='disc_model'
            )

            disc_inputs = tf.concat(0, [real_batch, sampler])
            disc_inputs = tf.reshape(
                disc_inputs,
                [disc_inputs.get_shape()[0].value] + options['img_shape'] + [options['input_channels']]
            )

            preds = disc_model(disc_inputs)
            preds = tf.clip_by_value(preds, 0.00001, 0.99999)

            # Disc Accuracy
            disc_accuracy = (1 / float(labels.get_shape()[0].value)) * tf.reduce_sum(
                tf.cast(
                    tf.equal(
                        tf.round(preds),
                        labels
                    ),
                    tf.float32
                )
            )

        # Define Losses -------------------------------------------------------------------------
        # Discrimnator Cross-Entropy
        disc_CE = (1 / float(labels.get_shape()[0].value)) * tf.reduce_sum(
            -tf.add(
                tf.mul(
                    labels,
                    tf.log(preds)
                ),
                tf.mul(
                    1.0 - labels,
                    tf.log(1.0 - preds)
                )
            )
        )

        gen_loss = -tf.mul(
            1.0 - labels,
            tf.log(preds)
        )

        # Define Optimizers ---------------------------------------------------------------------
        optimizer = tf.train.AdamOptimizer(
            learning_rate=options['lr']
        )

        # Get Generator and Disriminator Trainable Variables
        gen_train_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'gen_scope')
        disc_train_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'disc_scope')

        # Get generator gradients
        grads = optimizer.compute_gradients(gen_loss, gen_train_vars)
        grads = [gv for gv in grads if gv[0] != None]
        clip_grads = [(tf.clip_by_norm(gv[0], 5.0, name='gen_grad_clipping'), gv[1]) for gv in grads]
        gen_backpass = optimizer.apply_gradients(clip_grads)

        # Get Dsicriminator gradients
        grads = optimizer.compute_gradients(disc_CE, disc_train_vars)
        grads = [gv for gv in grads if gv[0] != None]
        clip_grads = [(tf.clip_by_norm(gv[0], 5.0, name='disc_grad_clipping'), gv[1]) for gv in grads]
        disc_backpass = optimizer.apply_gradients(clip_grads)

        log.info('Optimizer graph built')
        # --------------------------------------------------------------------------------------
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
        if options['reload_all']:
            saver.restore(sess, options['reload_file'])
            log.info('Shared variables restored')
        else:
            sess.run(init_op)
            log.info('Variables initialized')

        # Define last losses to compute a running average
        last_losses = np.zeros((10))
        last_accs = np.zeros((10))
        disc_tracker = np.ones((5000))

        batch_abs_idx = 0
        D_to_G = options['D_to_G']
        total_D2G = sum(D_to_G)
        base = options['initial_G_iters'] + options['initial_D_iters']
        # must_init = True
        feat_params = pickle.load(open(options['disc_params_path'], 'rb'))

        for epoch_idx in xrange(options['n_epochs']):
            batch_rel_idx = 0
            log.info('Epoch {}'.format(epoch_idx + 1))

            for inputs in train_provider:
                if isinstance(inputs, tuple):
                    inputs = inputs[0]

                batch_abs_idx += 1
                batch_rel_idx += 1

                if batch_abs_idx < options['initial_G_iters']:
                    backpass = gen_backpass
                    log_format_string = '{},{},{},,\n'
                elif options['initial_G_iters'] <= batch_abs_idx < base:
                    backpass = disc_backpass
                    log_format_string = '{},{},,,{}\n'
                else:
                    # if np.mean(disc_tracker) < 0.95:
                    #     disc_model._disc.layers[-2].re_init_weights(sess)
                    #     disc_tracker = np.ones((5000))

                    if (batch_abs_idx - base) % total_D2G < D_to_G[0]:
                        # if must_init:
                        #     # i = 0
                        #     # for j in xrange(options['num_feat_layers']):
                        #     #     if feat_params[j]['layer_type'] == 'conv':
                        #     #         disc_model._disc.layers[i].re_init_weights(sess)
                        #     #         # print('@' * 1000)
                        #     #         # print(disc_model._disc.layers[i])
                        #     #         i += 1 # for dealing with activation function
                        #     #     elif feat_params[j]['layer_type'] == 'fc':
                        #     #         disc_model._disc.layers[i].re_init_weights(sess)
                        #     #         # print('@' * 1000)
                        #     #         # print(disc_model._disc.layers[i])
                        #     #     i += 1
                        #     disc_model._disc.layers[-2].re_init_weights(sess)
                        #     # print('@' * 1000)
                        #     # print(disc_model._disc.layers[-2])
                        #     must_init = False
                        backpass = disc_backpass
                        log_format_string = '{},{},,,{}\n'
                    else:
                        # must_init = True
                        backpass = gen_backpass
                        log_format_string = '{},{},,{},\n'

                log_format_string = '{},{},{},,\n'
                result = sess.run(
                    [
                        disc_CE,
                        backpass,
                        disc_accuracy
                    ],
                    feed_dict = {
                        real_batch: inputs,
                        sampler_input_batch: MVN(
                            np.zeros(options['latent_dims']),
                            np.diag(np.ones(options['latent_dims'])),
                            size = options['batch_size']
                        )
                    }
                )

                cost = result[0]

                if batch_abs_idx % 10 == 0:
                    train_log.write(log_format_string.format(batch_abs_idx, '2016-04-22', np.mean(last_losses)))
                    train_acc.write(log_format_string.format(batch_abs_idx, '2016-04-22', np.mean(last_accs)))

                    train_log.flush()
                    train_acc.flush()

                # Check cost
                if np.isnan(cost) or np.isinf(cost):
                    log.info('NaN detected')

                # Update last losses
                last_losses = np.roll(last_losses, 1)
                last_losses[0] = cost

                last_accs = np.roll(last_accs, 1)
                last_accs[0] = result[-1]

                disc_tracker = np.roll(disc_tracker, 1)
                disc_tracker[0] = result[-1]

                # Display training information
                if np.mod(epoch_idx, options['freq_logging']) == 0:
                    log.info('Epoch {:02}/{:02} Batch {:03} Current Loss: {:0>15.4f} Mean last losses: {:0>15.4f}'.format(
                        epoch_idx + 1,
                        options['n_epochs'],
                        batch_abs_idx,
                        float(cost),
                        np.mean(last_losses)
                    ))
                    log.info('Epoch {:02}/{:02} Batch {:03} Current Loss: {:0>15.4f} Mean last accuracies: {:0>15.4f}'.format(
                        epoch_idx + 1,
                        options['n_epochs'],
                        batch_abs_idx,
                        float(cost),
                        np.mean(last_accs)
                    ))

                # Save model
                if np.mod(batch_abs_idx, options['freq_saving']) == 0:
                    saver.save(sess, os.path.join(options['model_dir'], 'model_at_%d.ckpt' % batch_abs_idx))
                    log.info('Model saved')

                # Validate model
                if np.mod(batch_abs_idx, options['freq_validation']) == 0:

                    valid_costs = []
                    valid_accs = []
                    seen_batches = 0
                    for val_batch in val_provider:
                        if isinstance(val_batch, tuple):
                            val_batch = val_batch[0]

                        result = sess.run(
                            [
                                disc_CE,
                                disc_accuracy
                            ],
                            feed_dict = {
                                real_batch: val_batch,
                                sampler_input_batch: MVN(
                                    np.zeros(options['latent_dims']),
                                    np.diag(np.ones(options['latent_dims'])),
                                    size = options['batch_size']
                                )
                            }
                        )
                        valid_costs.append(result[0])
                        valid_accs.append(result[-1])
                        seen_batches += 1

                        if seen_batches == options['valid_batches']:
                            break

                    # Print results
                    log.info('Validation loss: {:0>15.4f}'.format(
                        float(np.mean(valid_costs))
                    ))
                    log.info('Validation accuracies: {:0>15.4f}'.format(
                        float(np.mean(valid_accs))
                    ))

                    val_samples = sess.run(
                        sampler,
                        feed_dict = {
                            sampler_input_batch: MVN(
                                np.zeros(options['latent_dims']),
                                np.diag(np.ones(options['latent_dims'])),
                                size = options['batch_size']
                            )
                        }
                    )

                    val_log.write(log_format_string.format(batch_abs_idx, '2016-04-22', np.mean(valid_costs)))
                    val_acc.write(log_format_string.format(batch_abs_idx, '2016-04-22', np.mean(valid_accs)))
                    val_log.flush()
                    val_acc.flush()

                    save_ae_samples(
                        catalog,
                        np.ones([options['batch_size']]+options['img_shape']),
                        np.reshape(inputs, [options['batch_size']]+options['img_shape']),
                        np.reshape(val_samples, [options['batch_size']]+options['img_shape']),
                        batch_abs_idx,
                        options['dashboard_dir'],
                        num_to_save=5,
                        save_gray=True
                    )

            log.info('End of epoch {}'.format(epoch_idx + 1))
    # --------------------------------------------------------------------------



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')

    parser.add_argument(
        'experiment',
        type = str,
        help = 'Path to config of the experiment'
    )

    args = parser.parse_args()

    options = config.load_config(args.experiment)

    train(options)
