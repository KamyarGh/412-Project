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
from evaluate import save_samples, save_dash_samples
from numpy.random import multivariate_normal as MVN, uniform

def train(options):
    # Get logger
    log = utils.get_logger(os.path.join(options['model_dir'], 'log.txt'))
    options_file = open(os.path.join(options['dashboard_dir'], 'options'), 'w')
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
ll.csv,csv,Neg. Log-Likelihood
dkl.csv,csv,DKL
val_loss.csv,csv,Validation Loss
"""
    )
    catalog.flush()
    train_log = open(os.path.join(options['dashboard_dir'], 'train_loss.csv'), 'w')
    val_log = open(os.path.join(options['dashboard_dir'], 'val_loss.csv'), 'w')
    dkl_log = open(os.path.join(options['dashboard_dir'], 'dkl.csv'), 'w')
    ll_log = open(os.path.join(options['dashboard_dir'], 'll.csv'), 'w')

    train_log.write('step,time,Train Loss\n')
    val_log.write('step,time,Validation Loss\n')
    dkl_log.write('step,time,DKL\n')
    ll_log.write('step,time,-LL\n')

    # Print options
    utils.print_options(options, log)

    # Load dataset ----------------------------------------------------------------------
    # Train provider
    num_data_points = len(
        os.listdir(
            os.path.join(options['data_dir'], 'train')
        )
    )
    num_data_points -= 2

    train_provider = DataProvider(
        num_data_points,
        options['batch_size'],
        toolbox.ImageLoader(
            data_dir = os.path.join(options['data_dir'], 'train'),
            flat=True
        )
    )

    # Valid provider
    num_data_points = len(
        os.listdir(
            os.path.join(options['data_dir'], 'val')
        )
    )
    num_data_points -= 2

    val_provider = DataProvider(
        num_data_points,
        options['batch_size'],
        toolbox.ImageLoader(
            data_dir = os.path.join(options['data_dir'], 'val'),
            flat = True
        )
    )
    log.info('Data providers initialized.')


    # Initialize model ------------------------------------------------------------------
    with tf.device('/gpu:0'):
        model = cupboard(options['model'])(
            options['p_layers'],
            options['q_layers'],
            np.prod(options['img_shape']),
            options['latent_dims'],
            'vanilla_vae'
        )
        log.info('Model initialized')

        # Define inputs
        model_input_batch = tf.placeholder(
            tf.float32,
            shape = [options['batch_size'], np.prod(np.array(options['img_shape']))],
            name = 'enc_inputs'
        )
        sampler_input_batch = tf.placeholder(
            tf.float32,
            shape = [options['batch_size'], options['latent_dims']],
            name = 'dec_inputs'
        )
        log.info('Inputs defined')

        # Define forward pass
        cost_function = model(model_input_batch)
        log.info('Forward pass graph built')

        # Define sampler
        sampler = model.build_sampler(sampler_input_batch)
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

        # Define last losses to compute a running average
        last_losses = np.zeros((10))

        batch_abs_idx = 0
        for epoch_idx in xrange(options['n_epochs']):
            batch_rel_idx = 0
            log.info('Epoch {}'.format(epoch_idx + 1))

            for inputs in train_provider:
                batch_abs_idx += 1
                batch_rel_idx += 1

                result = sess.run(
                    # (cost_function, train_step, model.enc_std, model.enc_mean, model.encoder, model.dec_std, model.dec_mean, model.decoder, model.rec_loss, model.DKL),
                    #       0           1               2           3               4               5               6              7               8            9           10
                    (cost_function, train_step, model.DKL, model.rec_loss, model.dec_mean),
                    feed_dict = {
                        model_input_batch: inputs
                    }
                )

                cost = result[0]

                if batch_abs_idx % 10 == 0:
                    train_log.write('{},{},{}\n'.format(batch_abs_idx, '2016-04-22', np.mean(last_losses)))
                    dkl_log.write('{},{},{}\n'.format(batch_abs_idx, '2016-04-22', -np.mean(result[2])))
                    ll_log.write('{},{},{}\n'.format(batch_abs_idx, '2016-04-22', -np.mean(result[3])))

                    train_log.flush()
                    dkl_log.flush()
                    ll_log.flush()

                # print('\n\nENC_MEAN:')
                # print(result[3])
                # print('\n\nENC_STD:')
                # print(result[2])
                # print('\nDEC_MEAN:')
                # print(result[6])
                # print('\nDEC_STD:')
                # print(result[5])

                # print('\n\nENCODER WEIGHTS:')
                # print(model._encoder.layers[0].weights['w'].eval())
                # print('\n\DECODER WEIGHTS:')
                # print(model._decoder.layers[0].weights['w'].eval())

                # print(model._encoder.layers[0].weights['w'].eval())
                # print(result[2])
                # print(result[3])

                # print(result[3])
                # print(result[2])
                # print(result[-2])
                # print(result[-1])

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

                # Display training information
                if np.mod(epoch_idx, options['freq_logging']) == 0:
                    log.info('Epoch {:02}/{:02} Batch {:03} Current Loss: {:0>15.4f} Mean last losses: {:0>15.4f}'.format(
                        epoch_idx + 1,
                        options['n_epochs'],
                        batch_abs_idx,
                        float(cost),
                        np.mean(last_losses)
                    ))
                    log.info('Batch Mean LL: {:0>15.4f}'.format(np.mean(result[3], axis=0)))
                    log.info('Batch Mean -DKL: {:0>15.4f}'.format(np.mean(result[2], axis=0)))

                # Save model
                if np.mod(batch_abs_idx, options['freq_saving']) == 0:
                    saver.save(sess, os.path.join(options['model_dir'], 'model.ckpt'))
                    log.info('Model saved')

                # Validate model
                if np.mod(batch_abs_idx, options['freq_validation']) == 0:

                    model._decoder.layers[0].weights['w'].eval()[:5,:5]

                    valid_costs = []
                    seen_batches = 0
                    for val_batch in val_provider:

                        # Break if 10 batches seen for now
                        if seen_batches == options['valid_batches']:
                            break

                        val_cost = sess.run(
                            cost_function,
                            feed_dict = {
                                model_input_batch: val_batch
                            }
                        )
                        valid_costs.append(val_cost)
                        seen_batches += 1

                    # Print results
                    log.info('Validation loss: {:0>15.4f}'.format(
                        float(np.mean(valid_costs))
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

                    val_log.write('{},{},{}\n'.format(batch_abs_idx, '2016-04-22', np.mean(valid_costs)))
                    val_log.flush()

                    save_dash_samples(
                        catalog,
                        val_samples,
                        batch_abs_idx,
                        options['dashboard_dir'],
                        flat_samples=True,
                        img_shape=options['img_shape'],
                        num_to_save=5
                    )

                    save_samples(
                        val_samples,
                        int(batch_abs_idx/options['freq_validation']),
                        os.path.join(options['model_dir'], 'valid_samples'),
                        True,
                        options['img_shape'],
                        5
                    )

                    save_samples(
                        inputs,
                        int(batch_abs_idx/options['freq_validation']),
                        os.path.join(options['model_dir'], 'input_sanity'),
                        True,
                        options['img_shape'],
                        num_to_save=5
                    )

                    save_samples(
                        result[-1],
                        int(batch_abs_idx/options['freq_validation']),
                        os.path.join(options['model_dir'], 'rec_sanity'),
                        True,
                        options['img_shape'],
                        num_to_save=5
                    )


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