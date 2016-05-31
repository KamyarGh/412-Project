"""
Code modified from Lluis Castrjon's Poly-RNN train.py
"""

import numpy as np
import tensorflow as tf
import toolbox
from toolbox import cupboard
import config
import argparse
import utils
import os
from evaluate import save_samples, save_dash_samples, save_ae_samples
from numpy.random import normal
from toolbox import get_providers
import pickle

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
    train_provider, val_provider, test_provider = get_providers(options, log)

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
        # train_step = optimizer.minimize(cost_function)
        log.info('Optimizer graph built')

        # Get gradients
        grads = optimizer.compute_gradients(cost_function)
        grads = [gv for gv in grads if gv[0] != None]
        grad_tensors = [gv[0] for gv in grads]

        # Clip gradients
        clip_grads = [(tf.clip_by_norm(gv[0], 5.0, name='grad_clipping'), gv[1]) for gv in grads]

        # Update op
        backpass = optimizer.apply_gradients(clip_grads)

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

            for inputs,_ in train_provider:
                batch_abs_idx += 1
                batch_rel_idx += 1

                result = sess.run(
                    [cost_function, backpass] + [gv[0] for gv in grads],
                    feed_dict = {
                        model_clean_input_batch: inputs,
                        model_noisy_input_batch: np.float32(inputs) + \
                            normal(
                                loc=0.0,
                                scale=np.float32(options['noise_std']),
                                size=inputs.shape
                            )
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
                    saver.save(sess, os.path.join(options['model_dir'], 'model_at_%d.ckpt' % batch_abs_idx))
                    log.info('Model saved')

                    # Save Encoder Params
                    save_dict = {
                        'enc_W': [],
                        'enc_b': [],
                        'enc_act_fn': [],
                    }
                    if options['model'] == 'cnn_ae':
                        pass
                    else:
                        for i in range(len(model._encoder.layers)):
                            save_dict['enc_W'].append(model._encoder.layers[i].weights['w'].eval())
                            save_dict['enc_b'].append(model._encoder.layers[i].weights['b'].eval())
                            save_dict['enc_act_fn'].append(options['enc_params']['act_fn'][i])

                    pickle.dump(save_dict, open(os.path.join(options['model_dir'], 'enc_dict_%d' % batch_abs_idx), 'wb'))


                # Validate model
                if np.mod(batch_abs_idx, options['freq_validation']) == 0:

                    model._decoder.layers[0].weights['w'].eval()[:5,:5]

                    valid_costs = []
                    seen_batches = 0
                    for val_batch,_ in val_provider:

                        noisy_val_batch = val_batch + \
                            normal(
                                loc=0.0,
                                scale=np.float32(options['noise_std']),
                                size=val_batch.shape
                            )

                        val_results = sess.run(
                            (cost_function, model.decoder),
                            feed_dict = {
                                model_clean_input_batch: val_batch,
                                model_noisy_input_batch: noisy_val_batch
                            }
                        )
                        valid_costs.append(val_results[0])
                        seen_batches += 1

                        if seen_batches == options['valid_batches']:
                            break

                    # Print results
                    log.info('Validation loss: {:0>15.4f}'.format(
                        float(np.mean(valid_costs))
                    ))

                    val_log.write('{},{},{}\n'.format(batch_abs_idx, '2016-04-22', np.mean(valid_costs)))
                    val_log.flush()


                    if options['model'] == 'conv_ae':
                        val_recon = np.reshape(
                            val_results[-1],
                            val_batch.shape
                        )
                    else:
                        val_batch = np.reshape(
                            val_batch,
                            [val_batch.shape[0]] + options['img_shape'] + [options['input_channels']]
                        )
                        noisy_val_batch = np.reshape(
                            noisy_val_batch,
                            [val_batch.shape[0]] + options['img_shape'] + [options['input_channels']]
                        )
                        val_recon = np.reshape(
                            val_results[-1],
                            [val_batch.shape[0]] + options['img_shape'] + [options['input_channels']]
                        )

                    save_ae_samples(
                        catalog,
                        val_batch,
                        noisy_val_batch,
                        val_recon,
                        batch_abs_idx,
                        options['dashboard_dir'],
                        num_to_save=5,
                        save_gray=True
                    )

                    # save_samples(
                    #     val_recon,
                    #     int(batch_abs_idx/options['freq_validation']),
                    #     os.path.join(options['model_dir'], 'valid_samples'),
                    #     False,
                    #     options['img_shape'],
                    #     5
                    # )

                    # save_samples(
                    #     inputs,
                    #     int(batch_abs_idx/options['freq_validation']),
                    #     os.path.join(options['model_dir'], 'input_sanity'),
                    #     False,
                    #     options['img_shape'],
                    #     num_to_save=5
                    # )


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