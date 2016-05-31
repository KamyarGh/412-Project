"""
Code template from from Lluis Castrjon's Poly-RNN train.py
"""

import numpy as np
import tensorflow as tf
from data_provider import DataProvider
import toolbox
from toolbox import cupboard, visualize
import config
import argparse
import utils
import os
from evaluate import save_samples, save_dash_samples, save_ae_samples
from numpy.random import multivariate_normal as MVN, uniform
from containers.sequential import Sequential
from layers.conv import ConvLayer
from layers.pool import PoolLayer
from layers.fc_layer import ConstFC, FullyConnected
from toolbox import get_providers, test_LL_and_DKL
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

    with open(os.path.join(options['dashboard_dir'], 'description'), 'w') as desc_file:
        desc_file.write(options['description'])

    # Dashboard Catalog
    catalog = open(os.path.join(options['dashboard_dir'], 'catalog'), 'w')
    catalog.write(
"""filename,type,name
description,plain,Description
options,plain,Options
train_loss.csv,csv,Train Loss
ll.csv,csv,Neg. Log-Likelihood
dec_log_sig_sq.csv,csv,Decoder Log Simga^2
dec_std_log_sig_sq.csv,csv,STD of Decoder Log Simga^2
dec_mean.csv,csv,Decoder Mean
dkl.csv,csv,DKL
enc_log_sig_sq.csv,csv,Encoder Log Sigma^2
enc_std_log_sig_sq.csv,csv,STD of Encoder Log Sigma^2
enc_mean.csv,csv,Encoder Mean
val_loss.csv,csv,Validation Loss
"""
    )
    catalog.flush()
    train_log = open(os.path.join(options['dashboard_dir'], 'train_loss.csv'), 'w')
    val_log = open(os.path.join(options['dashboard_dir'], 'val_loss.csv'), 'w')
    dkl_log = open(os.path.join(options['dashboard_dir'], 'dkl.csv'), 'w')
    ll_log = open(os.path.join(options['dashboard_dir'], 'll.csv'), 'w')

    dec_sig_log = open(os.path.join(options['dashboard_dir'], 'dec_log_sig_sq.csv'), 'w')
    enc_sig_log = open(os.path.join(options['dashboard_dir'], 'enc_log_sig_sq.csv'), 'w')

    dec_std_sig_log = open(os.path.join(options['dashboard_dir'], 'dec_std_log_sig_sq.csv'), 'w')
    enc_std_sig_log = open(os.path.join(options['dashboard_dir'], 'enc_std_log_sig_sq.csv'), 'w')

    dec_mean_log = open(os.path.join(options['dashboard_dir'], 'dec_mean.csv'), 'w')
    enc_mean_log = open(os.path.join(options['dashboard_dir'], 'enc_mean.csv'), 'w')
    # val_sig_log = open(os.path.join(options['dashboard_dir'], 'val_log_sig_sq.csv'), 'w')

    train_log.write('step,time,Train Loss\n')
    val_log.write('step,time,Validation Loss\n')
    dkl_log.write('step,time,DKL\n')
    ll_log.write('step,time,-LL\n')

    dec_sig_log.write('step,time,Decoder Log Sigma^2\n')
    enc_sig_log.write('step,time,Encoder Log Sigma^2\n')

    dec_std_sig_log.write('step,time,STD of Decoder Log Sigma^2\n')
    enc_std_sig_log.write('step,time,STD of Encoder Log Sigma^2\n')

    dec_mean_log.write('step,time,Decoder Mean\n')
    enc_mean_log.write('step,time,Encoder Mean\n')

    # Print options
    utils.print_options(options, log)

    # Load dataset ----------------------------------------------------------------------
    # Train provider
    train_provider, val_provider, test_provider = get_providers(options, log, flat=True)

    # Initialize model ------------------------------------------------------------------
    with tf.device('/gpu:0'):

        # Define inputs ----------------------------------------------------------
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

        # Discriminator ---------------------------------------------------------
        # with tf.variable_scope('disc_scope'):
        #     disc_model = cupboard('fixed_conv_disc')(
        #         pickle.load(open(options['feat_params_path'], 'rb')),
        #         options['num_feat_layers'],
        #         'discriminator'
        #     )

        # VAE -------------------------------------------------------------------
        # VAE model
        # with tf.variable_scope('vae_scope'):
        vae_model = cupboard('vanilla_vae')(
            options['p_layers'],
            options['q_layers'],
            np.prod(options['img_shape']),
            options['latent_dims'],
            options['DKL_weight'],
            options['sigma_clip'],
            'vanilla_vae'
        )
        # VAE/GAN ---------------------------------------------------------------
        # vae_gan = cupboard('vae_gan')(
        #     vae_model,
        #     disc_model,
        #     options['img_shape'],
        #     options['input_channels'],
        #     'vae_scope',
        #     'disc_scope',
        #     name = 'vae_gan_model'
        # )

        log.info('Model initialized')

        # Define optimizer
        optimizer = tf.train.AdamOptimizer(learning_rate=options['lr'])
        # Define forward pass
        cost_function  = vae_model(model_input_batch)
        # backpass, grads = vae_gan(model_input_batch, sampler_input_batch, optimizer)
        log.info('Forward pass graph built')

        # Define sampler
        # sampler = vae_gan.sampler
        sampler = vae_model.build_sampler(sampler_input_batch)
        log.info('Sampler graph built')

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
            saver.restore(sess, options['reload_file'])
            log.info('Shared variables restored')

            test_LL_and_DKL(sess, test_provider, feat_vae.vae.DKL, feat_vae.vae.rec_loss, options, model_input_batch)
            return

            mean_img = np.load(os.path.join(options['data_dir'], 'mean' + options['extension']))
            std_img = np.load(os.path.join(options['data_dir'], 'std' + options['extension']))
            visualize(sess, feat_vae.vae.dec_mean, feat_vae.vae.dec_log_std_sq, sampler, sampler_input_batch,
                        model_input_batch, feat_vae.vae.enc_mean, feat_vae.vae.enc_log_std_sq,
                        train_provider, val_provider, options, catalog, mean_img, std_img)
            return
        else:
            sess.run(init_op)
            log.info('Shared variables initialized')

        # Define last losses to compute a running average
        last_losses = np.zeros((10))

        batch_abs_idx = 0
        D_to_G = options['D_to_G']
        total_D2G = sum(D_to_G)
        for epoch_idx in xrange(options['n_epochs']):
            batch_rel_idx = 0
            log.info('Epoch {}'.format(epoch_idx + 1))

            for inputs,_ in train_provider:
                batch_abs_idx += 1
                batch_rel_idx += 1

                # if batch_abs_idx < options['initial_G_iters']:
                #     optimizer = vae_optimizer
                # else:
                #     optimizer = disc_optimizer
                    # if batch_abs_idx % total_D2G < D_to_G[0]:
                    #     optimizer = disc_optimizer
                    # else:
                    #     optimizer = vae_optimizer
                result = sess.run(
                    [
                        cost_function,
                        backpass,
                        vae_model.DKL,
                        vae_model.rec_loss,
                        vae_model.dec_log_std_sq,
                        vae_model.enc_log_std_sq,
                        vae_model.enc_mean,
                        vae_model.dec_mean,
                    ] + [gv[0] for gv in grads],
                    feed_dict = {
                        model_input_batch: inputs
                    }
                )

                # print('#'*80)
                # print(result[-1])
                # print('#'*80)

                cost = result[0]

                if batch_abs_idx % 10 == 0:
                    train_log.write('{},{},{}\n'.format(batch_abs_idx, '2016-04-22', np.mean(last_losses)))
                    dkl_log.write('{},{},{}\n'.format(batch_abs_idx, '2016-04-22', -np.mean(result[2])))
                    ll_log.write('{},{},{}\n'.format(batch_abs_idx, '2016-04-22', -np.mean(result[3])))

                    train_log.flush()
                    dkl_log.flush()
                    ll_log.flush()

                    dec_sig_log.write('{},{},{}\n'.format(batch_abs_idx, '2016-04-22', np.mean(result[4])))
                    enc_sig_log.write('{},{},{}\n'.format(batch_abs_idx, '2016-04-22', np.mean(result[5])))
                    # val_sig_log.write('{},{},{}\n'.format(batch_abs_idx, '2016-04-22', np.mean(result[6])))

                    dec_sig_log.flush()
                    enc_sig_log.flush()

                    dec_std_sig_log.write('{},{},{}\n'.format(batch_abs_idx, '2016-04-22', np.std(result[4])))
                    enc_std_sig_log.write('{},{},{}\n'.format(batch_abs_idx, '2016-04-22', np.std(result[5])))

                    dec_mean_log.write('{},{},{}\n'.format(batch_abs_idx, '2016-04-22', np.mean(result[7])))
                    enc_mean_log.write('{},{},{}\n'.format(batch_abs_idx, '2016-04-22', np.mean(result[6])))

                    dec_std_sig_log.flush()
                    enc_std_sig_log.flush()

                    dec_mean_log.flush()
                    enc_mean_log.flush()                    
                    # val_sig_log.flush()

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
                    # log.info('Batch Mean Acc.: {:0>15.4f}'.format(result[-2], axis=0))

                # Save model
                if np.mod(batch_abs_idx, options['freq_saving']) == 0:
                    saver.save(sess, os.path.join(options['model_dir'], 'model_at_%d.ckpt' % batch_abs_idx))
                    log.info('Model saved')

                # Validate model
                if np.mod(batch_abs_idx, options['freq_validation']) == 0:

                    valid_costs = []
                    seen_batches = 0
                    for val_batch,_ in val_provider:

                        val_cost = sess.run(
                            vae_model.cost,
                            feed_dict = {
                                model_input_batch: val_batch,
                                sampler_input_batch: MVN(
                                    np.zeros(options['latent_dims']),
                                    np.diag(np.ones(options['latent_dims'])),
                                    size = options['batch_size']
                                )
                            }
                        )
                        valid_costs.append(val_cost)
                        seen_batches += 1

                        if seen_batches == options['valid_batches']:
                            break

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

                    save_ae_samples(
                        catalog,
                        np.reshape(result[7], [options['batch_size']]+options['img_shape']),
                        np.reshape(inputs, [options['batch_size']]+options['img_shape']),
                        np.reshape(val_samples, [options['batch_size']]+options['img_shape']),
                        batch_abs_idx,
                        options['dashboard_dir'],
                        num_to_save=5,
                        save_gray=True
                    )

                    # save_samples(
                    #     val_samples,
                    #     int(batch_abs_idx/options['freq_validation']),
                    #     os.path.join(options['model_dir'], 'valid_samples'),
                    #     True,
                    #     options['img_shape'],
                    #     5
                    # )

                    # save_samples(
                    #     inputs,
                    #     int(batch_abs_idx/options['freq_validation']),
                    #     os.path.join(options['model_dir'], 'input_sanity'),
                    #     True,
                    #     options['img_shape'],
                    #     num_to_save=5
                    # )

                    # save_samples(
                    #     result[8],
                    #     int(batch_abs_idx/options['freq_validation']),
                    #     os.path.join(options['model_dir'], 'rec_sanity'),
                    #     True,
                    #     options['img_shape'],
                    #     num_to_save=5
                    # )


            log.info('End of epoch {}'.format(epoch_idx + 1))
    # Test Model --------------------------------------------------------------------------
        test_results = []

        for inputs in test_provider:
            if isinstance(inputs, tuple):
                inputs = inputs[0]
            batch_results = sess.run(
                [
                    feat_vae.vae.DKL, feat_vae.vae.rec_loss,
                    feat_vae.vae.dec_log_std_sq, feat_vae.vae.enc_log_std_sq,
                    feat_vae.vae.dec_mean, feat_vae.vae.enc_mean
                ],
                feed_dict = {
                    model_input_batch: inputs
                }
            )

            test_results.append(map(lambda p: np.mean(p, axis=1) if len(p.shape) > 1 else np.mean(p), batch_results))
        test_results = map(list, zip(*test_results))

        # Print results
        log.info('Test Mean Rec. Loss: {:0>15.4f}'.format(
            float(np.mean(test_results[1]))
        ))
        log.info('Test DKL: {:0>15.4f}'.format(
            float(np.mean(test_results[0]))
        ))
        log.info('Test Dec. Mean Log Std Sq: {:0>15.4f}'.format(
            float(np.mean(test_results[2]))
        ))
        log.info('Test Enc. Mean Log Std Sq: {:0>15.4f}'.format(
            float(np.mean(test_results[3]))
        ))
        log.info('Test Dec. Mean Mean: {:0>15.4f}'.format(
            float(np.mean(test_results[4]))
        ))
        log.info('Test Enc. Mean Mean: {:0>15.4f}'.format(
            float(np.mean(test_results[5]))
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