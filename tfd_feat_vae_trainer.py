"""
Code modified from Lluis Castrjon's Poly-RNN train.py
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
from layers.fc_layer import ConstFC, FullyConnected
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
    if options['data_dir'] != 'MNIST':
        num_data_points = len(
            os.listdir(
                os.path.join(options['data_dir'], 'train', 'patches')
            )
        )
        num_data_points -= 2

        train_provider = DataProvider(
            num_data_points,
            options['batch_size'],
            toolbox.ImageLoader(
                data_dir = os.path.join(options['data_dir'], 'train', 'patches'),
                flat=True,
                extension=options['file_extension']
            )
        )

        # Valid provider
        num_data_points = len(
            os.listdir(
                os.path.join(options['data_dir'], 'valid', 'patches')
            )
        )
        num_data_points -= 2

        val_provider = DataProvider(
            num_data_points,
            options['batch_size'],
            toolbox.ImageLoader(
                data_dir = os.path.join(options['data_dir'], 'valid', 'patches'),
                flat = True,
                extension=options['file_extension']
            )
        )

    else:
        train_provider = DataProvider(
            55000,
            options['batch_size'],
            toolbox.MNISTLoader(
                mode='train',
                flat=True
            )
        )

        val_provider = DataProvider(
            5000,
            options['batch_size'],
            toolbox.MNISTLoader(
                mode='validation',
                flat = True
            )
        )

    log.info('Data providers initialized.')


    # Initialize model ------------------------------------------------------------------
    with tf.device('/gpu:0'):
        # Feature Extractor -----------------------------------------------------
        # feat_params = pickle.load(open(options['feat_params_path'], 'rb'))
        # _classifier = Sequential('CNN_Classifier')
        # _classifier += ConvLayer(
        #     options['input_channels'],
        #     feat_params[0]['n_filters_out'],
        #     feat_params[0]['input_dim'],
        #     feat_params[0]['filter_dim'],
        #     feat_params[0]['strides'],
        #     name='classifier_conv_0'
        # )
        # _classifier += feat_params[0]['act_fn']
        # _classifier.layers[-2].weights['W'] = tf.constant(feat_params[0]['W'])
        # _classifier.layers[-2].weights['b'] = tf.constant(feat_params[0]['b'])
        # print("1 conv layer")

        # i = 1
        # while i < options['num_feat_layers']:
        #     if 'filter_dim' in feat_params[i]:
        #         _classifier += ConvLayer(
        #             feat_params[i]['n_filters_in'],
        #             feat_params[i]['n_filters_out'],
        #             feat_params[i]['input_dim'],
        #             feat_params[i]['filter_dim'],
        #             feat_params[i]['strides'],
        #             name='classifier_conv_0'
        #         )
        #         _classifier += feat_params[i]['act_fn']
        #         _classifier.layers[-2].weights['W'] = tf.constant(feat_params[i]['W'])
        #         _classifier.layers[-2].weights['b'] = tf.constant(feat_params[i]['b'])
        #         print("1 conv layer")
        #     else:
        #         _classifier += ConstFC(
        #             feat_params[i]['W'],
        #             feat_params[i]['b'],
        #             activation=feat_params[i]['act_fn'],
        #             name='classifier_fc_0'
        #         )
        #         print("1 fc layer")
        #     i += 1

        if options['feat_type'] == 'fc':
            feat_model = Sequential('feat_extractor')
            feat_params = pickle.load(open(options['feat_params_path'], 'rb'))
            for i in range(options['num_feat_layers']):
                feat_model += ConstFC(
                    feat_params['enc_W'][i],
                    feat_params['enc_b'][i],
                    activation=feat_params['enc_act_fn'][i],
                    name='feat_layer_%d'%i
                )
        else:
            pass

        # VAE -------------------------------------------------------------------
        # VAE model
        vae_model = cupboard('vanilla_vae')(
            options['p_layers'],
            options['q_layers'],
            np.prod(options['img_shape']),
            options['latent_dims'],
            options['DKL_weight'],
            options['sigma_clip'],
            'vanilla_vae'
        )
        # -----------------------------------------------------------------------
        feat_vae = cupboard('feat_vae')(
            vae_model,
            feat_model,
            options['DKL_weight'],
            0.0,
            img_shape=options['img_shape'],
            input_channels=options['input_channels'],
            flat=True, 
            name='feat_vae_model'
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
        cost_function = feat_vae(model_input_batch)
        log.info('Forward pass graph built')

        # Define sampler
        sampler = feat_vae.build_sampler(sampler_input_batch)
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
            saver.restore(sess, options['reload_file'])
            log.info('Shared variables restored')
            if options['data_dir'] == 'MNIST':
                mean_img = np.zeros(np.prod(options['img_shape']))
                std_img = np.ones(np.prod(options['img_shape']))
            else:
                mean_img = np.load(os.path.join(options['data_dir'], 'mean' + options['extension']))
                std_img = np.load(os.path.join(options['data_dir'], 'std' + options['extension']))
            visualize(feat_vae.vae.sampler_mean, sess, feat_vae.vae.dec_mean, feat_vae.vae.dec_log_std_sq, sampler, sampler_input_batch,
                        model_input_batch, feat_vae.vae.enc_mean, feat_vae.vae.enc_log_std_sq,
                        train_provider, val_provider, options, catalog, mean_img, std_img)
            return
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
                    [cost_function, backpass, feat_vae.vae.DKL, feat_vae.vae.rec_loss, feat_vae.vae.dec_log_std_sq, feat_vae.vae.enc_log_std_sq, feat_vae.vae.enc_mean, feat_vae.vae.dec_mean] + [gv[0] for gv in grads],
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

                # Save model
                if np.mod(batch_abs_idx, options['freq_saving']) == 0:
                    saver.save(sess, os.path.join(options['model_dir'], 'model_at_%d.ckpt' % batch_abs_idx))
                    log.info('Model saved')

                # Validate model
                if np.mod(batch_abs_idx, options['freq_validation']) == 0:

                    valid_costs = []
                    seen_batches = 0
                    for val_batch in val_provider:

                        val_cost = sess.run(
                            cost_function,
                            feed_dict = {
                                model_input_batch: val_batch
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

                    # save_dash_samples(
                    #     catalog,
                    #     val_samples,
                    #     batch_abs_idx,
                    #     options['dashboard_dir'],
                    #     flat_samples=True,
                    #     img_shape=options['img_shape'],
                    #     num_to_save=5
                    # )

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
                        result[7],
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