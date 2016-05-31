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
train_loss.csv,csv,Discriminator Cross-Entropy
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

    train_log.write('step,time,Train CE (Training Vanilla),Train CE (Training Gen.),Train CE (Training Disc.)\n')
    val_log.write('step,time,Validation CE (Training Vanilla),Validation CE (Training Gen.),Validation CE (Training Disc.)\n')
    dkl_log.write('step,time,DKL (Training Vanilla),DKL (Training Gen.),DKL (Training Disc.)\n')
    ll_log.write('step,time,-LL (Training Vanilla),-LL (Training Gen.),-LL (Training Disc.)\n')

    dec_sig_log.write('step,time,Decoder Log Sigma^2 (Training Vanilla),Decoder Log Sigma^2 (Training Gen.),Decoder Log Sigma^2 (Training Disc.)\n')
    enc_sig_log.write('step,time,Encoder Log Sigma^2 (Training Vanilla),Encoder Log Sigma^2 (Training Gen.),Encoder Log Sigma^2 (Training Disc.)\n')

    dec_std_sig_log.write('step,time,STD of Decoder Log Sigma^2 (Training Vanilla),STD of Decoder Log Sigma^2 (Training Gen.),STD of Decoder Log Sigma^2 (Training Disc.)\n')
    enc_std_sig_log.write('step,time,STD of Encoder Log Sigma^2 (Training Vanilla),STD of Encoder Log Sigma^2 (Training Gen.),STD of Encoder Log Sigma^2 (Training Disc.)\n')

    dec_mean_log.write('step,time,Decoder Mean (Training Vanilla),Decoder Mean (Training Gen.),Decoder Mean (Training Disc.)\n')
    enc_mean_log.write('step,time,Encoder Mean (Training Vanilla),Encoder Mean (Training Gen.),Encoder Mean (Training Disc.)\n')

    # Print options
    utils.print_options(options, log)

    # Load dataset ----------------------------------------------------------------------
    # Train provider
    train_provider, val_provider, test_provider = get_providers(options, log, flat=True)


    # Initialize model ------------------------------------------------------------------
    with tf.device('/gpu:0'):
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

        # Define model
        with tf.variable_scope('vae_scope'):
            vae_model = cupboard('vanilla_vae')(
                options['p_layers'],
                options['q_layers'],
                np.prod(options['img_shape']),
                options['latent_dims'],
                options['DKL_weight'],
                options['sigma_clip'],
                'vae_model'
            )

        with tf.variable_scope('disc_scope'):
            disc_model = cupboard('fixed_conv_disc')(
                pickle.load(open(options['disc_params_path'], 'rb')),
                options['num_feat_layers'],
                name='disc_model'
            )

        vae_gan = cupboard('vae_gan')(
            vae_model,
            disc_model,
            options['disc_weight'],
            options['img_shape'],
            options['input_channels'],
            'vae_scope',
            'disc_scope',
            name='vae_gan_model'
        )

        # Define Optimizers ---------------------------------------------------------------------
        optimizer = tf.train.AdamOptimizer(
            learning_rate=options['lr']
        )

        vae_backpass, disc_backpass, vanilla_backpass = vae_gan(model_input_batch, sampler_input_batch, optimizer)

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

            if options['reload_vae']:
                vae_model.reload_vae(options['vae_params_path'])

        # Define last losses to compute a running average
        last_losses = np.zeros((10))

        batch_abs_idx = 0
        D_to_G = options['D_to_G']
        total_D2G = sum(D_to_G)
        base = options['initial_G_iters'] + options['initial_D_iters']

        for epoch_idx in xrange(options['n_epochs']):
            batch_rel_idx = 0
            log.info('Epoch {}'.format(epoch_idx + 1))

            for inputs in train_provider:
                if isinstance(inputs, tuple):
                    inputs = inputs[0]

                batch_abs_idx += 1
                batch_rel_idx += 1

                if batch_abs_idx < options['initial_G_iters']:
                    backpass = vanilla_backpass
                    log_format_string = '{},{},{},,\n'
                elif options['initial_G_iters'] <= batch_abs_idx < base:
                    backpass = disc_backpass
                    log_format_string = '{},{},,,{}\n'
                else:
                    if (batch_abs_idx - base) % total_D2G < D_to_G[0]:
                        backpass = disc_backpass
                        log_format_string = '{},{},,,{}\n'
                    else:
                        backpass = vae_backpass
                        log_format_string = '{},{},,{},\n'

                result = sess.run(
                    [
                        vae_gan.disc_CE,
                        backpass,
                        vae_gan._vae.DKL,
                        vae_gan._vae.rec_loss,
                        vae_gan._vae.dec_log_std_sq,
                        vae_gan._vae.enc_log_std_sq,
                        vae_gan._vae.enc_mean,
                        vae_gan._vae.dec_mean
                    ],
                    feed_dict = {
                        model_input_batch: inputs,
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
                    dkl_log.write(log_format_string.format(batch_abs_idx, '2016-04-22', -np.mean(result[2])))
                    ll_log.write(log_format_string.format(batch_abs_idx, '2016-04-22', -np.mean(result[3])))

                    train_log.flush()
                    dkl_log.flush()
                    ll_log.flush()

                    dec_sig_log.write(log_format_string.format(batch_abs_idx, '2016-04-22', np.mean(result[4])))
                    enc_sig_log.write(log_format_string.format(batch_abs_idx, '2016-04-22', np.mean(result[5])))
                    # val_sig_log.write('{},{},{}\n'.format(batch_abs_idx, '2016-04-22', np.mean(result[6])))

                    dec_sig_log.flush()
                    enc_sig_log.flush()

                    dec_std_sig_log.write(log_format_string.format(batch_abs_idx, '2016-04-22', np.std(result[4])))
                    enc_std_sig_log.write(log_format_string.format(batch_abs_idx, '2016-04-22', np.std(result[5])))

                    dec_mean_log.write(log_format_string.format(batch_abs_idx, '2016-04-22', np.mean(result[7])))
                    enc_mean_log.write(log_format_string.format(batch_abs_idx, '2016-04-22', np.mean(result[6])))

                    dec_std_sig_log.flush()
                    enc_std_sig_log.flush()

                    dec_mean_log.flush()
                    enc_mean_log.flush()

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
                    print(vae_gan._vae._encoder.layers[0].weights['w'].eval())
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

                    save_dict = {}
                    # Save encoder params ------------------------------------------------------------------
                    for i in range(len(vae_gan._vae._encoder.layers)):
                        layer_dict = {
                            'input_dim':vae_gan._vae._encoder.layers[i].input_dim,
                            'output_dim':vae_gan._vae._encoder.layers[i].output_dim,
                            'act_fn':vae_gan._vae._encoder.layers[i].activation,
                            'W':vae_gan._vae._encoder.layers[i].weights['w'].eval(),
                            'b':vae_gan._vae._encoder.layers[i].weights['b'].eval()
                        }
                        save_dict['encoder'] = layer_dict

                    layer_dict = {
                        'input_dim':vae_gan._vae._enc_mean.input_dim,
                        'output_dim':vae_gan._vae._enc_mean.output_dim,
                        'act_fn':vae_gan._vae._enc_mean.activation,
                        'W':vae_gan._vae._enc_mean.weights['w'].eval(),
                        'b':vae_gan._vae._enc_mean.weights['b'].eval()
                    }
                    save_dict['enc_mean'] = layer_dict

                    layer_dict = {
                        'input_dim':vae_gan._vae._enc_log_std_sq.input_dim,
                        'output_dim':vae_gan._vae._enc_log_std_sq.output_dim,
                        'act_fn':vae_gan._vae._enc_log_std_sq.activation,
                        'W':vae_gan._vae._enc_log_std_sq.weights['w'].eval(),
                        'b':vae_gan._vae._enc_log_std_sq.weights['b'].eval()
                    }
                    save_dict['enc_log_std_sq'] = layer_dict

                    # Save decoder params ------------------------------------------------------------------
                    for i in range(len(vae_gan._vae._decoder.layers)):
                        layer_dict = {
                            'input_dim':vae_gan._vae._decoder.layers[i].input_dim,
                            'output_dim':vae_gan._vae._decoder.layers[i].output_dim,
                            'act_fn':vae_gan._vae._decoder.layers[i].activation,
                            'W':vae_gan._vae._decoder.layers[i].weights['w'].eval(),
                            'b':vae_gan._vae._decoder.layers[i].weights['b'].eval()
                        }
                        save_dict['decoder'] = layer_dict

                    layer_dict = {
                        'input_dim':vae_gan._vae._dec_mean.input_dim,
                        'output_dim':vae_gan._vae._dec_mean.output_dim,
                        'act_fn':vae_gan._vae._dec_mean.activation,
                        'W':vae_gan._vae._dec_mean.weights['w'].eval(),
                        'b':vae_gan._vae._dec_mean.weights['b'].eval()
                    }
                    save_dict['dec_mean'] = layer_dict

                    layer_dict = {
                        'input_dim':vae_gan._vae._dec_log_std_sq.input_dim,
                        'output_dim':vae_gan._vae._dec_log_std_sq.output_dim,
                        'act_fn':vae_gan._vae._dec_log_std_sq.activation,
                        'W':vae_gan._vae._dec_log_std_sq.weights['w'].eval(),
                        'b':vae_gan._vae._dec_log_std_sq.weights['b'].eval()
                    }
                    save_dict['dec_log_std_sq'] = layer_dict

                    pickle.dump(save_dict, open(os.path.join(options['model_dir'], 'vae_dict_%d' % batch_abs_idx), 'wb'))

                # Validate model
                if np.mod(batch_abs_idx, options['freq_validation']) == 0:

                    vae_gan._vae._decoder.layers[0].weights['w'].eval()[:5,:5]

                    valid_costs = []
                    seen_batches = 0
                    for val_batch in val_provider:
                        if isinstance(val_batch, tuple):
                            val_batch = val_batch[0]

                        val_cost = sess.run(
                            vae_gan.disc_CE,
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
                        vae_gan.sampler,
                        feed_dict = {
                            sampler_input_batch: MVN(
                                np.zeros(options['latent_dims']),
                                np.diag(np.ones(options['latent_dims'])),
                                size = options['batch_size']
                            )
                        }
                    )

                    val_log.write(log_format_string.format(batch_abs_idx, '2016-04-22', np.mean(valid_costs)))
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
