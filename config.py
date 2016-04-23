"""
    Author: Lluis Castrejon
    config.py
    ~~~~~~~~~

    Manage different experiments configurations.
"""
# Imports ----------------------------------------------------------------------
import os
import json
# ------------------------------------------------------------------------------


def load_config(json_path):
    """
    Load an experiment configuration.
    """
    # Check that config file exists
    assert os.path.exists(json_path), 'Config file not found! {}'.format(json_path)

    # Load config
    with open(json_path, 'r') as f:
        options = json.load(f)

    if not os.path.exists(options['model_dir']):
        create_dir = ''
        while create_dir not in ['y', 'n']:
            create_dir = raw_input(
                'Experiment dir does not exist. Do you want to create it? [y/n]' +
                ' {}'.format(options['model_dir'])
            )

        if create_dir == 'y':
            os.makedirs(options['model_dir'])
            os.makedirs(os.path.join(options['model_dir'], 'valid_samples'))
            os.makedirs(os.path.join(options['model_dir'], 'input_sanity'))
            os.makedirs(os.path.join(options['model_dir'], 'rec_sanity'))
        else:
            exit()

    if not os.path.exists(options['dashboard_dir']):
        create_dir = ''
        while create_dir not in ['y', 'n']:
            create_dir = raw_input(
                'Visualization dir does not exist. Do you want to create it? [y/n]' +
                ' {}'.format(options['dashboard_dir'])
            )

        if create_dir == 'y':
            os.makedirs(options['dashboard_dir'])
            res_cat = open('/u/kamyar/public_html/results/catalog', 'a')
            res_cat.write(options['dashboard_dir'].split('/')[-1] + '\n')
        else:
            exit()

    return options