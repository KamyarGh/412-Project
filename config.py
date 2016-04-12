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
            os.makedirs(os.path.join(options['model_dir']), 'valid_samples')
        else:
            exit()

    return options