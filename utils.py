"""
    utils.py
    ~~~~~~~~
    Author: Lluis Castrejon
    General utilities.
"""
# Imports ----------------------------------------------------------------------
import sys
import logging
import logging.handlers
# ------------------------------------------------------------------------------

# Globals ----------------------------------------------------------------------
LINE_LEN = 50
# ------------------------------------------------------------------------------


def print_header(header, log):
    """
    Print a header.
    """
    n_dashes = LINE_LEN - len(header) - 2
    line = (n_dashes/2)*'-' + ' {} '.format(header)
    if n_dashes % 2 == 0:
        line += (n_dashes/2)*'-'
    else:
        line += ((n_dashes/2) - 1)*'-'

    log.info(line)


def print_footer(log):
    """
    Print a footer.
    """
    log.info(LINE_LEN*'-')


def print_options(options, log):
    """
    Print model options.
    """
    print_header('Options', log)
    for kk, vv in options.iteritems():
        log.info('{}: {}'.format(kk, vv))
    print_footer(log)


def get_logger(out_path):
    """
    Get a logger to write to both to a file and stdout.

    Returns:
        logger
    """
    log = logging.getLogger('VAE')
    log.setLevel(logging.INFO)
    format = logging.Formatter("[%(asctime)s] %(message)s")

    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(format)
    log.addHandler(ch)

    fh = logging.handlers.RotatingFileHandler(out_path, mode='w')
    fh.setFormatter(format)
    log.addHandler(fh)

    return log
