import tensorflow as tf
# from tensorflow.python import pywrap_tensorflow
# import numpy as np
import json
import os, re
import logging

logging.basicConfig(level=logging.INFO)

def calc_num_batches(total_num, batch_size):
    '''Calculates the number of batches.
    total_num: total sample number
    batch_size
    Returns
    number of batches, allowing for remainders.'''

    return total_num // batch_size + int(total_num % batch_size != 0)
