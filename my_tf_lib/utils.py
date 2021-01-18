import os
# next line is to limit tensorflow verbose output
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf


def check_gpu():
    """ returns True if a GPU was found by Tensorflow, False if not """
    try:
        if tf.test.is_built_with_cuda():
            print('# Cool, tensorflow is compiled with cuda (GPU) support')
            list_physical_devices = tf.config.list_physical_devices('GPU')
            print('# Here is the list of GPU found by tensorflow :')
            print(list_physical_devices)
            if len(list_physical_devices) > 0:
                return True
            else:
                return False
        else:
            print('# No way to check if gpu is present because your tensorflow is not compiled with cuda')
            return False
    except Exception as e:
        print("Exception occurred : {}".format(e))
        return False
