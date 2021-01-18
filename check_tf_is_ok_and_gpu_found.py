import os
# next line is to limit tensorflow verbose output
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
from my_tf_lib import utils as u

if __name__ == '__main__':
    print('# Tensorflow version : {}'.format(tf.__version__))
    if u.check_gpu():
        print('YES, you have a cool GPU card ready to rock TF code !')
    else:
        print('NO, sorry but no GPU card was found, TF will be slower !')
