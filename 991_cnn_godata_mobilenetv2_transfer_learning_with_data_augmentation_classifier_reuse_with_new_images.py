import pathlib
import matplotlib.pyplot as plt
import numpy as np
import os

# next line is to limit tensorflow verbose output
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from my_tf_lib import images_classification as ic

from my_tf_lib import images_classification as ic

CONST_MODEL_PATH = 'trained_models/tf2_model_cnn_godata_mobilenetv2_transfer_learning_with_data_augmentation_classifier'
CONST_CLASS_NAMES = ['correspondance', 'facturation', 'photo', 'plan', 'plan_projet', 'plan_situation']
CONST_IMAGE_SIZE = (160, 160)
CONST_BATCH_SIZE = 32

CONST_NEW_IMAGES = 'unseen_test_samples/godata'
# Normalisation is not needed here , because it is integrated in the first layer of model :
# layers.experimental.preprocessing.Rescaling(1. / 255, input_shape=(img_height, img_width, 3)),
CONST_IMAGE_RESCALE = 1.0

CONST_CLASS_NAMES = ['correspondance', 'facturation', 'photo', 'plan', 'plan_projet', 'plan_situation']


if __name__ == '__main__':
    print('# REUSING A SIMPLE CNN model ')
    print(" # Godata categories : \n{}".format(CONST_CLASS_NAMES))
    ic.make_images_predictions_from_model(
        path_to_images=CONST_NEW_IMAGES,
        path_to_model=CONST_MODEL_PATH,
        class_list=CONST_CLASS_NAMES,
        image_size=CONST_IMAGE_SIZE,
        normalization_factor=CONST_IMAGE_RESCALE
    )
