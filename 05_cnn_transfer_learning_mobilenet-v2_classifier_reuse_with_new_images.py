import glob
import os
import re
import pathlib
import PIL

# next line is to limit tensorflow verbose output
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
import numpy as np
from tensorflow.keras.models import load_model


CONST_MODEL_PATH = 'trained_models/tf2_model_cnn_transfer_learning_mobilenet-v2_with_data_augmentation_classifier'
CONST_CLASS_NAMES = ['cats', 'dogs']
CONST_IMAGE_SIZE = (160, 160)
CONST_NEW_IMAGES = 'unseen_test_samples/cats_dogs'
# Normalisation is not needed here , because it is integrated in the first layer of model :
# layers.experimental.preprocessing.Rescaling(1. / 255, input_shape=(img_height, img_width, 3)),
CONST_IMAGE_RESCALE = 1.0


def make_images_predictions_from_retrained_model(path_to_images, path_to_model, class_list, image_size,
                                                 normalization_factor):
    print('# Tensorflow version : {}'.format(tf.__version__))
    print('# Loading model already fitted.')
    print("### will try to load model from path : {}".format(path_to_model))
    model = load_model(path_to_model)
    # probability_model = tf.keras.Sequential([model, tf.keras.layers.Softmax()])
    probability_model = model
    print("### Now trying model.predict with a brand new set of images !")
    test_images_path = []
    for image_path in glob.glob('{}/*.jpeg'.format(path_to_images)):
        test_images_path.append(image_path)
    test_images_path.sort()
    print("# found {} images in path : {} ".format(len(test_images_path), path_to_images))
    num_correct_predictions = 0
    num_wrong_predictions = 0
    num_ignored_images = 0
    for image_path in test_images_path:
        test_image = tf.keras.preprocessing.image.load_img(image_path,
                                                           color_mode='rgb',
                                                           target_size=image_size,
                                                           interpolation='nearest')
        # print('test_image shape : {}'.format(np.shape(test_image)))
        # REMEMBER  TO 'NORMALIZE' YOUR DATA !
        test_image_normalized = np.asarray(test_image) * normalization_factor
        # print('test_image_normalized shape : {}'.format(np.shape(test_image_normalized)))
        test_image_normalized_arr = np.expand_dims(test_image_normalized, 0)
        # print('test_image_normalized_arr shape : {}'.format(np.shape(test_image_normalized_arr)))
        filename = os.path.basename(image_path)
        filename_without_ext = os.path.splitext(filename)[0]
        image_real_class_name = re.split(r'\d+', filename_without_ext)[0]
        try:
            image_real_class = class_list.index(image_real_class_name)
            predictions_single = probability_model.predict(test_image_normalized_arr)
            res = predictions_single[0]
            # Apply a sigmoid since our model returns logits
            predictions = tf.nn.sigmoid(res)
            predictions = tf.where(predictions < 0.5, 0, 1)
            # print('Predictions:\n', predictions.numpy())
            predicted_class = predictions.numpy()[0]
            predicted_class_name = class_list[predicted_class.item()]
            if predicted_class == image_real_class:
                num_correct_predictions += 1
                print('# ✅ ✅ prediction for {} is CORRECT  {} = {:10} '.format(
                    filename, predicted_class, predicted_class_name))
            else:
                num_wrong_predictions += 1
                print('# ❌ ❌  prediction for {} is  WRONG   {} = {:10}'.format(
                    filename, predicted_class, predicted_class_name))

        except ValueError as e:
            num_ignored_images += 1
            print('WARNING : Image name {} is not in the given categories'.format(filename))
            print('WARNING : Image name {} will not be in the test set !'.format(filename))
            print('Exception : {}'.format(e))
    print('=' * 80)
    print('{} CORRECT PREDICTIONS, {} WRONG PREDICTIONS'.format(num_correct_predictions, num_wrong_predictions))
    total = num_correct_predictions + num_wrong_predictions
    if total > 0:
        percent_success = (num_correct_predictions / total) * 100
        print('{:2.2f}% percent success !'.format(percent_success))


if __name__ == '__main__':
    print('# REUSING A Mobile Net V2  model modified with transfer learning on cats and dogs ')
    print(" # Cats and dogs categories are obviously : \n{}".format(CONST_CLASS_NAMES))
    make_images_predictions_from_retrained_model(
        path_to_images=CONST_NEW_IMAGES,
        path_to_model=CONST_MODEL_PATH,
        class_list=CONST_CLASS_NAMES,
        image_size=CONST_IMAGE_SIZE,
        normalization_factor=CONST_IMAGE_RESCALE
    )
