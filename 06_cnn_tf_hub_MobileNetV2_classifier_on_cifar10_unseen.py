import glob
import os
import re

# next line is to limit tensorflow verbose output
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
from my_tf_lib import images_classification as ic
import time

import PIL.Image as Image
import matplotlib.pylab as plt

import tensorflow as tf
import tensorflow_hub as hub
from tensorflow.keras.models import load_model

CONST_NEW_IMAGES = 'unseen_test_samples/cifar10'
CONST_CIFAR_IMAGE_SIZE = (32, 32)  # CIFAR-10 FASHION original training image sizes
CONST_IMAGE_RESCALE = 1. / 255.0  # Normalisation factor for colors values
CONST_CLASS_NAMES = ic.get_imagenet_classes_with_synonyms('labels/imageNetLabelsWithSyn.txt')
CONST_IMAGE_SIZE = (224, 224)
CONST_BATCH_SIZE = 32


def make_images_predictions_from_model(path_to_images, model, class_list, image_size, normalization_factor):
    if type(model) == str:
        print("### will try to load model from path : {}".format(model))
        model = load_model(model)
        probability_model = tf.keras.Sequential([model, tf.keras.layers.Softmax()])
    else:
        if type(model) == tf.keras.Sequential:
            print("### will use the model as is : {}".format(model.summary()))
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
            # image_real_class = class_list.index(image_real_class_name)
            predictions_single = probability_model.predict(test_image_normalized_arr)
            res = predictions_single[0]
            predicted_class = np.argmax(predictions_single)
            predicted_class_name = class_list[predicted_class.item()]
            # convert from logits to
            probabilities = tf.nn.softmax(predictions_single[0])
            if re.match('.*{}'.format(image_real_class_name), predicted_class_name):
                num_correct_predictions += 1
                print('# ✅ ✅ prediction for {} is CORRECT  {} = "{:10}", with {:2.2f} percent confidence'.format(
                    filename, predicted_class, predicted_class_name, (probabilities[predicted_class])))
            else:
                num_wrong_predictions += 1
                print('# ❌ ❌  prediction for {} is  WRONG   {} = "{:10}", with {:2.2f} percent confidence'.format(
                    filename, predicted_class, predicted_class_name, (probabilities[predicted_class])))

            top_5_values, top_5_indices = tf.nn.top_k(probabilities, k=5)
            for i in range(len(top_5_values)):
                print("[{:4}]:{:6} -->{:2.2f}%".format(top_5_indices[i], class_list[top_5_indices[i]], top_5_values[i]))

        except Exception as e:
            num_ignored_images += 1
            print('WARNING : Exception {} occurred with Image name {}'.format(e, filename))
    print('=' * 80)
    print('{} CORRECT PREDICTIONS, {} WRONG PREDICTIONS'.format(num_correct_predictions, num_wrong_predictions))
    total = num_correct_predictions + num_wrong_predictions
    if total > 0:
        percent_success = (num_correct_predictions / total) * 100
        print('{:2.2f}% percent success !'.format(percent_success))


if __name__ == '__main__':
    print('# Tensorflow version : {}'.format(tf.__version__))
    print('# TensorFlow 2 using as is MobileNetV2 from TensorFlow Hub')
    print('#  more info: https://www.tensorflow.org/tutorials/images/transfer_learning_with_hub')

    classifier_model = "https://tfhub.dev/google/tf2-preview/mobilenet_v2/classification/4"

    classifier = tf.keras.Sequential([
        hub.KerasLayer(classifier_model, input_shape=CONST_IMAGE_SIZE + (3,))
    ])

    print(CONST_CLASS_NAMES)

    make_images_predictions_from_model(
        path_to_images=CONST_NEW_IMAGES,
        image_size=CONST_IMAGE_SIZE,
        class_list=CONST_CLASS_NAMES,
        model=classifier,
        normalization_factor=CONST_IMAGE_RESCALE
    )
