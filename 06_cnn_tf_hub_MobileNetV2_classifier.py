import os
# next line is to limit tensorflow verbose output
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
import time

import PIL.Image as Image
import matplotlib.pylab as plt

import tensorflow as tf
import tensorflow_hub as hub

CONST_MODEL_PATH = 'trained_models/tf2_model_cnn_tf_hub_MobileNetV2_classifier'
CONST_CLASS_NAMES = ['cats', 'dogs']
CONST_IMAGE_SIZE = (224, 224)
CONST_BATCH_SIZE = 32

if __name__ == '__main__':
    print('# Tensorflow version : {}'.format(tf.__version__))
    print('# TensorFlow 2 using as is MobileNetV2 from TensorFlow Hub')
    print('#  more info: https://www.tensorflow.org/tutorials/images/transfer_learning_with_hub')

    classifier_model = "https://tfhub.dev/google/tf2-preview/mobilenet_v2/classification/4"

    classifier = tf.keras.Sequential([
        hub.KerasLayer(classifier_model, input_shape=CONST_IMAGE_SIZE + (3,), output_shape=[1001])
    ])

    print("""
#   Let's run it on a single image :
    """)
    grace_hopper = tf.keras.utils.get_file(
        'image.jpg',
        'https://storage.googleapis.com/download.tensorflow.org/example_images/grace_hopper.jpg'
    )
    grace_hopper = Image.open(grace_hopper).resize(CONST_IMAGE_SIZE)
    plt.imshow(grace_hopper)
    plt.show()
    grace_hopper = np.array(grace_hopper) / 255.0
    print("#    image shape: {}".format(grace_hopper.shape))
    result = classifier.predict(grace_hopper[np.newaxis, ...])
    print("#    prediction result shape: {}".format(result.shape))
    print("""
#   The result is a 1001 element vector of logits, rating the probability of each class for the image.
#   So the top class ID can be found with argmax:    
    """)
    predicted_class = np.argmax(result[0], axis=-1)
    print("#    predicted_class : {}".format(predicted_class))
    labels_path = tf.keras.utils.get_file(
        'ImageNetLabels.txt',
        'https://storage.googleapis.com/download.tensorflow.org/data/ImageNetLabels.txt')
    imagenet_labels = np.array(open(labels_path).read().splitlines())
    plt.imshow(grace_hopper)
    plt.axis('off')
    predicted_class_name = imagenet_labels[predicted_class]
    _ = plt.title("Prediction: " + predicted_class_name.title())
    plt.show()



