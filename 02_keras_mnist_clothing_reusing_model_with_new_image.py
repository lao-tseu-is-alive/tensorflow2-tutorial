import glob
import os
import re

# next line is to limit tensorflow verbose output
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
import numpy as np
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt

CONST_MODEL_PATH = 'trained_models/tf2_model_mnist_fashion_2Dense128x10'
CONST_NEW_IMAGES = 'unseen_test_samples/mnist_fashion_categories'
CONST_IMAGE_SIZE = (28, 28)  # MNIST FASHION image sizes
CONST_IMAGE_RESCALE = 1. / 255.0  # Normalisation factor for colors values

class_names = ['t-shirt', 'trouser', 'pullover', 'dress', 'coat',
               'sandal', 'shirt', 'sneaker', 'bag', 'ankle_boot']


def plot_image(predictions_array, true_label, img):
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])

    plt.imshow(img, cmap=plt.cm.binary)

    predicted_image_label = np.argmax(predictions_array)
    if predicted_image_label == true_label:
        color = 'blue'
    else:
        color = 'red'

    plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_image_label.item()],
                                         100 * np.max(predictions_array),
                                         class_names[true_label]),
               color=color)


def plot_value_array(predictions_array, true_label):
    plt.grid(False)
    plt.xticks(range(10))
    plt.yticks([])
    my_plot = plt.bar(range(10), predictions_array, color="#777777")
    plt.ylim([0, 1])
    predicted_label = np.argmax(predictions_array)
    my_plot[predicted_label].set_color('red')
    my_plot[true_label].set_color('blue')
    _ = plt.xticks(range(10), class_names, rotation=45)


if __name__ == '__main__':
    print('# Tensorflow version : {}'.format(tf.__version__))
    print('# TensorFlow 2 using a Model to Classify NEW images of clothing')

    print('# Loading model already fitted.')
    print("### will try to load model from path : {}".format(CONST_MODEL_PATH))
    model = load_model(CONST_MODEL_PATH)
    print('# MAKE PREDICTIONS :')
    print("""
    #   With the model trained, you can use it to make predictions about some images. The model's linear outputs, logits.
    #   Attach a softmax layer to convert the logits to probabilities, which are easier to interpret.  
    #   logits definition : https://developers.google.com/machine-learning/glossary#logits  
        """)
    probability_model = tf.keras.Sequential([model, tf.keras.layers.Softmax()])
    print("""
#   The images from the training set are 28x28 NumPy arrays, with pixel values ranging from 0 to 255. 
#   The labels are an array of integers, ranging from 0 to 9. 
#   These correspond to the class of clothing the image represents:""")

    print("### Now trying model.predict with a brand new set of images !!")
    test_images_path = []
    for image_path in glob.glob('{}/*.jpeg'.format(CONST_NEW_IMAGES)):
        test_images_path.append(image_path)

    test_images_path.sort()
    test_images_init = []
    test_labels = []
    test_filenames = []
    for image_path in test_images_path:
        test_image = tf.keras.preprocessing.image.load_img(image_path,
                                                           color_mode='grayscale',
                                                           target_size=CONST_IMAGE_SIZE,
                                                           interpolation='nearest')
        print('test_image shape : {}'.format(np.shape(test_image)))
        # REMEMBER  TO 'NORMALIZE' YOUR DATA !
        test_image_normalized = np.asarray(test_image) * CONST_IMAGE_RESCALE
        print('test_image_normalized shape : {}'.format(np.shape(test_image_normalized)))
        test_image_normalized_arr = np.expand_dims(test_image_normalized, 0)
        print('test_image_normalized_arr shape : {}'.format(np.shape(test_image_normalized_arr)))
        filename = os.path.basename(image_path)
        filename_without_ext = os.path.splitext(filename)[0]
        image_real_class_name = re.split(r'\d+', filename_without_ext)[0]
        try:
            image_real_class = class_names.index(image_real_class_name)
            predictions_single = probability_model.predict(test_image_normalized_arr)
            res = predictions_single[0]
            predicted_class = np.argmax(predictions_single)
            predicted_class_name = class_names[predicted_class.item()]
            print('# prediction for {} is {} = {:10} {:2.2f} percent confidence'.format(
                filename, predicted_class, predicted_class_name, (100 * res[predicted_class])))
            print(', '.join(['{}: {:2.2f}%'.format(class_names[i], 100 * x) for i, x in enumerate(res)]))
            plt.figure(figsize=(6, 3))
            plt.subplot(1, 2, 1)
            plot_image(res, image_real_class, test_image_normalized)
            plt.subplot(1, 2, 2)
            plot_value_array(res, image_real_class)
            plt.show()

            test_labels.append(image_real_class)
            test_images_init.append(test_image_normalized)
            test_filenames.append(filename)
        except ValueError as e:
            print('WARNING : Image name {} is not in the MNIST fashion classes'.format(filename))
            print('WARNING : Image name {} will not be in the test set !'.format(filename))

    print('test_images_init shape : {}'.format(np.shape(test_images_init)))
    test_images = np.array(test_images_init)
    test_images = test_images / 255.0  # normalize
    print('test_images shape : {}'.format(np.shape(test_images)))
