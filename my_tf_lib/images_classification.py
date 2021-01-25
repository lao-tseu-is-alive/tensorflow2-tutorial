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
import matplotlib.pyplot as plt


def make_images_predictions_from_model(path_to_images, path_to_model, class_list, image_size, normalization_factor):
    print('# Tensorflow version : {}'.format(tf.__version__))
    print('# Loading model already fitted.')
    print("### will try to load model from path : {}".format(path_to_model))
    model = load_model(path_to_model)
    probability_model = tf.keras.Sequential([model, tf.keras.layers.Softmax()])

    print("### Now trying model.predict with a brand new set of images !!")
    test_images_path = []
    for image_path in glob.glob('{}/*.jpeg'.format(path_to_images)):
        test_images_path.append(image_path)
    test_images_path.sort()
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
            predicted_class = np.argmax(predictions_single)
            predicted_class_name = class_list[predicted_class.item()]
            if predicted_class == image_real_class:
                num_correct_predictions += 1
                print('# ✅ ✅ prediction for {} is CORRECT  {} = {:10} {:2.2f} percent confidence'.format(
                    filename, predicted_class, predicted_class_name, (100 * res[predicted_class])))
            else:
                num_wrong_predictions += 1
                print('# ❌ ❌  prediction for {} is  WRONG   {} = {:10} {:2.2f} percent confidence'.format(
                    filename, predicted_class, predicted_class_name, (100 * res[predicted_class])))
            print(', '.join(['{}: {:2.2f}%'.format(class_list[i], 100 * x) for i, x in enumerate(res)]))

        except ValueError as e:
            num_ignored_images += 1
            print('WARNING : Image name {} is not in the CIFAR-10 fashion classes'.format(filename))
            print('WARNING : Image name {} will not be in the test set !'.format(filename))
            print('Exception : {}'.format(e))
    print('=' * 80)
    print('{} CORRECT PREDICTIONS, {} WRONG PREDICTIONS'.format(num_correct_predictions, num_wrong_predictions))
    total = num_correct_predictions + num_wrong_predictions
    percent_success = (num_correct_predictions / total) * 100
    print('{:2.2f}% percent success !'.format(percent_success))


def show_n_images_category_from_path(class_names, base_path, num_images_by_category=5):
    """
    will display a figure of 10x10 inch with a row for every category in class_names and num_images_by_category columns
    :param class_names: list of different categories names (directory names)
    :param base_path: the base path of the different categories of images, one dir by category
    :param num_images_by_category: the number of columns with the first images from the category
    :return: None
    """
    data_dir = pathlib.Path(base_path)
    plt.figure(figsize=(10, 10))
    num_categories = len(class_names)
    total = num_categories * num_images_by_category
    previous_category_index = -1
    for i in range(total):
        cat_index = i // num_categories
        img_index = i % num_categories
        if previous_category_index != cat_index:
            files = list(data_dir.glob('{}/*'.format(class_names[cat_index])))
            previous_category_index = cat_index
        plt.subplot(num_categories, num_images_by_category, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        img = PIL.Image.open(str(files[img_index]))
        label = '{}, {}'.format(class_names[cat_index], img.size)
        plt.imshow(img, cmap=plt.cm.binary)
        plt.xlabel(label)
    plt.show()


def show_n_images_from_dataset(dataset, num_images_by_category=3):
    """
    will display a figure of 10x10 inch with a row for every category in class_names and num_images_by_category columns
    :param dataset: a valid tf.data.Dataset
    :param num_images_by_category:
    :return:
    """
    class_names = dataset.class_names
    plt.figure(figsize=(10, 10))
    num_categories = len(class_names)
    total = num_categories * num_images_by_category
    for images, labels in dataset.take(1):
        for i in range(total):
            cat_index = i // num_categories
            img_index = i % num_categories
            ax = plt.subplot(num_categories, num_images_by_category, i + 1)
            plt.imshow(images[i].numpy().astype("uint8"))
            plt.title(class_names[labels[i]])
            plt.axis("off")
    plt.show()
