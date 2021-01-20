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
CONST_IMAGE_RESCALE = 1. / 255  # Normalisation factor for colors values

class_names = ['t-shirt', 'trouser', 'pullover', 'dress', 'coat',
               'sandal', 'shirt', 'sneaker', 'bag', 'ankle_boot']


def plot_image(index, predictions_array, true_label, img):
    true_label, img = true_label[index], img[index]
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


def plot_value_array(index, predictions_array, true_label):
    true_label = true_label[index]
    plt.grid(False)
    plt.xticks(range(10))
    plt.yticks([])
    my_plot = plt.bar(range(10), predictions_array, color="#777777")
    plt.ylim([0, 1])
    predicted_label = np.argmax(predictions_array)
    my_plot[predicted_label].set_color('red')
    my_plot[true_label].set_color('blue')


if __name__ == '__main__':
    print('# Tensorflow version : {}'.format(tf.__version__))
    print('# TensorFlow 2 using a Model to Classify NEW images of clothing')

    print('# Loading model already fitted.')
    print("### will try to load model from path : {}".format(CONST_MODEL_PATH))
    model = load_model(CONST_MODEL_PATH)
    print("""
#   The images from the training set are 28x28 NumPy arrays, with pixel values ranging from 0 to 255. 
#   The labels are an array of integers, ranging from 0 to 9. 
#   These correspond to the class of clothing the image represents:""")

    print("### Now trying model.predict with a brand new set of images !!")
    test_images_init = []
    test_labels = []
    test_filenames = []
    for image_path in glob.glob('{}/*.jpeg'.format(CONST_NEW_IMAGES)):
        test_image = tf.keras.preprocessing.image.load_img(image_path,
                                                           color_mode='grayscale',
                                                           target_size=CONST_IMAGE_SIZE,
                                                           interpolation='bilinear')
        print('test_image shape : {}'.format(np.shape(test_image)))
        # REMEMBER  TO 'NORMALIZE' YOUR DATA !
        test_image_normalized = tf.keras.preprocessing.image.img_to_array(test_image)
        print('test_image_normalized shape : {}'.format(np.shape(test_image_normalized)))
        filename = os.path.basename(image_path)
        filename_without_ext = os.path.splitext(filename)[0]
        image_real_class = re.split(r'\d+', filename_without_ext)[0]
        try:
            test_labels.append(class_names.index(image_real_class))
            test_images_init.append(test_image_normalized)
            test_filenames.append(filename)
        except ValueError as e:
            print('WARNING : Image name {} is not in the MNIST fashion classes'.format(filename))
            print('WARNING : Image name {} will not be in the test set !'.format(filename))

    print('test_images_init shape : {}'.format(np.shape(test_images_init)))
    test_images = np.array(test_images_init)
    test_images = test_images / 255.0  # normalize
    print('test_images shape : {}'.format(np.shape(test_images)))
    print('# MAKE PREDICTIONS :')
    print("""
#   With the model trained, you can use it to make predictions about some images. The model's linear outputs, logits.
#   Attach a softmax layer to convert the logits to probabilities, which are easier to interpret.  
#   logits definition : https://developers.google.com/machine-learning/glossary#logits  
    """)
    probability_model = tf.keras.Sequential([model, tf.keras.layers.Softmax()])
    predictions = probability_model.predict(test_images)
    print("""
#   At this point the model has predicted the label for each image in the testing set. 
#   Let's take a look at the first prediction:
    """)
    print('# prediction for first row of test set : {}'.format(predictions[0]))
    print("""
#   A prediction is an array of 10 numbers. 
#   They represent the model's "confidence" that the image corresponds to each of the 10 different articles of clothing.
#   You can see which label has the highest confidence value by using the numpy argmax:
    """)
    print('predicted class {}, corresponding to a {}'.format(
        np.argmax(predictions[0]), class_names[np.argmax(predictions[0]).item()]))
    print('real class {}, corresponding to a {}'.format(
        test_labels[0], class_names[test_labels[0]]))

    print('# VERIFY SOME PREDICTIONS')
    print("""
#   With the model trained, you can use it to make predictions about some images.
#   Let's look at the 0th image, predictions, and prediction array.
#   Correct prediction labels are blue and incorrect prediction labels are red. 
#   The number gives the percentage (out of 100) for the predicted label.
    """)
    i = 0
    plt.figure(figsize=(6, 3))
    plt.subplot(1, 2, 1)
    plot_image(i, predictions[i], test_labels, test_images)
    plt.subplot(1, 2, 2)
    plot_value_array(i, predictions[i], test_labels)
    plt.show()
    print('# what about the 12th image ?')
    i = 12
    plt.figure(figsize=(6, 3))
    plt.subplot(1, 2, 1)
    plot_image(i, predictions[i], test_labels, test_images)
    plt.subplot(1, 2, 2)
    plot_value_array(i, predictions[i], test_labels)
    plt.show()
    print("""
#   Let's plot several images with their predictions. 
#   Note that the model can be wrong even when very confident.
""")
    # Plot the first X test images, their predicted labels, and the true labels.
    # Color correct predictions in blue and incorrect predictions in red.
    num_rows = 10
    num_cols = 3
    num_images = num_rows * num_cols
    plt.figure(figsize=(2 * 2 * num_cols, 2 * num_rows))
    for i in range(num_images):
        plt.subplot(num_rows, 2 * num_cols, 2 * i + 1)
        plot_image(i, predictions[i], test_labels, test_images)
        plt.subplot(num_rows, 2 * num_cols, 2 * i + 2)
        plot_value_array(i, predictions[i], test_labels)
    plt.tight_layout()
    plt.show()
    print('predictions shape : {}'.format(np.shape(predictions)))
    for j in range(len(predictions)):

        iterator = np.nditer(predictions[j], flags=['f_index'])
        print("\n### %%% Predictions for {} ###".format(test_filenames[j]))
        predicted_label = np.argmax(predictions[j]).item()
        res = []
        for i in iterator:
            res.append(" {} {:2.2f}%".format(class_names[iterator.index], i * 100))
        print('#[{}]#'.format(','.join(res)))
        if test_labels[j] == predicted_label:
            print("### ✔ ✔  Predicted label for {:10} is CORRECT : {:10} {:2.2f} percent confidence".format(
                test_filenames[j], class_names[predicted_label], (100 * predictions[j][predicted_label])))
        else:
            print("### ⚠ ⚠  Predicted label for {:10} is WRONG : {:10} {:2.2f} percent confidence".format(
                test_filenames[j], class_names[predicted_label], (100 * predictions[j][predicted_label])))
        # print(classes[test_predicted_index])
