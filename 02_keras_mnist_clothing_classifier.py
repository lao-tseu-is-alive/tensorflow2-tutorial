import os

# next line is to limit tensorflow verbose output
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

CONST_MODEL_PATH = 'trained_models/tf2_model_mnist_fashion_2Dense128x10'


def plot_image(index, predictions_array, true_label, img):
    true_label, img = true_label[index], img[index]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])

    plt.imshow(img, cmap=plt.cm.binary)

    predicted_label = np.argmax(predictions_array)
    if predicted_label == true_label:
        color = 'blue'
    else:
        color = 'red'

    plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label.item()],
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
    print('# TensorFlow 2 Basic classification: Classify images of clothing')
    print('#  more info: https://www.tensorflow.org/tutorials/keras/classification')

    print('# Loading and prepare the Fashion MNIST dataset.')
    print("""
    # 60,000 images are used to train the network and 
    # 10,000 images to evaluate how accurately the network learned to classify images""")
    print(' # more info : https://github.com/zalandoresearch/fashion-mnist')
    fashion_mnist = tf.keras.datasets.fashion_mnist
    (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
    print("""
 #   Loading the dataset returns four NumPy arrays:
 #   - The train_images and train_labels arrays are the training set, the data the model uses to learn.
 #   - The model is tested against the test set, the test_images, and test_labels arrays.""")

    print(' # shape of the training set {}'.format(train_images.shape))
    print(' # type of the training set {}'.format(type(train_images)))
    print(' # len of the training label {}'.format(len(train_labels)))
    print(' # stats on the training label {}'.format(pd.DataFrame(train_labels).describe()))

    print(' # shape of the test set {}'.format(test_images.shape))
    print(' # len of the test label {}'.format(len(test_labels)))
    print(' # stats on the training label {}'.format(pd.DataFrame(test_labels).describe()))

    print("""
#   The images are 28x28 NumPy arrays, with pixel values ranging from 0 to 255. 
#   The labels are an array of integers, ranging from 0 to 9. 
#   These correspond to the class of clothing the image represents:""")

    class_names = ['T-shirt', 'Trouser', 'Pullover', 'Dress', 'Coat',
                   'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle_boot']

    print("# Let's have a look at first image from training set")
    plt.figure()
    plt.imshow(train_images[0])
    plt.colorbar()
    plt.grid(False)
    plt.show()
    print('# we can see that the pixel values fall in the range of 0 to 255')
    print('# the data must be preprocessed before training the network.')
    print('# we need to "normalize" the images in the training & testing sets')
    train_images = train_images / 255.0
    test_images = test_images / 255.0

    print("""
#   To verify that the data is in the correct format
#   and that you're ready to build and train the network,
#   let's display the first 25 images from the training set and 
#   display the class name below each image.""")
    plt.figure(figsize=(10, 10))
    for i in range(25):
        plt.subplot(5, 5, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(train_images[i], cmap=plt.cm.binary)
        plt.xlabel(class_names[train_labels[i]])
    plt.show()

    print('# BUILDING THE MODEL :')
    print("""
#   First let's setup the layers:
#   The basic building block of a neural network is the layer.
#   Layers extract representations from the data fed into them. 
#   Hopefully, these representations are meaningful for the problem at hand.

#   Most of deep learning consists of chaining together simple layers. 
#   Most layers, such as tf.keras.layers.Dense, have parameters that are learned during training.
#   more info : https://www.tensorflow.org/api_docs/python/tf/keras/layers/Dense
    """)

    model = tf.keras.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(10)
    ])

    print("""
#   The first layer in this network, tf.keras.layers.Flatten,
#   transforms the format of the images from a two-dimensional array (of 28 by 28 pixels)
#   to a one-dimensional array (of 28 * 28 = 784 pixels).
#   Think of this layer as unstacking rows of pixels in the image and lining them up.
#   This layer has no parameters to learn; it only reformats the data. 
#   more info : https://www.tensorflow.org/api_docs/python/tf/keras/layers/Flatten   
    """)
    print(""""
#   After the pixels are flattened, the network consists of a sequence of two tf.keras.layers.Dense layers.
#   These are densely connected, or fully connected, neural layers.
#   The first Dense layer has 128 nodes (or neurons). 
#   The second (and last) layer returns a logits array with length of 10. 
#   Each node contains a score that indicates the current image belongs to one of the 10 classes.  
#   more info : https://www.tensorflow.org/api_docs/python/tf/keras/layers/Dense  
    """)

    print("""
#   Ok we are ready to compile the previous model
#   Before the model is ready for training, it needs a few more settings.
#   These are added during the model's compile step:
#
#   - Choose a "Loss function" : This one measures how accurate the model is during training.
#                           You want to minimize this function to "steer" the model in the right direction.
#   - Choose an "Optimizer" : This is how the model is updated based on the data it sees and its loss function.
#   - Choose the "Metrics" :  Used to monitor the training and testing steps. 
#           The following example uses accuracy, the fraction of the images that are correctly classified.
#   more info : https://www.tensorflow.org/api_docs/python/tf/keras/Sequential#compile
    """)
    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])

    print('# TRAINING THE MODEL :')
    print(""""
#   Training the neural network model requires the following steps:
# 
#       1) Feed the training data to the model. here, the training data is in the train_images and train_labels arrays.
#       2) The model learns to associate images and labels.
#       3) Next we "EVALUATE" the model, by asking the model to make predictions with the test set (test_images).
#       4) Verify that the predictions match the labels from the test_labels array.    
    """)

    print('# To start training, call the model.fit method—so called because it "fits" the model to the training data:')
    print('# more info : https://www.tensorflow.org/api_docs/python/tf/keras/Sequential#fit')
    model.fit(train_images, train_labels, epochs=10)

    print('# EVALUATING THE MODEL :')
    test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)

    print('\nTest accuracy:', test_acc)
    print("""
#   It turns out that the accuracy on the test dataset is a little less than the accuracy on the training dataset.
#   This gap between training accuracy and test accuracy represents overfitting.
#   Overfitting happens when a machine learning model performs worse on new, previously unseen inputs 
#   than it does on the training data. An overfitted model "memorizes" the noise and details in the training dataset
#   to a point where it negatively impacts the performance of the model on the new data. 
#   For more information, see the following:
#   Demonstrate overfitting : https://www.tensorflow.org/tutorials/keras/overfit_and_underfit#demonstrate_overfitting
#   Strategies to prevent overfitting : 
    https://www.tensorflow.org/tutorials/keras/overfit_and_underfit#strategies_to_prevent_overfitting
    """)

    print('# SAVING THE MODEL FOR LATER USE')
    print("### Now will save model to path : {}".format(CONST_MODEL_PATH))
    tf.saved_model.save(model, CONST_MODEL_PATH)

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
    num_rows = 5
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
