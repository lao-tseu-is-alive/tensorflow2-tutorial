import os

# next line is to limit tensorflow verbose output
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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
    print(' # len of the training label {}'.format(len(train_labels)))
    print(' # stats on the training label {}'.format(pd.DataFrame(train_labels).describe()))

    print(' # shape of the test set {}'.format(test_images.shape))
    print(' # len of the test label {}'.format(len(test_labels)))
    print(' # stats on the training label {}'.format(pd.DataFrame(test_labels).describe()))

    print("""
#   The images are 28x28 NumPy arrays, with pixel values ranging from 0 to 255. 
#   The labels are an array of integers, ranging from 0 to 9. 
#   These correspond to the class of clothing the image represents:""")

    class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                   'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

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
