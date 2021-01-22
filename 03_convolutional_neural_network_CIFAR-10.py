import os

# next line is to limit tensorflow verbose output
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt

if __name__ == '__main__':
    print('# Tensorflow version : {}'.format(tf.__version__))
    print('# TensorFlow 2 Convolutional Neural Network (CNN) CIFAR-10 image classification')
    print('#  more info on this Tensorflow example: https://www.tensorflow.org/tutorials/images/cnn')

    print('# Loading and prepare the CIFAR-10 dataset (https://www.cs.toronto.edu/~kriz/cifar.html)')
    (train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()
    print('# Normalize pixel values from the samples to floats between 0 and 1')
    train_images, test_images = train_images / 255.0, test_images / 255.0

    class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                   'dog', 'frog', 'horse', 'ship', 'truck']
    print(" # CIFAR categories : \n{}".format(class_names))

    plt.figure(figsize=(10, 10))
    for i in range(25):
        plt.subplot(5, 5, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(train_images[i], cmap=plt.cm.binary)
        # The CIFAR labels happen to be arrays,
        # which is why you need the extra index
        plt.xlabel(class_names[train_labels[i][0]])
    plt.show()

    print('# building the CNN model by  using a common pattern: a stack of Conv2D and MaxPooling2D layers.')
    print('#  more info on Conv2: https://www.tensorflow.org/api_docs/python/tf/keras/layers/Conv2D')
    print('#  more info on MaxPooling2D: https://www.tensorflow.org/api_docs/python/tf/keras/layers/MaxPool2D')

    print("""
# As input, a CNN takes tensors of shape (image_height, image_width, color_channels), ignoring the batch size.
# If you are new to these dimensions, color_channels refers to (R,G,B).
# In this example, you will configure our CNN to process inputs of shape:(32, 32, 3),it's the format of CIFAR images.
# You can do this by passing the argument "input_shape" to our first layer.    
    """)
    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    print('Here is the model.summary() : {}'.format(model.summary()))

    print(""""
#   Above, you can see that the output of every Conv2D and MaxPooling2D layer is 
#   a 3D tensor of shape (height, width, channels).
#   The width and height dimensions tend to shrink as you go deeper in the network. 
#   The number of output channels for each Conv2D layer is controlled by the first argument (e.g., 32 or 64).
#   Typically, as the width and height shrink, you can afford (computationally) to add more output channels
#   in each Conv2D layer.    
    """)
    print('# Add Dense layers on top')
    print("""
#   To complete our model, you will feed the last output tensor from the convolutional base (of shape (4, 4, 64))
#   into one or more Dense layers to perform classification. 
#   Dense layers take vectors as input (which are 1D), while the current output is a 3D tensor. 
#   First, you will flatten (or unroll) the 3D output to 1D, then add one or more Dense layers on top. 
#   CIFAR has 10 output classes, so you use a final Dense layer with 10 outputs.
    """)
    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(10))

    print('Here is the model.summary() : {}'.format(model.summary()))
    print("""
#   As you can see, our (4, 4, 64) outputs were flattened 
#   into vectors of shape (1024) before going through two Dense layers.
    """)

    print('# Compile and train the model')
    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])

    history = model.fit(train_images, train_labels, epochs=10,
                        validation_data=(test_images, test_labels))

    print('# Evaluate the model')
    plt.plot(history.history['accuracy'], label='accuracy')
    plt.plot(history.history['val_accuracy'], label='val_accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.ylim([0.5, 1])
    plt.legend(loc='lower right')

    test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
    print('test_acc : {}'.format(test_acc))
    print(""""
#   Our simple CNN has achieved a test accuracy of over 70%. Not bad for a few lines of code! 
#   For another CNN style, see an example using the Keras subclassing API and a tf.GradientTape here :
    https://www.tensorflow.org/tutorials/quickstart/advanced
    """)
