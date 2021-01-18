import os
# next line is to limit tensorflow verbose output
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf

if __name__ == '__main__':
    print('# Tensorflow version : {}'.format(tf.__version__))
    print('# TensorFlow 2 quickstart for beginners')
    print('#  more info: https://www.tensorflow.org/tutorials/quickstart/beginner')

    print('# Loading and prepare the MNIST dataset (http://yann.lecun.com/exdb/mnist/)')
    mnist = tf.keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    print('# converting the samples to normalized floats')
    x_train, x_test = x_train / 255.0, x_test / 255.0

    print('# building the tf.keras.Sequential model by stacking layers.')
    print('#  more info : https://www.tensorflow.org/api_docs/python/tf/keras/Sequential')
    model = tf.keras.models.Sequential([
      tf.keras.layers.Flatten(input_shape=(28, 28)),
      tf.keras.layers.Dense(128, activation='relu'),
      tf.keras.layers.Dropout(0.2),
      tf.keras.layers.Dense(10)
    ])

    print('For each example the model returns a vector of "logits" or "log-odds" scores, one for each class.')
    predictions = model(x_train[:1]).numpy()
    print('predictions = {}'.format(predictions))
    print('The tf.nn.softmax function converts these logits to "probabilities" for each class:')
    print('tf.nn.softmax(predictions).numpy() : {}'.format(tf.nn.softmax(predictions).numpy()))

    print("""Note: It is possible to bake this tf.nn.softmax in as the activation function
      for the last layer of the network. While this can make the model output more directly interpretable,
      this approach is discouraged as it's impossible to provide an exact and numerically stable loss calculation 
      for all models when using a softmax output.""")
    print("""The losses.SparseCategoricalCrossentropy loss takes a vector of logits
     and a True index and returns a scalar loss for each example.""")
    print('# more info: https://www.tensorflow.org/api_docs/python/tf/keras/losses/SparseCategoricalCrossentropy')
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    print("""This loss is equal to the negative log probability of the true class: 
    It is zero if the model is sure of the correct class. """)
    print("""This untrained model gives probabilities close to random (1/10 for each class),
     so the initial loss should be close to -tf.log(1/10) ~= 2.3.""")
    print('loss_fn(y_train[:1], predictions).numpy() : {} '.format(loss_fn(y_train[:1], predictions).numpy()))

    print('# compiling the model :')
    model.compile(optimizer='adam', loss=loss_fn, metrics=['accuracy'])
    print('# and use Model.fit method to adjusts the model parameters to minimize the loss:')
    print('# more info : https://www.tensorflow.org/api_docs/python/tf/keras/Model#fit')
    model.fit(x_train, y_train, epochs=5)

    print('The Model.evaluate method checks the models performance, usually on a "Validation-set" or "Test-set".')
    print('# more info: https://www.tensorflow.org/api_docs/python/tf/keras/Model#evaluate')
    print('# Validation-set: https://developers.google.com/machine-learning/glossary#validation-set')
    print('model.evaluate(x_test,  y_test, verbose=2) : {}'.format(model.evaluate(x_test,  y_test, verbose=2)))
    print('CONGRATULATIONS ! Your image classifier is now trained to ~98% accuracy on this MNIST dataset.')

