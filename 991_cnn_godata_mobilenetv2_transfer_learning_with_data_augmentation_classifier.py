import pathlib
import matplotlib.pyplot as plt
import numpy as np
import os

# next line is to limit tensorflow verbose output
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential

# inspired by https://www.tensorflow.org/tutorials/images/transfer_learning

from my_tf_lib import images_classification as ic

CONST_MODEL_PATH = 'trained_models/tf2_model_cnn_godata_mobilenetv2_transfer_learning_with_data_augmentation_classifier'
CONST_CLASS_NAMES = ['correspondance', 'facturation', 'photo', 'plan', 'plan_projet', 'plan_situation']
CONST_IMAGE_SIZE = (160, 160)
CONST_BATCH_SIZE = 32

base_path = '/home/cgil/PycharmProjects/tensorflow2-tutorial/godata_resized'
data_dir = pathlib.Path(base_path)
image_count = len(list(data_dir.glob('*/*.jpeg')))
print('Total number of images : {}'.format(image_count))
ic.show_n_images_category_from_path(CONST_CLASS_NAMES, base_path)
print('# creating the tf.data.Dataset from disk')

train_dataset = tf.keras.preprocessing.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=CONST_IMAGE_SIZE,
    batch_size=CONST_BATCH_SIZE)
validation_dataset = tf.keras.preprocessing.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=CONST_IMAGE_SIZE,
    batch_size=CONST_BATCH_SIZE)

print("""
#   You can find the class names in the class_names attribute on these datasets. 
#   These correspond to the directory names in alphabetical order.
""")
class_names = train_dataset.class_names
print(class_names)
CONST_CLASS_NAMES.sort()
print(CONST_CLASS_NAMES)
ic.show_n_images_from_dataset(train_dataset)
print('# Configure the dataset for performance')
AUTOTUNE = tf.data.AUTOTUNE
train_dataset = train_dataset.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
validation_dataset = validation_dataset.cache().prefetch(buffer_size=AUTOTUNE)

print("""
#   When you don't have a large image dataset, it's a good practice to artificially introduce sample diversity
#   by applying random, yet realistic, transformations to the training images, such as rotation and horizontal flipping.
#   This helps expose the model to different aspects of the training data and reduce overfitting. 
#   You can learn more about data augmentation in this tutorial:
#   https://www.tensorflow.org/tutorials/images/data_augmentation
""")
data_augmentation = tf.keras.Sequential([
    tf.keras.layers.experimental.preprocessing.RandomFlip('horizontal'),
    tf.keras.layers.experimental.preprocessing.RandomRotation(0.2),
])

preprocess_input = tf.keras.applications.mobilenet_v2.preprocess_input
# Note: Alternatively, you could rescale pixel values from [0,255] to [-1, 1] using a Rescaling layer.
# normalization_layer = tf.keras.layers.experimental.preprocessing.Rescaling(1. / 127.5, offset=-1)

num_classes = len(class_names)

img_height, img_width = CONST_IMAGE_SIZE
# Create the base model from the pre-trained model MobileNet V2
IMG_SHAPE = CONST_IMAGE_SIZE + (3,)
base_model = tf.keras.applications.MobileNetV2(input_shape=IMG_SHAPE,
                                               include_top=False,
                                               weights='imagenet')
image_batch, label_batch = next(iter(train_dataset))
feature_batch = base_model(image_batch)
print('#    feature_batch.shape : {}'.format(feature_batch.shape))

base_model.trainable = False

# Let's take a look at the base model architecture
print('#    base_model.summary :')
base_model.summary()

global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
feature_batch_average = global_average_layer(feature_batch)
print('#    feature_batch_average.shape : {}'.format(feature_batch_average.shape))

prediction_layer = tf.keras.layers.Dense(num_classes, activation='relu')
prediction_batch = prediction_layer(feature_batch_average)
print('#    prediction_batch.shape : {}'.format(prediction_batch.shape))

print('# IMG_SHAPE is {} , should be (160, 160, 3)'.format(IMG_SHAPE))

inputs = tf.keras.Input(shape=(160, 160, 3))
x = data_augmentation(inputs)
x = preprocess_input(x)
x = base_model(x, training=False)
x = global_average_layer(x)
x = tf.keras.layers.Dropout(0.2)(x)
outputs = prediction_layer(x)
model = tf.keras.Model(inputs, outputs)

# model = Sequential([
#     layers.experimental.preprocessing.Rescaling(1. / 255, input_shape=(img_height, img_width, 3)),
#     layers.Conv2D(16, 3, padding='same', activation='relu'),
#     layers.MaxPooling2D(),
#     layers.Conv2D(32, 3, padding='same', activation='relu'),
#     layers.MaxPooling2D(),
#     layers.Conv2D(64, 3, padding='same', activation='relu'),
#     layers.MaxPooling2D(),
#     layers.Flatten(),
#     layers.Dense(128, activation='relu'),
#     layers.Dense(num_classes)
# ])

base_learning_rate = 0.0001
model.compile(optimizer=tf.keras.optimizers.Adam(lr=base_learning_rate),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

print('#    model.summary : {}'.format(model.summary()))

print('#    len(model.trainable_variables : {}'.format(len(model.trainable_variables)))

initial_epochs = 10
loss0, accuracy0 = model.evaluate(validation_dataset)
print("#    initial loss: {:.2f}".format(loss0))
print("#    initial accuracy: {:.2f}".format(accuracy0))

history = model.fit(
    train_dataset,
    validation_data=validation_dataset,
    epochs=initial_epochs
)
# let's see how we are doing for the first round
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

plt.figure(figsize=(8, 8))
plt.subplot(2, 1, 1)
plt.plot(acc, label='Training Accuracy')
plt.plot(val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.ylabel('Accuracy')
plt.ylim([min(plt.ylim()), 1])
plt.title('Training and Validation Accuracy')

plt.subplot(2, 1, 2)
plt.plot(loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.ylabel('Cross Entropy')
plt.ylim([0, 1.0])
plt.title('Training and Validation Loss')
plt.xlabel('epoch')
plt.show()

print(" FINE TUNING :")

base_model.trainable = True
# Let's take a look to see how many layers are in the base model
print("Number of layers in the base model: ", len(base_model.layers))

# Fine-tune from this layer onwards
fine_tune_at = 100

# Freeze all the layers before the `fine_tune_at` layer
for layer in base_model.layers[:fine_tune_at]:
    layer.trainable = False

model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              optimizer=tf.keras.optimizers.RMSprop(lr=base_learning_rate / 10),
              metrics=['accuracy'])

print('#    model.summary : {}'.format(model.summary()))
print('#    len(model.trainable_variables : {}'.format(len(model.trainable_variables)))

fine_tune_epochs = 10
total_epochs = initial_epochs + fine_tune_epochs

history_fine = model.fit(train_dataset,
                         epochs=total_epochs,
                         initial_epoch=history.epoch[-1],
                         validation_data=validation_dataset)
acc += history_fine.history['accuracy']
val_acc += history_fine.history['val_accuracy']

plt.figure(figsize=(8, 8))
plt.subplot(2, 1, 1)
plt.plot(acc, label='Training Accuracy')
plt.plot(val_acc, label='Validation Accuracy')
plt.ylim([0.8, 1])
plt.plot([initial_epochs-1,initial_epochs-1],
          plt.ylim(), label='Start Fine Tuning')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(2, 1, 2)
plt.plot(loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.ylim([0, 1.0])
plt.plot([initial_epochs-1,initial_epochs-1],
         plt.ylim(), label='Start Fine Tuning')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.xlabel('epoch')
plt.show()

loss += history_fine.history['loss']
val_loss += history_fine.history['val_loss']


print('# SAVING THE MODEL FOR LATER USE')
print("### Now will save model to path : {}".format(CONST_MODEL_PATH))
tf.saved_model.save(model, CONST_MODEL_PATH)
